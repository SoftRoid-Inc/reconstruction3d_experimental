"""島の切れ目を跨ぐ「橋渡しチャンク」で、島間の相対Sim(3)を1発で出す

島は階段・静止で切るため共有フレームを持たず、幾何的に完全独立になる。
従来の逐次縫合は切れ目で破綻する:
  **チャンク縫合は階段区間で破綻** — 共有カメラが一直線に並ぶため回転が不定
  (重力拘束+ヨーのみのUmeyamaでも階段部の共有フレーム自体の姿勢が不一致)

橋渡しチャンクは**逐次縫合を介さない**。切れ目の両側のフレームを1チャンクに
混ぜてPanoVGGTに1度通すと、両側の姿勢が同一ゲージで出る。あとは各側を
元の島のチェーンにSim(3)フィットすれば、島Aゲージ→島Bゲージの変換が決まる。

**既知のリスク**: 疎サンプリング一括推論は破綻する実績がある
(代表16フレーム・30間隔では姿勢推定が崩壊)。本モジュールの span 方式は
stride 2〜4 なので条件が違うが、成立するかは実測で確かめること。
そのための3方式:
  ends  : 切れ目の手前8枚 + 直後8枚 (階段区間そのものは飛ばす)
  span  : 切れ目を跨いで等間隔サンプル (階段区間も含む。stride 2〜4)
  both  : 両端を厚めに取りつつ中間も拾う

使い方:
  IMG_SESSION=0912 python -m img_align.core.bridge --list
  IMG_SESSION=0912 python -m img_align.core.bridge --junction 237:278 --mode span
"""
import argparse
import json

import numpy as np

from img_align.config import FRAMES_DIR, SESSION
from img_align.core.chaining import load_chunk
from img_align.core.chunking import CHUNK_SIZE, imu_segments
from img_align.core.geometry import sim3_umeyama, apply
from img_align.core.panovggt import PCD_DIR, run_panovggt

BRIDGE_DIR = PCD_DIR / "bridges"


def n_frames():
    return len(sorted(FRAMES_DIR.glob("frame_number_*.jpg")))


def junctions(segs=None):
    """島の切れ目 [(a_end, b_start)]。segs は imu_segments の出力"""
    segs = segs if segs is not None else imu_segments(n_frames())
    return [(segs[i][2], segs[i + 1][1]) for i in range(len(segs) - 1)]


def bridge_frames(a_end, b_start, mode="span", size=CHUNK_SIZE):
    """橋渡しチャンクに使うフレーム番号列 (昇順、重複なし)

    ends系の stride を上げると、同じ枚数でも**基線が伸びて姿勢が決まりやすい**。
    連続8枚 (=4秒歩行) はほぼ直線で、島チェーンへのフィットが悪条件になる。
    """
    half = size // 2
    if mode.startswith("ends") and len(mode) > 4:
        st = int(mode[4:])
        fr = list(range(a_end - (half - 1) * st, a_end + 1, st)) + \
             list(range(b_start, b_start + half * st, st))
    elif mode == "ends":
        fr = list(range(a_end - half + 1, a_end + 1)) + \
             list(range(b_start, b_start + half))
    elif mode == "span":
        lo, hi = a_end - half + 1, b_start + half - 1
        fr = np.linspace(lo, hi, size).round().astype(int).tolist()
    elif mode == "both":
        q = size // 4
        mid = np.linspace(a_end + 1, b_start - 1, size - 2 * q)
        fr = (list(range(a_end - q + 1, a_end + 1))
              + mid.round().astype(int).tolist()
              + list(range(b_start, b_start + q)))
    else:
        raise ValueError(f"未知のmode: {mode}")
    return sorted({f for f in fr if 0 <= f < n_frames()})


def chain_end_frames(tag, side, n=8, stride=2):
    """島チェーンの端から n 枚 (stride刻み、昇順)

    橋は**2本のチェーンを繋ぐ**ものなので、フレームはチェーンに実在する
    ものから選ぶ。島の宣言区間 [a,b] から機械的に取ると、チャンク被覆の
    都合で端がチェーンに無く、フィットできずに橋が成立しない
    (2026-10-15: 11切れ目中7つがこれで評価不能だった)。
    side: "a"=終端側から遡る / "b"=始端側から進む
    """
    fr = sorted(_chain_poses(tag))
    sel = fr[::-1][:n * stride:stride] if side == "a" else fr[:n * stride:stride]
    return sorted(sel)


def _chain_poses(tag):
    """島チェーンの {frame: カメラ中心(3,)}"""
    from img_align.config import BENCH_DIR
    d = BENCH_DIR / tag
    cam = np.load(d / "global_cameras.npz")
    names = [str(n) for n in cam["names"]]
    pos = cam["poses"][:, :3, 3]
    return {int(n.replace("frame_", "")): pos[i] for i, n in enumerate(names)}


def fit_side(bridge_pos, chain_pos, frames):
    """橋の姿勢を島チェーンにSim(3)フィット。(s, R, t, 残差m, 使用数)"""
    common = [f for f in frames if f in chain_pos and f in bridge_pos]
    if len(common) < 4:
        return None
    src = np.stack([bridge_pos[f] for f in common])
    dst = np.stack([chain_pos[f] for f in common])
    s, R, t = sim3_umeyama(src, dst)
    resid = float(np.median(np.linalg.norm(
        apply((s, R, t), src) - dst, axis=1)))
    return s, R, t, resid, len(common)


def run_frames(frames, name, use_mask=True):
    """指定フレーム集合を1チャンクとして推論し {frame: カメラ中心} を返す"""
    out = BRIDGE_DIR / f"{SESSION}_{name}"
    if not (out / "cameras.npz").exists():
        entries = [{"index": f,
                    "path": str(FRAMES_DIR / f"frame_number_{f:06d}.jpg")}
                   for f in frames
                   if (FRAMES_DIR / f"frame_number_{f:06d}.jpg").exists()]
        run_panovggt(entries, out, use_mask=use_mask)
    cam = np.load(out / "cameras.npz")
    names = [str(n) for n in cam["names"]]
    pos = cam["poses"][:, :3, 3]
    return {int(n.replace("frame_", "")): pos[i] for i, n in enumerate(names)}


def run_chain_bridge(tag_a, tag_b, n=8, stride=2, use_mask=True):
    """2つの島チェーンの端どうしを1チャンクで繋ぐ (推奨経路)"""
    fa = chain_end_frames(tag_a, "a", n, stride)
    fb = chain_end_frames(tag_b, "b", n, stride)
    name = f"{tag_a}__{tag_b}__n{n}s{stride}"
    return run_frames(sorted(set(fa + fb)), name, use_mask), fa, fb


def run_bridge(a_end, b_start, mode="span", use_mask=True):
    """橋渡しチャンクを推論して {frame: カメラ中心} を返す (既存なら再利用)"""
    frames = bridge_frames(a_end, b_start, mode)
    out = BRIDGE_DIR / f"{SESSION}_{a_end}_{b_start}_{mode}"
    if not (out / "cameras.npz").exists():
        entries = [{"index": f,
                    "path": str(FRAMES_DIR / f"frame_number_{f:06d}.jpg")}
                   for f in frames]
        entries = [e for e in entries if FRAMES_DIR.joinpath(
            f"frame_number_{e['index']:06d}.jpg").exists()]
        run_panovggt(entries, out, use_mask=use_mask)
    cam = np.load(out / "cameras.npz")
    names = [str(n) for n in cam["names"]]
    pos = cam["poses"][:, :3, 3]
    return {int(n.replace("frame_", "")): pos[i] for i, n in enumerate(names)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="切れ目を列挙して終了")
    ap.add_argument("--junction", help="a_end:b_start (省略時は全切れ目)")
    ap.add_argument("--mode", default="span",
                    help="ends / ends2 / ends4 (端をstride刻みで拾う) / span / both")
    ap.add_argument("--no-mask", action="store_true")
    args = ap.parse_args()

    js = ([tuple(int(v) for v in args.junction.split(":"))] if args.junction
          else junctions())
    if args.list:
        for a, b in js:
            print(f"  切れ目 {a} -> {b} (間隔 {b - a - 1}フレーム)")
            for m in ("ends", "ends2", "ends4", "span", "both"):
                print(f"    {m:5s}: {bridge_frames(a, b, m)}")
        return
    for a, b in js:
        print(f"\n=== 切れ目 {a} -> {b} ({args.mode})")
        pos = run_bridge(a, b, args.mode, use_mask=not args.no_mask)
        print(f"  橋 {len(pos)}フレーム: {sorted(pos)}")


if __name__ == "__main__":
    main()
