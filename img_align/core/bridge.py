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

from img_align.config import BENCH_DIR, FRAMES_DIR, IMU_DIR, SESSION
from img_align.core.chunking import CHUNK_SIZE, imu_segments
from img_align.core.floor_assign import apply as apply_floor
from img_align.core.geometry import apply, compose, sim3_umeyama
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


def manifest_islands(name="pv_manifest.json"):
    """階が割り当たった島を開始フレーム順に返す"""
    man = apply_floor(json.loads((BENCH_DIR / name).read_text()), quiet=True)
    return sorted([m for m in man if m.get("floor")], key=lambda m: m["a"])


def islands_at(a_end, b_start):
    """切れ目の両側の島タグ (A側=終端が a_end 寄り / B側=始端が b_start 寄り)"""
    man = manifest_islands()
    a = [m for m in man if m["b"] <= a_end + 2]
    b = [m for m in man if m["a"] >= b_start - 2]
    if not a or not b:
        return None, None
    return max(a, key=lambda m: m["b"]), min(b, key=lambda m: m["a"])


def junction_kind(a_end, b_start):
    """切れ目の種類を返す ("階段" / "静止" / "分割" / "不明")

    島は①IMUの階段イベント ②静止 (ZUPT) ③--max-seg の等分割 で切れる。
    ③は**連続したフレームの人為的な分割**なので視覚的な重なりがあり、
    ①②とは橋の難易度が本質的に違う。混ぜて集計すると傾向が見えない。
    """
    gap = b_start - a_end - 1
    if gap <= 1:
        return "分割"
    tr = IMU_DIR / "transitions.json"
    if tr.exists():
        for e in json.loads(tr.read_text())["events"]:
            if int(e["frame_start"]) <= b_start and int(e["frame_end"]) >= a_end:
                return "階段"
    il = IMU_DIR / "idle_spans.json"
    if il.exists():
        for sp in json.loads(il.read_text())["spans"]:
            if int(sp["frame_start"]) <= b_start and \
                    int(sp["frame_end"]) >= a_end:
                return "静止"
    return "不明"


def bridge_transform(a_end, b_start, mode):
    """島Bゲージ → 島Aゲージ の Sim(3)。(T, タグ対)。失敗時は (None, None)

    橋を鎖状に合成して「群れ」を作るために、位置でなく変換を返す。
    """
    ma, mb = islands_at(a_end, b_start)
    if ma is None:
        return None, None
    ca, cb = _chain_poses(ma["tag"]), _chain_poses(mb["tag"])
    if mode.startswith("n") and "s" in mode:
        n, st = (int(v) for v in mode[1:].split("s"))
        pos, af, bf_ = run_chain_bridge(ma["tag"], mb["tag"], n, st)
        fa, fb = fit_side(pos, ca, af), fit_side(pos, cb, bf_)
    else:
        pos = run_bridge(a_end, b_start, mode)
        bf = sorted(pos)
        mid = (a_end + b_start) / 2
        fa = fit_side(pos, ca, [f for f in bf if f <= mid])
        fb = fit_side(pos, cb, [f for f in bf if f > mid])
    if fa is None or fb is None:
        return None, None
    sa, Ra, ta, _, _ = fa
    sb, Rb, tb, _, _ = fb
    inv = (1.0 / sb, Rb.T, -(Rb.T @ tb) / sb)
    return compose((sa, Ra, ta), inv), (ma["tag"], mb["tag"])


def transported(a_end, b_start, mode):
    """橋で島Bを島Aの座標系へ運んだ位置 (GT不要)。(位置(N,3), タグ対)"""
    ma, mb = islands_at(a_end, b_start)
    if ma is None:
        return None, None
    ca, cb = _chain_poses(ma["tag"]), _chain_poses(mb["tag"])
    if mode.startswith("n") and "s" in mode:
        n, st = (int(v) for v in mode[1:].split("s"))
        pos, af, bf_ = run_chain_bridge(ma["tag"], mb["tag"], n, st)
        fa, fb = fit_side(pos, ca, af), fit_side(pos, cb, bf_)
    else:
        pos = run_bridge(a_end, b_start, mode)
        bf = sorted(pos)
        mid = (a_end + b_start) / 2
        fa = fit_side(pos, ca, [f for f in bf if f <= mid])
        fb = fit_side(pos, cb, [f for f in bf if f > mid])
    if fa is None or fb is None:
        return None, None
    sa, Ra, ta, _, _ = fa
    sb, Rb, tb, _, _ = fb
    kb = np.stack([cb[f] for f in sorted(cb)])
    inv = (1.0 / sb, Rb.T, -(Rb.T @ tb) / sb)
    return apply((sa, Ra, ta), apply(inv, kb)), (ma["tag"], mb["tag"])


def consensus(a_end, b_start, modes, metric=True):
    """方式間で運び先がどれだけ一致するか。GTを使わない採否判定の候補

    **metric=True でメートルに正規化する**。島ゲージの尺度は物件で大きく違い
    (実測: 28-BBBBBが0.45〜0.49 m/unit に対し外部データは1.23〜1.60)、
    生の単位でしきい値を置くと**物件をまたいだ瞬間に意味が変わる**
    (2.5単位が1.2mにも4.0mにもなる)。

    橋の残差 (両側チェーンへのフィット) は**当てにならない** (2026-10-15実測:
    残差0.128/0.032で結合誤差4.30m)。橋が「両側には合うが相対関係が誤り」
    の状態を残差は検出できないため。一致度なら別の情報になりうる。
    """
    outs = []
    for m in modes:
        t, tags = transported(a_end, b_start, m)
        if t is not None:
            outs.append(t)
    if len(outs) < 2:
        return None
    n = min(len(o) for o in outs)
    d = []
    for i in range(len(outs)):
        for j in range(i + 1, len(outs)):
            d.append(np.median(np.linalg.norm(outs[i][:n] - outs[j][:n],
                                              axis=1)))
    v = float(np.median(d))
    if metric:
        ma, _ = islands_at(a_end, b_start)
        if ma is not None:
            from img_align.postproc.scale_prior import m_per_unit
            try:
                v *= m_per_unit(ma["tag"])[0]
            except Exception:
                pass
    return v



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
