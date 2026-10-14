"""階遷移の列から各島の階を決める (GTフリー、絶対階は建物の階数で決まる)

従来 `postproc/floor_pick` は島ごとに全フロアの図面と照合してスコア最良を採る
方式だったが、階の間のスコア差が0.012〜0.036しかなく判別できなかった
(2026-10-13実測、0912で2/3誤答)。

ここは**別の立て方**をする:
  1. stairnet (imu/transitions) が階遷移を検出する — 再現率は実測100% (6/6)
  2. 過検出 (脚立・ステップ台) は**鉛直変位**で落とす。実測で真の遷移は
     |Δ|≥1.98、過検出は≤0.244 と8倍離れており閾値に迷いがない
  3. up/down を積算すると**相対的な階**が決まる (絶対階は未知の定数だけずれる)
  4. 訪問した相対階の**幅**が建物の階数と一致すれば、絶対階は一意に決まる
     (例: 3階建てで3レベル訪問 → 置き方は1通り)

つまり残る未知数はセッションあたり1つ (開始階) だけで、しかも多くの場合は
建物の階数が決めてくれる。島ごとの判別より桁違いに制約が強い。

注意: 鉛直変位の**絶対値は使えない** (逐次連結でスケールがドリフトし、
1階分が1.98〜4.06とばらつく)。符号と閾値超えだけが使える情報。

使い方:
  IMG_SESSION=0912 python -m img_align.core.floor_seq
出力: results/<s>/imu/floor_seq.json (フレーム範囲→相対階/絶対階)
"""
import argparse
import json

import numpy as np

from img_align.config import IMU_DIR, PROP, RESULTS_DIR

MIN_RISE = 0.8      # これ未満の鉛直変位は往復/誤検出とみなす (島単位。実測0.24 vs 1.98)
PAD = 25            # イベント前後で高さを測る窓 (フレーム)


def vertical_profile(chunks_dir="chunks"):
    """全チャンクを逐次連結した鉛直プロファイル (frames, 高さ[島単位・上が正])

    階段チャンクも含めて連結する。水平はドリフトしても、階の段差は桁違いに
    大きいので符号判定には足りる。
    """
    from img_align.core.chaining import chain, load_chunk
    from img_align.core.geometry import apply, rotation_to_y
    pcd = RESULTS_DIR / "pcd" / chunks_dir
    cdirs = sorted(pcd.glob("chunk_*"), key=lambda p: int(p.name.split("_")[1]))
    chunks = [load_chunk(c) for c in cdirs]
    _, T, _ = chain(cdirs)
    ups = []
    for _, _, poses, _, _ in chunks:
        u = poses[:, :3, 1]
        ups.append(np.median(u / np.linalg.norm(u, axis=1, keepdims=True), 0))
    g = np.median(np.stack(ups), axis=0)
    rg = rotation_to_y(g / np.linalg.norm(g))
    fy = {}
    for k, (names, pos, _, _, unt) in enumerate(chunks):
        gp = apply(T[k], pos) @ rg.T
        for i, nm in enumerate(names):
            if nm not in unt:
                fy.setdefault(int(nm.replace("frame_", "")), []).append(gp[i, 1])
    fr = np.array(sorted(fy))
    # PanoVGGTのY+は物理的に下 → 反転して「上が正」
    return fr, -np.array([np.mean(fy[f]) for f in fr])


def real_transitions(chunks_dir="chunks", min_rise=MIN_RISE):
    """IMUイベントのうち、鉛直変位を伴う**真の階遷移**だけを返す

    [{frame_start, frame_end, dy, step}] (step=+1/-1)
    """
    ev = json.loads((IMU_DIR / "transitions.json").read_text())["events"]
    fr, y = vertical_profile(chunks_dir)
    out = []
    for e in ev:
        a, b = int(e["frame_start"]), int(e["frame_end"])
        pre = y[(fr >= a - PAD) & (fr < a)]
        post = y[(fr > b) & (fr <= b + PAD)]
        if len(pre) < 3 or len(post) < 3:
            continue
        dy = float(np.median(post) - np.median(pre))
        if abs(dy) < min_rise:
            continue
        out.append({"frame_start": a, "frame_end": b, "dy": round(dy, 3),
                    "step": 1 if dy > 0 else -1})
    return out


def relative_levels(n_frames, trans):
    """真の遷移列 → 区間ごとの相対階 [(a, b, level)] (開始を0とする)"""
    segs, lvl, prev = [], 0, 0
    for t in sorted(trans, key=lambda t: t["frame_start"]):
        if t["frame_start"] - 1 >= prev:
            segs.append((prev, t["frame_start"] - 1, lvl))
        lvl += t["step"]
        prev = t["frame_end"] + 1
    if prev <= n_frames - 1:
        segs.append((prev, n_frames - 1, lvl))
    return segs


def assign_floors(segs, floor_names):
    """相対階 → 絶対階。訪問した幅が建物の階数と一致すれば一意に決まる

    floor_names: 下から順の階名 (例 ["1F","2F","3F"])
    返り値: (区間+階名, 一意に決まったか, 候補オフセット一覧)
    """
    lv = sorted({s[2] for s in segs})
    span = lv[-1] - lv[0] + 1
    n = len(floor_names)
    # オフセット = 最下レベルを何階に置くか
    offsets = [o for o in range(n - span + 1)]
    unique = len(offsets) == 1
    off = offsets[0] if offsets else 0
    out = [{"a": a, "b": b, "level": l,
            "floor": floor_names[l - lv[0] + off]
            if 0 <= l - lv[0] + off < n else None}
           for a, b, l in segs]
    return out, unique, offsets


def floor_names():
    """図面がある階を下から順に (1F, 2F, 3F…)"""
    names = sorted(p.stem for p in PROP.zumen_dir.glob("*.jpg"))
    return sorted(names, key=lambda s: int("".join(c for c in s if c.isdigit())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks-dir", default="chunks")
    ap.add_argument("--min-rise", type=float, default=MIN_RISE)
    args = ap.parse_args()
    from img_align.config import FRAMES_DIR
    n = len(sorted(FRAMES_DIR.glob("frame_number_*.jpg")))
    trans = real_transitions(args.chunks_dir, args.min_rise)
    print(f"真の階遷移 {len(trans)}件:")
    for t in trans:
        print(f"  f{t['frame_start']:4d}-{t['frame_end']:<4d} "
              f"dy={t['dy']:+.2f} step={t['step']:+d}")
    segs = relative_levels(n, trans)
    names = floor_names()
    out, unique, offsets = assign_floors(segs, names)
    print(f"\n建物の階 {names} / 訪問レベル幅 "
          f"{max(s[2] for s in segs) - min(s[2] for s in segs) + 1}")
    print(f"絶対階の置き方: {len(offsets)}通り "
          f"{'→ 一意に決まる' if unique else '→ 曖昧 (開始階が未知)'}")
    for s in out:
        print(f"  f{s['a']:4d}-{s['b']:<4d} level {s['level']:+d} -> {s['floor']}")
    (IMU_DIR / "floor_seq.json").write_text(json.dumps(
        {"transitions": trans, "segments": out, "unique": unique,
         "n_offsets": len(offsets), "floor_names": names},
        indent=1, ensure_ascii=False))
    print(f"-> {IMU_DIR / 'floor_seq.json'}")


if __name__ == "__main__":
    main()
