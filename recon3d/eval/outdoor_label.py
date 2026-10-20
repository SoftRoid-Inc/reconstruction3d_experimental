"""別リポジトリ `outdoor_classification` のフレーム判定を島単位に集計する

`eval/outdoor_proxy` (天頂の空の割合) は**外周撮影と庇つきテラスを拾えない**
(2026-10-16実測: 外周島 s001r2=0.20、テラス s02=0.00 を「室内」と誤判定)。
空は原理的に不十分で、建築中の躯体内部は空が見えるのに屋内、軒下・ベランダは
頭上が塞がっているのに屋外になる。

`outdoor_classification` は**囲繞性** (水平全方位が構造物で囲まれているか) を
主軸にしており、ERPだからこそ計算できる特徴で上の両方を正しく扱う。
こちらを正とし、proxy は使わない。

島の判定は**フレームの屋外率**で行う。島は数十〜百数十フレームあり、出入口
付近では混ざるので、全部屋外である必要はない (OUTDOOR_FRAC を超えたら除外)。

使い方:
  IMG_SESSION=0925 python -m recon3d.eval.outdoor_label
  IMG_SESSION=0925 python -m recon3d.eval.outdoor_label --frac 0.3
"""
import argparse
import json
from pathlib import Path

import numpy as np

from recon3d.config import BENCH_DIR, OUT_DIR, SESSION

OC_RESULTS = Path("/mnt/data/programming/outdoor_classification/results")
OUTDOOR_FRAC = 0.5     # 島のフレームのこの割合超が屋外なら島ごと除外


def frame_labels(session=None):
    """{frame_idx: ("indoor"|"outdoor", p_outdoor)}。無ければ None"""
    p = OC_RESULTS / (session or SESSION) / "prediction.jsonl"
    if not p.exists():
        return None
    out = {}
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        out[int(r["frame_idx"])] = (r["label"], float(r.get("p_outdoor", 0.0)))
    return out


def island_outdoor(manifest="pv_manifest.json", session=None):
    """島ごとの屋外率 [{tag, n, frac, p_med, outdoor}]"""
    lab = frame_labels(session)
    if lab is None:
        return None
    man = json.loads((BENCH_DIR / manifest).read_text())
    rows = []
    for m in man:
        cam = BENCH_DIR / m["tag"] / "global_cameras.npz"
        if cam.exists():     # チェーンに実在するフレームで数える
            fr = [int(str(n).replace("frame_", ""))
                  for n in np.load(cam)["names"]]
        else:
            fr = list(range(m["a"], m["b"] + 1))
        v = [lab[f] for f in fr if f in lab]
        if not v:
            continue
        frac = sum(1 for l, _ in v if l == "outdoor") / len(v)
        rows.append({"tag": m["tag"], "floor": m.get("floor"), "n": len(v),
                     "frac": frac,
                     "p_med": float(np.median([p for _, p in v])),
                     "outdoor": frac > OUTDOOR_FRAC})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="pv_manifest.json")
    ap.add_argument("--frac", type=float, default=OUTDOOR_FRAC)
    args = ap.parse_args()
    rows = island_outdoor(args.manifest)
    if rows is None:
        print(f"{SESSION}: outdoor_classification の結果が無い "
              f"({OC_RESULTS / SESSION})")
        return
    print(f"{'island':20s} {'枚数':>5s} {'屋外率':>7s} {'p中央':>7s}  判定")
    for r in rows:
        r["outdoor"] = r["frac"] > args.frac
        print(f"{r['tag']:20s} {r['n']:5d} {r['frac']:7.1%} {r['p_med']:7.3f}"
              f"  {'屋外' if r['outdoor'] else '室内'}")
    p = OUT_DIR / "outdoor_label.json"
    p.write_text(json.dumps({"frac_threshold": args.frac, "islands": rows},
                            ensure_ascii=False, indent=1))
    print(f"-> {p}")


if __name__ == "__main__":
    main()
