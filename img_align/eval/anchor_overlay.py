"""図面アンカーの結果を図面に重ねて確認する (GTフィットを挟まない素の出力)

`eval/floor_overlay` は島をGTへ相似フィットしてから描くので「前段の形状が
どれだけ正しいか」しか見えない。こちらは **corner_anchor が出した図面px座標を
そのまま**描くので、**アンカーそのものの成否**が見える。

corner_anchor は `backend/corner_anchor_<tag>.json`
(frame名 → 図面ピクセル) を書くので、それを読むだけ。GTは参照用に薄く重ねる。

注意: floor_pick (全フロア試行) を走らせると corner_anchor_<tag>.json は
**最後に試した階**の結果で上書きされる。可視化前に正しい階で打ち直すこと。

使い方:
  IMG_SESSION=0912 python -m img_align.eval.anchor_overlay
出力: <物件>/results/<sess>/backend/anchor_overlay_<階>.png
"""
import argparse
import json

import numpy as np
from PIL import Image

from img_align.config import BENCH_DIR, OUT_DIR, PROP, SESSION
from img_align.eval.gt_traj import MM_PER_PX, load_gt

PAD_PX = 160
COLORS = ("#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e", "#17becf")


def load_track(tag):
    """corner_anchor の出力 → (frames, 図面px)"""
    p = OUT_DIR / f"corner_anchor_{tag}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    fr = np.array(sorted(int(k.replace("frame_", "")) for k in d))
    xy = np.array([d[f"frame_{f:06d}"] for f in fr])
    return fr, xy


def gt_xy(gt, floor):
    g = [v for _, v in sorted(gt.items()) if v["floor"] == floor]
    return np.array([[v["x_px"], v["y_px"]] for v in g])


def err_vs_gt(fr, xy, gt):
    e = [np.linalg.norm(xy[i] - [gt[f]["x_px"], gt[f]["y_px"]])
         for i, f in enumerate(fr) if f in gt]
    return float(np.median(e)) * MM_PER_PX / 1000 if len(e) >= 4 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="pv_full_manifest.json")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]

    gt = load_gt()
    man = json.loads((BENCH_DIR / args.manifest).read_text())
    by_floor = {}
    for m in man:
        if m.get("floor"):
            by_floor.setdefault(m["floor"], []).append(m["tag"])

    for floor, tags in sorted(by_floor.items()):
        bg = np.asarray(Image.open(PROP.zumen_dir / f"{floor}.jpg").convert("L"))
        fig, ax = plt.subplots(figsize=(13, 9))
        ax.imshow(bg, cmap="gray", vmin=0, vmax=255, alpha=0.38, zorder=0)
        g = gt_xy(gt, floor)
        ax.plot(g[:, 0], g[:, 1], "-", color="k", lw=4.0, alpha=0.28,
                label="GT (参考)", zorder=1)
        labels = []
        for i, tag in enumerate(tags):
            r = load_track(tag)
            if r is None:
                print(f"[skip] {tag}: corner_anchor_{tag}.json なし")
                continue
            fr, xy = r
            e = err_vs_gt(fr, xy, gt)
            ax.plot(xy[:, 0], xy[:, 1], "-o", ms=2.2, lw=1.4, zorder=4,
                    color=COLORS[i % len(COLORS)],
                    label=f"{tag} ({e:.2f}m)" if e else tag)
            labels.append(f"{tag} {e:.2f}m" if e else tag)
        allxy = np.vstack([g] + [load_track(t)[1] for t in tags
                                 if load_track(t) is not None])
        ax.set_xlim(max(0, allxy[:, 0].min() - PAD_PX),
                    min(bg.shape[1], allxy[:, 0].max() + PAD_PX))
        ax.set_ylim(min(bg.shape[0], allxy[:, 1].max() + PAD_PX),
                    max(0, allxy[:, 1].min() - PAD_PX))
        ax.set_axis_off()
        ax.legend(fontsize=9, loc="upper right", framealpha=0.85)
        ax.set_title(f"{SESSION} {floor} — 図面アンカーの素の出力 "
                     f"(GTフィット無し)   {' / '.join(labels)}", fontsize=11)
        out = OUT_DIR / f"anchor_overlay_{floor}.png"
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"-> {out}")


if __name__ == "__main__":
    main()
