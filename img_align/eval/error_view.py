"""誤差の大きい島を図面上で強調し、密度地図をラベルマップに重ねて原因を見る

アンカーの成否は島ごとにばらつく (2026-10-14: 20島中13本成功)。失敗島が
「どこに貼り付いたか」と「密度地図が図面のどの壁と対応しそうか」を並べて
見られるようにする。数値だけでは原因が分からないため。

2枚組で出力する:
  1. **軌跡図** — 図面 + GT + 各島のアンカー結果。成功島は細い灰青、
     **失敗島は太い赤系で強調**し、GT残差をラベルに出す
  2. **密度 × ラベルマップ** — 島の壁密度 (赤) を labelme のラベル
     (wall/column/door/window) に重ねる。アンカー変換を当てた後の位置で描くので、
     **どの壁と対応が取れていないか**が直接見える

使い方:
  IMG_SESSION=0925 python -m img_align.eval.error_view
  IMG_SESSION=0925 python -m img_align.eval.error_view --err-m 1.0
出力: backend/error_view_<階>.png / error_density_<島>.png
"""
import argparse
import json
import math

import numpy as np
from PIL import Image

from img_align.config import BENCH_DIR, OUT_DIR, PROP, SESSION
from img_align.eval.gt_traj import MM_PER_PX, load_gt

PAD_PX = 180
LABEL_COLOR = {"wall": "#1f77b4", "column": "#9467bd",
               "door": "#2ca02c", "window": "#ff7f0e", "boundary": "#7f7f7f"}
BAD_COLORS = ("#d62728", "#e377c2", "#ff7f0e", "#8c564b", "#e31a1c")


def load_track(tag):
    p = OUT_DIR / f"corner_anchor_{tag}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    fr = np.array(sorted(int(k.replace("frame_", "")) for k in d))
    return fr, np.array([d[f"frame_{f:06d}"] for f in fr])


def gt_err(fr, xy, gt):
    e = [math.dist(p, (gt[f]["x_px"], gt[f]["y_px"]))
         for f, p in zip(fr, xy) if f in gt]
    return float(np.median(e)) * MM_PER_PX / 1000 if len(e) >= 4 else None


def plan_shapes(floor):
    """labelme のラベル矩形 → {label: [(x1,y1,x2,y2)]}"""
    p = PROP.zumen_dir / f"{floor}.json"
    if not p.exists():
        return {}
    out = {}
    for s in json.loads(p.read_text()).get("shapes", []):
        (x1, y1), (x2, y2) = s["points"][0], s["points"][1]
        out.setdefault(s["label"], []).append(
            (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
    return out


def draw_labels(ax, floor, lw=1.6, alpha=0.9):
    """ラベルマップ (wall/column/door/window) を描く"""
    from matplotlib.patches import Rectangle
    shown = set()
    for lab, rects in plan_shapes(floor).items():
        c = LABEL_COLOR.get(lab, "#333333")
        for x1, y1, x2, y2 in rects:
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                   edgecolor=c, lw=lw, alpha=alpha, zorder=3,
                                   label=lab if lab not in shown else None))
            shown.add(lab)


def island_density_px(tag, floor):
    """島の壁密度セルを、アンカー変換で図面ピクセルへ落とす"""
    from img_align.core.plan_grid import PlanGrid
    from img_align.postproc.anchor import prepare_island
    from img_align.postproc.corners import DENS_THR
    from img_align.postproc.plan_icp import cells_to_px, transform_pts
    a = OUT_DIR / f"anchor_{tag}_corners.json"
    if not a.exists():
        return None
    an = json.loads(a.read_text())
    d = prepare_island(tag, mpu=an["m_per_unit"])
    zz, xx = np.nonzero(d["density"] > DENS_THR)
    pts = np.stack([zz, xx], axis=1).astype(np.float64)
    pts = pts[:: max(1, len(pts) // 40000)]
    moved = transform_pts(pts, an["s"], an["th"], np.array(an["t"]))
    return cells_to_px(moved, PlanGrid(floor))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="pv_manifest.json")
    ap.add_argument("--err-m", type=float, default=1.0, help="これ超を失敗とする")
    ap.add_argument("--no-density", action="store_true")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]

    gt = load_gt()
    man = json.loads((BENCH_DIR / args.manifest).read_text())
    by_floor = {}
    for m in man:
        if not m.get("floor"):
            continue
        r = load_track(m["tag"])
        if r is None:
            continue
        fr, xy = r
        by_floor.setdefault(m["floor"], []).append(
            {"tag": m["tag"], "fr": fr, "xy": xy, "err": gt_err(fr, xy, gt),
             "n": m["b"] - m["a"] + 1})

    for floor, isl in sorted(by_floor.items()):
        bg = np.asarray(Image.open(PROP.zumen_dir / f"{floor}.jpg").convert("L"))
        fig, ax = plt.subplots(figsize=(13, 9))
        ax.imshow(bg, cmap="gray", vmin=0, vmax=255, alpha=0.30, zorder=0)
        draw_labels(ax, floor, lw=1.2, alpha=0.55)
        g = np.array([[v["x_px"], v["y_px"]] for _, v in sorted(gt.items())
                      if v["floor"] == floor])
        ax.plot(g[:, 0], g[:, 1], "-", color="k", lw=4.0, alpha=0.25,
                label="GT", zorder=2)
        nbad = 0
        pts_all = [g]
        for i in sorted(isl, key=lambda d: (d["err"] or 0)):
            bad = i["err"] is not None and i["err"] > args.err_m
            col = BAD_COLORS[nbad % len(BAD_COLORS)] if bad else "#4c72b0"
            nbad += bad
            ax.plot(i["xy"][:, 0], i["xy"][:, 1], "-",
                    lw=2.6 if bad else 1.1, alpha=0.95 if bad else 0.55,
                    color=col, zorder=6 if bad else 4,
                    label=f"{'✗' if bad else '✓'} {i['tag'].split('_')[1]} "
                          f"{i['err']:.2f}m ({i['n']}f)")
            pts_all.append(i["xy"])
        allp = np.vstack(pts_all)
        ax.set_xlim(max(0, allp[:, 0].min() - PAD_PX),
                    min(bg.shape[1], allp[:, 0].max() + PAD_PX))
        ax.set_ylim(min(bg.shape[0], allp[:, 1].max() + PAD_PX),
                    max(0, allp[:, 1].min() - PAD_PX))
        ax.set_axis_off()
        ax.legend(fontsize=8, loc="upper right", framealpha=0.88, ncol=1)
        ax.set_title(f"{SESSION} {floor} — アンカー結果 "
                     f"(赤系=GT残差 {args.err_m}m超)  細線/青=成功", fontsize=11)
        out = OUT_DIR / f"error_view_{floor}.png"
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"-> {out}  失敗{nbad}/{len(isl)}島")

        if args.no_density:
            continue
        bad_isl = [i for i in isl
                   if i["err"] is not None and i["err"] > args.err_m]
        for i in bad_isl:
            px = island_density_px(i["tag"], floor)
            if px is None:
                continue
            fig, ax = plt.subplots(figsize=(12, 8.5))
            ax.imshow(bg, cmap="gray", vmin=0, vmax=255, alpha=0.25, zorder=0)
            draw_labels(ax, floor, lw=1.8, alpha=0.95)
            ax.scatter(px[:, 0], px[:, 1], s=0.6, c="#d62728", alpha=0.45,
                       zorder=5, label="島の壁密度")
            ax.plot(i["xy"][:, 0], i["xy"][:, 1], "-", lw=1.6, color="#d62728",
                    zorder=6, label="アンカー軌跡")
            ax.plot(g[:, 0], g[:, 1], "-", color="k", lw=3.0, alpha=0.25,
                    zorder=2, label="GT")
            p2 = np.vstack([px, i["xy"], g])
            ax.set_xlim(max(0, p2[:, 0].min() - PAD_PX),
                        min(bg.shape[1], p2[:, 0].max() + PAD_PX))
            ax.set_ylim(min(bg.shape[0], p2[:, 1].max() + PAD_PX),
                        max(0, p2[:, 1].min() - PAD_PX))
            ax.set_axis_off()
            ax.legend(fontsize=9, loc="upper right", framealpha=0.88)
            ax.set_title(f"{SESSION} {i['tag']} ({floor}) — "
                         f"密度 × ラベルマップ  GT残差 {i['err']:.2f}m",
                         fontsize=11)
            out = OUT_DIR / f"error_density_{i['tag']}.png"
            fig.savefig(out, dpi=140, bbox_inches="tight")
            plt.close(fig)
            print(f"-> {out}")


if __name__ == "__main__":
    main()
