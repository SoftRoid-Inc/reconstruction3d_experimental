"""仮想2Dスキャンを間取り図(GT)に照合してフレーム位置(x, y, θ)を推定する (スパイク実装)

図面側: recon3d.loader のGTパースと実測スケール(9100mm校正)を再利用し、
壁+柱の距離場と開口(窓∪扉)の距離場を作る。
照合: (x, y, θ) のグリッドサーチ。スキャン端点が壁/開口の近くに落ちるほど高スコア
(ガウシアンカーネル)。1F/2F両方でスコアを出し、高い方の階を採用する。
"""
import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage

from recon3d.calibrate import scale_mm_per_px
from recon3d.loader import load_labelme

from img_align.config import PROP, RESULTS_DIR
from img_align.legacy.sample_frames import load_selected
from img_align.legacy.scan import TAG_OPENING

LOC_DIR = RESULTS_DIR / "loc"
from img_align.core.plan_grid import (KNOWN_WIDTH_MM, PlanGrid,
                                      RES_M, ZUMEN_DIR)  # noqa: E402,F401

THETA_STEP_DEG = 4      # 向きの探索刻み
SIGMA_WALL_M = 0.20     # 壁一致カーネル幅
SIGMA_OPEN_M = 0.30     # 開口一致カーネル幅
FLOOR_RAY_W = 0.6       # 床由来レイの重み (家具バイアスがあるため低め)
CEIL_RAY_W = 1.0        # 天井由来レイの重み
MAX_DIST_M = 10.0       # これ以上の距離のレイは捨てる
PASS_FRACS = (0.4, 0.7)  # レイ透過チェックの中間点 (端点までの割合)
PASS_PENALTY = 0.8      # 中間点が壁の中にある場合のペナルティ係数
def shifted(k, dy, dx):
    """out[y,x] = k[y+dy, x+dx] (はみ出しは0)"""
    h, w = k.shape
    out = np.zeros_like(k)
    ys0, ys1 = max(0, -dy), min(h, h - dy)
    xs0, xs1 = max(0, -dx), min(w, w - dx)
    if ys1 > ys0 and xs1 > xs0:
        out[ys0:ys1, xs0:xs1] = k[ys0 + dy: ys1 + dy, xs0 + dx: xs1 + dx]
    return out


def frame_points(scans, name):
    """スキャンnpzから (az, dist, is_opening, weight) の点列を作る"""
    pts = []
    for suffix, w in (("floor", FLOOR_RAY_W), ("ceil", CEIL_RAY_W)):
        az = scans[f"{name}_az"]
        dist = scans[f"{name}_dist_{suffix}"]
        tag = scans[f"{name}_tag_{suffix}"]
        m = ~np.isnan(dist) & (dist < MAX_DIST_M)
        pts.append(np.stack([az[m], dist[m],
                             (tag[m] == TAG_OPENING).astype(float),
                             np.full(m.sum(), w)], axis=1))
    return np.concatenate(pts, axis=0)


def localize(grid, pts):
    """グリッドサーチ。返り値: (best_score, cy, cx, theta_deg, score_map)"""
    best_score = np.zeros((grid.ny, grid.nx))
    best_theta = np.zeros((grid.ny, grid.nx))
    total_w = pts[:, 3].sum()
    for theta_deg in range(0, 360, THETA_STEP_DEG):
        theta = math.radians(theta_deg)
        ang = pts[:, 0] + theta
        dx = np.rint(pts[:, 1] * np.cos(ang) / RES_M).astype(int)
        dy = np.rint(pts[:, 1] * np.sin(ang) / RES_M).astype(int)
        acc = np.zeros((grid.ny, grid.nx))
        # 同一オフセット・同一タグのレイをまとめて shift 回数を減らす
        keys = {}
        for i in range(len(pts)):
            k = (dy[i], dx[i], int(pts[i, 2]))
            keys[k] = keys.get(k, 0.0) + pts[i, 3]
        for (oy, ox, is_open), w in keys.items():
            kmap = grid.k_open if is_open else grid.k_wall
            acc += w * shifted(kmap, oy, ox)
            # 自由空間整合: レイの中間点が壁の中なら減点 (壁の透過を禁止)
            for frac in PASS_FRACS:
                acc -= (w * PASS_PENALTY / len(PASS_FRACS)) * shifted(
                    grid.solid, round(oy * frac), round(ox * frac))
        score = np.maximum(acc, 0.0) / total_w
        upd = score > best_score
        best_score[upd] = score[upd]
        best_theta[upd] = theta_deg
    best_score[~grid.valid_pose] = 0.0
    cy, cx = np.unravel_index(np.argmax(best_score), best_score.shape)
    return best_score[cy, cx], cy, cx, best_theta[cy, cx], best_score


def draw_result(grid, bg, score_map, cy, cx, theta_deg, path, title):
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.imshow(bg, cmap="gray")
    extent = (grid.x0_px, grid.x0_px + grid.nx * grid.cell_px,
              grid.y0_px + grid.ny * grid.cell_px, grid.y0_px)
    ax.imshow(score_map, cmap="hot", alpha=0.55, extent=extent,
              vmin=0, vmax=max(score_map.max(), 1e-6))
    px, py = grid.cell_to_px(cx, cy)
    arrow = 800 * 0.05 / grid.cell_px  # ≈0.8m の矢印
    ax.plot(px, py, "o", color="lime", markersize=8)
    ax.arrow(px, py,
             arrow * math.cos(math.radians(theta_deg)),
             arrow * math.sin(math.radians(theta_deg)),
             color="lime", width=4, head_width=25)
    ax.set_title(title)
    ax.set_axis_off()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="",
                    help="例 '_gt': scan_gt を読み loc_gt へ出力")
    args = ap.parse_args()
    scan_dir = RESULTS_DIR / f"scan{args.variant}"
    loc_dir = RESULTS_DIR / f"loc{args.variant}"

    loc_dir.mkdir(parents=True, exist_ok=True)
    grids = {f: PlanGrid(f) for f in ("1F", "2F")}
    bgs = {f: np.asarray(Image.open(ZUMEN_DIR / f"{f}.jpg").convert("L"))
           for f in grids}
    scans = np.load(scan_dir / "scans.npz")

    summary = {}
    for entry in load_selected():
        name = f"frame_{entry['index']:06d}"
        pts = frame_points(scans, name)
        if len(pts) < 30:
            summary[name] = {"status": "too_few_rays", "n_rays": len(pts)}
            continue
        results = {f: localize(g, pts) for f, g in grids.items()}
        floor = max(results, key=lambda f: results[f][0])
        other_score = min(results[f][0] for f in results)
        score, cy, cx, theta, smap = results[floor]
        g = grids[floor]
        px, py = g.cell_to_px(cx, cy)
        draw_result(
            g, bgs[floor], smap, cy, cx, theta,
            loc_dir / f"{name}_{floor}.jpg",
            f"{name}  floor={floor}  score={score:.3f}  "
            f"(other={other_score:.3f})",
        )
        summary[name] = {
            "floor": floor,
            "score": round(float(score), 4),
            "score_other_floor": round(float(other_score), 4),
            "x_px": round(float(px), 1),
            "y_px": round(float(py), 1),
            "theta_deg": float(theta),
            "n_rays": len(pts),
        }
        print(name, summary[name])

    (loc_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # 軌跡図: 階ごとに、採用されたフレーム位置を撮影順に結ぶ
    for floor, g in grids.items():
        pts_xy = [(s["x_px"], s["y_px"], n) for n, s in summary.items()
                  if s.get("floor") == floor]
        if not pts_xy:
            continue
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.imshow(bgs[floor], cmap="gray")
        xs, ys, names = zip(*pts_xy)
        ax.plot(xs, ys, "-o", color="tab:red", linewidth=1.5)
        for x, y, n in pts_xy:
            ax.annotate(n.replace("frame_0000", "").replace("frame_000", ""),
                        (x, y), fontsize=8, color="blue")
        ax.set_title(f"trajectory candidates on {floor}")
        ax.set_axis_off()
        fig.savefig(loc_dir / f"trajectory_{floor}.jpg", dpi=110,
                    bbox_inches="tight")
        plt.close(fig)
    print(f"done -> {LOC_DIR}")


if __name__ == "__main__":
    main()
