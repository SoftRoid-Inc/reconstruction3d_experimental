"""壁密度マップをGT図面へ相似変換で照合する (SceneAligner-lite 後段)

探索: 回転(3°刻み) × スケール(±30%対数) × 並進(FFT相互相関で一括)。
スコア: 壁カーネル一致 - 屋内自由空間に密度が落ちるペナルティ(ゼロ平均化)。
1F/2F両方でスコアし高い方を採用。カメラ位置列も同じ変換で図面に載せ、
撮影軌跡として描画する。

出力: results/pcd/align_{1F,2F}.jpg / trajectory_pano.jpg / align_summary.json
"""
import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.signal import fftconvolve

from recon3d.core.plan_grid import PlanGrid

THETA_STEP_DEG = 3
SCALES = np.geomspace(0.5, 1.1, 12)   # PanoVGGT点群は実寸より膨張する傾向が
                                      # あるため下方に広く探索
                                      # (縮小退化はカメラ実在ゲートが防ぐ)
BG_PENALTY = 0.25                     # 屋内自由空間に密度が乗った時の減点
CAM_FEAS_MIN = 0.8                    # カメラの8割以上が屋内自由空間に載ること


def warp_density(density, theta_deg, s):
    """密度マップを回転+スケールした配列と、順変換関数を返す

    順変換: u = s * R(θ) @ (v - c_in) + shift  (v,uは(row,col)配列座標)
    """
    h, w = density.shape
    th = math.radians(theta_deg)
    rot = np.array([[math.cos(th), -math.sin(th)],
                    [math.sin(th), math.cos(th)]])
    fwd = s * rot
    c_in = np.array([(h - 1) / 2, (w - 1) / 2])
    corners = np.array([[0, 0], [0, w - 1], [h - 1, 0], [h - 1, w - 1]], float)
    mapped = (corners - c_in) @ fwd.T
    lo = mapped.min(axis=0)
    hi = mapped.max(axis=0)
    out_shape = (int(np.ceil(hi[0] - lo[0])) + 1, int(np.ceil(hi[1] - lo[1])) + 1)
    shift = -lo  # 出力座標 = fwd @ (v - c_in) + shift

    inv = np.linalg.inv(fwd)
    warped = ndimage.affine_transform(
        density, inv, offset=c_in - inv @ shift, output_shape=out_shape, order=1)

    def transform_pts(pts_rc):
        return (np.asarray(pts_rc, float) - c_in) @ fwd.T + shift

    return warped, transform_pts


def camera_feasibility(grid, corr_shape, warped_shape, cam_cells):
    """配置(=corrの添字)ごとに、屋内自由空間に載るカメラの割合を返す"""
    feas = np.zeros(corr_shape, dtype=np.float32)
    vp = grid.valid_pose.astype(np.float32)
    hd1, wd1 = warped_shape[0] - 1, warped_shape[1] - 1
    for cy, cx in cam_cells:
        oy, ox = int(round(cy)) - hd1, int(round(cx)) - wd1
        m0, m1 = max(0, -oy), min(corr_shape[0], grid.ny - oy)
        n0, n1 = max(0, -ox), min(corr_shape[1], grid.nx - ox)
        if m1 > m0 and n1 > n0:
            feas[m0:m1, n0:n1] += vp[m0 + oy:m1 + oy, n0 + ox:n1 + ox]
    return feas / len(cam_cells)


def mirror_variants(density, cam_rc):
    """鏡映なし/あり の2仮説 (点群world座標系の掌性が図面と逆の場合に対応)"""
    yield False, density, cam_rc
    rc = cam_rc.copy()
    rc[:, 1] = (density.shape[1] - 1) - rc[:, 1]
    yield True, np.ascontiguousarray(density[:, ::-1]), rc


def density_and_rc_for(d, best, rc):
    """bestのmirrorフラグに応じて密度マップとrc座標を変換して返す"""
    dens = d["density"]
    if best.get("mirror"):
        dens = np.ascontiguousarray(dens[:, ::-1])
        rc = rc.copy()
        rc[:, 1] = (d["density"].shape[1] - 1) - rc[:, 1]
    return dens, rc


def match_floor(grid, density, cam_rc, cell_ratio,
                allow_mirror=True, thetas=None, scales=None):
    """1フロア分の全探索。cell_ratio = 密度セル(m)/図面セル(m) (通常1.0)

    スコア = precision (密度質量が図面の壁近傍に乗る割合)
           × recall (図面の壁セル全体のうち密度の壁で覆われた割合)。
    片方向スコアだけだと「縮小して壁密集地帯に置く」退化解が最良になるため、
    recall項が必須 (縮小すると図面の壁を覆えなくなる)。

    allow_mirror=False / thetas=[角度列] で探索を拘束できる
    (小さい島の照合を、既照合の島との相対ヨー(方位ロックで既知)で縛る用)。
    """
    k_score = grid.k_wall - BG_PENALTY * (grid.valid_pose & (grid.k_wall < 0.1))
    wall_r = (grid.k_wall >= 0.5).astype(np.float32)
    wall_total = wall_r.sum()
    best = None
    for mirror, dens, rc in mirror_variants(density, cam_rc):
      if mirror and not allow_mirror:
          continue
      for s in (scales if scales is not None else SCALES):
        for theta in (thetas if thetas is not None
                      else range(0, 360, THETA_STEP_DEG)):
            warped, transform_pts = warp_density(dens, theta, s * cell_ratio)
            if warped.shape[0] >= grid.ny * 4 or warped.shape[1] >= grid.nx * 4:
                continue
            flip = warped[::-1, ::-1]
            precision = fftconvolve(k_score, flip, mode="full")
            precision /= max(warped.sum(), 1e-6)
            warpbin = (warped > 0.12).astype(np.float32)[::-1, ::-1]
            recall = fftconvolve(wall_r, warpbin, mode="full") / wall_total
            score_map = precision * np.clip(recall, 0, 1)
            feas = camera_feasibility(grid, score_map.shape, warped.shape,
                                      transform_pts(rc))
            score_map[feas < CAM_FEAS_MIN] = -np.inf
            idx = np.unravel_index(np.argmax(score_map), score_map.shape)
            score = score_map[idx]
            if not np.isfinite(score):
                continue
            if best is None or score > best["score"]:
                best = {
                    "score": float(score), "theta": theta, "s": float(s),
                    "mirror": mirror,
                    "precision": float(precision[idx]),
                    "recall": float(min(recall[idx], 1.0)),
                    "cam_feas": float(feas[idx]),
                    # warped[0,0] が図面グリッドセル (dy,dx) に対応
                    "dy": int(idx[0]) - (warped.shape[0] - 1),
                    "dx": int(idx[1]) - (warped.shape[1] - 1),
                }
    return best


def model_to_plan_px(pts_xz, d, best, grid, cell_ratio):
    """モデル座標(x,z) → 図面px座標"""
    rc = np.stack([(pts_xz[:, 1] - d["origin"][1]) / d["cell_units"],
                   (pts_xz[:, 0] - d["origin"][0]) / d["cell_units"]], axis=1)
    dens, rc = density_and_rc_for(d, best, rc)
    _, transform_pts = warp_density(dens, best["theta"],
                                    best["s"] * cell_ratio)
    cells = transform_pts(rc) + np.array([best["dy"], best["dx"]])
    px = np.stack([grid.x0_px + cells[:, 1] * grid.cell_px,
                   grid.y0_px + cells[:, 0] * grid.cell_px], axis=1)
    return px


def draw(grid, bg, d, best, cell_ratio, cam_px, names, path, title):
    dens, _ = density_and_rc_for(d, best, np.zeros((1, 2)))
    warped, _ = warp_density(dens, best["theta"], best["s"] * cell_ratio)
    canvas = np.zeros((grid.ny, grid.nx))
    y0, x0 = best["dy"], best["dx"]
    ys0, xs0 = max(0, y0), max(0, x0)
    ys1 = min(grid.ny, y0 + warped.shape[0])
    xs1 = min(grid.nx, x0 + warped.shape[1])
    if ys1 > ys0 and xs1 > xs0:
        canvas[ys0:ys1, xs0:xs1] = warped[ys0 - y0: ys1 - y0, xs0 - x0: xs1 - x0]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(bg, cmap="gray")
    extent = (grid.x0_px, grid.x0_px + grid.nx * grid.cell_px,
              grid.y0_px + grid.ny * grid.cell_px, grid.y0_px)
    masked = np.ma.masked_where(canvas < 0.05, canvas)
    ax.imshow(masked, cmap="autumn", alpha=0.65, extent=extent)
    ax.plot(cam_px[:, 0], cam_px[:, 1], "-o", color="lime", markersize=5)
    for (x, y), n in zip(cam_px, names):
        ax.annotate(str(n).replace("frame_0000", "").replace("frame_000", ""),
                    (x, y), fontsize=7, color="blue")
    ax.set_title(title)
    ax.set_axis_off()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main(paths, argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None)
    args = ap.parse_args(argv)
    out_dir = paths.pcd / args.tag if args.tag else paths.pcd

    d = dict(np.load(out_dir / "density.npz", allow_pickle=False))
    # 密度セルは CELL_M=0.05m相当で作ってあり図面セル(RES_M=0.05m)と名目1:1。
    # スケール推定の誤差は SCALES の±30%探索で吸収する
    cell_ratio = 1.0

    grids = {f: PlanGrid(paths, f) for f in ("1F", "2F")}
    bgs = {f: np.asarray(Image.open(paths.floors / f"{f}.jpg").convert("L"))
           for f in grids}

    cam_rc = np.stack(
        [(d["cam_pos_aligned"][:, 2] - d["origin"][1]) / d["cell_units"],
         (d["cam_pos_aligned"][:, 0] - d["origin"][0]) / d["cell_units"]],
        axis=1)

    summary = {}
    results = {}
    for f, grid in grids.items():
        best = match_floor(grid, d["density"], cam_rc, cell_ratio)
        if best is None:
            raise RuntimeError(
                f"{f}: 有効な配置なし (密度マップが大きすぎるか、"
                "カメラ実在ゲートを満たす配置が無い)")
        results[f] = best
        print(f, best)

    floor = max(results, key=lambda f: results[f]["score"])
    for f, best in results.items():
        cam_px = model_to_plan_px(
            d["cam_pos_aligned"][:, [0, 2]], d, best, grids[f], cell_ratio)
        draw(grids[f], bgs[f], d, best, cell_ratio, cam_px, d["names"],
             out_dir / f"align_{f}.jpg",
             f"{f} score={best['score']:.3f} theta={best['theta']} s={best['s']:.2f}"
             f"{'  <== selected' if f == floor else ''}")
        summary[f] = {**best, "selected": f == floor,
                      "cam_px": cam_px.round(1).tolist()}

    (out_dir / "align_summary.json").write_text(json.dumps(
        {"selected_floor": floor, "results": summary,
         "names": [str(n) for n in d["names"]]}, indent=2))
    print(f"selected floor: {floor}")
    print(f"done -> {out_dir}")

