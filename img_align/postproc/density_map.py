"""点群の重力整列と壁密度マップ生成 (SceneAligner-lite 中段, 本家§3準拠)

1. 鉛直軸の同定: 床・天井の平面が作る座標ヒストグラムのピーク鋭度で判定
2. 床平面フィットで傾き補正 (微小回転)
3. 床-天井間距離=2.4m仮定でスケール推定 (照合側でも±30%探索するので粗くて良い)
4. 床・天井スラブ近傍の点を除去して壁を残し、水平面2Dヒストグラム+γ補正

出力: results/pcd/density.npz (density, origin, cell, m_per_unit, R_align, 鉛直軸情報)
      + density.png (目視確認用)
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

from img_align.config import CAM_HEIGHT_M
from img_align.core.geometry import (camera_centers,
                                     gravity_from_cameras,
                                     rotation_to_y)
from img_align.core.io_ply import read_ply_xyzrgb
from img_align.core.panovggt import PCD_DIR

CELL_M = 0.05          # 密度マップのセルサイズ (m相当)
SLAB_MARGIN_M = 0.20   # 床・天井ピーク近傍の除去幅
GAMMA = 0.5
N_HIST_BINS = 200


def scale_from_local_floors(out_dir, cams):
    """各フレーム直下の床までの高さ(カメラ高)の中央値からスケールを推定する

    stitch_chunks 出力なら floor_heights (チャンクスケール補正済み) を使い、
    単一チャンク出力なら per-frame PLY から都度計算する。
    """
    if "floor_heights" in cams:
        heights = cams["floor_heights"]
        heights = heights[np.isfinite(heights)]
    else:
        from img_align.core.panovggt import local_floor_height
        heights = [local_floor_height(
            out_dir / "panovggt_out" / "pointclouds" / "per_frame" / f"{n}.ply")
            for n in cams["names"]]
        heights = np.array([h for h in heights if h is not None])
    h = float(np.median(heights))
    print(f"camera height above floor: median {h:.3f} units "
          f"(n={len(heights)}, IQR {np.percentile(heights, 25):.2f}"
          f"-{np.percentile(heights, 75):.2f})")
    return CAM_HEIGHT_M / h




def slab_peaks(y):
    """鉛直座標ヒストグラムから床・天井等の水平面ピークを検出"""
    hist, edges = np.histogram(y, bins=N_HIST_BINS)
    centers = (edges[:-1] + edges[1:]) / 2
    peaks, _ = find_peaks(hist, prominence=hist.max() * 0.15,
                          distance=N_HIST_BINS // 40)
    order = np.argsort(centers[peaks])
    return centers[peaks][order], hist, centers


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    out_dir = PCD_DIR / args.tag if args.tag else PCD_DIR

    xyz, _ = read_ply_xyzrgb(out_dir / "points.ply")
    cams = np.load(out_dir / "cameras.npz")
    print(f"{len(xyz):,} points")

    conv, up = gravity_from_cameras(cams["poses"])
    print(f"pose convention = {conv}")
    R_align = rotation_to_y(up)
    pts = xyz @ R_align.T
    cam_pos = camera_centers(cams["poses"], conv) @ R_align.T

    # PanoVGGTのworld系は図面と掌性が逆 (鏡映仮説が全島で高スコア、2026-09-04実測)
    # → zを反転して「上から見た右手系」に合わせる
    pts[:, 2] *= -1
    cam_pos = cam_pos.copy()
    cam_pos[:, 2] *= -1

    m_per_unit = scale_from_local_floors(out_dir, cams)
    print(f"m_per_unit = {m_per_unit:.4f} (cam height {CAM_HEIGHT_M}m assumed)")

    peaks, hist, centers = slab_peaks(pts[:, 1])
    print(f"horizontal slabs at y = {np.round(peaks, 3).tolist()}")

    # スラブ近傍を除去して壁を残す
    margin_u = SLAB_MARGIN_M / m_per_unit
    keep = np.ones(len(pts), dtype=bool)
    for p in peaks:
        keep &= np.abs(pts[:, 1] - p) > margin_u
    walls = pts[keep]
    print(f"wall-band points: {len(walls):,} ({100 * len(walls) / len(pts):.1f}%)")

    # 空間バウンディング (本家のρ_xz相当): 窓越しの屋外点等の外れ裾を落とす
    lo_x, hi_x = np.percentile(walls[:, 0], [2.5, 97.5])
    lo_z, hi_z = np.percentile(walls[:, 2], [2.5, 97.5])
    inb = ((walls[:, 0] >= lo_x) & (walls[:, 0] <= hi_x)
           & (walls[:, 2] >= lo_z) & (walls[:, 2] <= hi_z))
    print(f"spatial bounding: keep {100 * inb.mean():.1f}%")
    walls = walls[inb]

    # 水平面 (x,z) 2Dヒストグラム
    cell_u = CELL_M / m_per_unit
    x, z = walls[:, 0], walls[:, 2]
    ox, oz = x.min() - 10 * cell_u, z.min() - 10 * cell_u
    nx = int((x.max() - ox) / cell_u) + 10
    nz = int((z.max() - oz) / cell_u) + 10
    density, _, _ = np.histogram2d(
        z, x, bins=(nz, nx),
        range=((oz, oz + nz * cell_u), (ox, ox + nx * cell_u)))
    density = (density / max(density.max(), 1)) ** GAMMA

    np.savez_compressed(
        out_dir / "density.npz",
        density=density, origin=np.array([ox, oz]), cell_units=cell_u,
        m_per_unit=m_per_unit, R_align=R_align,
        cam_pos_aligned=cam_pos, names=cams["names"], slab_peaks=peaks)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(density, cmap="gray_r", origin="upper")
    axes[0].plot((cam_pos[:, 0] - ox) / cell_u, (cam_pos[:, 2] - oz) / cell_u,
                 "r.-", markersize=6)
    axes[0].set_title(f"wall density map ({nz}x{nx} cells, {CELL_M}m/cell)")
    axes[1].plot(centers, hist)
    for p in peaks:
        axes[1].axvline(p, color="r", linestyle="--")
    axes[1].set_title("vertical histogram (red = slab peaks)")
    fig.savefig(out_dir / "density.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"done -> {out_dir / 'density.png'}")


if __name__ == "__main__":
    main()
