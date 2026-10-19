"""改善版SceneAligner-liteの点群を図面座標系(メートル)へ変換して1つのPLYに書き出す

チャンクごとの色付き点群(chunks_masked)を、最終補正後のカメラ位置
(results/eval_gt/assembled_masked.json) に整合する相似変換で図面座標へ変換する。
参照用にGT図面の壁も点列(青=1F, 水色=2F)として同梱する。

出力: results/pcd/aligned_house.ply (CloudCompare等で閲覧)
座標系: x,y = 図面平面 (メートル、図面px×11.968mm)、z = 高さ (1F床=0、2F床=+2.9m)
"""
import json

import numpy as np

from img_align.config import RESULTS_DIR
from img_align.core.geometry import camera_centers, gravity_from_cameras, rotation_to_y
from img_align.eval.gt_traj import mm_per_px
from img_align.core.plan_grid import PlanGrid
from img_align.core.io_ply import read_ply_xyzrgb
from img_align.core.panovggt import PCD_DIR
from img_align.postproc.plan_icp import cams_to_cells, fit_similarity, transform_pts, wall_segments_px

# chunk12(islandC)は内部地図が歪んでおり、領域もchunk0-4でカバー済みのため
# 地図(点群)からは除外する (軌跡評価には含まれる)
CHUNKS = [(k, "1F", 0.0) for k in range(0, 5)] \
    + [(k, "2F", 2.9) for k in range(6, 12)]
CHUNKS_DIR = "chunks_masked"
SUBSAMPLE = 6
CELL_M = 0.05


def export():
    # チャンク変換は同時最適化の結果 (joint_refine.py) を使う
    trans = json.loads((PCD_DIR / "chunk_transforms.json").read_text())
    grids = {f: PlanGrid(f) for f in ("1F", "2F")}

    all_xyz, all_rgb = [], []
    for k, floor, z_off in CHUNKS:
        tr = trans[str(k)]
        s, th, t = tr["s"], tr["th"], np.array(tr["t"])
        d_dir = PCD_DIR / CHUNKS_DIR / f"chunk_{k:02d}"
        d2 = dict(np.load(d_dir / "density2.npz"))
        grid = grids[floor]

        xyz, rgb = read_ply_xyzrgb(d_dir / "points.ply")
        xyz, rgb = xyz[::SUBSAMPLE], rgb[::SUBSAMPLE]
        cams = np.load(d_dir / "cameras.npz")
        conv, up = gravity_from_cameras(cams["poses"])
        R = rotation_to_y(up)
        p = xyz @ R.T
        p[:, 2] *= -1
        mpu = float(d2["m_per_unit"])
        cells = np.stack([(p[:, 2] - d2["origin"][1]) / d2["cell_units"],
                          (p[:, 0] - d2["origin"][0]) / d2["cell_units"]], axis=1)
        pc = transform_pts(cells, s, th, t)  # plan cell (row, col)
        # 図面絶対px → メートル (壁参照点と同じ座標系に揃える)
        plan = np.stack([(grid.y0_px + pc[:, 0] * grid.cell_px) * mm_per_px() / 1000,
                         (grid.x0_px + pc[:, 1] * grid.cell_px) * mm_per_px() / 1000],
                        axis=1)
        # 高さ: 床スラブピーク (joint_refineで推定) を z=0 に
        h = (p[:, 1] - tr["floor_y_units"]) * mpu + z_off
        keep = (h > -0.5 + z_off) & (h < 3.2 + z_off)
        out = np.stack([plan[keep, 1], plan[keep, 0], h[keep]], axis=1)
        all_xyz.append(out)
        all_rgb.append(rgb[keep])
        print(f"chunk {k:02d} ({floor}): {keep.sum():,} pts")

    # 図面の壁 (参照用の点カーテン)
    for floor, z_off, color in (("1F", 0.0, (60, 90, 255)),
                                ("2F", 2.9, (90, 200, 255))):
        segs = wall_segments_px(floor) * mm_per_px() / 1000
        pts = []
        for x1, y1, x2, y2 in segs:
            n = max(int(np.hypot(x2 - x1, y2 - y1) / 0.05), 2)
            tt = np.linspace(0, 1, n)
            for hz in np.arange(0, 2.4, 0.3):
                pts.append(np.stack([x1 + tt * (x2 - x1),
                                     y1 + tt * (y2 - y1),
                                     np.full(n, hz + z_off)], axis=1))
        pts = np.concatenate(pts)
        all_xyz.append(pts)
        all_rgb.append(np.tile(np.array(color, np.uint8), (len(pts), 1)))
        print(f"walls {floor}: {len(pts):,} pts")

    xyz = np.concatenate(all_xyz).astype(np.float32)
    rgb = np.concatenate(all_rgb).astype(np.uint8)
    out_path = PCD_DIR / "aligned_house.ply"
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(xyz)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "end_header\n")
    data = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                     ("r", "u1"), ("g", "u1"), ("b", "u1")])
    data["x"], data["y"], data["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    data["r"], data["g"], data["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(out_path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())
    print(f"total {len(xyz):,} pts -> {out_path}")


if __name__ == "__main__":
    export()
