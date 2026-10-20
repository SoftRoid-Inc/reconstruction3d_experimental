"""図面×点群の2層ハイブリッド衝突モデル生成 (ロボットシミュレーション用)

層1 = 構造 (図面由来): recon3dの箱モデル(壁/柱/スラブ/建具、水密な直方体群)を
      aligned座標系 (x,y=図面平面[m](yは画像下向き), z=高さ[m], 2F床=+2.9m) へ変換。
      直方体集合はそのまま衝突形状として理想的 (凸分解不要)。
層2 = 障害物 (点群由来): 位置合わせ済み点群から「構造で説明できない点」
      (資材・脚立・設備・家具・階段など図面に無い実在物) を抽出し、
      DBSCANクラスタごとに凸包メッシュ化 (衝突用途には凸包で十分)。

副産物: 視覚確認用のPoissonメッシュ (色付き)。

実行: IMG_SESSION=0312 conda run -n panovggt python -m recon3d.mesh.collision_mesh
出力: results/backend/collision_<sess>.{glb,obj} / mesh_visual_<sess>.ply /
      collision_check_<sess>.png
"""
import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from scipy import ndimage

from recon3d.plan2model.calibrate import scale_mm_per_px
from recon3d.plan2model.config import ReconConfig
from recon3d.plan2model.geometry import plan_to_boxes, rect_px_to_mm
from recon3d.plan2model.loader import load_labelme
from recon3d.plan2model.mesh import boxes_to_scene, footprint_polygon

from recon3d.config import BENCH_DIR
from recon3d.config import SESSION
from recon3d.eval.gt_traj import mm_per_px
from recon3d.core.plan_grid import ZUMEN_DIR
from recon3d.core.io_ply import read_ply_xyzrgb
from recon3d.postproc.plan_icp import wall_segments_px

OUT_DIR = BENCH_DIR.parent / "backend"
FLOORS = [("1F", 0.0), ("2F", 2.9)]
PLY_NAME = {"0828": "aligned_house.ply", "0312": "aligned_house_0312.ply"}

# 合成参照点 (export_aligned2 の壁カーテン) の色
CURTAIN_COLORS = [(60, 90, 255), (90, 200, 255)]

WALL_CLEAR_M = 0.15      # 構造(壁/柱/建具)からこの距離以内の点は構造扱い
H_MIN, H_MAX = 0.10, 2.3  # 障害物とみなす床上高さ帯
CELL_M = 0.05
DBSCAN_EPS = 0.12
DBSCAN_MIN = 40
MIN_CLUSTER_PTS = 200
MIN_CLUSTER_H = 0.10
OBSTACLE_COLOR = (240, 140, 40, 255)
# 凸包の充実度(2Dフットプリント占有率)がこれ未満なら主成分軸で再帰分割
# (複数物体を繋いだクラスタの凸包は物体間の空間まで覆い過大になるため)
SOLIDITY_MIN = 0.45
SOLIDITY_CELL = 0.10
MAX_PART_VOL = 1.5   # 単一障害物の包体積上限[m3] (超えたら分割。散在資材の連結対策)
SPLIT_MAX_DEPTH = 6
MIN_PART_PTS = 120


# ---------------------------------------------------------------- 構造層

def build_structure_scene():
    """recon3d箱モデルを構築し、aligned座標系[m]へ変換したtrimesh.Sceneを返す"""
    cfg = ReconConfig()
    plans = [load_labelme(ZUMEN_DIR / f"{f}.json", f) for f, _ in FLOORS]
    mm_per_px = scale_mm_per_px(plans[0], cfg.known_width_mm)
    boxes, slabs = [], []
    for i, plan in enumerate(plans):
        base_z = i * cfg.floor_height_mm
        boxes.extend(plan_to_boxes(plan, mm_per_px, cfg, base_z))
        solid = plan.walls + plan.columns + plan.windows + plan.doors
        rects_mm = [rect_px_to_mm(r, mm_per_px, plan.image_size[1])
                    for r in solid]
        fp = footprint_polygon(rects_mm)
        slabs.append((fp, base_z - cfg.slab_thickness_mm, base_z))
        if i == len(plans) - 1:
            top = base_z + cfg.floor_height_mm
            slabs.append((fp, top, top + cfg.slab_thickness_mm))
    scene_mm = boxes_to_scene(boxes, slabs)

    # recon3d座標(mm, y=画像高−y_px) → aligned座標(m, y=y_px系)
    h_mm = plans[0].image_size[1] * mm_per_px
    out = trimesh.Scene()
    for name, geom in scene_mm.geometry.items():
        g = geom.copy()
        v = g.vertices.copy()
        v[:, 0] /= 1000.0
        v[:, 1] = (h_mm - v[:, 1]) / 1000.0
        v[:, 2] /= 1000.0
        g.vertices = v
        g.invert()   # y反転で裏返った法線を戻す
        out.add_geometry(g, geom_name=f"structure_{name}")
    return out, plans, mm_per_px


# ---------------------------------------------------------------- 障害物層

def structure_dist_raster(plan, mm_per_px):
    """壁/柱/建具矩形までの2D距離場[m] (aligned座標のx,y[m]でルックアップ)"""
    w_px, h_px = plan.image_size
    cell_px = CELL_M * 1000 / mm_per_px
    nx = int(w_px / cell_px) + 2
    ny = int(h_px / cell_px) + 2
    mask = np.zeros((ny, nx), bool)
    for r in plan.walls + plan.columns + plan.windows + plan.doors:
        x0, x1 = int(r.x1 / cell_px), int(math.ceil(r.x2 / cell_px))
        y0, y1 = int(r.y1 / cell_px), int(math.ceil(r.y2 / cell_px))
        mask[max(y0, 0):y1 + 1, max(x0, 0):x1 + 1] = True
    dist = ndimage.distance_transform_edt(~mask) * CELL_M

    def lookup(x_m, y_m):
        c = np.clip((x_m * 1000 / mm_per_px / cell_px).astype(int), 0, nx - 1)
        r_ = np.clip((y_m * 1000 / mm_per_px / cell_px).astype(int), 0, ny - 1)
        return dist[r_, c]
    return lookup


def extract_obstacles(xyz, rgb, plans, mm_per_px):
    """点群から非構造点を抽出し、クラスタごとの凸包メッシュ列を返す"""
    import open3d as o3d
    keep = np.ones(len(xyz), bool)
    for c in CURTAIN_COLORS:
        keep &= ~np.all(rgb == np.array(c, np.uint8), axis=1)
    xyz = xyz[keep]
    print(f"points after curtain filter: {len(xyz):,}")

    lookups = {f: structure_dist_raster(p, mm_per_px)
               for (f, _), p in zip(FLOORS, plans)}
    floor_of = np.where(xyz[:, 2] < 2.6, 0, 1)
    obst_mask = np.zeros(len(xyz), bool)
    for i, (f, z_off) in enumerate(FLOORS):
        m = floor_of == i
        h = xyz[m, 2] - z_off
        d = lookups[f](xyz[m, 0], xyz[m, 1])
        obst_mask[np.nonzero(m)[0]] = (h > H_MIN) & (h < H_MAX) \
            & (d > WALL_CLEAR_M)
    pts = xyz[obst_mask]
    print(f"non-structure points: {len(pts):,}")

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)
    pc = pc.voxel_down_sample(0.03)
    p = np.asarray(pc.points)
    labels = np.asarray(pc.cluster_dbscan(eps=DBSCAN_EPS,
                                          min_points=DBSCAN_MIN))

    def solidity2d(cl, hull2d_area):
        ij = np.unique(np.round(cl[:, :2] / SOLIDITY_CELL).astype(int), axis=0)
        return len(ij) * SOLIDITY_CELL ** 2 / max(hull2d_area, 1e-6)

    def decompose(cl, depth=0):
        """充実度が低いクラスタ(複数物体の橋渡し)は主成分軸で分割する"""
        if len(cl) < MIN_PART_PTS or np.ptp(cl[:, 2]) < MIN_CLUSTER_H:
            return []
        try:
            hull = trimesh.convex.convex_hull(cl)
        except Exception:
            return []
        # 2Dフットプリントの充実度
        from scipy.spatial import ConvexHull as CH2
        try:
            area = CH2(cl[:, :2]).volume   # 2DではvolumeがQhullの面積
        except Exception:
            return []
        sol = solidity2d(cl, area)
        if (sol >= SOLIDITY_MIN and hull.volume <= MAX_PART_VOL) \
                or depth >= SPLIT_MAX_DEPTH:
            hull.visual.face_colors = OBSTACLE_COLOR
            return [(hull, cl)]
        c = cl[:, :2] - cl[:, :2].mean(axis=0)
        ax = np.linalg.svd(c, full_matrices=False)[2][0]
        t = c @ ax
        cut = np.median(t)
        return decompose(cl[t <= cut], depth + 1) \
            + decompose(cl[t > cut], depth + 1)

    hulls = []
    n_covered = 0
    for lab in range(labels.max() + 1):
        cl = p[labels == lab]
        if len(cl) < MIN_CLUSTER_PTS:
            continue
        # クラスタ内の外れ値(フライングピクセル等)を除去して包を締める
        cpc = o3d.geometry.PointCloud()
        cpc.points = o3d.utility.Vector3dVector(cl)
        cpc, _ = cpc.remove_statistical_outlier(nb_neighbors=16,
                                                std_ratio=1.8)
        cl = np.asarray(cpc.points)
        parts = decompose(cl)
        for hull, part in parts:
            hulls.append((hull, part))
            n_covered += len(part)
    print(f"obstacle parts: {len(hulls)} "
          f"(coverage {n_covered / max(len(p), 1):.0%} of non-structure pts)")
    return hulls, pts


# ---------------------------------------------------------------- 視覚メッシュ

def poisson_visual(xyz, rgb, out_path):
    import open3d as o3d
    keep = np.ones(len(xyz), bool)
    for c in CURTAIN_COLORS:
        keep &= ~np.all(rgb == np.array(c, np.uint8), axis=1)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz[keep])
    pc.colors = o3d.utility.Vector3dVector(rgb[keep] / 255.0)
    pc = pc.voxel_down_sample(0.02)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(0.08, 30))
    pc.orient_normals_consistent_tangent_plane(20)
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pc, depth=10)
    dens = np.asarray(dens)
    mesh.remove_vertices_by_mask(dens < np.quantile(dens, 0.05))
    o3d.io.write_triangle_mesh(str(out_path), mesh)
    print(f"-> {out_path} ({len(mesh.vertices):,} verts)")


# ---------------------------------------------------------------- 検証画像

def check_figure(structure, hulls, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(20, 11))
    for ax, (floor, z_off) in zip(axes, FLOORS):
        for x1, y1, x2, y2 in wall_segments_px(floor) * mm_per_px() / 1000:
            ax.plot([x1, x2], [y1, y2], "-", color="cyan", lw=1.2)
        for name, g in structure.geometry.items():
            v = g.vertices
            if not ((v[:, 2].min() < z_off + 2.4)
                    and (v[:, 2].max() > z_off + 0.1)):
                continue
            ax.scatter(v[:, 0], v[:, 1], s=0.5, c="gray", alpha=0.4)
        from scipy.spatial import ConvexHull as CH2
        for hull, cl in hulls:
            zc = cl[:, 2].mean()
            if not (z_off - 0.3 < zc < z_off + 2.6):
                continue
            ax.scatter(cl[::8, 0], cl[::8, 1], s=0.8, c="orange", alpha=0.5)
            try:
                h2 = CH2(hull.vertices[:, :2])
                loop = np.append(h2.vertices, h2.vertices[0])
                ax.plot(hull.vertices[loop, 0], hull.vertices[loop, 1],
                        "-", color="red", lw=1.0, alpha=0.9)
            except Exception:
                pass
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_title(f"{floor}: structure(gray) obstacles(orange) "
                     f"GT walls(cyan)")
        ax.set_facecolor("#101014")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, facecolor="#181820")
    plt.close(fig)
    print(f"-> {out_path}")


def main():
    sess = SESSION
    xyz, rgb = read_ply_xyzrgb(OUT_DIR / PLY_NAME[sess])
    print(f"session {sess}: {len(xyz):,} points")

    structure, plans, mm_per_px = build_structure_scene()
    hulls, _ = extract_obstacles(xyz, rgb, plans, mm_per_px)

    scene = trimesh.Scene()
    for name, g in structure.geometry.items():
        scene.add_geometry(g, geom_name=name)
    for i, (hull, _) in enumerate(hulls):
        scene.add_geometry(hull, geom_name=f"obstacle_{i:03d}")
    for ext in ("glb", "obj"):
        path = OUT_DIR / f"collision_{sess}.{ext}"
        scene.export(path)
        print(f"-> {path}")
    stats = {
        "session": sess,
        "structure_geoms": len(structure.geometry),
        "obstacles": len(hulls),
        "obstacle_faces": int(sum(len(h.faces) for h, _ in hulls)),
        "obstacle_volume_m3": round(float(sum(h.volume for h, _ in hulls)), 2),
    }
    (OUT_DIR / f"collision_{sess}.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats))

    check_figure(structure, hulls, OUT_DIR / f"collision_check_{sess}.png")
    poisson_visual(xyz, rgb, OUT_DIR / f"mesh_visual_{sess}.ply")


if __name__ == "__main__":
    main()
