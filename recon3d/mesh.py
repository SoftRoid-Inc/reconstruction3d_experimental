"""Box3D列+スラブからtrimesh.Sceneを組み立て、glb/objへ書き出す。"""
import os

import numpy as np
import trimesh
from shapely.geometry import Polygon, box as shapely_box
from shapely.ops import unary_union

COLORS = {
    "wall": (205, 205, 210, 255),
    "column": (150, 120, 90, 255),
    "pane_window": (140, 200, 255, 120),
    "pane_door": (170, 130, 80, 200),
    "slab": (235, 230, 220, 255),
}


def footprint_polygon(walls_mm):
    union = unary_union([shapely_box(r.x1, r.y1, r.x2, r.y2) for r in walls_mm])
    if union.geom_type == "MultiPolygon":
        union = max(union.geoms, key=lambda g: g.area)
    return Polygon(union.exterior)  # 中庭等の穴は埋める


def slab_mesh(poly, z1, z2):
    m = trimesh.creation.extrude_polygon(poly, height=z2 - z1)
    m.apply_translation((0, 0, z1))
    return m


def _box_mesh(b):
    m = trimesh.creation.box(extents=(b.x2 - b.x1, b.y2 - b.y1, b.z2 - b.z1))
    m.apply_translation(((b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2, (b.z1 + b.z2) / 2))
    return m


def _paint(mesh, category):
    mesh.visual.face_colors = np.tile(COLORS[category], (len(mesh.faces), 1))
    return mesh


def boxes_to_scene(boxes, slabs):
    scene = trimesh.Scene()
    by_cat = {}
    for b in boxes:
        by_cat.setdefault(b.category, []).append(_box_mesh(b))
    for cat, meshes in by_cat.items():
        scene.add_geometry(_paint(trimesh.util.concatenate(meshes), cat),
                           geom_name=cat)
    for i, (poly, z1, z2) in enumerate(slabs):
        scene.add_geometry(_paint(slab_mesh(poly, z1, z2), "slab"),
                           geom_name=f"slab_{i}")
    return scene


def export_scene(scene, out_dir, stem="model"):
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for ext in ("glb", "obj"):
        p = os.path.join(out_dir, f"{stem}.{ext}")
        scene.export(p)
        paths[ext] = p
    return paths
