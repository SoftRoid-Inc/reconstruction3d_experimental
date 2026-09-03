import os

import pytest
import trimesh

from recon3d.geometry import Box3D
from recon3d.loader import Rect
from recon3d.mesh import boxes_to_scene, export_scene, footprint_polygon, slab_mesh


def test_footprint_union_of_rects():
    # ロの字(中庭あり)の外周壁 → 穴は埋めて外周のみ
    walls = [Rect(0, 0, 1000, 100), Rect(0, 900, 1000, 1000),
             Rect(0, 0, 100, 1000), Rect(900, 0, 1000, 1000)]
    poly = footprint_polygon(walls)
    assert poly.area == pytest.approx(1000 * 1000)
    assert len(poly.interiors) == 0


def test_slab_mesh_extrudes():
    walls = [Rect(0, 0, 1000, 100), Rect(0, 900, 1000, 1000),
             Rect(0, 0, 100, 1000), Rect(900, 0, 1000, 1000)]
    m = slab_mesh(footprint_polygon(walls), -150.0, 0.0)
    assert m.is_volume
    assert m.bounds[0][2] == pytest.approx(-150.0)
    assert m.bounds[1][2] == pytest.approx(0.0)
    assert m.volume == pytest.approx(1000 * 1000 * 150, rel=1e-3)


def test_boxes_to_scene_and_export(tmp_path):
    boxes = [Box3D(0, 0, 0, 1000, 150, 2900, "wall"),
             Box3D(0, 0, 0, 150, 150, 2900, "column")]
    walls = [Rect(0, 0, 1000, 150)]
    scene = boxes_to_scene(boxes, [(footprint_polygon(walls), -150.0, 0.0)])
    assert isinstance(scene, trimesh.Scene)
    assert len(scene.geometry) == 3
    paths = export_scene(scene, str(tmp_path))
    for p in paths.values():
        assert os.path.getsize(p) > 0
    reloaded = trimesh.load(paths["glb"])
    assert len(reloaded.geometry) == 3
