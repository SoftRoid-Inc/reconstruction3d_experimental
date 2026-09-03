import pytest

from recon3d.config import ReconConfig
from recon3d.geometry import (
    Box3D, assign_openings, opening_to_boxes, plan_to_boxes, rect_px_to_mm,
    subtract_openings_1d, wall_to_boxes,
)
from recon3d.loader import FloorPlan, Rect

CFG = ReconConfig()


def test_rect_px_to_mm_flips_y():
    r = Rect(10, 10, 90, 14)               # 画像座標(y下向き), 高さ80px画像
    m = rect_px_to_mm(r, 10.0, 80)
    assert (m.x1, m.x2) == (100.0, 900.0)
    assert (m.y1, m.y2) == (660.0, 700.0)  # y=(80-14)*10 .. (80-10)*10


def test_subtract_openings_1d():
    assert subtract_openings_1d(0, 100, []) == [(0, 100)]
    assert subtract_openings_1d(0, 100, [(30, 50)]) == [(0, 30), (50, 100)]
    # 端に接する穴・複数穴
    assert subtract_openings_1d(0, 100, [(0, 20), (80, 100)]) == [(20, 80)]
    # 区間外にはみ出す穴はクリップ
    assert subtract_openings_1d(0, 100, [(-10, 20)]) == [(20, 100)]
    # 全部覆う
    assert subtract_openings_1d(0, 100, [(-5, 105)]) == []


def test_assign_openings_split_by_overlap():
    walls = [Rect(0, 0, 100, 10), Rect(0, 20, 100, 30)]
    openings = [Rect(40, 19, 60, 31),      # 壁1と83%重なり → 引き算
                Rect(10, -1, 20, 11),      # 壁0と83%重なり → 引き算
                Rect(300, 300, 310, 310),  # 重なりなし → 充填
                Rect(0, 8, 30, 20)]        # 壁0と17%重なり → 充填
    assigned, free = assign_openings(walls, openings)
    assert assigned == {0: [openings[1]], 1: [openings[0]]}
    assert free == [openings[2], openings[3]]


def test_opening_to_boxes_window_infill():
    # 壁の隙間の窓(横向き): 腰壁+垂れ壁(wall_infill) + pane
    boxes = opening_to_boxes(Rect(1000, 0, 2000, 150), "window", CFG, base_z=0.0)
    infill = [b for b in boxes if b.category == "wall_infill"]
    panes = [b for b in boxes if b.category == "pane_window"]
    assert len(infill) == 2 and len(panes) == 1
    zs = sorted((b.z1, b.z2) for b in infill)
    assert zs == [(0.0, 900.0), (2000.0, 2900.0)]
    assert all(b.x1 == 1000.0 and b.x2 == 2000.0 for b in infill)
    assert (panes[0].z1, panes[0].z2) == (900.0, 2000.0)
    assert panes[0].y2 - panes[0].y1 == pytest.approx(CFG.pane_thickness_mm)


def test_opening_to_boxes_door_infill_vertical():
    # 縦向きドア: 垂れ壁のみ + pane
    boxes = opening_to_boxes(Rect(0, 500, 150, 1400), "door", CFG, base_z=0.0)
    infill = [b for b in boxes if b.category == "wall_infill"]
    panes = [b for b in boxes if b.category == "pane_door"]
    assert len(infill) == 1 and len(panes) == 1
    assert (infill[0].z1, infill[0].z2) == (2000.0, 2900.0)
    assert (infill[0].y1, infill[0].y2) == (500.0, 1400.0)
    assert panes[0].x2 - panes[0].x1 == pytest.approx(CFG.pane_thickness_mm)


def test_wall_solid_no_opening():
    wall = Rect(0, 0, 4000, 150)           # mm, 横壁
    boxes = wall_to_boxes(wall, [], CFG, base_z=0.0)
    assert len(boxes) == 1
    b = boxes[0]
    assert (b.z1, b.z2) == (0.0, 2900.0) and b.category == "wall"


def test_wall_with_window():
    wall = Rect(0, 0, 4000, 150)
    win = (Rect(1000, 0, 2000, 150), "window")
    boxes = wall_to_boxes(wall, [win], CFG, base_z=0.0)
    walls = [b for b in boxes if b.category == "wall"]
    panes = [b for b in boxes if b.category == "pane_window"]
    # 左残り・右残り・腰壁(z0-900)・垂れ壁(z2000-2900) = 4
    assert len(walls) == 4 and len(panes) == 1
    xs = sorted((b.x1, b.x2) for b in walls)
    assert (0.0, 1000.0) in xs and (2000.0, 4000.0) in xs
    zs = sorted((b.z1, b.z2) for b in walls if b.x1 == 1000.0)
    assert zs == [(0.0, 900.0), (2000.0, 2900.0)]
    p = panes[0]
    assert (p.z1, p.z2) == (900.0, 2000.0)
    assert p.y2 - p.y1 == pytest.approx(CFG.pane_thickness_mm)


def test_wall_with_door_vertical():
    wall = Rect(0, 0, 150, 4000)           # 縦壁
    door = (Rect(0, 500, 150, 1400), "door")
    boxes = wall_to_boxes(wall, [door], CFG, base_z=0.0)
    walls = [b for b in boxes if b.category == "wall"]
    # 下残り・上残り・垂れ壁(z2000-2900)。ドアは床からなので腰壁なし = 3
    assert len(walls) == 3
    lintel = [b for b in walls if b.y1 == 500.0 and b.y2 == 1400.0]
    assert len(lintel) == 1 and (lintel[0].z1, lintel[0].z2) == (2000.0, 2900.0)


def test_two_openings_same_wall():
    wall = Rect(0, 0, 4000, 150)
    ops = [(Rect(500, 0, 1000, 150), "door"), (Rect(2000, 0, 3000, 150), "window")]
    boxes = wall_to_boxes(wall, ops, CFG, base_z=0.0)
    walls = [b for b in boxes if b.category == "wall"]
    # 長手残り3 + ドア垂れ壁1 + 窓腰壁/垂れ壁2 = 6
    assert len(walls) == 6


def test_plan_to_boxes_counts():
    fp = FloorPlan(name="1F", image_size=(100, 80))
    fp.walls = [Rect(10, 10, 90, 14)]
    fp.columns = [Rect(10, 10, 14, 14)]
    fp.windows = [Rect(30, 9, 50, 15)]
    boxes = plan_to_boxes(fp, 10.0, CFG, base_z=0.0)
    cats = sorted(b.category for b in boxes)
    # 壁: 左右残り+腰+垂れ=4, pane=1, 柱=1
    assert cats.count("wall") == 4
    assert cats.count("pane_window") == 1
    assert cats.count("column") == 1
    col = [b for b in boxes if b.category == "column"][0]
    assert (col.z1, col.z2) == (0.0, 2900.0)
