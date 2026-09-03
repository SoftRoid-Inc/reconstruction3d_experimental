from recon3d.loader import Rect, load_labelme, load_prediction, load_floor
from recon3d.tests.conftest import GT_DIR


def test_rect_normalizes_and_props():
    r = Rect.from_points(90.0, 70.0, 86.0, 10.0)
    assert (r.x1, r.y1, r.x2, r.y2) == (86.0, 10.0, 90.0, 70.0)
    assert r.w == 4.0 and r.h == 60.0
    assert r.cx == 88.0 and r.cy == 40.0
    assert r.is_vertical()


def test_load_labelme(tiny_labelme):
    fp = load_labelme(tiny_labelme, "1F")
    assert fp.name == "1F" and fp.image_size == (100, 80)
    assert len(fp.walls) == 2 and len(fp.columns) == 1
    assert len(fp.windows) == 1 and len(fp.doors) == 1
    assert fp.walls[1].x1 == 86.0  # 点順が逆でも正規化される


def test_load_prediction(tiny_prediction):
    fp = load_prediction(tiny_prediction, "1F")
    assert fp.image_size == (100, 80)
    assert len(fp.walls) == 1          # is_column=True は柱へ
    assert len(fp.columns) == 1
    assert fp.columns[0].w == 4.0
    assert len(fp.windows) == 1 and len(fp.doors) == 0


def test_load_floor_dispatch(tiny_labelme, tiny_prediction):
    assert len(load_floor(tiny_labelme, "1F").walls) == 2
    assert len(load_floor(tiny_prediction, "1F").walls) == 1


def test_load_real_gt():
    fp = load_floor(GT_DIR + "/1F.json", "1F")
    assert fp.image_size == (2480, 1753)
    assert len(fp.walls) == 38 and len(fp.columns) == 39
    assert len(fp.windows) == 6 and len(fp.doors) == 10
    for r in fp.walls:
        assert r.x2 > r.x1 and r.y2 > r.y1
