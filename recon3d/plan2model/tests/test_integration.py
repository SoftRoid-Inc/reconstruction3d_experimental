import json
import os

import pytest
import trimesh

from recon3d.plan2model.build import build_model
from recon3d.plan2model.config import ReconConfig
from recon3d.plan2model.tests.conftest import GT_DIR, NO_GT, NO_GT_REASON


@pytest.mark.skipif(NO_GT, reason=NO_GT_REASON)
def test_build_28_11177(tmp_path):
    cfg = ReconConfig()
    report = build_model(
        [("1F", GT_DIR + "/1F.json"), ("2F", GT_DIR + "/2F.json")],
        cfg, str(tmp_path))
    assert 11.85 < report["mm_per_px"] < 12.1
    assert report["scale_cross_check_err"] < 0.01
    assert report["counts"]["1F"]["walls"] == 38
    assert report["counts"]["2F"]["columns"] == 22
    scene = trimesh.load(report["outputs"]["glb"])
    # 建物全体の寸法がGT壁bbox(1F: 773x924.9px)×スケールと一致する
    # (y方向は通り芯9555mmの外側にポーチ等の壁が突出するため9555より大きい)
    ext = scene.bounds[1] - scene.bounds[0]
    s = report["mm_per_px"]
    assert ext[0] == pytest.approx(773.0 * s, rel=0.01)
    assert ext[1] == pytest.approx(924.9 * s, rel=0.01)
    # 高さ: 2階分5800 + 床スラブ150 + 天井スラブ150
    assert ext[2] == pytest.approx(6100.0, abs=1.0)
    # 1F床スラブが建物全幅を覆う(壁が開口で途切れてもフットプリントが閉じる)
    slab0 = scene.geometry["slab_0"]
    slab_ext = slab0.bounds[1] - slab0.bounds[0]
    assert slab_ext[0] == pytest.approx(773.0 * s, rel=0.01)
    assert slab_ext[1] == pytest.approx(924.9 * s, rel=0.01)
    assert os.path.exists(os.path.join(str(tmp_path), "report.json"))
    with open(os.path.join(str(tmp_path), "report.json")) as f:
        assert json.load(f)["mm_per_px"] == report["mm_per_px"]


from recon3d.plan2model.calibrate import scale_mm_per_px
from recon3d.plan2model.geometry import plan_to_boxes
from recon3d.plan2model.loader import load_floor
from recon3d.plan2model.verify import reprojection_iou, save_overlay


@pytest.mark.skipif(NO_GT, reason=NO_GT_REASON)
def test_reprojection_matches_gt(tmp_path):
    cfg = ReconConfig()
    plan_1f = load_floor(GT_DIR + "/1F.json", "1F")
    s = scale_mm_per_px(plan_1f, cfg.known_width_mm)
    for name in ("1F", "2F"):
        plan = load_floor(f"{GT_DIR}/{name}.json", name)
        boxes = plan_to_boxes(plan, s, cfg, base_z=0.0)
        iou = reprojection_iou(plan, boxes, s)
        assert iou > 0.99, f"{name}: IoU={iou}"
        png = str(tmp_path / f"{name}.png")
        save_overlay(plan, boxes, s, png)
        assert os.path.getsize(png) > 0
