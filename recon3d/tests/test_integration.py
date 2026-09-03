import json
import os

import pytest
import trimesh

from recon3d.build import build_model
from recon3d.config import ReconConfig
from recon3d.tests.conftest import GT_DIR


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
    assert os.path.exists(os.path.join(str(tmp_path), "report.json"))
    with open(os.path.join(str(tmp_path), "report.json")) as f:
        assert json.load(f)["mm_per_px"] == report["mm_per_px"]
