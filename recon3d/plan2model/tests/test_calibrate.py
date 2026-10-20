import pytest

from recon3d.plan2model.calibrate import centerline_span_px, cross_check_scale, scale_mm_per_px
from recon3d.plan2model.config import ReconConfig
from recon3d.plan2model.loader import load_floor
from recon3d.plan2model.tests.conftest import GT_DIR


def test_centerline_span_synthetic(tiny_labelme):
    fp = load_floor(tiny_labelme, "1F")
    # tiny_labelmeの縦壁は1本のみ → スパン計算には2本必要でエラー
    with pytest.raises(ValueError):
        centerline_span_px(fp, "x")
    # 横壁も1本のみ
    with pytest.raises(ValueError):
        centerline_span_px(fp, "y")


def test_scale_on_real_gt_1f():
    fp = load_floor(GT_DIR + "/1F.json", "1F")
    span = centerline_span_px(fp, "x")
    assert 755 < span < 765               # 実測760.3px
    s = scale_mm_per_px(fp, 9100.0)
    assert 11.85 < s < 12.1               # 実測≈11.968
    err = cross_check_scale(fp, s, 9555.0)
    assert err < 0.01                     # 縦寸法との整合 1%未満


def test_2f_span_matches_module():
    fp = load_floor(GT_DIR + "/2F.json", "2F")
    fp1 = load_floor(GT_DIR + "/1F.json", "1F")
    s = scale_mm_per_px(fp1, 9100.0)      # スケールは1Fで校正し全階共通
    width_2f = centerline_span_px(fp, "x") * s
    # 2F幅は910mmモジュールの整数倍(8×910=7280mm)に近い
    assert abs(width_2f - 7280.0) < 40.0


def test_default_config():
    cfg = ReconConfig()
    assert cfg.floor_height_mm == 2900.0
    assert cfg.window_sill_mm + cfg.window_height_mm == 2000.0
    assert cfg.door_height_mm == 2000.0
