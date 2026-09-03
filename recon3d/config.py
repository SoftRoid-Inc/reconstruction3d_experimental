"""3D再構成の設定値。高さ関係は図面注記から確定できないため明示的な仮定値
(MODEL_RECONSTRUCTION.md の推奨値)。物件ごとの既知寸法はCLIから上書きする。"""
from dataclasses import dataclass


@dataclass
class ReconConfig:
    floor_height_mm: float = 2900.0
    window_sill_mm: float = 900.0
    window_height_mm: float = 1100.0
    door_height_mm: float = 2000.0
    slab_thickness_mm: float = 150.0
    pane_thickness_mm: float = 30.0
    known_width_mm: float = 9100.0
    known_depth_mm: float = 9555.0
