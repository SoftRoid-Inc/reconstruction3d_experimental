"""データセットに依らない定数

ここに置くのは**どの入力でも同じ値**だけ。入力ごとに変わるものは
`recon3d/paths.py` の `Paths` が持ち、引数で引き回す。
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent   # reconstruction_3d/
PKG_DIR = PROJECT_ROOT / "recon3d"
# 全データセット横断のSOTAログ (git管理)。出力とは別に据え置く
REPORT_PATH = PKG_DIR / "results" / "REPORT.md"

NUM_SAMPLE_FRAMES = 16  # 代表フレーム数 (等間隔サンプリング)
CAM_HEIGHT_M = 1.5      # カメラ高の仮定値 (スケール推定のフォールバック用)

# 人物マスク生成 (seg_ade20k) は本パッケージから外したが、既存の seg 出力を
# 読む経路が残っているのでラベル定義は保つ
TARGET_LABELS = {
    "wall": ["wall"],
    "floor": ["floor"],
    "ceiling": ["ceiling"],
    "window": ["windowpane", "window"],
    "door": ["door", "screen door"],
    "stairs": ["stairs", "stairway", "step"],
    "column": ["column", "pillar"],
    "person": ["person"],
}
GROUP_IDS = {name: i + 1 for i, name in enumerate(TARGET_LABELS)}  # 0=その他
GROUP_COLORS = {
    "wall": (230, 180, 60),
    "floor": (80, 170, 80),
    "ceiling": (100, 130, 220),
    "window": (60, 220, 220),
    "door": (220, 70, 70),
    "stairs": (200, 90, 200),
    "column": (250, 250, 60),
    "person": (128, 128, 128),
}
