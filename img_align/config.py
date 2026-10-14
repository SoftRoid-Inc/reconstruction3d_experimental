"""img_align 共通設定: パス・モデル・対象クラス"""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # reconstruction_3d/
IMG_ALIGN_DIR = PROJECT_ROOT / "img_align"
# 全セッション横断のSOTAログ (git管理)。セッション別出力とは別に据え置く
REPORT_PATH = IMG_ALIGN_DIR / "results" / "REPORT.md"

# 撮影セッション。環境変数 IMG_SESSION で切替。
# 0828=28-AAAAA竣工後(既定) / 0312=28-AAAAA工事中 / 19176=28-CCCCC(建築中)
SESSION = os.environ.get("IMG_SESSION", "0828")

# 物件レジストリ: セッション → 入力データの所在と図面パラメータ
_28_11177 = PROJECT_ROOT / "28-AAAAA"
_28_19176 = PROJECT_ROOT / "28-CCCCC"
_28_41237 = PROJECT_ROOT / "28-BBBBB"


@dataclass(frozen=True)
class Property:
    root: Path                  # データセット(物件)ディレクトリ。出力は root/results/<session>
    frames_dir: Path            # 360°等距円筒フレーム (5376x2688)
    zumen_dir: Path             # 図面JPEG + labelme GT
    # スケール校正に使う既知寸法 (最外壁芯間のx方向スパン)。
    # 階でスパンが違う物件 (セットバック等) は {"1F": mm, ...} の辞書で指定
    known_width_mm: object
    mm_per_px: Optional[float]  # 実測スケール (Noneなら zumen GT から遅延導出)
    person_mask_dir: Optional[Path] = None  # 既存人物マスク (白=有効/黒=人物)。無ければ seg_ade20k
    colmap_model: Optional[Path] = None     # COLMAP sparse モデル (colmap_prior用)
    drawing_wh: Tuple[int, int] = (2480, 1753)  # 図面画像サイズ


PROPERTY = {
    "0828": Property(root=_28_11177,
                     frames_dir=_28_11177 / "video_data" / "0828" / "images",
                     zumen_dir=_28_11177 / "zumen",
                     known_width_mm=9100.0, mm_per_px=11.968),
    "0312": Property(root=_28_11177,
                     frames_dir=_28_11177 / "video_data" / "0312" / "images",
                     zumen_dir=_28_11177 / "zumen",
                     known_width_mm=9100.0, mm_per_px=11.968),
    "19176": Property(root=_28_19176,
                      frames_dir=_28_19176 / "images",
                      zumen_dir=_28_19176 / "zumen",
                      known_width_mm=7280.0, mm_per_px=None,
                      person_mask_dir=_28_19176 / "masks",
                      colmap_model=_28_19176 / "sfm" / "colmap_coarse" / "0"),
    "0428": Property(root=_28_41237,
                     frames_dir=_28_41237 / "0428" / "images",
                     zumen_dir=_28_41237 / "zumen",
                     # 図面上辺の全体寸法 (3Fはセットバックでスパンが短い)
                     known_width_mm={"1F": 13877.5, "2F": 13877.5,
                                     "3F": 11375.0},
                     mm_per_px=None,
                     person_mask_dir=_28_41237 / "0428" / "masks",
                     colmap_model=_28_41237 / "0428" / "sfm"
                     / "colmap_coarse" / "0"),
    "0616": Property(root=_28_41237,
                     frames_dir=_28_41237 / "0616" / "images",
                     zumen_dir=_28_41237 / "zumen",
                     known_width_mm={"1F": 13877.5, "2F": 13877.5,
                                     "3F": 11375.0},
                     mm_per_px=None,
                     person_mask_dir=_28_41237 / "0616" / "masks",
                     colmap_model=_28_41237 / "0616" / "sfm"
                     / "colmap_coarse" / "0"),
}
# 28-BBBBB の追加撮影日は構成が同一なのでまとめて登録
for _d in ("0715", "0912", "0925"):
    PROPERTY[_d] = Property(root=_28_41237,
                            frames_dir=_28_41237 / _d / "images",
                            zumen_dir=_28_41237 / "zumen",
                            known_width_mm={"1F": 13877.5, "2F": 13877.5,
                                            "3F": 11375.0},
                            mm_per_px=None,
                            person_mask_dir=_28_41237 / _d / "masks",
                            colmap_model=_28_41237 / _d / "sfm"
                            / "colmap_coarse" / "0")
PROP = PROPERTY[SESSION]

FRAMES_DIR = PROP.frames_dir
# 出力はデータセット直下のセッション別results (root/results/<session>)。
# サブ構造はフラット: seg/ imu/ pcd/ frontend_bench/ backend/ boundary_mask/ ...
RESULTS_DIR = PROP.root / "results" / SESSION
SEG_DIR = RESULTS_DIR / "seg"
IMU_DIR = RESULTS_DIR / "imu"
BENCH_DIR = RESULTS_DIR / "frontend_bench"   # 島タグ (global_cameras.npz等)
OUT_DIR = RESULTS_DIR / "backend"            # アンカーjson・aligned_house等
NUM_SAMPLE_FRAMES = 16  # 代表フレーム数(等間隔サンプリング)
CAM_HEIGHT_M = 1.5      # カメラ高の仮定値 (スケール推定のフォールバック用)

# セッション別設定 (フレーム列と階段区間。階段区間はIMU/GT階タイムライン由来)
SESSION_CFG = {
    "0828": {"frames": range(0, 454, 3),
             "ascend": (226, 243), "descend": (406, 414)},
    "0312": {"frames": range(0, 260),
             "ascend": (120, 133), "descend": (228, 241)},
    # 19176: 全216フレームstep1。階段区間はカメラ高さ実測 (64-83昇り/152-167降り)
    "19176": {"frames": range(0, 216),
              "ascend": (64, 83), "descend": (152, 167)},
    # 0428 (28-BBBBB): 基礎工事段階の屋外撮影。階段なし (単一レベル)
    "0428": {"frames": range(0, 374),
             "ascend": (0, 0), "descend": (0, 0)},
    # 0616 (28-BBBBB): 建方後3フロア。1F→2F=144-160, 2F→3F=283-295, 降り336-384
    "0616": {"frames": range(0, 403),
             "ascend": (144, 160), "descend": (336, 384)},
    "0715": {"frames": range(0, 416),
             "ascend": (124, 136), "descend": (352, 400)},
    "0912": {"frames": range(0, 757),
             "ascend": (252, 280), "descend": (728, 756)},
    "0925": {"frames": range(0, 1378), "ascend": (0, 0), "descend": (0, 0)},
}
CFG = SESSION_CFG[SESSION]
STAIR_ASCEND = CFG["ascend"]    # 1F→2F
STAIR_DESCEND = CFG["descend"]  # 2F→1F

# 手法A: ADE20K セマンティックセグメンテーション
SEG_MODEL = "facebook/mask2former-swin-large-ade-semantic"
CUBE_FACE_SIZE = 1024   # キューブマップ面の解像度
EQUI_W = 2048           # 統合ラベルマップ/オーバーレイの等距円筒幅 (高さは半分)

# 手法C: HorizonNet
THIRD_PARTY = Path("/home/maren/programming/third_party")
HORIZONNET_DIR = THIRD_PARTY / "HorizonNet"
HORIZONNET_REPO = "https://github.com/sunset1995/HorizonNet.git"
HORIZONNET_WEIGHTS = THIRD_PARTY / "weights" / "resnet50_rnn__st3d.pth"

# 対象クラス: ADE20Kラベル名で指定し、実行時にモデルの id2label からIDを解決する
# (personは撮影者マスク用)
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

# グループID (統合ラベルマップの値。0=その他)
GROUP_IDS = {name: i + 1 for i, name in enumerate(TARGET_LABELS)}

# オーバーレイ色 (RGB)
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
