"""recon3d 共通設定: パス・モデル・対象クラス"""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # reconstruction_3d/
PKG_DIR = PROJECT_ROOT / "recon3d"
# 全セッション横断のSOTAログ (git管理)。セッション別出力とは別に据え置く
REPORT_PATH = PKG_DIR / "results" / "REPORT.md"

# 撮影セッション。環境変数 IMG_SESSION で切替。
# 0828=28-AAAAA竣工後(既定) / 0312=28-AAAAA工事中 / 19176=28-CCCCC(建築中)
#
# **IMG_SESSION はパスも受け付ける**。レジストリに無い値はデータセット
# ディレクトリとして解釈し、その場で Property を組み立てて登録する
# (下の _register_dataset)。環境変数を2本に増やさないのが要点:
#   - tools/ext_batch.py は `{**os.environ, "IMG_SESSION": sess}` で子に渡す。
#     別変数を足すと明示上書きされず**全件に漏れる**
#   - 「両方セットされたときどちらが勝つか」という曖昧さが生まれない
SESSION_REF = os.environ.get("IMG_SESSION", "0828")
# 子プロセス/コンテナを起こすときは SESSION ではなく **SESSION_REF** を渡す
# こと。SESSION は派生名なので、env に戻すとレジストリに無くて落ちる

# 物件レジストリ: セッション → 入力データの所在と図面パラメータ
_28_11177 = PROJECT_ROOT / "28-AAAAA"
_28_19176 = PROJECT_ROOT / "28-CCCCC"
_28_41237 = PROJECT_ROOT / "28-BBBBB"
_EXT = PROJECT_ROOT / "ext"       # 外部テストデータセット (polish-pipeline sfm_testdataset)


@dataclass(frozen=True)
class Property:
    root: Path                  # データセット(物件)ディレクトリ。出力は root/results/<session>
    frames_dir: Path            # 360°等距円筒フレーム (5376x2688)
    zumen_dir: Path             # 図面JPEG + labelme GT
    # スケール校正に使う既知寸法 (最外壁芯間のx方向スパン)。
    # 階でスパンが違う物件 (セットバック等) は {"1F": mm, ...} の辞書で指定
    known_width_mm: object
    mm_per_px: Optional[float]  # 実測スケール (Noneなら zumen GT から遅延導出)
    # GT CSV 内でこのセッションを指す MMDD キー。外部データセット (ext/) は
    # セッション名と撮影日が一致しないので明示する
    gt_session: Optional[str] = None
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

# --- 外部テストデータセット (polish-pipeline の sfm_testdataset、INSTA 156件)
# 図面の壁アノテーション (labelme) が無いので**図面アンカーは動かない**。
# GT軌跡だけで測れるもの (島の形状品質・ポーズグラフ・橋渡し) の検証に使う。
#
# `ext/<name>/` を**自動で走査して登録**する (大量データを回すので1件ずつ
# 手で書かない)。各ディレクトリは tools/ext_batch.py が用意する:
#   images/ masks/ zumen/ trajectory_points.csv
#   trajectory_data_by_scan_data_id.csv floor_info.csv meta.json
# セッション名は meta.json の "session"、gt_session は撮影日 (MMDD) で別物
# (同じ日に撮った別スキャンがあるため名前が衝突する)
def _load_meta(path):
    """meta.json を読む (無ければ {})。壊れていたら黙らずに落とす"""
    import json as _json
    if not path.is_file():
        return {}
    try:
        m = _json.loads(path.read_text())
    except Exception as e:
        raise SystemExit(f"{path} を読めない: {type(e).__name__}: {e}")
    if not isinstance(m, dict):
        raise SystemExit(f"{path}: トップレベルはオブジェクトである必要がある")
    return m


# meta.json が解釈するキー。これ以外 (ext_batch が書く s3key/n_frames 等) は
# 覚え書きとして素通りさせる。**型だけは入口で検査する** — known_width_mm に
# 文字列が入ると PlanGrid が TypeError を出し、アンカー経路が import ごと
# 死ぬ (その例外が出るのは図面を読む瞬間なので原因が分かりにくい)
def _check_meta(path, m):
    def bad(k, why):
        raise SystemExit(f"{path}: \"{k}\" が不正 ({why}) -> {m[k]!r}")

    kw = m.get("known_width_mm")
    if kw is not None:
        vals = kw.values() if isinstance(kw, dict) else [kw]
        if isinstance(kw, dict) and not all(isinstance(k, str) for k in kw):
            bad("known_width_mm", "階名は文字列")
        if not all(isinstance(v, (int, float)) and v > 0 for v in vals):
            bad("known_width_mm", "最外壁芯間のx方向スパン[mm]。正の数、"
                                  "階で違うなら {\"1F\": 13877.5, ...}")
    if m.get("mm_per_px") is not None and not (
            isinstance(m["mm_per_px"], (int, float)) and m["mm_per_px"] > 0):
        bad("mm_per_px", "正の数 [mm/px]")
    wh = m.get("drawing_wh")
    if wh is not None and not (isinstance(wh, (list, tuple)) and len(wh) == 2
                               and all(isinstance(v, int) and v > 0
                                       for v in wh)):
        bad("drawing_wh", "図面画像サイズ [幅, 高さ] の整数2つ")
    for k in ("session", "gt_session"):
        if m.get(k) is not None and not isinstance(m[k], str):
            bad(k, "文字列")


def _register_dataset(root, session=None, overwrite=False):
    """データセットディレクトリを PROPERTY に登録し、セッション名を返す

    期待する構成 (外部パイプラインが用意する。欠けていれば能力が下がるだけ):
        <root>/images/   frame_number_*.jpg          必須
              /masks/    frame_number_*.jpg          人物マスク (白=有効)
              /imu/      transitions.json 等          IMU派生物 (外部生成)
              /floors/   1F.json 1F.jpg … meta.json   間取り図ラベル
              /results/<session>/                     出力 (ここだけ書く)

    図面の実寸 `known_width_mm` は **floors/meta.json** に置く。壁注釈だけでは
    px→m が決まらず図面アンカーは動かない (core/plan_grid.PlanGrid 参照)。
    `ext/` 互換のため <root>/meta.json と <root>/zumen/ も見る。
    """
    # symlink は**解決しない**。ext/ は /home/maren/temp/recon3d_ext への
    # symlink で、解決するとコンテナの /code2 マウント外のパスになる
    root = Path(os.path.abspath(os.path.expanduser(str(root))))
    if not root.is_dir():
        raise SystemExit(f"データセットが見つからない: {root}")
    # 新しい入力契約は floors/。ext/ 互換で zumen/ も見る
    # (どちらも無ければ floors/ を名乗る — エラーメッセージが分かりやすい)
    zumen = root / "zumen" if (root / "zumen").is_dir() else root / "floors"
    meta = {**_load_meta(root / "meta.json"), **_load_meta(zumen / "meta.json")}
    _check_meta(zumen / "meta.json", meta)
    name = session or meta.get("session") or root.name
    # 同じ root を指す再登録は素通り。`ext/sfm11196` は _register_ext が
    # 先に e11196 として登録しているので、パスで渡したときに衝突扱いに
    # しない (同じデータセットを2通りの書き方で指しているだけ)
    if name in PROPERTY and PROPERTY[name].root == root:
        return name
    if name in PROPERTY and not overwrite:
        raise SystemExit(
            f"セッション名 '{name}' は既に登録済み ({PROPERTY[name].root})。"
            f"meta.json の \"session\" か --session で別名を付けてください")
    frames = root / "images"
    masks = root / "masks"
    wh = meta.get("drawing_wh")
    PROPERTY[name] = Property(
        root=root, frames_dir=frames,
        zumen_dir=zumen,
        known_width_mm=meta.get("known_width_mm"),
        mm_per_px=meta.get("mm_per_px"),
        gt_session=meta.get("gt_session"),
        person_mask_dir=masks if masks.is_dir() else None,
        **({"drawing_wh": tuple(wh)} if wh else {}))
    return name


def _register_ext():
    """ext/ 配下を自動走査して登録 (外部テストデータセット156件を手で書かない)"""
    if not _EXT.exists():
        return
    for _d in sorted(_EXT.iterdir()):
        if (_d / "meta.json").is_file():
            _register_dataset(_d, overwrite=True)


_register_ext()
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

def _resolve_session(ref):
    """IMG_SESSION の値 -> セッション名

    受け付ける形:
      "0912"                     登録済みセッション名
      "/data/reconstruction"     データセットディレクトリ (名前はディレクトリ名)
      "myrun=/data/reconstruction"  名前を明示 (同じ入力を条件違いで回すとき)
    """
    name = None
    if ref not in PROPERTY and "=" in ref:
        name, ref = ref.split("=", 1)
    if ref in PROPERTY:
        if name and name != ref:
            raise SystemExit(f"'{ref}' は登録済みセッション。別名は付けられない")
        return ref
    p = Path(ref).expanduser()
    if not p.is_dir():
        raise SystemExit(
            f"IMG_SESSION='{ref}' は登録済みセッションでもディレクトリでもない"
            f"\n  登録済み: {sorted(PROPERTY)}")
    return _register_dataset(p, session=name)


SESSION = _resolve_session(SESSION_REF)
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

def frame_indices(frames_dir=None):
    """フレーム番号の昇順リスト。ゼロ詰め桁数には依存しない"""
    d = Path(frames_dir or FRAMES_DIR)
    if not d.is_dir():
        return []
    out = []
    for p in d.glob("frame_number_*.jpg"):
        s = p.stem[len("frame_number_"):]
        if s.isdigit():
            out.append(int(s))
    return sorted(out)


def _frame_pad(frames_dir):
    """ファイル名のゼロ詰め桁数を実物から決める (既定6)"""
    d = Path(frames_dir)
    for p in sorted(d.glob("frame_number_*.jpg")) if d.is_dir() else ():
        s = p.stem[len("frame_number_"):]
        if s.isdigit():
            return len(s)
    return 6


FRAME_PAD = _frame_pad(FRAMES_DIR)


def frame_name(i, ext=".jpg"):
    """フレーム番号 -> ファイル名。桁数を決め打ちしない"""
    return f"frame_number_{i:0{FRAME_PAD}d}{ext}"


def frame_path(i, base=None, ext=".jpg"):
    return Path(base or FRAMES_DIR) / frame_name(i, ext)


# 未登録セッション (ext/ や入力ディレクトリ渡し) は階段区間の手動注釈を
# 持たない。フレーム範囲は画像から決め、階段区間は空にする
# (島の切り方はIMU/GTから決めるので要らない)
#
# 0枚のときもここでは落とさない。config の import はレポート生成など
# フレームを要らない経路でも走る (ext_batch は評価後に images/ を消す)。
# **走らせる側**で弾くこと → core/caps.py が frames 能力を見て止める
if SESSION not in SESSION_CFG:
    _idx = frame_indices()
    SESSION_CFG[SESSION] = {"frames": range(0, _idx[-1] + 1 if _idx else 0),
                            "ascend": (0, 0), "descend": (0, 0)}
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
