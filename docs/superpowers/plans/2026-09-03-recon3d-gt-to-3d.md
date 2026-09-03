# recon3d: GT平面図JSON → 3Dモデル再構成 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 28-AAAAAのGT JSON(1F/2F, labelme矩形)から壁・柱・開口・スラブを持つ3Dモデル(glTF/OBJ)を生成し、2D再投影でGTと一致することを検証する。

**Architecture:** 全要素が軸平行矩形である性質を利用し、ブーリアン演算エンジン不要の「軸平行ボックス分割」方式を取る。壁ボックスから開口ボックスを区間分割で引き算し、結果は常にボックスの集合になる。ローダは labelme GT と CubiCasa prediction.json の両方を共通スキーマ `FloorPlan` に正規化する(再学習モデルへの差し替え口)。

**Tech Stack:** Python 3 (cubicasaコンテナ内), numpy 1.24.4, shapely 1.8.5(既存), trimesh + mapbox_earcut + pytest(本計画で追加インストール)

**Spec:** `/home/maren/programming/reconstruction_3d/MODEL_RECONSTRUCTION.md`

## Global Constraints

- 実行は全て cubicasa コンテナ内: `cd /home/maren/programming && docker compose exec -T cubicasa ...`。ホスト側パス `/home/maren/programming/reconstruction_3d` はコンテナ内 `/code2/reconstruction_3d`
- 既存依存を上げない: scipy 1.9.3 / shapely 1.8.5 / matplotlib<3.9 / numpy 1.24.4 は固定。pip install時に既存を巻き込まないこと(`--no-deps`は不要だが、インストール後にバージョン再確認する)
- 元図面・GT(`28-*/`)・`third_party/CubiCasa5k` は無改変。成果物は `reconstruction_3d/recon3d/` 配下のみ
- コンテナが生成したファイルはroot所有 → 生成後 `docker compose exec -T cubicasa chown -R 1001:1001 <path>` する
- 座標系: 画像px(y下向き) → 世界mm(y上向き, z上向き)。変換は `y_mm = (image_height - y_px) * mm_per_px`
- **スケールは実測で校正済みの前提値を使う**: 図面の理論値4.233mm/pxは誤り。1F外周垂直壁の中心線xスパン760.3pxが寸法線9100mmに対応 → **mm_per_px ≈ 11.968**。コードでは必ず動的に算出し、9100mmから計算する(ハードコードしない)
- 高さの仮定値(図面注記から確定できないため明示的な設定値とする): 階高2900mm / 窓 腰高900+H1100 / ドア H2000 / スラブ厚150mm

## File Structure

```
reconstruction_3d/
├── recon3d/
│   ├── __init__.py
│   ├── config.py        # ReconConfig: 高さ仮定値・スラブ厚・既知寸法
│   ├── loader.py        # labelme GT / prediction.json → FloorPlan (共通スキーマ)
│   ├── calibrate.py     # px→mm校正 (通り芯スパン), 交差検証
│   ├── geometry.py      # Rect/Box3D, 開口割当て, ボックス分割(開口引き算)
│   ├── mesh.py          # Box3D列+スラブ → trimesh.Scene → glb/obj出力
│   ├── build.py         # CLI: 物件→3Dモデル一式生成 (python -m recon3d.build)
│   ├── verify.py        # 2D再投影IoU・寸法整合チェック・オーバーレイPNG
│   ├── requirements.txt # 追加依存の記録 (trimesh, mapbox_earcut, pytest)
│   ├── output/          # 生成物 (gitignore)
│   └── tests/
│       ├── __init__.py
│       ├── conftest.py          # 合成labelmeフィクスチャ
│       ├── test_loader.py
│       ├── test_calibrate.py
│       ├── test_geometry.py
│       ├── test_mesh.py
│       └── test_integration.py  # 実GTでのE2E
└── docs/superpowers/plans/  (本計画)
```

**テスト実行コマンド(全タスク共通):**
```bash
cd /home/maren/programming && docker compose exec -T cubicasa \
  env PYTHONPATH=/code2/reconstruction_3d python -m pytest \
  /code2/reconstruction_3d/recon3d/tests -v
```

---

### Task 0: 環境セットアップ (git初期化・パッケージ骨格・依存追加)

**Files:**
- Create: `.gitignore`, `recon3d/__init__.py`, `recon3d/tests/__init__.py`, `recon3d/requirements.txt`

**Interfaces:**
- Produces: importable package `recon3d`、コンテナ内で trimesh / mapbox_earcut / pytest が使える状態

- [ ] **Step 1: git初期化と.gitignore**

`reconstruction_3d/` はgit管理外なので初期化する(図面・GT・結果はコミットしない):

```bash
cd /home/maren/programming/reconstruction_3d && git init
```

`.gitignore`:
```gitignore
28-*/
cubicasa/results/
__pycache__/
*.pyc
recon3d/output/
.pytest_cache/
```

- [ ] **Step 2: パッケージ骨格**

`recon3d/__init__.py` と `recon3d/tests/__init__.py` は空ファイル。

`recon3d/requirements.txt`:
```
# cubicasaコンテナへの追加依存 (docker compose exec cubicasa pip install -r ...)
# 既存ピン (numpy1.24.4/scipy1.9.3/shapely1.8.5/matplotlib<3.9) を壊さないこと
trimesh
mapbox_earcut
pytest
```

- [ ] **Step 3: コンテナに依存をインストール**

```bash
cd /home/maren/programming && docker compose exec -T cubicasa \
  pip install trimesh mapbox_earcut pytest
```

- [ ] **Step 4: インストール検証(既存ピンが無事なことも確認)**

```bash
cd /home/maren/programming && docker compose exec -T cubicasa python -c "
import trimesh, mapbox_earcut, pytest, numpy, scipy, shapely
print('trimesh', trimesh.__version__)
print('numpy', numpy.__version__, '/ scipy', scipy.__version__, '/ shapely', shapely.__version__)
assert numpy.__version__ == '1.24.4' and scipy.__version__ == '1.9.3'
print('OK')"
```
Expected: `OK`(numpy/scipyのバージョンが変わっていたら `pip install numpy==1.24.4 scipy==1.9.3` で戻す)

- [ ] **Step 5: Commit**

```bash
cd /home/maren/programming/reconstruction_3d && git add -A && \
  git commit -m "chore: init recon3d package skeleton and container deps"
```

---

### Task 1: loader.py — GT/predictionを共通スキーマFloorPlanへ

**Files:**
- Create: `recon3d/loader.py`
- Test: `recon3d/tests/test_loader.py`, `recon3d/tests/conftest.py`

**Interfaces:**
- Produces:
  ```python
  @dataclass(frozen=True)
  class Rect:  # 軸平行矩形, px座標
      x1: float; y1: float; x2: float; y2: float  # 正規化済み x1<x2, y1<y2
      # プロパティ: w, h, cx, cy / メソッド: is_vertical() -> bool (h > w)

  @dataclass
  class FloorPlan:
      name: str                     # 例 "1F"
      image_size: tuple  # (width, height)
      walls: list    # list[Rect]
      columns: list  # list[Rect]
      windows: list  # list[Rect]
      doors: list    # list[Rect]

  def load_labelme(path: str, name: str) -> FloorPlan
  def load_prediction(path: str, name: str) -> FloorPlan
  def load_floor(path: str, name: str) -> FloorPlan  # キーで自動判別
  ```
- 入力仕様: labelme=`shapes[].label∈{wall,column,window,door}`, `points`=対角2点。prediction.json=`walls[].polygon`(4点・軸平行, `is_column`で柱), `windows[]`, `doors[]` 各`polygon`。どちらもbboxへ潰してRect化する

- [ ] **Step 1: フィクスチャと失敗するテストを書く**

`recon3d/tests/conftest.py`:
```python
import json
import pytest

GT_DIR = "/code2/reconstruction_3d/28-AAAAA/zumen"


@pytest.fixture
def tiny_labelme(tmp_path):
    """壁2(縦横)・柱1・窓1・ドア1の最小labelme。画像100x80px想定。"""
    data = {
        "imageWidth": 100, "imageHeight": 80,
        "shapes": [
            {"label": "wall", "shape_type": "rectangle",
             "points": [[10.0, 10.0], [90.0, 14.0]]},          # 横壁
            {"label": "wall", "shape_type": "rectangle",
             "points": [[90.0, 70.0], [86.0, 10.0]]},          # 縦壁(点順が逆)
            {"label": "column", "shape_type": "rectangle",
             "points": [[10.0, 10.0], [14.0, 14.0]]},
            {"label": "window", "shape_type": "rectangle",
             "points": [[30.0, 9.0], [50.0, 15.0]]},
            {"label": "door", "shape_type": "rectangle",
             "points": [[60.0, 9.5], [75.0, 14.5]]},
        ],
    }
    p = tmp_path / "t.json"
    p.write_text(json.dumps(data))
    return str(p)


@pytest.fixture
def tiny_prediction(tmp_path):
    data = {
        "image_size": [100, 80],
        "walls": [
            {"id": 0, "polygon": [[10, 10], [90, 10], [90, 14], [10, 14]],
             "is_column": False},
            {"id": 1, "polygon": [[86, 10], [90, 10], [90, 14], [86, 14]],
             "is_column": True},
        ],
        "windows": [{"polygon": [[30, 9], [50, 9], [50, 15], [30, 15]]}],
        "doors": [],
        "rooms": [],
    }
    p = tmp_path / "p.json"
    p.write_text(json.dumps(data))
    return str(p)
```

`recon3d/tests/test_loader.py`:
```python
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
```

- [ ] **Step 2: 失敗確認**

Run: 共通テストコマンドで `tests/test_loader.py -v`
Expected: FAIL (`ModuleNotFoundError: recon3d.loader` 等)

- [ ] **Step 3: 実装**

`recon3d/loader.py`:
```python
"""GT(labelme)とCubiCasa prediction.jsonを共通スキーマFloorPlanへ正規化する。"""
import json
from dataclasses import dataclass, field

LABELS = ("wall", "column", "window", "door")


@dataclass(frozen=True)
class Rect:
    x1: float
    y1: float
    x2: float
    y2: float

    @classmethod
    def from_points(cls, xa, ya, xb, yb):
        return cls(min(xa, xb), min(ya, yb), max(xa, xb), max(ya, yb))

    @classmethod
    def from_polygon(cls, poly):
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        return cls(min(xs), min(ys), max(xs), max(ys))

    @property
    def w(self):
        return self.x2 - self.x1

    @property
    def h(self):
        return self.y2 - self.y1

    @property
    def cx(self):
        return (self.x1 + self.x2) / 2

    @property
    def cy(self):
        return (self.y1 + self.y2) / 2

    def is_vertical(self):
        return self.h > self.w


@dataclass
class FloorPlan:
    name: str
    image_size: tuple
    walls: list = field(default_factory=list)
    columns: list = field(default_factory=list)
    windows: list = field(default_factory=list)
    doors: list = field(default_factory=list)


def load_labelme(path, name):
    with open(path) as f:
        d = json.load(f)
    fp = FloorPlan(name=name, image_size=(d["imageWidth"], d["imageHeight"]))
    buckets = {"wall": fp.walls, "column": fp.columns,
               "window": fp.windows, "door": fp.doors}
    for s in d["shapes"]:
        if s.get("shape_type") != "rectangle" or s["label"] not in buckets:
            continue
        (xa, ya), (xb, yb) = s["points"]
        buckets[s["label"]].append(Rect.from_points(xa, ya, xb, yb))
    return fp


def load_prediction(path, name):
    with open(path) as f:
        d = json.load(f)
    fp = FloorPlan(name=name, image_size=tuple(d["image_size"]))
    for w in d.get("walls", []):
        r = Rect.from_polygon(w["polygon"])
        (fp.columns if w.get("is_column") else fp.walls).append(r)
    for key, bucket in (("windows", fp.windows), ("doors", fp.doors)):
        for o in d.get(key, []):
            bucket.append(Rect.from_polygon(o["polygon"]))
    return fp


def load_floor(path, name):
    with open(path) as f:
        d = json.load(f)
    if "shapes" in d:
        return load_labelme(path, name)
    return load_prediction(path, name)
```

- [ ] **Step 4: テスト通過確認**

Run: 共通テストコマンド `tests/test_loader.py -v` → Expected: 全PASS

- [ ] **Step 5: Commit**

```bash
cd /home/maren/programming/reconstruction_3d && git add recon3d && \
  git commit -m "feat: loader normalizing labelme GT and prediction.json to FloorPlan"
```

---

### Task 2: config.py + calibrate.py — px→mm校正と寸法交差検証

**Files:**
- Create: `recon3d/config.py`, `recon3d/calibrate.py`
- Test: `recon3d/tests/test_calibrate.py`

**Interfaces:**
- Consumes: `loader.FloorPlan`, `loader.Rect`
- Produces:
  ```python
  @dataclass
  class ReconConfig:
      floor_height_mm: float = 2900.0
      window_sill_mm: float = 900.0
      window_height_mm: float = 1100.0
      door_height_mm: float = 2000.0
      slab_thickness_mm: float = 150.0
      pane_thickness_mm: float = 30.0
      known_width_mm: float = 9100.0   # 28-AAAAA 通り芯 横全幅 (図面寸法線)
      known_depth_mm: float = 9555.0   # 同 縦全幅 (交差検証用)

  def centerline_span_px(plan: FloorPlan, axis: str) -> float
      # axis='x': 縦壁(is_vertical)のcxのmax-min / axis='y': 横壁のcyのmax-min
  def scale_mm_per_px(plan: FloorPlan, known_width_mm: float) -> float
  def cross_check_scale(plan, mm_per_px, known_depth_mm) -> float  # 相対誤差(符号なし)
  ```
- 校正の根拠(実測済み): 1F縦壁cxスパン=760.3px ↔ 9100mm → mm_per_px≈11.968。
  横壁cyスパン=800.6px×11.968=9581mm ↔ 9555mm(誤差0.27%)。
  2F cxスパン=609.0px→7288mm≈8×910mmモジュールで整合

- [ ] **Step 1: 失敗するテストを書く**

`recon3d/tests/test_calibrate.py`:
```python
import pytest

from recon3d.calibrate import centerline_span_px, cross_check_scale, scale_mm_per_px
from recon3d.config import ReconConfig
from recon3d.loader import load_floor
from recon3d.tests.conftest import GT_DIR


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
```

- [ ] **Step 2: 失敗確認**

Run: `tests/test_calibrate.py -v` → Expected: FAIL (module not found)

- [ ] **Step 3: 実装**

`recon3d/config.py`:
```python
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
```

`recon3d/calibrate.py`:
```python
"""px→mmスケール校正。図面の通り芯寸法(既知mm)と、GT外周壁の中心線スパン(px)の
比から係数を求める。理論値(1/50・300dpi→4.233mm/px)は本データでは成立しない
(実測≈11.968mm/px)ため、必ず寸法線ベースで校正する。"""


def centerline_span_px(plan, axis):
    if axis == "x":
        coords = [w.cx for w in plan.walls if w.is_vertical()]
    elif axis == "y":
        coords = [w.cy for w in plan.walls if not w.is_vertical()]
    else:
        raise ValueError(f"axis must be 'x' or 'y': {axis}")
    if len(coords) < 2:
        raise ValueError(f"axis={axis} の壁が2本未満でスパンを測れない")
    span = max(coords) - min(coords)
    if span <= 0:
        raise ValueError(f"axis={axis} のスパンが0")
    return span


def scale_mm_per_px(plan, known_width_mm):
    return known_width_mm / centerline_span_px(plan, "x")


def cross_check_scale(plan, mm_per_px, known_depth_mm):
    depth_mm = centerline_span_px(plan, "y") * mm_per_px
    return abs(depth_mm - known_depth_mm) / known_depth_mm
```

- [ ] **Step 4: テスト通過確認**

Run: `tests/test_calibrate.py -v` → Expected: 全PASS

- [ ] **Step 5: Commit**

```bash
cd /home/maren/programming/reconstruction_3d && git add recon3d && \
  git commit -m "feat: px-to-mm calibration from grid dimension lines with cross-check"
```

---

### Task 3: geometry.py — 開口割当てとボックス分割

**Files:**
- Create: `recon3d/geometry.py`
- Test: `recon3d/tests/test_geometry.py`

**Interfaces:**
- Consumes: `loader.Rect`, `loader.FloorPlan`, `config.ReconConfig`
- Produces:
  ```python
  @dataclass(frozen=True)
  class Box3D:
      x1: float; y1: float; z1: float; x2: float; y2: float; z2: float  # mm
      category: str  # "wall" | "column" | "pane_window" | "pane_door"

  def rect_px_to_mm(r: Rect, mm_per_px: float, image_height: int) -> Rect
      # y軸反転: y_mm = (image_height - y_px) * mm_per_px。返すRectはmm座標
  def assign_openings(walls: list, openings: list) -> dict
      # {wall_index: [opening_rect,...]}  交差面積最大の壁へ割当て。
      # どの壁とも交差しない開口は捨てて warnings.warn する
  def subtract_openings_1d(start, end, holes) -> list
      # 1次元区間 [start,end] から穴区間リストを引いた残り区間リスト
  def wall_to_boxes(wall_mm, openings_mm, cfg, base_z) -> list
      # 壁1枚 → Box3Dリスト。openings_mm = [(rect_mm, kind)] kind∈{"window","door"}
      # 窓: z∈[base+sill, base+sill+wh] を開口 / ドア: z∈[base, base+dh]
      # 開口部にはpane(枠内いっぱい・壁厚方向中央pane_thickness_mm)を生成
  def plan_to_boxes(plan, mm_per_px, cfg, base_z) -> list
      # 階全体: 壁(開口処理込み)+柱(base_z..base_z+floor_height)のBox3D列
  ```
- 分割アルゴリズム: 壁の長手軸に沿って開口区間を引き算し(左右残り)、
  開口区間内は z 方向で下(腰壁)・上(垂れ壁)の残りボックスを作る。
  すべて軸平行ボックスなのでブーリアンエンジン不要

- [ ] **Step 1: 失敗するテストを書く**

`recon3d/tests/test_geometry.py`:
```python
import warnings

import pytest

from recon3d.config import ReconConfig
from recon3d.geometry import (
    Box3D, assign_openings, plan_to_boxes, rect_px_to_mm,
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


def test_assign_openings_picks_max_overlap():
    walls = [Rect(0, 0, 100, 10), Rect(0, 20, 100, 30)]
    openings = [Rect(40, 19, 60, 31),      # 壁1と大きく交差
                Rect(10, -1, 20, 11),      # 壁0
                Rect(300, 300, 310, 310)]  # どの壁とも交差しない
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        m = assign_openings(walls, openings)
        assert any("交差しない" in str(w.message) for w in rec)
    assert m == {0: [openings[1]], 1: [openings[0]]}


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
```

- [ ] **Step 2: 失敗確認**

Run: `tests/test_geometry.py -v` → Expected: FAIL (module not found)

- [ ] **Step 3: 実装**

`recon3d/geometry.py`:
```python
"""軸平行ボックス分割による形状生成。全入力が軸平行矩形なので、開口の引き算は
長手方向の1次元区間分割+z方向の腰壁/垂れ壁生成で完結する(ブーリアン不要)。"""
import warnings
from dataclasses import dataclass

from recon3d.loader import Rect


@dataclass(frozen=True)
class Box3D:
    x1: float
    y1: float
    z1: float
    x2: float
    y2: float
    z2: float
    category: str


def rect_px_to_mm(r, mm_per_px, image_height):
    return Rect.from_points(
        r.x1 * mm_per_px, (image_height - r.y1) * mm_per_px,
        r.x2 * mm_per_px, (image_height - r.y2) * mm_per_px,
    )


def _overlap_area(a, b):
    w = min(a.x2, b.x2) - max(a.x1, b.x1)
    h = min(a.y2, b.y2) - max(a.y1, b.y1)
    return w * h if (w > 0 and h > 0) else 0.0


def assign_openings(walls, openings):
    assigned = {}
    for o in openings:
        areas = [_overlap_area(w, o) for w in walls]
        best = max(range(len(walls)), key=lambda i: areas[i]) if walls else None
        if best is None or areas[best] <= 0:
            warnings.warn(f"開口 {o} はどの壁とも交差しないため無視する")
            continue
        assigned.setdefault(best, []).append(o)
    return assigned


def subtract_openings_1d(start, end, holes):
    segs = [(start, end)]
    for h1, h2 in sorted(holes):
        out = []
        for s1, s2 in segs:
            a, b = max(s1, h1), min(s2, h2)
            if a >= b:
                out.append((s1, s2))
                continue
            if s1 < a:
                out.append((s1, a))
            if b < s2:
                out.append((b, s2))
        segs = out
    return segs


def _make_box(along, thick, z1, z2, horizontal, category):
    (a1, a2), (t1, t2) = along, thick
    if horizontal:
        return Box3D(a1, t1, z1, a2, t2, z2, category)
    return Box3D(t1, a1, z1, t2, a2, z2, category)


def wall_to_boxes(wall_mm, openings_mm, cfg, base_z):
    horizontal = not wall_mm.is_vertical()
    if horizontal:
        along, thick = (wall_mm.x1, wall_mm.x2), (wall_mm.y1, wall_mm.y2)
    else:
        along, thick = (wall_mm.y1, wall_mm.y2), (wall_mm.x1, wall_mm.x2)
    top = base_z + cfg.floor_height_mm
    boxes = []
    holes = []
    for rect, kind in openings_mm:
        h1, h2 = (rect.x1, rect.x2) if horizontal else (rect.y1, rect.y2)
        h1, h2 = max(h1, along[0]), min(h2, along[1])
        if h1 >= h2:
            continue
        if kind == "window":
            z1 = base_z + cfg.window_sill_mm
            z2 = z1 + cfg.window_height_mm
        else:
            z1, z2 = base_z, base_z + cfg.door_height_mm
        holes.append((h1, h2, z1, z2, kind))
    # 長手方向の残り(全高)
    for s1, s2 in subtract_openings_1d(along[0], along[1], [(h[0], h[1]) for h in holes]):
        boxes.append(_make_box((s1, s2), thick, base_z, top, horizontal, "wall"))
    # 開口区間の腰壁・垂れ壁とpane
    tc = (thick[0] + thick[1]) / 2
    pane_thick = (tc - cfg.pane_thickness_mm / 2, tc + cfg.pane_thickness_mm / 2)
    for h1, h2, z1, z2, kind in holes:
        if z1 > base_z:
            boxes.append(_make_box((h1, h2), thick, base_z, z1, horizontal, "wall"))
        if z2 < top:
            boxes.append(_make_box((h1, h2), thick, z2, top, horizontal, "wall"))
        boxes.append(_make_box((h1, h2), pane_thick, z1, z2, horizontal, f"pane_{kind}"))
    return boxes


def plan_to_boxes(plan, mm_per_px, cfg, base_z):
    ih = plan.image_size[1]
    walls_mm = [rect_px_to_mm(w, mm_per_px, ih) for w in plan.walls]
    openings = ([(rect_px_to_mm(o, mm_per_px, ih), "window") for o in plan.windows]
                + [(rect_px_to_mm(o, mm_per_px, ih), "door") for o in plan.doors])
    assigned = assign_openings(walls_mm, [r for r, _ in openings])
    kind_of = {id(r): k for r, k in openings}
    boxes = []
    for i, w in enumerate(walls_mm):
        ops = [(r, kind_of[id(r)]) for r in assigned.get(i, [])]
        boxes.extend(wall_to_boxes(w, ops, cfg, base_z))
    for c in plan.columns:
        r = rect_px_to_mm(c, mm_per_px, ih)
        boxes.append(Box3D(r.x1, r.y1, base_z, r.x2, r.y2,
                           base_z + cfg.floor_height_mm, "column"))
    return boxes
```

- [ ] **Step 4: テスト通過確認**

Run: `tests/test_geometry.py -v` → Expected: 全PASS

- [ ] **Step 5: Commit**

```bash
cd /home/maren/programming/reconstruction_3d && git add recon3d && \
  git commit -m "feat: opening assignment and axis-aligned box subtraction geometry"
```

---

### Task 4: mesh.py — trimeshシーン生成とスラブ・glb/obj出力

**Files:**
- Create: `recon3d/mesh.py`
- Test: `recon3d/tests/test_mesh.py`

**Interfaces:**
- Consumes: `geometry.Box3D`, `loader.Rect`(mm), `config.ReconConfig`
- Produces:
  ```python
  COLORS = {"wall": (205, 205, 210, 255), "column": (150, 120, 90, 255),
            "pane_window": (140, 200, 255, 120), "pane_door": (170, 130, 80, 200),
            "slab": (235, 230, 220, 255)}

  def footprint_polygon(walls_mm: list) -> shapely.geometry.Polygon
      # 壁矩形のunion→最大ポリゴンの外周(穴は埋める)。スラブ用
  def slab_mesh(poly, z1: float, z2: float) -> trimesh.Trimesh
      # extrude_polygon(earcut)でz1..z2のスラブ
  def boxes_to_scene(boxes: list, slabs: list) -> trimesh.Scene
      # slabs = [(polygon, z1, z2)]。カテゴリ色を頂点色で付与
  def export_scene(scene, out_dir: str, stem: str = "model") -> dict
      # {"glb": path, "obj": path} を書き出して返す
  ```

- [ ] **Step 1: 失敗するテストを書く**

`recon3d/tests/test_mesh.py`:
```python
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
```

- [ ] **Step 2: 失敗確認**

Run: `tests/test_mesh.py -v` → Expected: FAIL (module not found)

- [ ] **Step 3: 実装**

`recon3d/mesh.py`:
```python
"""Box3D列+スラブからtrimesh.Sceneを組み立て、glb/objへ書き出す。"""
import os

import numpy as np
import trimesh
from shapely.geometry import Polygon, box as shapely_box
from shapely.ops import unary_union

COLORS = {
    "wall": (205, 205, 210, 255),
    "column": (150, 120, 90, 255),
    "pane_window": (140, 200, 255, 120),
    "pane_door": (170, 130, 80, 200),
    "slab": (235, 230, 220, 255),
}


def footprint_polygon(walls_mm):
    union = unary_union([shapely_box(r.x1, r.y1, r.x2, r.y2) for r in walls_mm])
    if union.geom_type == "MultiPolygon":
        union = max(union.geoms, key=lambda g: g.area)
    return Polygon(union.exterior)  # 中庭等の穴は埋める


def slab_mesh(poly, z1, z2):
    m = trimesh.creation.extrude_polygon(poly, height=z2 - z1)
    m.apply_translation((0, 0, z1))
    return m


def _box_mesh(b):
    m = trimesh.creation.box(extents=(b.x2 - b.x1, b.y2 - b.y1, b.z2 - b.z1))
    m.apply_translation(((b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2, (b.z1 + b.z2) / 2))
    return m


def _paint(mesh, category):
    mesh.visual.face_colors = np.tile(COLORS[category], (len(mesh.faces), 1))
    return mesh


def boxes_to_scene(boxes, slabs):
    scene = trimesh.Scene()
    by_cat = {}
    for b in boxes:
        by_cat.setdefault(b.category, []).append(_box_mesh(b))
    for cat, meshes in by_cat.items():
        scene.add_geometry(_paint(trimesh.util.concatenate(meshes), cat),
                           geom_name=cat)
    for i, (poly, z1, z2) in enumerate(slabs):
        scene.add_geometry(_paint(slab_mesh(poly, z1, z2), "slab"),
                           geom_name=f"slab_{i}")
    return scene


def export_scene(scene, out_dir, stem="model"):
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for ext in ("glb", "obj"):
        p = os.path.join(out_dir, f"{stem}.{ext}")
        scene.export(p)
        paths[ext] = p
    return paths
```

- [ ] **Step 4: テスト通過確認**

Run: `tests/test_mesh.py -v` → Expected: 全PASS
(`extrude_polygon` がtriangulationエンジン不足で落ちる場合は mapbox_earcut のインストールを再確認)

- [ ] **Step 5: Commit**

```bash
cd /home/maren/programming/reconstruction_3d && git add recon3d && \
  git commit -m "feat: trimesh scene assembly with slabs and glb/obj export"
```

---

### Task 5: build.py — CLI統合(28-AAAAAの2階建てモデル生成)

**Files:**
- Create: `recon3d/build.py`
- Test: `recon3d/tests/test_integration.py`

**Interfaces:**
- Consumes: Task1-4の全API
- Produces:
  ```python
  def build_model(floor_paths: list, cfg: ReconConfig, out_dir: str) -> dict
      # floor_paths = [("1F", path), ("2F", path), ...] 下階から順。
      # スケールは最初の階で校正し全階共通。階iのbase_z = i*floor_height。
      # 各階: plan_to_boxes + 床スラブ(footprint, base_z-150..base_z)。
      # 最上階には天井スラブ(top..top+150)も追加。
      # 戻り値レポート: {"mm_per_px", "scale_cross_check_err", "counts", "outputs"}
      # レポートは out_dir/report.json にも書く
  # CLI: python -m recon3d.build --floors 1F=<path> 2F=<path> \
  #        --width-mm 9100 --depth-mm 9555 --out <dir>
  ```

- [ ] **Step 1: 失敗するテストを書く**

`recon3d/tests/test_integration.py`:
```python
import json
import os

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
    # 建物全体の寸法が通り芯+壁厚の範囲に収まる(10.5m四方・高さ5.8m+スラブ)
    ext = scene.bounds[1] - scene.bounds[0]
    assert 9000 < ext[0] < 9600      # 幅: 9100+壁厚
    assert 9400 < ext[1] < 10000     # 奥行: 9555+壁厚
    assert 5700 < ext[2] < 6300      # 高さ: 2階分5800+スラブ
    assert os.path.exists(os.path.join(str(tmp_path), "report.json"))
    with open(os.path.join(str(tmp_path), "report.json")) as f:
        assert json.load(f)["mm_per_px"] == report["mm_per_px"]
```

- [ ] **Step 2: 失敗確認**

Run: `tests/test_integration.py -v` → Expected: FAIL (module not found)

- [ ] **Step 3: 実装**

`recon3d/build.py`:
```python
"""物件の複数階GT/predictionから3Dモデル一式(glb/obj/report.json)を生成する。

使い方(コンテナ内):
  python -m recon3d.build --floors 1F=/code2/.../1F.json 2F=/code2/.../2F.json \
      --width-mm 9100 --depth-mm 9555 --out /code2/reconstruction_3d/recon3d/output/28-AAAAA
"""
import argparse
import json

from recon3d.calibrate import cross_check_scale, scale_mm_per_px
from recon3d.config import ReconConfig
from recon3d.geometry import plan_to_boxes, rect_px_to_mm
from recon3d.loader import load_floor
from recon3d.mesh import boxes_to_scene, export_scene, footprint_polygon


def build_model(floor_paths, cfg, out_dir):
    plans = [load_floor(p, name) for name, p in floor_paths]
    mm_per_px = scale_mm_per_px(plans[0], cfg.known_width_mm)
    err = cross_check_scale(plans[0], mm_per_px, cfg.known_depth_mm)

    boxes, slabs, counts = [], [], {}
    for i, plan in enumerate(plans):
        base_z = i * cfg.floor_height_mm
        boxes.extend(plan_to_boxes(plan, mm_per_px, cfg, base_z))
        walls_mm = [rect_px_to_mm(w, mm_per_px, plan.image_size[1])
                    for w in plan.walls]
        fp_poly = footprint_polygon(walls_mm)
        slabs.append((fp_poly, base_z - cfg.slab_thickness_mm, base_z))
        if i == len(plans) - 1:  # 最上階の天井スラブ
            top = base_z + cfg.floor_height_mm
            slabs.append((fp_poly, top, top + cfg.slab_thickness_mm))
        counts[plan.name] = {
            "walls": len(plan.walls), "columns": len(plan.columns),
            "windows": len(plan.windows), "doors": len(plan.doors),
        }

    scene = boxes_to_scene(boxes, slabs)
    outputs = export_scene(scene, out_dir)
    report = {
        "mm_per_px": mm_per_px,
        "scale_cross_check_err": err,
        "counts": counts,
        "num_boxes": len(boxes),
        "assumptions": {
            "floor_height_mm": cfg.floor_height_mm,
            "window_sill_mm": cfg.window_sill_mm,
            "window_height_mm": cfg.window_height_mm,
            "door_height_mm": cfg.door_height_mm,
        },
        "outputs": outputs,
    }
    with open(f"{out_dir}/report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return report


def main():
    ap = argparse.ArgumentParser(description="平面図JSON→3Dモデル再構成")
    ap.add_argument("--floors", nargs="+", required=True,
                    help="下階から順に NAME=PATH (例: 1F=/code2/.../1F.json)")
    ap.add_argument("--width-mm", type=float, default=9100.0,
                    help="通り芯 横全幅mm(スケール校正用)")
    ap.add_argument("--depth-mm", type=float, default=9555.0,
                    help="通り芯 縦全幅mm(交差検証用)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = ReconConfig(known_width_mm=args.width_mm, known_depth_mm=args.depth_mm)
    floor_paths = [tuple(f.split("=", 1)) for f in args.floors]
    report = build_model(floor_paths, cfg, args.out)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: テスト通過確認**

Run: 共通テストコマンド(全テスト) → Expected: 全PASS

- [ ] **Step 5: 実際にモデルを生成して確認**

```bash
cd /home/maren/programming && docker compose exec -T cubicasa \
  env PYTHONPATH=/code2/reconstruction_3d python -m recon3d.build \
  --floors 1F=/code2/reconstruction_3d/28-AAAAA/zumen/1F.json \
           2F=/code2/reconstruction_3d/28-AAAAA/zumen/2F.json \
  --out /code2/reconstruction_3d/recon3d/output/28-AAAAA
docker compose exec -T cubicasa chown -R 1001:1001 /code2/reconstruction_3d/recon3d/output
```
Expected: `model.glb` / `model.obj` / `report.json` が生成され、reportの mm_per_px≈11.97

- [ ] **Step 6: Commit**

```bash
cd /home/maren/programming/reconstruction_3d && git add recon3d && \
  git commit -m "feat: build CLI assembling multi-floor 3D model with slabs"
```

---

### Task 6: verify.py — 2D再投影検証とオーバーレイ画像

**Files:**
- Create: `recon3d/verify.py`
- Test: `recon3d/tests/test_integration.py` に追記

**Interfaces:**
- Consumes: `geometry.plan_to_boxes` の出力(Box3D列)、`loader.FloorPlan`
- Produces:
  ```python
  def rasterize_rects(rects_px: list, image_size, grid: int = 1) -> np.ndarray  # bool配列
  def reprojection_iou(plan: FloorPlan, boxes: list, mm_per_px: float) -> float
      # wall/columnカテゴリのBox3Dをpxへ逆投影しGT(壁∪柱)とpixel IoU
  def save_overlay(plan, boxes, mm_per_px, out_png: str) -> None
      # GT=緑 / 再投影=赤枠 の重ね合わせPNG (matplotlib, Agg)
  # CLI: python -m recon3d.verify --floor 1F=<path> --width-mm 9100 --out <dir>
  ```
- 逆投影: `x_px = x_mm / mm_per_px`, `y_px = image_height - y_mm / mm_per_px`
  (geometry.rect_px_to_mm の逆変換)。壁は開口があっても平面フットプリントが保存される
  設計(腰壁・垂れ壁が区間を覆う)なので IoU ≈ 1.0 が期待値

- [ ] **Step 1: 失敗するテストを書く**

`recon3d/tests/test_integration.py` に追記:
```python
from recon3d.geometry import plan_to_boxes
from recon3d.loader import load_floor
from recon3d.verify import reprojection_iou, save_overlay
from recon3d.calibrate import scale_mm_per_px


def test_reprojection_matches_gt(tmp_path):
    cfg = ReconConfig()
    for name in ("1F", "2F"):
        plan = load_floor(f"{GT_DIR}/{name}.json", name)
        plan_1f = load_floor(f"{GT_DIR}/1F.json", "1F")
        s = scale_mm_per_px(plan_1f, cfg.known_width_mm)
        boxes = plan_to_boxes(plan, s, cfg, base_z=0.0)
        iou = reprojection_iou(plan, boxes, s)
        assert iou > 0.99, f"{name}: IoU={iou}"
        png = str(tmp_path / f"{name}.png")
        save_overlay(plan, boxes, s, png)
        assert os.path.getsize(png) > 0
```

- [ ] **Step 2: 失敗確認**

Run: `tests/test_integration.py -v` → Expected: 新テストがFAIL (import error)

- [ ] **Step 3: 実装**

`recon3d/verify.py`:
```python
"""3Dモデルの2D再投影がGTと一致するかの検証。壁+柱のフットプリントIoUと
オーバーレイPNGを出力する。"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from recon3d.calibrate import scale_mm_per_px
from recon3d.config import ReconConfig
from recon3d.geometry import plan_to_boxes
from recon3d.loader import Rect, load_floor

SOLID = ("wall", "column")


def rasterize_rects(rects_px, image_size, grid=1):
    w, h = image_size
    mask = np.zeros((h * grid, w * grid), dtype=bool)
    for r in rects_px:
        x1, y1 = int(round(r.x1 * grid)), int(round(r.y1 * grid))
        x2, y2 = int(round(r.x2 * grid)), int(round(r.y2 * grid))
        mask[max(y1, 0):min(y2, h * grid), max(x1, 0):min(x2, w * grid)] = True
    return mask


def _boxes_to_rects_px(boxes, mm_per_px, image_height):
    rects = []
    for b in boxes:
        if b.category not in SOLID:
            continue
        rects.append(Rect.from_points(
            b.x1 / mm_per_px, image_height - b.y1 / mm_per_px,
            b.x2 / mm_per_px, image_height - b.y2 / mm_per_px))
    return rects


def reprojection_iou(plan, boxes, mm_per_px):
    gt = rasterize_rects(plan.walls + plan.columns, plan.image_size)
    pred = rasterize_rects(
        _boxes_to_rects_px(boxes, mm_per_px, plan.image_size[1]),
        plan.image_size)
    inter = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()
    return float(inter) / float(union) if union else 0.0


def save_overlay(plan, boxes, mm_per_px, out_png):
    fig, ax = plt.subplots(figsize=(12, 9))
    for r in plan.walls + plan.columns:
        ax.add_patch(plt.Rectangle((r.x1, r.y1), r.w, r.h,
                                   facecolor="green", alpha=0.4, edgecolor="none"))
    for r in _boxes_to_rects_px(boxes, mm_per_px, plan.image_size[1]):
        ax.add_patch(plt.Rectangle((r.x1, r.y1), r.w, r.h,
                                   fill=False, edgecolor="red", linewidth=0.6))
    ax.set_xlim(0, plan.image_size[0])
    ax.set_ylim(plan.image_size[1], 0)
    ax.set_aspect("equal")
    ax.set_title(f"{plan.name}: GT(green) vs reprojection(red)")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", required=True, help="NAME=PATH")
    ap.add_argument("--calib-floor", default=None,
                    help="校正に使う階 NAME=PATH (省略時は--floorと同じ)")
    ap.add_argument("--width-mm", type=float, default=9100.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = ReconConfig(known_width_mm=args.width_mm)
    name, path = args.floor.split("=", 1)
    plan = load_floor(path, name)
    cname, cpath = (args.calib_floor or args.floor).split("=", 1)
    s = scale_mm_per_px(load_floor(cpath, cname), cfg.known_width_mm)
    boxes = plan_to_boxes(plan, s, cfg, base_z=0.0)
    iou = reprojection_iou(plan, boxes, s)
    os.makedirs(args.out, exist_ok=True)
    save_overlay(plan, boxes, s, os.path.join(args.out, f"overlay_{name}.png"))
    print(json.dumps({"floor": name, "iou": iou, "mm_per_px": s}))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: テスト通過確認**

Run: 共通テストコマンド(全テスト) → Expected: 全PASS

- [ ] **Step 5: 実GTでオーバーレイ生成・成果物確定**

```bash
cd /home/maren/programming && for f in 1F 2F; do \
  docker compose exec -T cubicasa env PYTHONPATH=/code2/reconstruction_3d \
  python -m recon3d.verify \
  --floor $f=/code2/reconstruction_3d/28-AAAAA/zumen/$f.json \
  --calib-floor 1F=/code2/reconstruction_3d/28-AAAAA/zumen/1F.json \
  --out /code2/reconstruction_3d/recon3d/output/28-AAAAA; done
docker compose exec -T cubicasa chown -R 1001:1001 /code2/reconstruction_3d/recon3d/output
```
Expected: 両階 IoU>0.99、overlay_1F.png / overlay_2F.png 生成。PNGを目視確認する

- [ ] **Step 6: Commit**

```bash
cd /home/maren/programming/reconstruction_3d && git add recon3d && \
  git commit -m "feat: 2D reprojection verification with IoU and overlay images"
```

---

## Self-Review結果

- **Spec coverage:** 入力(GT開始・prediction互換ローダ)/前処理(px→mm・y反転)/形状生成(壁押し出し・開口・柱・スラブ)/出力(glTF+OBJ)/検証(再投影IoU・寸法整合) — 全てタスクに対応あり。1F/2F位置合わせはGTが同一CAD座標系(右端通り芯cx差0.5px実測)のため恒等変換とし、IoU検証で担保
- **仮定値の明示:** ReconConfigに集約し report.json の assumptions に記録(Task5)
- **型整合:** Rect/FloorPlan(Task1) → calibrate(Task2) → Box3D/plan_to_boxes(Task3) → boxes_to_scene(Task4) → build/verify(Task5/6) のシグネチャ一貫を確認済み

---

## 実行時の逸脱記録 (2026-09-03 実装完了)

1. **開口の扱いを変更**: GT実測で窓/ドアは壁の隙間を埋める配置(壁との重なり最大20%)と判明。
   「壁からの引き算」から「開口矩形への充填(wall_infill+pane)」へ変更。
   重なり≥50%の場合のみ引き算(prediction.json互換のため両対応)。`OVERLAP_SUBTRACT=0.5`
2. **フットプリント計算を修正**: 壁のみのunionでは開口の隙間で外周が閉じず断片化する。
   壁+柱+開口のunionに変更(Task 5実装後の統合テストで発覚)
3. **統合テストの寸法期待値を修正**: y方向の建物全長は11069mm(通り芯9555mmの外側に
   ポーチ等の壁が突出)。期待値をGT壁bbox(773×924.9px)×スケール基準に変更
4. **検証結果**: 全25テストPASS、再投影IoU 1F/2F共に1.0、スケール11.968mm/px(交差検証誤差0.28%)
