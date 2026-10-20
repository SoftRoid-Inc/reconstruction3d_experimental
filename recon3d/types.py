"""このパイプラインで繰り返し出てくる型の別名

numpy の配列は形も意味もバラバラなので、**何の配列か**が分かる名前を付ける。
`NDArray[np.float64]` と書いただけでは (N,3) の点群と (N,4,4) の姿勢が
区別できず、注釈を足した意味が薄い。

実行時のチェックはしない (ランタイムの挙動は変えない)。あくまで読み手と
pyright のための記述。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TypeAlias

import numpy as np
import numpy.typing as npt

# ------------------------------------------------------------------ 基本
PathLike: TypeAlias = "str | Path"
F64: TypeAlias = npt.NDArray[np.float64]
I64: TypeAlias = npt.NDArray[np.int64]
U8: TypeAlias = npt.NDArray[np.uint8]
BoolArr: TypeAlias = npt.NDArray[np.bool_]
#: 実数スカラ。numpy の float64 も入るので、角度やスケールの引数はこれで受ける
Scalar: TypeAlias = "float | np.floating[Any]"

# ------------------------------------------------------------------ 幾何
Vec3: TypeAlias = F64            # (3,)
Mat3: TypeAlias = F64            # (3,3) 回転
Mat4: TypeAlias = F64            # (4,4) 同次変換
Points3: TypeAlias = F64         # (N,3) 点群・カメラ中心列
Points2: TypeAlias = F64         # (N,2) 図面px や水平面の座標列
Poses: TypeAlias = F64           # (N,4,4) フレーム姿勢の列
Rgb: TypeAlias = U8              # (N,3) 点の色

#: 相似変換 (スケール, 回転, 並進)。`core.geometry.apply` が食う形
Sim3: TypeAlias = "tuple[float, Mat3, Vec3]"

# ------------------------------------------------------------------ 台帳
#: 島1本の記述。frontend_bench の `*_manifest.json` の要素
#:   {"tag": "pv0912_s001", "a": 33, "b": 134,
#:    "chunks": [3, 4, …], "floor": "1F" | None, "parent": "…"?}
Island: TypeAlias = "dict[str, Any]"
Manifest: TypeAlias = "list[Island]"

#: 島タグ -> その島のゲージへの変換 (`core.island_group` の群れ)
GroupT: TypeAlias = "dict[str, Sim3]"

#: アンカー結果 (`backend/anchor_<tag>*.json`)。
#: R_align / origin / cell_units / s / th / t / floor_phys_y_units …
Anchor: TypeAlias = "dict[str, Any]"

#: GT軌跡。frame -> {"floor": "1F", "x_px": …, "y_px": …}
GT: TypeAlias = "dict[int, dict[str, Any]]"

# ------------------------------------------------------------------ 設定
#: 島の切り方
Segments: TypeAlias = Literal["gt", "imu"]
#: 前段モデル (チャンク置き場とタグが分かれる)
Model: TypeAlias = Literal["pv", "pi3", "holo"]
#: `m_per_unit` の供給源 (優先順: calib > mapany > slab)
ScaleSource: TypeAlias = Literal["calib", "mapany", "slab", "gt"]

#: hydra から来る入れ子の設定。`Ctx.get("chunking.stride")` で引く
Opt: TypeAlias = "dict[str, Any]"

__all__ = [
    "PathLike", "F64", "I64", "U8", "BoolArr", "Scalar",
    "Vec3", "Mat3", "Mat4", "Points3", "Points2", "Poses", "Rgb", "Sim3",
    "Island", "Manifest", "GroupT", "Anchor", "GT",
    "Segments", "Model", "ScaleSource", "Opt",
]
