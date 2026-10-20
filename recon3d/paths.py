"""データセットの所在 — 「どこに何があるか」を一箇所にまとめる

入力は**固定構造の1ディレクトリ**で、それがデータセットそのもの。

    <root>/
      images/   frame_number_*.jpg          必須。360°等距円筒
      masks/    frame_number_*.jpg          人物マスク (白=有効/黒=人物)
      imu/      transitions.json 等          IMU派生物 (外部生成)
      floors/   1F.json 1F.jpg … meta.json   間取り図ラベル
      results/                               出力はここだけ。入力は不変

`Paths` は**引数で引き回す**。モジュール変数に固めない — 固めると
「import より前に値が決まっていること」が暗黙の前提になり、環境変数で
それを担保する羽目になる。実際かつては `IMG_SESSION` がその役目を負い、
未設定だと既定セッションに落ちて**別物件を黙って処理する**事故があった
(2026-10-09)。引数なら渡し忘れが `TypeError` で即座に出る。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

from recon3d.types import PathLike


@dataclass(frozen=True)
class Paths:
    root: Path

    # ---------------------------------------------------------------- 入力
    @property
    def images(self) -> Path:
        return self.root / "images"

    @property
    def masks(self) -> Path | None:
        """人物マスク (白=有効)。無ければ None"""
        d = self.root / "masks"
        return d if d.is_dir() and next(d.glob("frame_number_*"), None) else None

    @property
    def imu_in(self) -> Path:
        """外部生成の IMU 派生物。`core/ingest` が results/imu へ取り込む"""
        return self.root / "imu"

    @property
    def floors(self) -> Path:
        """間取り図 (labelme JSON + jpg + meta.json)"""
        return self.root / "floors"

    # ---------------------------------------------------------------- 出力
    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def pcd(self) -> Path:
        return self.results / "pcd"

    @property
    def seg(self) -> Path:
        return self.results / "seg"

    @property
    def imu(self) -> Path:
        return self.results / "imu"

    @property
    def bench(self) -> Path:
        """島タグの置き場 (global_cameras.npz / global_points.ply / chain_log)"""
        return self.results / "frontend_bench"

    @property
    def backend(self) -> Path:
        """アンカーjson・aligned_house 等"""
        return self.results / "backend"

    # ------------------------------------------------------------ 図面の実寸
    @cached_property
    def meta(self) -> dict[str, Any]:
        """floors/meta.json (無ければ {})

        図面の実寸 `known_width_mm` (最外壁芯間のx方向スパン[mm]) を持つ。
        **壁注釈だけでは px→m が決まらない**ので、これが無いと図面アンカーは
        動かない (plan2model.calibrate.scale_mm_per_px)。
        """
        p = self.floors / "meta.json"
        if not p.is_file():
            return {}
        try:
            m = json.loads(p.read_text())
        except Exception as e:                      # noqa: BLE001
            raise SystemExit(f"{p} を読めない: {type(e).__name__}: {e}")
        if not isinstance(m, dict):
            raise SystemExit(f"{p}: トップレベルはオブジェクトである必要がある")
        _check_meta(p, m)
        return m

    def floor_jsons(self) -> list[tuple[str, Path]]:
        """間取り図の labelme JSON を**下階から順に**返す [(階名, Path)]

        `meta.json` は図面ではなく実寸の覚え書きなので外す。これを忘れると
        `load_floor` が prediction.json として読もうとして
        `KeyError: image_size` になる。B1F は地下なので負、RF/PH のように
        数字を持たない階名は末尾 (`core/floor_assign.floor_names` と同じ規則)。
        """
        def key(name: str) -> tuple[int, int, str]:
            d = "".join(c for c in name if c.isdigit())
            if not d:
                return (1, 0, name)
            n = int(d)
            return (0, -n if name.upper().startswith("B") else n, name)

        if not self.floors.is_dir():
            return []
        return [(p.stem, p) for p in
                sorted((p for p in self.floors.glob("*.json")
                        if p.stem != "meta" and not p.stem.startswith("_")),
                       key=lambda p: key(p.stem))]

    @property
    def known_width_mm(self) -> float | dict[str, float] | None:
        """階名→mm の辞書、数値ひとつ、または None"""
        return self.meta.get("known_width_mm")

    @property
    def mm_per_px(self) -> float | None:
        """実測スケールが分かっていれば。無ければ図面から導出する"""
        return self.meta.get("mm_per_px")

    @property
    def drawing_wh(self) -> tuple[int, int]:
        wh = self.meta.get("drawing_wh")
        return tuple(wh) if wh else (2480, 1753)    # A4横300dpi

    # -------------------------------------------------------------- 識別子
    @property
    def name(self) -> str:
        """タグ接頭辞などに使う。入力ディレクトリ名"""
        return self.root.name

    # -------------------------------------------------------------- フレーム
    @cached_property
    def frame_indices(self) -> list[int]:
        """フレーム番号の昇順リスト。ゼロ詰め桁数には依存しない"""
        if not self.images.is_dir():
            return []
        out = []
        for p in self.images.glob("frame_number_*.jpg"):
            s = p.stem[len("frame_number_"):]
            if s.isdigit():
                out.append(int(s))
        return sorted(out)

    @cached_property
    def frame_pad(self) -> int:
        """ファイル名のゼロ詰め桁数を実物から決める (既定6)"""
        for p in sorted(self.images.glob("frame_number_*.jpg")) \
                if self.images.is_dir() else ():
            s = p.stem[len("frame_number_"):]
            if s.isdigit():
                return len(s)
        return 6

    def frame_name(self, i: int, ext: str = ".jpg") -> str:
        return f"frame_number_{i:0{self.frame_pad}d}{ext}"

    def frame_path(self, i: int, base: PathLike | None = None,
                   ext: str = ".jpg") -> Path:
        return Path(base or self.images) / self.frame_name(i, ext)

    @property
    def n_frames(self) -> int:
        """フレーム範囲の上限 (= 最大番号+1)。0枚なら 0"""
        idx = self.frame_indices
        return idx[-1] + 1 if idx else 0


def from_input(root: PathLike) -> Paths:
    """入力ディレクトリから Paths を作る。存在しなければ SystemExit"""
    # symlink は**解決しない**。コンテナの /code2 マウント外のパスになるため
    import os
    root = Path(os.path.abspath(os.path.expanduser(str(root))))
    if not root.is_dir():
        raise SystemExit(f"入力ディレクトリが見つからない: {root}")
    return Paths(root=root)


_META_HINT = ('最外壁芯間のx方向スパン[mm]。正の数、'
              '階で違うなら {"1F": 13877.5, "2F": 11375.0}')


def _check_meta(path: Path, m: dict[str, Any]) -> None:
    """型だけ入口で見る

    `known_width_mm` に文字列が入ると `scale_mm_per_px` が TypeError を出すが、
    それが飛ぶのは図面を読む瞬間なので原因が分かりにくい。
    `ext/` の meta.json が持つ s3key / n_frames 等の覚え書きは素通りさせる。
    """
    def bad(k: str, why: str) -> None:
        raise SystemExit(f"{path}: \"{k}\" が不正 ({why}) -> {m[k]!r}")

    kw = m.get("known_width_mm")
    if kw is not None:
        vals = kw.values() if isinstance(kw, dict) else [kw]
        if isinstance(kw, dict) and not all(isinstance(k, str) for k in kw):
            bad("known_width_mm", "階名は文字列")
        if not all(isinstance(v, (int, float)) and v > 0 for v in vals):
            bad("known_width_mm", _META_HINT)
    if m.get("mm_per_px") is not None and not (
            isinstance(m["mm_per_px"], (int, float)) and m["mm_per_px"] > 0):
        bad("mm_per_px", "正の数 [mm/px]")
    wh = m.get("drawing_wh")
    if wh is not None and not (isinstance(wh, (list, tuple)) and len(wh) == 2
                               and all(isinstance(v, int) and v > 0
                                       for v in wh)):
        bad("drawing_wh", "図面画像サイズ [幅, 高さ] の整数2つ")
