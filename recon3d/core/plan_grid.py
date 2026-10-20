"""図面(labelme)のグリッド表現 PlanGrid (全系統共通)

plan2model のGTパースと実測スケール校正を再利用し、壁+柱/開口の距離場カーネル、
屋内自由空間 (valid_pose) を持つセルグリッドを作る。
"""
import math

import numpy as np
from scipy import ndimage

from recon3d.plan2model.calibrate import scale_mm_per_px
from recon3d.plan2model.loader import load_labelme

RES_M = 0.05            # グリッド解像度 (m/セル)
SIGMA_WALL_M = 0.20     # 壁一致カーネル幅
SIGMA_OPEN_M = 0.30     # 開口一致カーネル幅
MIN_WALL_CLEAR_M = 0.2  # カメラ位置は壁からこれ以上離れていること


class PlanGrid:
    def __init__(self, paths, floor_name):
        plan = load_labelme(paths.floors / f"{floor_name}.json", floor_name)
        self.name = floor_name
        kw = paths.known_width_mm
        if kw is None:
            raise SystemExit(
                f"図面の実寸が未設定なので {floor_name} をグリッド化できない。"
                f'{paths.floors / "meta.json"} に '
                '{"known_width_mm": 13877.5} を書いてください '
                "(壁注釈だけでは px→m が決まらない)")
        if isinstance(kw, dict):
            kw = kw.get(floor_name)
            if kw is None:
                raise SystemExit(
                    f"known_width_mm に階 {floor_name!r} が無い "
                    f'({paths.floors / "meta.json"})')
        self.m_per_px = scale_mm_per_px(plan, kw) / 1000.0

        rects_wall = plan.walls + plan.columns
        rects_open = plan.windows + plan.doors
        xs = [r.x1 for r in rects_wall] + [r.x2 for r in rects_wall]
        ys = [r.y1 for r in rects_wall] + [r.y2 for r in rects_wall]
        margin = 0.5 / self.m_per_px
        self.x0_px, self.y0_px = min(xs) - margin, min(ys) - margin
        w_px = max(xs) - min(xs) + 2 * margin
        h_px = max(ys) - min(ys) + 2 * margin
        self.cell_px = RES_M / self.m_per_px
        self.nx = int(w_px / self.cell_px) + 1
        self.ny = int(h_px / self.cell_px) + 1

        wall = self._raster(rects_wall)
        opening = self._raster(rects_open)
        d_wall = ndimage.distance_transform_edt(~wall) * RES_M
        d_open = ndimage.distance_transform_edt(~opening) * RES_M
        self.k_wall = np.exp(-((d_wall / SIGMA_WALL_M) ** 2))
        self.k_open = np.exp(-((d_open / SIGMA_OPEN_M) ** 2))
        # レイ透過ペナルティ用: 壁(開口以外)のセル
        self.solid = (wall & ~opening).astype(float)

        # GTの壁は開口(窓・扉)の位置で途切れているため、外周の閉包には開口も含める
        ring = ndimage.binary_closing(wall | opening, iterations=2)
        interior = ndimage.binary_fill_holes(ring) & ~wall & ~opening
        self.valid_pose = interior & (d_wall > MIN_WALL_CLEAR_M)

    def _raster(self, rects):
        mask = np.zeros((self.ny, self.nx), dtype=bool)
        for r in rects:
            cx0 = int((r.x1 - self.x0_px) / self.cell_px)
            cx1 = int(math.ceil((r.x2 - self.x0_px) / self.cell_px))
            cy0 = int((r.y1 - self.y0_px) / self.cell_px)
            cy1 = int(math.ceil((r.y2 - self.y0_px) / self.cell_px))
            mask[max(0, cy0): cy1 + 1, max(0, cx0): cx1 + 1] = True
        return mask

    def cell_to_px(self, cx, cy):
        return self.x0_px + cx * self.cell_px, self.y0_px + cy * self.cell_px
