"""新パイプライン(連結島+図面ICP)の色付き点群を図面座標(メートル)へ書き出す

backend_align が保存するアンカー変換 (results/backend/anchor_<tag>.json) を
連結島の色付き点群 (frontend_bench/<tag>/global_points.ply) に適用する。
参照用にGT図面の壁点列を同梱 (青=1F、水色=2F)。

出力: results/backend/aligned_house.ply
座標系: x,y=図面平面[m] (図面px×11.968mm)、z=高さ (各階の床=0 / 2Fは+2.9m)
"""
import json

import numpy as np

from recon3d.core.io_ply import write_ply
from recon3d.eval.gt_traj import mm_per_px
from recon3d.core.plan_grid import PlanGrid
from recon3d.core.io_ply import read_ply_xyzrgb
from recon3d.postproc.plan_icp import transform_pts, wall_segments_px

SUBSAMPLE = 3


def export(paths, islands, out_name, anchor_suffix="", include_c=False):
    """アンカー済みの島を図面座標へ書き出す

    islands: [(tag, z_off, anchor_suffix[, (h_lo, h_hi)]), …]
    かつてはセッション名で引く既定構成 (SESSION_EXPORT) を持っていたが、
    呼び出し側 (steps._export) が必ず渡すので廃した。
    """
    grids = {}
    all_xyz, all_rgb = [], []
    for isl in islands:
        tag, z_off, sfx = isl[:3]
        # 第4要素で高さクロップを上書き可 (最上階は小屋組・屋根が+3m超に出る)
        h_lo, h_hi = isl[3] if len(isl) > 3 else (-0.3, 3.0)
        a = json.loads(
            (paths.backend / f"anchor_{tag}{sfx or anchor_suffix}.json").read_text())
        grid = grids.setdefault(a["floor"], PlanGrid(paths, a["floor"]))
        xyz, rgb = read_ply_xyzrgb(paths.bench / tag / "global_points.ply")
        xyz, rgb = xyz[::SUBSAMPLE], rgb[::SUBSAMPLE]
        R = np.array(a["R_align"])
        p = xyz @ R.T
        p[:, 2] *= -1
        cells = np.stack([(p[:, 2] - a["origin"][1]) / a["cell_units"],
                          (p[:, 0] - a["origin"][0]) / a["cell_units"]], axis=1)
        if a["mirror_col"] is not None:
            cells[:, 1] = a["mirror_col"] - cells[:, 1]
        from recon3d.mesh.meshing import anchor_transform_cells
        pc = anchor_transform_cells(cells, a)
        xm = (grid.x0_px + pc[:, 1] * grid.cell_px) * mm_per_px(paths) / 1000
        ym = (grid.y0_px + pc[:, 0] * grid.cell_px) * mm_per_px(paths) / 1000
        # 鉛直スケールはアンカーの水平スケールから導出 (島座標は等方。
        # カメラ高さ仮定由来のm_per_unitは約2倍過大だった)。
        # 1セルの実メートルは階ごとに異なる (2Fは図面のmm/pxが違い0.04m/セル)。
        # RES_M固定にすると2Fの高さが25%伸びるので cell_px から実測値を使う
        # 高さの向き: 整列フレームはy+が物理下向き (PanoVGGT equirect由来) のため
        # 物理床 floor_phys_y_units から「下ほど低い」向きで測る
        m_per_cell = grid.cell_px * mm_per_px(paths) / 1000
        mpu_z = a["s"] * m_per_cell / a["cell_units"]
        floor_ref = a.get("floor_phys_y_units",
                          a["floor_y_units"])   # 旧anchorはフォールバック
        h = (floor_ref - p[:, 1]) * mpu_z + z_off
        keep = (h > z_off + h_lo) & (h < z_off + h_hi)
        all_xyz.append(np.stack([xm[keep], ym[keep], h[keep]], axis=1))
        all_rgb.append(rgb[keep])
        print(f"{tag} ({a['floor']}): {int(keep.sum()):,} pts "
              f"(wall_resid {a.get('wall_resid_m', float('nan')):.3f}m)")

    # 図面の壁 (参照点カーテン)
    for floor, z_off, color in (("1F", 0.0, (60, 90, 255)),
                                ("2F", 2.9, (90, 200, 255))):
        segs = wall_segments_px(paths, floor) * mm_per_px(paths) / 1000
        pts = []
        for x1, y1, x2, y2 in segs:
            n = max(int(np.hypot(x2 - x1, y2 - y1) / 0.05), 2)
            tt = np.linspace(0, 1, n)
            for hz in np.arange(0, 2.4, 0.3):
                pts.append(np.stack([x1 + tt * (x2 - x1), y1 + tt * (y2 - y1),
                                     np.full(n, hz + z_off)], axis=1))
        pts = np.concatenate(pts)
        all_xyz.append(pts)
        all_rgb.append(np.tile(np.array(color, np.uint8), (len(pts), 1)))

    xyz = np.concatenate(all_xyz).astype(np.float32)
    rgb = np.concatenate(all_rgb).astype(np.uint8)
    write_ply(paths.backend / out_name, xyz, rgb)
    print(f"total {len(xyz):,} pts -> {paths.backend / out_name}")



def main(paths, argv=None):
    """アンカー済みの島をまとめて図面座標の1つのPLYへ書き出す

    島の指定が無ければ `backend/islands.json` (無ければマニフェスト) から
    「アンカー結果があるもの全部」を拾う。
    """
    import argparse
    import json
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--islands", nargs="*", default=None,
                    help="TAG[:Z_OFF] の並び (既定: アンカー済み全島)")
    ap.add_argument("--suffix", default="_corners", help="アンカー接尾辞")
    ap.add_argument("--out", default=None, help="出力PLY名")
    args = ap.parse_args(argv)
    if args.islands:
        isl = []
        for spec in args.islands:
            tag, _, z = spec.partition(":")
            isl.append((tag, float(z or 0.0), args.suffix))
    else:
        from recon3d.mesh.meshing import islands_for
        isl = [(t, 0.0, args.suffix) for t in islands_for(paths)]
    if not isl:
        raise SystemExit("書き出す島が無い (先に anchor を回すこと)")
    export(paths, isl, args.out or f"aligned_house_{paths.name}.ply")
