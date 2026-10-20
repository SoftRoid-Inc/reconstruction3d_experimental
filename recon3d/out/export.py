"""新パイプライン(連結島+図面ICP)の色付き点群を図面座標(メートル)へ書き出す

backend_align が保存するアンカー変換 (results/backend/anchor_<tag>.json) を
連結島の色付き点群 (frontend_bench/<tag>/global_points.ply) に適用する。
参照用にGT図面の壁点列を同梱 (青=1F、水色=2F)。

出力: results/backend/aligned_house.ply
座標系: x,y=図面平面[m] (図面px×11.968mm)、z=高さ (各階の床=0 / 2Fは+2.9m)
"""
import json

import numpy as np

from recon3d.config import BENCH_DIR
from recon3d.core.io_ply import write_ply
from recon3d.eval.gt_traj import mm_per_px
from recon3d.core.plan_grid import PlanGrid
from recon3d.core.io_ply import read_ply_xyzrgb
from recon3d.postproc.plan_icp import transform_pts, wall_segments_px

OUT_DIR = BENCH_DIR.parent / "backend"
# セッション別の書き出し構成: (tag, z_off, アンカー接尾辞)
# 0828: 島C=C_full(コーナーアンカー~0.1m)含む。0312: 島Cは~2mずれ未解決のため除外
from recon3d.config import SESSION  # noqa: E402

SESSION_EXPORT = {
    "0828": {"out": "aligned_house.ply",
             "islands": [("panovggt_A_yawfix", 0.0, "_rays"),
                         ("panovggt_Bcore_yawfix", 2.9, "_rays"),
                         ("panovggt_C_full", 0.0, "_rays")]},
    "0312": {"out": "aligned_house_0312.ply",
             "islands": [("cm0312_1F", 0.0, "_face"),
                         ("cm0312_2F", 2.9, "_face")]},
    "19176": {"out": "aligned_house_19176.ply",
              "islands": [("cm19176_1F", 0.0, ""),
                          ("cm19176_2F", 2.9, "")]},
    "0428": {"out": "aligned_house_0428.ply",
             "islands": [("cm0428_ALL", 0.0, "")]},
    "0616": {"out": "aligned_house_0616.ply",
             "islands": [("cm0616_1F", 0.0, ""),
                         ("cm0616_2F", 2.9, ""),
                         ("cm0616_3F", 5.8, "")]},
    "0715": {"out": "aligned_house_0715.ply",
             "islands": [("cm0715_1F", 0.0, ""),
                         ("cm0715_2F", 2.9, ""),
                         ("cm0715_3F", 5.8, "")]},
    "0925": {"out": "aligned_house_0925.ply",
             "islands": [("cm0925_1F", 0.0, ""),
                         ("cm0925_2F", 2.9, ""),
                         ("cm0925_3F", 5.8, "")]},
    "0912": {"out": "aligned_house_0912.ply",
             "islands": [("cm0912_1F", 0.0, "_corners"),
                         ("cm0912_2F", 2.9, ""),
                         ("cm0912_3F", 5.8, "")]},
}
ISLANDS = [(t, z) for t, z, _ in SESSION_EXPORT["0828"]["islands"]]  # 互換用
SUBSAMPLE = 3


def export(anchor_suffix="", include_c=False, islands=None, out_name=None):
    """islands/out_name 指定でセッション既定以外の構成も書き出せる
    (例: 自動アンカー検証 islands=[("pv0616_1F",0.0,"_corners"),…])"""
    # 未登録セッション (入力ディレクトリ渡し) には手書きの既定構成が無い。
    # 呼び出し側 (steps._export) が islands/out_name を渡すので、辞書が
    # 無くても困らない。両方とも無いときだけ落とす
    cfg = SESSION_EXPORT.get(SESSION)
    if cfg is None and (islands is None or out_name is None):
        raise SystemExit(
            f"セッション '{SESSION}' には既定の書き出し構成が無い。"
            "islands= と out_name= を渡すか、SESSION_EXPORT に足してください")
    islands = islands if islands is not None else cfg["islands"]
    out_name = out_name or cfg["out"]
    grids = {}
    all_xyz, all_rgb = [], []
    for isl in islands:
        tag, z_off, sfx = isl[:3]
        # 第4要素で高さクロップを上書き可 (最上階は小屋組・屋根が+3m超に出る)
        h_lo, h_hi = isl[3] if len(isl) > 3 else (-0.3, 3.0)
        a = json.loads(
            (OUT_DIR / f"anchor_{tag}{sfx or anchor_suffix}.json").read_text())
        grid = grids.setdefault(a["floor"], PlanGrid(a["floor"]))
        xyz, rgb = read_ply_xyzrgb(BENCH_DIR / tag / "global_points.ply")
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
        xm = (grid.x0_px + pc[:, 1] * grid.cell_px) * mm_per_px() / 1000
        ym = (grid.y0_px + pc[:, 0] * grid.cell_px) * mm_per_px() / 1000
        # 鉛直スケールはアンカーの水平スケールから導出 (島座標は等方。
        # カメラ高さ仮定由来のm_per_unitは約2倍過大だった)。
        # 1セルの実メートルは階ごとに異なる (2Fは図面のmm/pxが違い0.04m/セル)。
        # RES_M固定にすると2Fの高さが25%伸びるので cell_px から実測値を使う
        # 高さの向き: 整列フレームはy+が物理下向き (PanoVGGT equirect由来) のため
        # 物理床 floor_phys_y_units から「下ほど低い」向きで測る
        m_per_cell = grid.cell_px * mm_per_px() / 1000
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
        segs = wall_segments_px(floor) * mm_per_px() / 1000
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
    write_ply(OUT_DIR / out_name, xyz, rgb)
    print(f"total {len(xyz):,} pts -> {OUT_DIR / out_name}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor-suffix", default="",
                    help="'_rays' で開口ランドマーク精密化済みアンカーを使う")
    ap.add_argument("--include-c", action="store_true",
                    help="島C込みの完全版を aligned_house_full.ply に出力")
    args = ap.parse_args()
    export(args.anchor_suffix, include_c=args.include_c)
