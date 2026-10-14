"""小島を「配置済みの参照島の密度地図」に照合してアンカーする

図面の壁線だけでは小島(壁カバレッジが少ない)の照合は曖昧で偽解に落ちる。
参照島は同じ空間の家具・内装まで含む密な地図なので、地図同士の照合は
はるかに条件が良い。回転は方位ロックの相対ヨーで2候補に固定し、
スケールを粗探索、平行移動はFFT相互相関で解く。最後に図面壁ICPで微修正。

使い方: python -m img_align.legacy.anchor_by_map --tag panovggt_C_full --floor 1F \
          --ref panovggt_A_yawfix
出力: results/backend/anchor_<tag>.json を上書き (backend_alignと同形式)
"""
import argparse
import json
import math

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve

from img_align.config import OUT_DIR
from img_align.postproc.anchor import island_map_yaw, prepare_island
from img_align.core.plan_grid import RES_M, PlanGrid
from img_align.postproc.plan_icp import cells_to_px, icp_to_walls, segs_to_cells, transform_pts, wall_segments_px

# スケールは参照島とのスラブ推定「比」から事前値を作り±35%だけ探索する
# (絶対値はカメラ高さ仮定の系統誤差で信用できないが、比なら相殺される。
#  自由探索は縮退する: 大スケール=セル数増で有利 / 小スケール=精度型で有利)
SCALE_REL = np.geomspace(0.74, 1.35, 15)
TH_OFF_DEG = (-2, 0, 2)
# sgn=0: PanoVGGTは先頭フレーム基準座標系のため方位ロック下で島間ゲージが
# ほぼ共通 → θ_ref自体が最有力。±Δψはψ汚染時の保険
TH_BRANCHES = (0, 1, -1)


def island_wall_cells(tag):
    d = prepare_island(tag)
    zz, xx = np.nonzero(d["density"] > 0.15)
    pts = np.stack([zz, xx], axis=1).astype(np.float64)
    return d, pts


def raster(pts, shape):
    """plan cell座標の点列を占有ラスタへ"""
    img = np.zeros(shape, np.float32)
    ij = np.round(pts).astype(int)
    m = (ij[:, 0] >= 0) & (ij[:, 0] < shape[0]) \
        & (ij[:, 1] >= 0) & (ij[:, 1] < shape[1])
    img[ij[m, 0], ij[m, 1]] = 1.0
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--floor", required=True)
    ap.add_argument("--ref", required=True, help="配置済みの参照島tag")
    args = ap.parse_args()

    grid = PlanGrid(args.floor)
    ref_anchor = json.loads(
        (OUT_DIR / f"anchor_{args.ref}.json").read_text())
    d_ref, pts_ref = island_wall_cells(args.ref)
    if ref_anchor["mirror_col"] is not None:
        pts_ref[:, 1] = ref_anchor["mirror_col"] - pts_ref[:, 1]
    ref_plan = transform_pts(pts_ref, ref_anchor["s"], ref_anchor["th"],
                             np.array(ref_anchor["t"]))
    shape = (grid.ny, grid.nx)
    # 参照は数セル(≈10cm)の位置ずれを許容するようぼかす
    img_ref = gaussian_filter(raster(ref_plan, shape), sigma=2.0)

    d, pts = island_wall_cells(args.tag)
    cam_rc = np.stack(
        [(d["cam_pos_aligned"][:, 2] - d["origin"][1]) / d["cell_units"],
         (d["cam_pos_aligned"][:, 0] - d["origin"][0]) / d["cell_units"]],
        axis=1)
    psi_ref = island_map_yaw(args.ref)
    psi = island_map_yaw(args.tag)

    s_prior = ref_anchor["s"] * ref_anchor["m_per_unit"] / d["m_per_unit"]
    print(f"scale prior {s_prior:.3f} (ref s {ref_anchor['s']:.3f})")
    best = None
    for sgn in TH_BRANCHES:
        for off in TH_OFF_DEG:
            th = ref_anchor["th"] + sgn * (psi_ref - psi) + math.radians(off)
            for s in s_prior * SCALE_REL:
                p = transform_pts(pts, s, th, np.zeros(2))
                # FFT相互相関で最良の平行移動
                img = raster(p - p.min(axis=0), shape)
                corr = fftconvolve(img_ref, img[::-1, ::-1], mode="full")
                idx = np.unravel_index(np.argmax(corr), corr.shape)
                # precision型: 占有セルあたりの一致度 (未正規化だと
                # 大スケール=セル数増で不当に有利になる)
                score = float(corr[idx]) / max(img.sum(), 1.0)
                t = (np.array(idx) - (np.array(img.shape) - 1)) - p.min(axis=0)
                if best is None or score > best[0]:
                    best = (score, s, th, t, sgn)
    score, s, th, t, sgn = best
    print(f"map-match: score={score:.0f} s={s:.3f} "
          f"th={math.degrees(th) % 360:.1f}deg branch={sgn:+d}")

    # 図面壁ICPで微修正
    segs = segs_to_cells(wall_segments_px(args.floor), grid)
    sub = pts[:: max(1, len(pts) // 20000)]
    s2, th2, t2, resid = icp_to_walls(sub, segs, s, th, t)
    print(f"wall ICP: resid {resid:.3f}m")

    px = cells_to_px(transform_pts(cam_rc, s2, th2, t2), grid)
    out = {
        "tag": args.tag, "floor": args.floor,
        "R_align": np.asarray(d["R_align"]).tolist(),
        "floor_y_units": d["floor_y_units"],
        "floor_phys_y_units": d["floor_phys_y_units"],   # 高さ基準 (corners.py参照)
        "m_per_unit": d["m_per_unit"],
        "origin": np.asarray(d["origin"]).tolist(),
        "cell_units": float(d["cell_units"]),
        "mirror_col": None,
        "s": float(s2), "th": float(th2), "t": [float(v) for v in t2],
        "wall_resid_m": resid, "anchored_by": f"map:{args.ref}",
    }
    (OUT_DIR / f"anchor_{args.tag}.json").write_text(json.dumps(out, indent=1))
    names = [str(n) for n in d["names"]]
    track = {n: (float(x), float(y)) for n, (x, y) in zip(names, px)}
    (OUT_DIR / f"map_anchor_{args.tag}.json").write_text(
        json.dumps(track, indent=1))
    print(f"-> anchor_{args.tag}.json / map_anchor_{args.tag}.json")


if __name__ == "__main__":
    main()
