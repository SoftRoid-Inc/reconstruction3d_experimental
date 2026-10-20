"""階層アンカー: 親島(全周)のアンカー軌跡をサブ島へ移送し壁ICPで精密化する

分割サブ島(低ドリフト)への直接コーナーRANSACは「フロア全周のコーナー配置」という
指紋が痩せて偽解が通る(0912/0925で実証)。そこでグローバル配置は全周の親島
(postproc.corners.corner_anchor)が担い、サブ島は親と共有するフレームの図面位置から
相似変換を解いて壁ICPのみで仕上げる。RANSACを使わないため偽解の余地がない。
実証: 0912 2F 親のみ0.528m → サブ精錬後 2F1=0.457 / 2F2=0.281m。

使い方:
  IMG_SESSION=0912 python -m recon3d.postproc.hier_anchor \
      --parent pv0912_2F --floor 2F --subs pv0912_2F1 pv0912_2F2
出力: backend/anchor_<sub>_hier.json (out.export互換) + hier_anchor_<sub>.json (frame→px)
"""
import argparse
import json
import math

import numpy as np

from recon3d.core.plan_grid import PlanGrid
from recon3d.postproc.anchor import prepare_island
from recon3d.postproc.plan_icp import (cells_to_px, icp_to_walls,
                                         segs_to_cells, transform_pts,
                                         wall_segments_px)

MIN_SHARED = 6     # 親と共有する最少フレーム数


def hier_anchor(paths, sub_tag, floor, parent_track, mpu=None):
    """親の軌跡(frame→図面px)からサブ島のアンカーを解く。返り値: (track, resid)"""
    d = prepare_island(paths, sub_tag, mpu=mpu)
    names = [str(n) for n in d["names"]]
    cam = d["cam_pos_aligned"]
    cells = np.stack([(cam[:, 2] - d["origin"][1]) / d["cell_units"],
                      (cam[:, 0] - d["origin"][0]) / d["cell_units"]], axis=1)
    grid = PlanGrid(paths, floor)
    src, dst = [], []
    for i, n in enumerate(names):
        if n in parent_track:
            x, y = parent_track[n]
            src.append(cells[i])
            dst.append([(y - grid.y0_px) / grid.cell_px,
                        (x - grid.x0_px) / grid.cell_px])
    if len(src) < MIN_SHARED:
        raise RuntimeError(f"{sub_tag}: 親との共有フレーム不足 ({len(src)})")
    src, dst = np.array(src), np.array(dst)

    # 2D Umeyama相似 (鏡映なし。共有フレームが直線縮退なら det<0 になり得るが
    # その場合も壁ICPで検証される)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd(dc.T @ sc / len(sc))
    S = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[1, 1] = -1
    R = U @ S @ Vt
    s0 = float(np.trace(np.diag(D) @ S) / (sc ** 2).sum() * len(sc))
    th0 = math.atan2(R[1, 0], R[0, 0])
    t0 = mu_d - s0 * (R @ mu_s)

    zz, xx = np.nonzero(d["density"] > 0.15)
    wall_pts = np.stack([zz, xx], 1).astype(float)
    wall_pts = wall_pts[:: max(1, len(wall_pts) // 20000)]
    segs = segs_to_cells(wall_segments_px(paths, floor), grid)
    s2, th2, t2, resid = icp_to_walls(wall_pts, segs, s0, th0, t0)
    px = cells_to_px(transform_pts(cells, s2, th2, t2), grid)
    print(f"{sub_tag}: shared={len(src)} wall_resid={resid:.3f}m")

    out = {"tag": sub_tag, "floor": floor,
           "R_align": np.asarray(d["R_align"]).tolist(),
           "floor_y_units": d["floor_y_units"],
           "floor_phys_y_units": d["floor_phys_y_units"],
           "m_per_unit": d["m_per_unit"],
           "origin": np.asarray(d["origin"]).tolist(),
           "cell_units": float(d["cell_units"]), "mirror_col": None,
           "s": float(s2), "th": float(th2), "t": [float(v) for v in t2],
           "wall_resid_m": resid,
           "anchored_by": f"hier({len(src)} shared frames)"}
    (paths.backend / f"anchor_{sub_tag}_hier.json").write_text(
        json.dumps(out, indent=1))
    track = {n: (float(a), float(b)) for n, (a, b) in zip(names, px)}
    (paths.backend / f"hier_anchor_{sub_tag}.json").write_text(
        json.dumps(track, indent=1))
    return track, resid


def _eval_vs_gt(paths, track, floor):
    from recon3d.eval.gt_traj import mm_per_px, load_gt
    try:
        gt = load_gt(paths)
    except Exception:
        return None
    errs = [np.hypot(x - gt[fr]["x_px"], y - gt[fr]["y_px"]) * mm_per_px(paths) / 1000
            for n, (x, y) in track.items()
            for fr in [int(n.replace("frame_", ""))]
            if fr in gt and gt[fr]["floor"] == floor]
    if not errs:
        return None
    e = np.array(errs)
    return {"vs_manual_median_m": round(float(np.median(e)), 3),
            "vs_manual_p90_m": round(float(np.percentile(e, 90)), 3),
            "n": len(e)}


def main(paths, argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", required=True,
                    help="親島タグ (corner_anchor_<tag>.json を軌跡源に使う)")
    ap.add_argument("--floor", required=True)
    ap.add_argument("--subs", nargs="+", required=True)
    ap.add_argument("--parent-track", default=None,
                    help="軌跡jsonを直接指定 (既定: corner_anchor_<parent>.json)")
    ap.add_argument("--mpu", type=float, default=None,
                    help="メトリック系サブ島は1.0を指定")
    args = ap.parse_args(argv)
    from pathlib import Path
    tp = (Path(args.parent_track) if args.parent_track
          else paths.backend / f"corner_anchor_{args.parent}.json")
    if not tp.is_absolute():
        tp = paths.backend / tp
    parent_track = json.loads(tp.read_text())
    for sub in args.subs:
        track, _ = hier_anchor(paths, sub, args.floor, parent_track,
                               mpu=args.mpu)
        r = _eval_vs_gt(paths, track, args.floor)
        if r:
            print(f"  {sub}: {json.dumps(r)}")

