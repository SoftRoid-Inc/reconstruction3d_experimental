"""図面位置合わせの精密化 (改善1〜3)

1. 密度マップv2: 高さビン占有数を重みにする (壁=床〜天井まで貫く面だけが強く残る)
2. point-to-line ICP: グリッド照合の解を初期値に、密度の壁セル↔GT壁中心線分で
   相似変換(s,θ,t)を連続最適化
3. チャンク単位補正: 島の精密化変換を初期値に、チャンクごとにICPを再実行して
   島内ドリフトを吸収 → フレーム単位軌跡を組み立ててGT評価

使い方: python main.py tool plan_icp --input /data/x --island islandA --floor 1F
        (--chunks 0:5 --chunks-dir chunks を足すとチャンク単位補正+軌跡評価まで)
"""
from __future__ import annotations

from pathlib import Path

import argparse
import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from recon3d.constants import CAM_HEIGHT_M
from PIL import Image

from recon3d.plan2model.calibrate import scale_mm_per_px
from recon3d.plan2model.loader import load_labelme

from recon3d.core.geometry import camera_centers, gravity_from_cameras, rotation_to_y
from recon3d.postproc.density_map import scale_from_local_floors, slab_peaks
from recon3d.eval.gt_traj import mm_per_px, evaluate_track, load_gt
from recon3d.core.plan_grid import RES_M, PlanGrid
from recon3d.core.io_ply import read_ply_xyzrgb
from recon3d.paths import Paths
from recon3d.types import F64, Points2, Scalar

CELL_M = 0.05
Y_BIN_M = 0.20           # 高さビン幅
MIN_Y_BINS = 4           # 壁とみなす最小の高さビン占有数
SLAB_MARGIN_M = 0.20
ICP_ITERS = 25
ICP_CUTOFF_START = 10.0  # セル単位 (=50cm)
ICP_CUTOFF_END = 3.0


# ---------------------------------------------------------------- 密度 v2

def build_density2(paths: Paths, tag: str) -> Path:
    """島/チャンクの points.ply から高さスパン重み密度を作る (density2.npz)"""
    d = paths.pcd / tag
    xyz, _ = read_ply_xyzrgb(d / "points.ply")
    cams = np.load(d / "cameras.npz")
    conv, up = gravity_from_cameras(cams["poses"])
    R = rotation_to_y(up)
    pts = xyz @ R.T
    cam_pos = camera_centers(cams["poses"], conv) @ R.T
    pts[:, 2] *= -1
    cam_pos = cam_pos.copy()
    cam_pos[:, 2] *= -1
    mpu = scale_from_local_floors(d, cams)

    peaks, _, _ = slab_peaks(pts[:, 1])
    keep = np.ones(len(pts), dtype=bool)
    for p in peaks:
        keep &= np.abs(pts[:, 1] - p) > SLAB_MARGIN_M / mpu
    w = pts[keep]
    lo_x, hi_x = np.percentile(w[:, 0], [2.5, 97.5])
    lo_z, hi_z = np.percentile(w[:, 2], [2.5, 97.5])
    w = w[(w[:, 0] >= lo_x) & (w[:, 0] <= hi_x)
          & (w[:, 2] >= lo_z) & (w[:, 2] <= hi_z)]

    cell_u = CELL_M / mpu
    ybin_u = Y_BIN_M / mpu
    ox, oz = w[:, 0].min() - 10 * cell_u, w[:, 2].min() - 10 * cell_u
    ix = ((w[:, 0] - ox) / cell_u).astype(np.int32)
    iz = ((w[:, 2] - oz) / cell_u).astype(np.int32)
    iy = ((w[:, 1] - w[:, 1].min()) / ybin_u).astype(np.int32)
    nx, nz, ny = ix.max() + 10, iz.max() + 10, iy.max() + 1
    # (z, x) セルごとの高さビン占有数
    flat = (iz.astype(np.int64) * nx + ix) * ny + iy
    occupied = np.unique(flat)
    zx = occupied // ny
    span = np.bincount(zx, minlength=nz * nx).reshape(nz, nx)
    density = np.where(span >= MIN_Y_BINS, span, 0).astype(np.float32)
    density /= max(density.max(), 1)

    np.savez_compressed(
        d / "density2.npz", density=density,
        origin=np.array([ox, oz]), cell_units=cell_u, m_per_unit=mpu,
        cam_pos_aligned=cam_pos, names=cams["names"])
    print(f"{tag}: density2 {density.shape}, nonzero {int((density > 0).sum())}")
    return d / "density2.npz"


# ---------------------------------------------------------------- 図面の壁線分

def wall_segments_px(paths: Paths, floor: str) -> F64:
    """GT壁+柱の中心線分 (図面px座標)"""
    plan = load_labelme(paths.floors / f"{floor}.json", floor)
    segs = []
    for r in plan.walls + plan.columns:
        if r.w >= r.h:
            segs.append((r.x1, r.cy, r.x2, r.cy))
        else:
            segs.append((r.cx, r.y1, r.cx, r.y2))
    return np.array(segs, dtype=np.float64)


def wall_face_segments_px(paths: Paths, floor: str) -> F64:
    """GT壁+柱の両面ライン (壁厚を尊重)。工事中は間柱「面」が写るため、
    中心線ICPより面ICPの方がスケールバイアスが出ない (0312で実証)"""
    plan = load_labelme(paths.floors / f"{floor}.json", floor)
    segs = []
    for r in plan.walls + plan.columns:
        if r.w >= r.h:
            segs.append((r.x1, r.y1, r.x2, r.y1))
            segs.append((r.x1, r.y2, r.x2, r.y2))
        else:
            segs.append((r.x1, r.y1, r.x1, r.y2))
            segs.append((r.x2, r.y1, r.x2, r.y2))
    return np.array(segs, dtype=np.float64)


# ---------------------------------------------------------------- ICP

def similarity_from_grid(best, d, grid) -> tuple[float, float, F64]:
    """align_floorplanのグリッド解 → 明示的な相似変換 (density_m座標→plan cell座標)

    plan_cell = s/RES_M * R(θ) @ (p_m - c_m) + t_cell
    (density配列セル v: p_m = v * CELL_M。warp: u = s R (v - c_in) + shift, +(dy,dx))
    mirror解の場合は呼び出し側で density/セル座標を列反転してから使うこと。
    """
    dens = d["density"]
    h, wd = dens.shape
    th = math.radians(best["theta"])
    s = best["s"]
    c_in = np.array([(h - 1) / 2, (wd - 1) / 2])
    R = np.array([[math.cos(th), -math.sin(th)],
                  [math.sin(th), math.cos(th)]])
    corners = np.array([[0, 0], [0, wd - 1], [h - 1, 0], [h - 1, wd - 1]], float)
    lo = ((corners - c_in) @ (s * R).T).min(axis=0)
    t = -lo + np.array([best["dy"], best["dx"]]) - (s * R) @ c_in
    return s, th, t  # cell→cell: u = s R v + t


def transform_pts(pts: Points2, s: Scalar, th: Scalar, t) -> Points2:
    R = np.array([[math.cos(th), -math.sin(th)],
                  [math.sin(th), math.cos(th)]])
    return (s * (pts @ R.T)) + t


def icp_to_walls(pts_cell: Points2, segs_cell: F64, s: Scalar,
                 th: Scalar, t) -> tuple[Scalar, Scalar, F64, float]:
    """point-to-line ICP。pts_cell: (N,2) density配列セル座標 (row,col)。
    segs_cell: (M,4) plan cell座標 (row1,col1,row2,col2)。返り値: (s,θ,t,残差m)"""
    a = segs_cell[:, :2]
    b = segs_cell[:, 2:]
    ab = b - a
    ab2 = (ab ** 2).sum(axis=1).clip(min=1e-9)
    dmin = np.zeros(len(pts_cell))
    m = np.zeros(len(pts_cell), dtype=bool)
    for it in range(ICP_ITERS):
        cutoff = ICP_CUTOFF_START + (ICP_CUTOFF_END - ICP_CUTOFF_START) \
            * it / max(ICP_ITERS - 1, 1)
        p = transform_pts(pts_cell, s, th, t)  # (N,2)
        # 各点→全線分の footpoint と距離 (N,M)
        diff = p[:, None, :] - a[None, :, :]
        tt = np.clip((diff * ab[None, :, :]).sum(-1) / ab2[None, :], 0, 1)
        foot = a[None, :, :] + tt[..., None] * ab[None, :, :]
        dist = np.linalg.norm(p[:, None, :] - foot, axis=-1)
        j = dist.argmin(axis=1)
        dmin = dist[np.arange(len(p)), j]
        m = dmin < cutoff
        if m.sum() < 50:
            break
        src = pts_cell[m]
        dst = foot[np.arange(len(p)), j][m]
        # 2D相似 (点→footpoint) を閉形式で解く
        mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
        sc, dc = src - mu_s, dst - mu_d
        cov = dc.T @ sc / len(sc)
        U, D, Vt = np.linalg.svd(cov)
        S = np.eye(2)
        if np.linalg.det(U) * np.linalg.det(Vt) < 0:
            S[1, 1] = -1
        R2 = U @ S @ Vt
        s = float(np.trace(np.diag(D) @ S) / (sc ** 2).sum() * len(sc))
        th = math.atan2(R2[1, 0], R2[0, 0])
        t = mu_d - s * R2 @ mu_s
    resid_m = float(np.median(dmin[m])) * RES_M if m.sum() else float("nan")
    return s, th, t, resid_m


# ---------------------------------------------------------------- メイン

def density_wall_points(d: dict, thr: float = 0.15) -> Points2:
    zz, xx = np.nonzero(d["density"] > thr)
    return np.stack([zz, xx], axis=1).astype(np.float64)


def cams_to_cells(d: dict) -> Points2:
    cp = d["cam_pos_aligned"]
    return np.stack([(cp[:, 2] - d["origin"][1]) / d["cell_units"],
                     (cp[:, 0] - d["origin"][0]) / d["cell_units"]], axis=1)


def cells_to_px(cells: Points2, grid) -> Points2:
    return np.stack([grid.x0_px + cells[:, 1] * grid.cell_px,
                     grid.y0_px + cells[:, 0] * grid.cell_px], axis=1)


def segs_to_cells(segs_px: F64, grid) -> F64:
    out = np.empty_like(segs_px)
    out[:, 0] = (segs_px[:, 1] - grid.y0_px) / grid.cell_px
    out[:, 1] = (segs_px[:, 0] - grid.x0_px) / grid.cell_px
    out[:, 2] = (segs_px[:, 3] - grid.y0_px) / grid.cell_px
    out[:, 3] = (segs_px[:, 2] - grid.x0_px) / grid.cell_px
    return out


def refine_island(paths: Paths, tag: str, floor: str, gt) -> dict:
    """密度v2再構築 → グリッド照合(align_floorplan) → ICP精密化 → GT評価"""
    import subprocess
    import sys
    d_dir = paths.pcd / tag
    if not (d_dir / "align_summary.json").exists():
        build_density2(paths, tag)
        # グリッド照合は既存のalign_floorplanを density2 で流用するため一時リネーム
        (d_dir / "density.npz").rename(d_dir / "density_v1.npz") \
            if (d_dir / "density.npz").exists() else None
        (d_dir / "density2.npz").rename(d_dir / "density.npz")
        subprocess.run([sys.executable, "-m", "recon3d.postproc.plan_match",
                        "--tag", tag], check=True)

    d = dict(np.load(d_dir / "density.npz"))
    summ = json.loads((d_dir / "align_summary.json").read_text())
    best = summ["results"][floor]
    grid = PlanGrid(paths, floor)

    mirror_col = None
    if best.get("mirror"):
        # 鏡映解: 配列とセル座標を列反転して非鏡映として扱う
        mirror_col = d["density"].shape[1] - 1
        d["density"] = np.ascontiguousarray(d["density"][:, ::-1])
        print(f"  (mirror解 → 列反転で処理)")

    s, th, t = similarity_from_grid(best, d, grid)
    pts = density_wall_points(d)
    if len(pts) > 20000:
        pts = pts[:: len(pts) // 20000 + 1]
    segs = segs_to_cells(wall_segments_px(paths, floor), grid)

    cam_cells = cams_to_cells(d)
    if mirror_col is not None:
        cam_cells[:, 1] = mirror_col - cam_cells[:, 1]

    def track(s_, th_, t_):
        px = cells_to_px(transform_pts(cam_cells, s_, th_, t_), grid)
        return [{"frame": int(str(n).replace("frame_", "")), "floor": floor,
                 "x_px": float(x), "y_px": float(y)}
                for n, (x, y) in zip(d["names"], px)]

    print(f"--- {tag} ({floor}) grid init:")
    evaluate_track(paths, f"{tag} grid(v2density)", track(s, th, t), gt)
    s2, th2, t2, resid = icp_to_walls(pts, segs, s, th, t)
    print(f"ICP: s {s:.3f}->{s2:.3f}, θ {math.degrees(th):.1f}->"
          f"{math.degrees(th2):.1f}deg, wall resid {resid:.3f}m")
    res = evaluate_track(paths, f"{tag} ICP refined", track(s2, th2, t2), gt)
    return {"tag": tag, "floor": floor, "s": s2, "th": th2,
            "t": t2.tolist(), "resid_m": resid, "eval": res,
            "grid": grid, "d": d, "params": (s2, th2, t2),
            "mirror_col": mirror_col}


def fit_similarity(src: Points2, dst: Points2) -> tuple[float, float, F64]:
    """2D相似 (dst ≈ s R src + t) の閉形式解"""
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    sc, dc = src - mu_s, dst - mu_d
    cov = dc.T @ sc / len(sc)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[1, 1] = -1
    R2 = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / (sc ** 2).sum() * len(sc))
    th = math.atan2(R2[1, 0], R2[0, 0])
    t = mu_d - s * R2 @ mu_s
    return s, th, t


def refine_chunks(paths: Paths, floor, chunk_ids, chunks_dir,
                  island_cam_cells,
                  grid, gt):
    """改善2: チャンクごとにICP補正して島内ドリフトを吸収し、フレーム位置を集める

    island_cam_cells: {frame_name: plan cell (row,col)} 島の精密化変換による初期位置
    返り値: {frame: [(plan_cell, resid_m)]}
    """
    segs = segs_to_cells(wall_segments_px(paths, floor), grid)
    frame_pos = {}
    for k in chunk_ids:
        tag = f"{chunks_dir}/chunk_{k:02d}"
        d_dir = paths.pcd / tag
        if not (d_dir / "density2.npz").exists():
            build_density2(paths, tag)
        d = dict(np.load(d_dir / "density2.npz"))
        names = [str(n) for n in d["names"]]
        cam_cells = cams_to_cells(d)
        shared = [i for i, n in enumerate(names) if n in island_cam_cells]
        if len(shared) < 4:
            continue
        src = cam_cells[shared]
        dst = np.stack([island_cam_cells[names[i]] for i in shared])
        s0, th0, t0 = fit_similarity(src, dst)

        pts = density_wall_points(d)
        if len(pts) > 12000:
            pts = pts[:: len(pts) // 12000 + 1]
        s1, th1, t1, resid = icp_to_walls(pts, segs, s0, th0, t0)
        # 発散ガード: カメラが初期位置から大きく動いたらICPを捨てる
        c0 = transform_pts(cam_cells, s0, th0, t0)
        c1 = transform_pts(cam_cells, s1, th1, t1)
        jump = float(np.median(np.linalg.norm(c1 - c0, axis=1))) * RES_M
        if not math.isfinite(resid) or jump > 1.0:
            s1, th1, t1, c1 = s0, th0, t0, c0
            resid = float("nan")
        print(f"  chunk {k:02d}: resid {resid:.3f}m jump {jump:.2f}m")
        for n, cell in zip(names, c1):
            fr = int(n.replace("frame_", ""))
            frame_pos.setdefault(fr, []).append((cell, resid))
    return frame_pos


ISLANDS = [("islandA{s}", "1F", range(0, 5)),
           ("islandB{s}", "2F", range(6, 12)),
           ("islandC{s}", "1F", range(12, 13))]


def run_full(paths: Paths, masked, gt) -> None:
    sfx = "_m" if masked else ""
    chunks_dir = "chunks_masked" if masked else "chunks"
    assembled = {}
    for tag_t, floor, chunk_ids in ISLANDS:
        tag = tag_t.format(s=sfx)
        res = refine_island(paths, tag, floor, gt)
        # 島の精密化カメラ位置 (plan cell) を初期値にチャンク補正
        cam_cells = {}
        d = res["d"]
        cc = cams_to_cells(d)
        if res.get("mirror_col") is not None:
            cc[:, 1] = res["mirror_col"] - cc[:, 1]
        cells = transform_pts(cc, *res["params"])
        for n, c in zip(d["names"], cells):
            cam_cells[str(n)] = c
        print(f"--- {tag} チャンク補正:")
        frame_pos = refine_chunks(paths, floor, chunk_ids, chunks_dir,
                                  cam_cells, res["grid"], gt)
        for fr, lst in frame_pos.items():
            cell = np.mean([c for c, _ in lst], axis=0)
            px = cells_to_px(cell[None], res["grid"])[0]
            cand = {"frame": fr, "floor": floor,
                    "x_px": float(px[0]), "y_px": float(px[1])}
            # 島間の重複フレームは後勝ちでなく1F/2F境界をGTに依らず残す:
            # 既存エントリがあれば両方保持せず、チャンク数が多い方を採用
            if fr not in assembled or len(lst) >= assembled[fr]["_n"]:
                cand["_n"] = len(lst)
                assembled[fr] = cand
    track = [dict((k, v) for k, v in t.items() if k != "_n")
             for t in sorted(assembled.values(), key=lambda x: x["frame"])]
    print("=== 全体軌跡 (チャンク補正済み) ===")
    res = evaluate_track(paths, f"assembled{'_masked' if masked else ''}", track, gt)
    out = paths.results / "eval_gt" / f"assembled{'_masked' if masked else ''}.json"
    out.write_text(json.dumps({"eval": res, "track": track}, indent=1))
    print(f"-> {out}")


def main(paths: Paths, argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--island")
    ap.add_argument("--floor")
    ap.add_argument("--full", action="store_true",
                    help="全島+チャンク補正+全体軌跡の組み立てと評価")
    ap.add_argument("--masked", action="store_true",
                    help="マスク付きチャンク(chunks_masked / island*_m)を使う")
    args = ap.parse_args(argv)
    gt = load_gt(paths)
    if args.full:
        run_full(paths, args.masked, gt)
    else:
        refine_island(paths, args.island, args.floor, gt)

