"""チャンク変換の同時最適化 (改善2の完全形)

各チャンクの図面座標への相似変換 T_k を、
  (a) 壁ICP: 密度v2の壁セル → GT壁線分の footpoint (単項制約)
  (b) 隣接チャンクとの共有カメラ位置の一致 (二項制約)
の重み付き最小二乗で交互に更新する (ブロック座標降下)。
(a)だけだと各チャンクが独立に図面へ合い、チャンク間で壁が二重化する。

高さは各チャンクの床スラブピークを z=0 (2Fは+2.9m) に正規化する。

出力: results/pcd/chunk_transforms.json (export_aligned が使用) + GT評価
"""
import json
import math

import numpy as np

from img_align.config import RESULTS_DIR
from img_align.core.geometry import camera_centers, gravity_from_cameras, rotation_to_y
from img_align.postproc.density_map import slab_peaks
from img_align.eval.gt_traj import evaluate_track, load_gt
from img_align.core.plan_grid import PlanGrid
from img_align.core.panovggt import PCD_DIR
from img_align.postproc.plan_icp import cams_to_cells, cells_to_px, density_wall_points, fit_similarity, segs_to_cells, transform_pts, wall_segments_px

CHUNKS_DIR = "chunks_masked"
ISLAND_OF = {**{k: ("1F", 0.0) for k in range(0, 5)},
             **{k: ("2F", 2.9) for k in range(6, 12)},
             12: ("1F", 0.0)}
NEIGHBORS = [(0, 1), (1, 2), (2, 3), (3, 4),
             (6, 7), (7, 8), (8, 9), (9, 10), (10, 11),
             (11, 12)]  # 11-12は階跨ぎだが共有フレームがあれば使う
CAM_WEIGHT = 400.0   # 共有カメラ1点の重み (壁セル何点分か)
PRIOR_WEIGHT = 150.0  # 島剛体変換への事前アンカー (カメラ位置1点あたり)
                      # 数十cmの微調整は許すが、壁ICPの回転局所解への逃亡を防ぐ
# chunk12 (islandC) は内部地図自体が歪んでおり、アンカーも隣接制約も逆効果
# (実測: アンカー強→p90 1.6m, 弱→2.1m, 単独壁ICP→0.67m)。単独ICPで解く
LONE_CHUNKS = {12}
ROUNDS = 8
CUTOFF_CELL = 6.0    # 壁対応の距離カットオフ (セル=5cm単位)
DAMP = 0.5           # 更新のダンピング


def fit_similarity_w(src, dst, w):
    """重み付き2D相似 (dst ≈ s R src + t)"""
    w = w / w.sum()
    mu_s = (src * w[:, None]).sum(axis=0)
    mu_d = (dst * w[:, None]).sum(axis=0)
    sc, dc = src - mu_s, dst - mu_d
    cov = (dc * w[:, None]).T @ sc
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[1, 1] = -1
    R2 = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / ((sc ** 2) * w[:, None]).sum())
    th = math.atan2(R2[1, 0], R2[0, 0])
    t = mu_d - s * R2 @ mu_s
    return s, th, t


def wall_foot(pts_plan, segs):
    """各点の最近傍壁線分footpointと距離"""
    a, b = segs[:, :2], segs[:, 2:]
    ab = b - a
    ab2 = (ab ** 2).sum(axis=1).clip(min=1e-9)
    diff = pts_plan[:, None, :] - a[None, :, :]
    tt = np.clip((diff * ab[None, :, :]).sum(-1) / ab2[None, :], 0, 1)
    foot = a[None, :, :] + tt[..., None] * ab[None, :, :]
    dist = np.linalg.norm(pts_plan[:, None, :] - foot, axis=-1)
    j = dist.argmin(axis=1)
    n = np.arange(len(pts_plan))
    return foot[n, j], dist[n, j]


class Chunk:
    def __init__(self, k):
        self.k = k
        self.floor, self.z_off = ISLAND_OF[k]
        d_dir = PCD_DIR / CHUNKS_DIR / f"chunk_{k:02d}"
        self.d2 = dict(np.load(d_dir / "density2.npz"))
        self.names = [str(n) for n in self.d2["names"]]
        self.cam_cells = cams_to_cells(self.d2)
        pts = density_wall_points(self.d2)
        if len(pts) > 8000:
            pts = pts[:: len(pts) // 8000 + 1]
        self.wall_pts = pts


def main():
    gt = load_gt()
    track0 = json.loads(
        (RESULTS_DIR / "eval_gt" / "assembled_masked.json").read_text())["track"]
    pos_px = {t["frame"]: (t["x_px"], t["y_px"]) for t in track0}
    grids = {f: PlanGrid(f) for f in ("1F", "2F")}
    segs = {f: segs_to_cells(wall_segments_px(f), grids[f]) for f in grids}

    # 初期値・事前アンカー: 島剛体照合(refine_islandのICP解)によるカメラ位置
    from img_align.postproc.plan_icp import cells_to_px as _c2p, refine_island, transform_pts as _tp
    island_px = {}
    for tag, floor in (("islandA_m", "1F"), ("islandB_m", "2F"),
                       ("islandC_m", "1F")):
        r = refine_island(tag, floor, gt)
        cc = cams_to_cells(r["d"])
        if r.get("mirror_col") is not None:
            cc[:, 1] = r["mirror_col"] - cc[:, 1]
        px = _c2p(_tp(cc, *r["params"]), r["grid"])
        for n, p in zip(r["d"]["names"], px):
            island_px[str(n)] = p  # 図面px (重複は後の島が上書き)

    chunks = {k: Chunk(k) for k in ISLAND_OF}
    T, anchor = {}, {}
    for k, c in chunks.items():
        grid = grids[c.floor]
        src, dst = [], []
        for i, n in enumerate(c.names):
            if n in island_px:
                px, py = island_px[n]
                src.append(c.cam_cells[i])
                dst.append([(py - grid.y0_px) / grid.cell_px,
                            (px - grid.x0_px) / grid.cell_px])
        T[k] = fit_similarity(np.array(src), np.array(dst))
        anchor[k] = transform_pts(c.cam_cells, *T[k])  # 事前アンカー位置 (plan cell)

    def shared_metric():
        """隣接チャンク間で共有カメラの位置がどれだけ食い違うか (m)"""
        ds = []
        for a, b in NEIGHBORS:
            ca, cb = chunks[a], chunks[b]
            if ca.floor != cb.floor:
                continue
            shared = set(ca.names) & set(cb.names)
            for n in shared:
                pa = transform_pts(ca.cam_cells[[ca.names.index(n)]], *T[a])
                pb = transform_pts(cb.cam_cells[[cb.names.index(n)]], *T[b])
                ds.append(float(np.linalg.norm(pa - pb)) * 0.05)
        return np.array(ds)

    print(f"before: shared-cam disagreement median "
          f"{np.median(shared_metric()):.3f}m / max {shared_metric().max():.3f}m")

    for r in range(ROUNDS):
        resids = []
        for k, c in chunks.items():
            foot, dist = wall_foot(transform_pts(c.wall_pts, *T[k]),
                                   segs[c.floor])
            m = dist < CUTOFF_CELL
            src = [c.wall_pts[m]]
            dst = [foot[m]]
            w = [np.ones(int(m.sum()))]
            # 島剛体解への事前アンカー (壁ICPの回転局所解への逃亡を防ぐ)
            if k not in LONE_CHUNKS:
                src.append(c.cam_cells)
                dst.append(anchor[k])
                w.append(np.full(len(c.cam_cells), PRIOR_WEIGHT))
            for a, b in NEIGHBORS:
                if k not in (a, b):
                    continue
                nb = chunks[b if k == a else a]
                if k in LONE_CHUNKS or nb.k in LONE_CHUNKS:
                    continue
                shared = sorted(set(c.names) & set(nb.names))
                if not shared:
                    continue
                idx_c = [c.names.index(n) for n in shared]
                idx_n = [nb.names.index(n) for n in shared]
                src.append(c.cam_cells[idx_c])
                dst.append(transform_pts(nb.cam_cells[idx_n], *T[nb.k]))
                w.append(np.full(len(shared), CAM_WEIGHT))
            s2, th2, t2 = fit_similarity_w(np.concatenate(src),
                                           np.concatenate(dst),
                                           np.concatenate(w))
            s0, th0, t0 = T[k]
            T[k] = (s0 + DAMP * (s2 - s0),
                    th0 + DAMP * (th2 - th0),
                    t0 + DAMP * (t2 - t0))
            resids.append(float(np.median(dist[m])) * 0.05)
        print(f"round {r}: wall resid median {np.median(resids):.3f}m, "
              f"shared-cam {np.median(shared_metric()):.3f}m")

    print(f"after: shared-cam disagreement median "
          f"{np.median(shared_metric()):.3f}m / max {shared_metric().max():.3f}m")

    # 高さ正規化 (床スラブピーク) と保存
    out = {}
    for k, c in chunks.items():
        d_dir = PCD_DIR / CHUNKS_DIR / f"chunk_{k:02d}"
        from img_align.core.io_ply import read_ply_xyzrgb
        cams = np.load(d_dir / "cameras.npz")
        conv, up = gravity_from_cameras(cams["poses"])
        R = rotation_to_y(up)
        xyz, _ = read_ply_xyzrgb(d_dir / "points.ply")
        y = (xyz[::11] @ R.T)[:, 1]
        peaks, _, _ = slab_peaks(y)
        cam_y = np.median((camera_centers(cams["poses"], conv) @ R.T)[:, 1])
        below = peaks[peaks < cam_y]
        floor_y = float(below.max()) if len(below) else float(cam_y - 1.5)
        s, th, t = T[k]
        out[str(k)] = {"floor": c.floor, "z_off": c.z_off,
                       "s": s, "th": th, "t": list(map(float, t)),
                       "floor_y_units": floor_y}
    (PCD_DIR / "chunk_transforms.json").write_text(json.dumps(out, indent=1))

    # フレーム軌跡の評価
    frame_pos = {}
    for k, c in chunks.items():
        cells = transform_pts(c.cam_cells, *T[k])
        px = cells_to_px(cells, grids[c.floor])
        for n, p in zip(c.names, px):
            fr = int(n.replace("frame_", ""))
            frame_pos.setdefault(fr, []).append((c.floor, p))
    track = []
    for fr in sorted(frame_pos):
        lst = frame_pos[fr]
        floor = max(set(f for f, _ in lst),
                    key=lambda f: sum(1 for x, _ in lst if x == f))
        ps = np.array([p for f, p in lst if f == floor])
        track.append({"frame": fr, "floor": floor,
                      "x_px": float(ps[:, 0].mean()),
                      "y_px": float(ps[:, 1].mean())})
    res = evaluate_track("joint_refined", track, gt)
    (RESULTS_DIR / "eval_gt" / "joint_refined.json").write_text(
        json.dumps({"eval": res, "track": track}, indent=1))
    print(f"-> {PCD_DIR / 'chunk_transforms.json'}")


if __name__ == "__main__":
    main()
