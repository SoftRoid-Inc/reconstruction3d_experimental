"""後段の精密化: チャンク単位の図面ICP補正 + ループ中心チャンクによるループクロージャ

改善2': 島の剛体アンカーを初期値に、チャンクごとに壁ICPで再スナップ
        (島端に累積するチェーンドリフトを吸収)
改善1': 島の始端・終端フレームを混成した「ループ中心チャンク」(loop_A/loop_B,
        PanoVGGTで推論済み)をグラフに加え、共有フレーム整合で端と端を直接拘束
        (VGGT-Longのloop-centric reconstructionと同型)

ブロック座標降下: 各チャンクの相似変換を
  壁ICP(単項) + 共有フレームのカメラ位置整合(二項) + 島アンカーへの事前(単項)
の重み付き最小二乗で交互更新。

使い方: python -m img_align.legacy.backend_refine
出力: results/backend/trajectory_refined.json / trajectory_refined_{1F,2F}.jpg
"""
import json

import numpy as np

from img_align.config import OUT_DIR
from img_align.postproc.anchor import assemble_and_eval
from img_align.eval.gt_traj import load_gt
from img_align.legacy.joint_refine import fit_similarity_w, wall_foot
from img_align.core.plan_grid import PlanGrid
from img_align.core.panovggt import PCD_DIR
from img_align.postproc.plan_icp import build_density2, cams_to_cells, cells_to_px, density_wall_points, fit_similarity, segs_to_cells, transform_pts, wall_segments_px

CHUNK_FLOOR = {**{k: "1F" for k in (0, 1, 2, 3, 4)},
               **{k: "2F" for k in (7, 8, 9, 10)},
               "loop_A": "1F", "loop_B": "2F"}
CHUNKS_DIR = "chunks_masked"
ROUNDS = 8
CUTOFF_CELL = 6.0
CAM_WEIGHT = 400.0
PRIOR_WEIGHT = 150.0
DAMP = 0.5


class Ch:
    def __init__(self, key):
        name = f"chunk_{key:02d}" if isinstance(key, int) else key
        self.key, self.floor = key, CHUNK_FLOOR[key]
        self.dir = PCD_DIR / CHUNKS_DIR / name
        if not (self.dir / "density2.npz").exists():
            build_density2(f"{CHUNKS_DIR}/{name}")
        self.d2 = dict(np.load(self.dir / "density2.npz"))
        self.names = [str(n) for n in self.d2["names"]]
        self.cam_cells = cams_to_cells(self.d2)
        pts = density_wall_points(self.d2)
        if len(pts) > 8000:
            pts = pts[:: len(pts) // 8000 + 1]
        self.wall_pts = pts


def main():
    gt = load_gt()
    base = json.loads((OUT_DIR / "trajectory.json").read_text())["track"]
    anchor_px = {t["frame"]: (t["x_px"], t["y_px"]) for t in base
                 if t["source"] == "island"}
    grids = {f: PlanGrid(f) for f in ("1F", "2F")}
    segs = {f: segs_to_cells(wall_segments_px(f), grids[f]) for f in grids}

    chunks = {k: Ch(k) for k in CHUNK_FLOOR}
    # 初期変換+事前アンカー: 島剛体ICP解のフレーム位置にフィット
    T, anchor = {}, {}
    for k, c in chunks.items():
        grid = grids[c.floor]
        src, dst = [], []
        for i, n in enumerate(c.names):
            fr = int(n.replace("frame_", ""))
            if fr in anchor_px:
                px, py = anchor_px[fr]
                src.append(c.cam_cells[i])
                dst.append([(py - grid.y0_px) / grid.cell_px,
                            (px - grid.x0_px) / grid.cell_px])
        T[k] = fit_similarity(np.array(src), np.array(dst))
        anchor[k] = transform_pts(c.cam_cells, *T[k])

    # 共有フレームを持つ同一階チャンクペア (ループチャンクは端チャンクと自動で繋がる)
    keys = list(chunks)
    pairs = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if chunks[a].floor != chunks[b].floor:
                continue
            if set(chunks[a].names) & set(chunks[b].names):
                pairs.append((a, b))
    print("graph pairs:", pairs)

    for r in range(ROUNDS):
        resids = []
        for k, c in chunks.items():
            foot, dist = wall_foot(transform_pts(c.wall_pts, *T[k]),
                                   segs[c.floor])
            m = dist < CUTOFF_CELL
            src = [c.wall_pts[m]]
            dst = [foot[m]]
            w = [np.ones(int(m.sum()))]
            src.append(c.cam_cells)
            dst.append(anchor[k])
            w.append(np.full(len(c.cam_cells), PRIOR_WEIGHT))
            for a, b in pairs:
                if k not in (a, b):
                    continue
                nb = chunks[b if k == a else a]
                shared = sorted(set(c.names) & set(nb.names))
                idx_c = [c.names.index(n) for n in shared]
                idx_n = [nb.names.index(n) for n in shared]
                src.append(c.cam_cells[idx_c])
                dst.append(transform_pts(nb.cam_cells[idx_n], *T[nb.key]))
                w.append(np.full(len(shared), CAM_WEIGHT))
            s2, th2, t2 = fit_similarity_w(np.concatenate(src),
                                           np.concatenate(dst),
                                           np.concatenate(w))
            s0, th0, t0 = T[k]
            T[k] = (s0 + DAMP * (s2 - s0), th0 + DAMP * (th2 - th0),
                    t0 + DAMP * (t2 - t0))
            resids.append(float(np.median(dist[m])) * 0.05)
        print(f"round {r}: wall resid median {np.median(resids):.3f}m")

    # フレーム位置: 所属チャンクの変換結果を平均 (ループチャンク含む)
    frame_pos, floors = {}, {}
    for k, c in chunks.items():
        px = cells_to_px(transform_pts(c.cam_cells, *T[k]), grids[c.floor])
        for n, p in zip(c.names, px):
            fr = int(n.replace("frame_", ""))
            frame_pos.setdefault(fr, []).append(p)
            floors[fr] = c.floor
    anchored = {fr: tuple(np.mean(v, axis=0)) for fr, v in frame_pos.items()}
    assemble_and_eval(anchored, floors, gt, label="_refined")


if __name__ == "__main__":
    main()
