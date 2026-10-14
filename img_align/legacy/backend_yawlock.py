"""方位ロック拘束つき後段最適化

Insta360のFlowState方位ロックにより全フレームのヨーは世界基準で一定
(実測: 健全チャンクで±1°)。したがって図面上の向きも単一の未知定数θ0。
各チャンクの図面アンカー回転を th_k = θ0 − median(チャンク内フレームヨー) に固定し、
ブロック座標降下では位置・スケールのみを解く。回転の自由度が全域で消えるため、
ポーズ回転が壊れたチャンク(chunk_04等)の系統誤差を構造的に排除する。

ヨーがチャンク中央値から大きく外れるフレーム(ポーズ不良)は、
共有カメラ整合と最終位置平均の重みを下げる。

使い方: python -m img_align.legacy.backend_yawlock
出力: results/backend/trajectory_yawlock.json / trajectory_yawlock_{1F,2F}.jpg
"""
import json
import math

import numpy as np

from img_align.config import OUT_DIR
from img_align.postproc.anchor import assemble_and_eval
from img_align.legacy.backend_refine import CAM_WEIGHT, CHUNK_FLOOR, CUTOFF_CELL, PRIOR_WEIGHT, Ch
from img_align.core.geometry import gravity_from_cameras, rotation_to_y
from img_align.eval.gt_traj import load_gt
from img_align.legacy.joint_refine import wall_foot
from img_align.core.plan_grid import PlanGrid
from img_align.postproc.plan_icp import cells_to_px, fit_similarity, segs_to_cells, transform_pts, wall_segments_px

ROUNDS = 8
DAMP = 0.6
YAW_BAD_DEG = 5.0    # チャンク中央値からの逸脱がこれ超のフレームはポーズ不良
BAD_FRAME_W = 0.1    # 不良フレームの重み係数


def rot2(th):
    return np.array([[math.cos(th), -math.sin(th)],
                     [math.sin(th), math.cos(th)]])


def fit_scale_trans_w(src, dst, w, R):
    """回転固定の重み付きフィット: dst ≈ s (R @ src) + t"""
    rs = src @ R.T
    w = w / w.sum()
    mu_s = (rs * w[:, None]).sum(axis=0)
    mu_d = (dst * w[:, None]).sum(axis=0)
    num = float((w[:, None] * (dst - mu_d) * (rs - mu_s)).sum())
    den = float((w[:, None] * (rs - mu_s) ** 2).sum())
    s = num / max(den, 1e-9)
    t = mu_d - s * mu_s
    return s, t


def chunk_yaw(ch):
    """チャンク内フレームのヨー((row,col)規約) と 中央値・逸脱"""
    cams = np.load(ch.dir / "cameras.npz")
    conv, up = gravity_from_cameras(cams["poses"])
    R = rotation_to_y(up / np.linalg.norm(up))
    fwd = cams["poses"][:, :3, 2] @ R.T
    # 密度座標: col=x, row=−z → 角度 = atan2(row成分, col成分)
    phi = np.arctan2(-fwd[:, 2], fwd[:, 0])
    c = np.exp(1j * phi)
    med = math.atan2(np.median(np.imag(c)), np.median(np.real(c)))
    dev = np.degrees(np.angle(c * np.exp(-1j * med)))
    return phi, med, dev


def main():
    gt = load_gt()
    base = json.loads((OUT_DIR / "trajectory.json").read_text())["track"]
    anchor_px = {t["frame"]: (t["x_px"], t["y_px"]) for t in base
                 if t["source"] == "island"}
    grids = {f: PlanGrid(f) for f in ("1F", "2F")}
    segs = {f: segs_to_cells(wall_segments_px(f), grids[f]) for f in grids}

    chunks = {k: Ch(k) for k in CHUNK_FLOOR}
    yaw = {}
    for k, c in chunks.items():
        phi, med, dev = chunk_yaw(c)
        w = np.where(np.abs(dev) > YAW_BAD_DEG, BAD_FRAME_W, 1.0)
        yaw[k] = {"med": med, "dev": dev, "w": w}
        print(f"{str(k):<8} yaw std {dev.std():5.1f}deg, "
              f"bad frames {int((np.abs(dev) > YAW_BAD_DEG).sum())}/{len(dev)}")

    # 初期変換 (自由回転フィット) → θ0 を健全チャンクの円周中央値で推定
    T = {}
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
    healthy = [k for k in chunks if yaw[k]["dev"].std() < 3.0]
    th0s = [yaw[k]["med"] + T[k][1] for k in healthy]
    c0 = np.exp(1j * np.array(th0s))
    theta0 = math.atan2(np.imag(c0).mean(), np.real(c0).mean())
    print(f"healthy chunks: {healthy}")
    print(f"theta0 = {math.degrees(theta0):.1f} deg "
          f"(spread {np.degrees(np.angle(c0*np.exp(-1j*theta0))).std():.1f})")

    # 回転を固定: th_k = θ0 − チャンクヨー中央値
    anchor = {}
    for k, c in chunks.items():
        th_k = theta0 - yaw[k]["med"]
        s0, _, t0 = T[k]
        # 位置合わせ直し (回転固定で初期 s,t を再フィット)
        grid = grids[c.floor]
        src, dst, w = [], [], []
        for i, n in enumerate(c.names):
            fr = int(n.replace("frame_", ""))
            if fr in anchor_px:
                px, py = anchor_px[fr]
                src.append(c.cam_cells[i])
                dst.append([(py - grid.y0_px) / grid.cell_px,
                            (px - grid.x0_px) / grid.cell_px])
                w.append(yaw[k]["w"][i])
        s, t = fit_scale_trans_w(np.array(src), np.array(dst),
                                 np.array(w), rot2(th_k))
        T[k] = (s, th_k, t)
        anchor[k] = transform_pts(c.cam_cells, *T[k])

    # 共有フレームペア
    keys = list(chunks)
    pairs = [(a, b) for i, a in enumerate(keys) for b in keys[i + 1:]
             if chunks[a].floor == chunks[b].floor
             and set(chunks[a].names) & set(chunks[b].names)]

    for r in range(ROUNDS):
        resids = []
        for k, c in chunks.items():
            R = rot2(T[k][1])
            foot, dist = wall_foot(transform_pts(c.wall_pts, *T[k]),
                                   segs[c.floor])
            m = dist < CUTOFF_CELL
            src = [c.wall_pts[m]]
            dst = [foot[m]]
            w = [np.ones(int(m.sum()))]
            src.append(c.cam_cells)
            dst.append(anchor[k])
            w.append(PRIOR_WEIGHT * yaw[k]["w"])
            for a, b in pairs:
                if k not in (a, b):
                    continue
                nb = chunks[b if k == a else a]
                shared = sorted(set(c.names) & set(nb.names))
                idx_c = [c.names.index(n) for n in shared]
                idx_n = [nb.names.index(n) for n in shared]
                src.append(c.cam_cells[idx_c])
                dst.append(transform_pts(nb.cam_cells[idx_n], *T[nb.key]))
                w.append(CAM_WEIGHT * yaw[k]["w"][idx_c]
                         * yaw[nb.key]["w"][idx_n])
            s2, t2 = fit_scale_trans_w(np.concatenate(src),
                                       np.concatenate(dst),
                                       np.concatenate(w), R)
            s0, th0k, t0 = T[k]
            T[k] = (s0 + DAMP * (s2 - s0), th0k, t0 + DAMP * (t2 - t0))
            resids.append(float(np.median(dist[m])) * 0.05)
        print(f"round {r}: wall resid median {np.median(resids):.3f}m")

    # フレーム位置 (ヨー不良フレームは重み低)
    frame_pos, frame_w, floors = {}, {}, {}
    for k, c in chunks.items():
        px = cells_to_px(transform_pts(c.cam_cells, *T[k]), grids[c.floor])
        for i, n in enumerate(c.names):
            fr = int(n.replace("frame_", ""))
            frame_pos.setdefault(fr, []).append(px[i])
            frame_w.setdefault(fr, []).append(yaw[k]["w"][i])
            floors[fr] = c.floor
    anchored = {}
    for fr, ps in frame_pos.items():
        ws = np.array(frame_w[fr])
        ps = np.array(ps)
        anchored[fr] = tuple((ps * ws[:, None]).sum(axis=0) / ws.sum())
    assemble_and_eval(anchored, floors, gt, label="_yawlock")


if __name__ == "__main__":
    main()
