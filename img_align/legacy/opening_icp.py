"""開口(窓・扉)ランドマークつき図面ICP

point-to-line ICP (icp_to_walls) は平行壁区間で壁沿い方向が不定(零空間)。
図面の開口矩形は壁沿い位置の絶対ランドマークになる:
点群側では開口が壁密度の「隙間(ギャップ)」として現れるため、
開口軸線に沿う1D占有プロファイルからギャップを検出し、
ギャップ端↔図面開口端 の点対点対応をICPに追加する。

不確かな開口は使わない:
- ギャップ幅が図面開口幅と大きく違う(閉じたドア・カーテン窓など)
- ギャップ両脇に壁スタブが無い(壁の端・未観測領域)

使い方: backend_align から icp_with_openings() を呼ぶ (--openings)
"""
import math

import numpy as np

from recon3d.loader import load_labelme

from img_align.core.plan_grid import RES_M, ZUMEN_DIR

EXT_M = 1.5          # プロファイルを開口の両側へ延長する長さ
CORR_M = 0.20        # 軸線からの許容直交距離 (プロファイル対象点)
SLACK_M = 0.8        # ギャップ探索の許容ずれ (現在の系統誤差~0.5mを覆う)
STUB_M = 0.35        # ギャップ両脇で壁点を要求する幅 / ランドマーク採取幅
MIN_OPEN_M = 0.6     # これより狭い開口は使わない
WIDTH_RATIO = (0.45, 1.8)   # ギャップ幅/開口幅の許容比
ICP_ROUNDS = 4
ICP_ITERS_PER = 6
CUTOFF_CELL = 3.0
W_CAP = 150.0        # ランドマーク1点あたり重みの上限


def opening_axes_cells(floor, grid):
    """図面の開口 → (端点a, 端点b) plan cell座標 (row,col) のリスト"""
    plan = load_labelme(ZUMEN_DIR / f"{floor}.json", floor)
    out = []
    for r in plan.windows + plan.doors:
        if r.w >= r.h:
            p1, p2, width = (r.x1, r.cy), (r.x2, r.cy), r.w
        else:
            p1, p2, width = (r.cx, r.y1), (r.cx, r.y2), r.h
        if width * grid.m_per_px < MIN_OPEN_M:
            continue
        a = np.array([(p1[1] - grid.y0_px) / grid.cell_px,
                      (p1[0] - grid.x0_px) / grid.cell_px])
        b = np.array([(p2[1] - grid.y0_px) / grid.cell_px,
                      (p2[0] - grid.x0_px) / grid.cell_px])
        out.append((a, b))
    return out


def detect_landmarks(pts_t, axes):
    """変換済み点群 pts_t (plan cell) から開口ギャップを検出し、
    (点index, 目標plan cell点) の対応と診断情報を返す"""
    ext = EXT_M / RES_M
    corr = CORR_M / RES_M
    slack = SLACK_M / RES_M
    stub = STUB_M / RES_M
    idx_all, dst_all, diags = [], [], []
    for a, b in axes:
        d = b - a
        L = float(np.linalg.norm(d))
        u_dir = d / L
        perp = np.array([-u_dir[1], u_dir[0]])
        rel = pts_t - a
        u = rel @ u_dir
        v = rel @ perp
        m = (u >= -ext) & (u <= L + ext) & (np.abs(v) <= corr)
        if m.sum() < 20:
            continue
        um = u[m]
        # 1セルビンの占有プロファイル
        lo = -ext
        nb = int(math.ceil(L + 2 * ext))
        occ = np.bincount(np.clip(((um - lo)).astype(int), 0, nb - 1),
                          minlength=nb) > 0
        # 空きビンの連続区間を列挙
        runs = []
        i = 0
        while i < nb:
            if not occ[i]:
                j = i
                while j < nb and not occ[j]:
                    j += 1
                runs.append((lo + i, lo + j))  # u座標 [ga, gb)
                i = j
            else:
                i += 1
        # 図面開口 [0, L] と最も重なるギャップを選ぶ
        best = None
        for ga, gb in runs:
            ov = min(gb, L) - max(ga, 0)
            center_ok = -slack <= (ga + gb) / 2 - L / 2 <= slack
            if ov > 0 and center_ok and (best is None or ov > best[2]):
                best = (ga, gb, ov)
        if best is None:
            diags.append({"L_m": L * RES_M, "status": "no_gap"})
            continue
        ga, gb, _ = best
        ratio = (gb - ga) / L
        if not (WIDTH_RATIO[0] <= ratio <= WIDTH_RATIO[1]):
            diags.append({"L_m": L * RES_M, "status": f"width_ratio {ratio:.2f}"})
            continue
        # 両脇の壁スタブ確認 (STUB幅の過半が占有)
        def stub_ok(u0, u1):
            i0 = int(np.clip(u0 - lo, 0, nb))
            i1 = int(np.clip(u1 - lo, 0, nb))
            return i1 > i0 and occ[i0:i1].mean() >= 0.5
        if not (stub_ok(ga - stub, ga) and stub_ok(gb, gb + stub)):
            diags.append({"L_m": L * RES_M, "status": "no_stub"})
            continue
        # ランドマーク対応: ギャップ両脇のスタブ点を、開口端が合う位置へ
        # 壁沿いシフト (+直交方向は中心線へスナップ)
        pts_idx = np.nonzero(m)[0]
        for shift, sel in (
                (0.0 - ga, (um >= ga - stub) & (um < ga)),
                (L - gb, (um > gb) & (um <= gb + stub))):
            for k in pts_idx[sel]:
                idx_all.append(k)
                dst_all.append(a + (u[k] + shift) * u_dir)
        diags.append({"L_m": round(L * RES_M, 2),
                      "gap_m": round((gb - ga) * RES_M, 2),
                      "shift_m": (round(-ga * RES_M, 2),
                                  round((L - gb) * RES_M, 2)),
                      "status": "matched"})
    if idx_all:
        return np.array(idx_all), np.stack(dst_all), diags
    return np.array([], int), np.zeros((0, 2)), diags


def fit_similarity_w(src, dst, w):
    """重み付き2D相似 (dst ≈ s R src + t) の閉形式解"""
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
    s = float(np.trace(np.diag(D) @ S) / (w[:, None] * sc ** 2).sum())
    th = math.atan2(R2[1, 0], R2[0, 0])
    t = mu_d - s * R2 @ mu_s
    return s, th, t


def icp_with_openings(pts_cell, segs_cell, axes, s, th, t):
    """壁point-to-line + 開口ランドマークの交互ICP

    pts_cell: density配列セル座標 (フル解像度推奨)
    segs_cell: 壁中心線分 (plan cell)。axes: opening_axes_cells の出力
    返り値: (s, th, t, 残差m, 採用開口数)
    """
    from img_align.postproc.plan_icp import transform_pts
    a = segs_cell[:, :2]
    b = segs_cell[:, 2:]
    ab = b - a
    ab2 = (ab ** 2).sum(axis=1).clip(min=1e-9)
    n_matched = 0
    for rnd in range(ICP_ROUNDS):
        p = transform_pts(pts_cell, s, th, t)
        idx_l, dst_l, diags = detect_landmarks(p, axes)
        n_matched = sum(1 for d in diags if d["status"] == "matched")
        if rnd == 0:
            for d in diags:
                print(f"    opening {d}")
        if len(idx_l) == 0:
            print("    opening: no landmarks matched")
            break
        # ランドマーク総重み ≈ 壁点数の50% (1点あたり上限W_CAP)
        w_land = min(W_CAP, 0.5 * len(pts_cell) / len(idx_l))
        for _ in range(ICP_ITERS_PER):
            p = transform_pts(pts_cell, s, th, t)
            diff = p[:, None, :] - a[None, :, :]
            tt = np.clip((diff * ab[None, :, :]).sum(-1) / ab2[None, :], 0, 1)
            foot = a[None, :, :] + tt[..., None] * ab[None, :, :]
            dist = np.linalg.norm(p[:, None, :] - foot, axis=-1)
            j = dist.argmin(axis=1)
            dmin = dist[np.arange(len(p)), j]
            m = dmin < CUTOFF_CELL
            src = np.concatenate([pts_cell[m], pts_cell[idx_l]])
            dst = np.concatenate([foot[np.arange(len(p)), j][m], dst_l])
            w = np.concatenate([np.ones(int(m.sum())),
                                np.full(len(idx_l), w_land)])
            s, th, t = fit_similarity_w(src, dst, w)
    p = transform_pts(pts_cell, s, th, t)
    diff = p[:, None, :] - a[None, :, :]
    tt = np.clip((diff * ab[None, :, :]).sum(-1) / ab2[None, :], 0, 1)
    foot = a[None, :, :] + tt[..., None] * ab[None, :, :]
    dmin = np.linalg.norm(p[:, None, :] - foot, axis=-1).min(axis=1)
    resid = float(np.median(dmin[dmin < CUTOFF_CELL])) * RES_M
    return s, th, t, resid, n_matched
