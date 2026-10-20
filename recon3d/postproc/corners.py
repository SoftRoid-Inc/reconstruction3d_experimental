"""コーナーランドマーク検出 (図面側 + 点群密度側)

図面の出隅・入隅(壁のL字交点)は、開口と違い工事中でも存在する構造的
ランドマークで、小島のアンカーやチャンク単位補正の壁沿い零空間を塞ぐ。

図面側: GT壁中心線分(軸整列)の端点同士の直交衝合から L字/T字交点を列挙。
点群側: 密度地図の支配的壁方位を推定して軸整列し、行/列方向の壁ラン
        (小ギャップ許容)を抽出、ラン端点の直交衝合から交点を検出する。

各コーナーは象限シグネチャ (交点から壁が伸びる向き dx,dy ∈ {+1,-1,0}) を持ち、
照合時の対応候補フィルタに使う (L字は一意な向きを持つ)。

使い方: python -m recon3d.postproc.corners            # 図面+島の検出と検証画像
出力: results/backend/corners_plan_{1F,2F}.jpg / corners_island_<tag>.jpg
"""
import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from recon3d.core.plan_grid import ZUMEN_DIR
from recon3d.postproc.plan_icp import wall_segments_px

PLAN_TOL_PX = 16          # 図面: 端点↔線分の許容距離 (壁厚相当 ~19cm)
DENS_THR = 0.35           # 密度: 強壁セルの閾値 (家具は高さスパンが浅く落ちる)
DENS_MIN_ARM = 16         # 密度: コーナー両腕の最小長 (セル=80cm)
DENS_GAP = 3              # 密度: ラン内の許容ギャップ (セル)
DENS_TOL = 4              # 密度: 端点↔線の許容距離 (セル=20cm)
DENS_BAND = 2             # 密度: 壁線の太さ許容 (±セル)


# ---------------------------------------------------------------- 図面側

def plan_corners(floor):
    """GT壁中心線分から交点を列挙。返り値: list of dict
    {x, y, kind: 'L'|'T', quad: (sx, sy)}  (sx,sy = 壁が伸びる向きの符号)"""
    segs = wall_segments_px(floor)
    horiz = [s for s in segs if abs(s[3] - s[1]) <= abs(s[2] - s[0])]
    vert = [s for s in segs if abs(s[3] - s[1]) > abs(s[2] - s[0])]
    out = []
    for hx1, hy, hx2, _ in horiz:
        hx1, hx2 = sorted((hx1, hx2))
        for vx, vy1, _, vy2 in vert:
            vy1, vy2 = sorted((vy1, vy2))
            # 交点候補 (vx, hy) が両線分の近傍にあるか
            if not (hx1 - PLAN_TOL_PX <= vx <= hx2 + PLAN_TOL_PX
                    and vy1 - PLAN_TOL_PX <= hy <= vy2 + PLAN_TOL_PX):
                continue
            # 各線分が交点からどちらへ伸びるか (両方向に伸びる=貫通)
            h_end = min(abs(vx - hx1), abs(vx - hx2)) <= PLAN_TOL_PX
            v_end = min(abs(hy - vy1), abs(hy - vy2)) <= PLAN_TOL_PX
            sx = 0 if not h_end else (1 if abs(vx - hx1) <= PLAN_TOL_PX else -1)
            sy = 0 if not v_end else (1 if abs(hy - vy1) <= PLAN_TOL_PX else -1)
            if h_end and v_end:
                kind = "L"
            elif h_end or v_end:
                kind = "T"
            else:
                continue   # 十字貫通は稀かつ曖昧なので使わない
            out.append({"x": float(vx), "y": float(hy),
                        "kind": kind, "quad": (int(sx), int(sy))})
    # 近接重複 (同じ角が複数ペアで検出される) をマージ
    return _dedup(out, PLAN_TOL_PX)


def _dedup(corners, tol):
    kept = []
    for c in sorted(corners, key=lambda c: (c["kind"] != "L")):
        if all(math.dist((c["x"], c["y"]), (k["x"], k["y"])) > tol
               for k in kept):
            kept.append(c)
    return kept


# ---------------------------------------------------------------- 密度側

def dominant_angle(pts):
    """壁点群 (row,col) の支配的方位 [0,90) deg。射影ヒストグラム集中度で探索"""
    best = (None, -1)
    for deg in np.arange(0, 90, 1.0):
        th = math.radians(deg)
        R = np.array([[math.cos(th), -math.sin(th)],
                      [math.sin(th), math.cos(th)]])
        q = pts @ R.T
        score = 0.0
        for ax in (0, 1):
            h = np.bincount((q[:, ax] - q[:, ax].min()).astype(int))
            score += float((h.astype(np.float64) ** 2).sum()) / max(h.sum(), 1)
        if score > best[1]:
            best = (deg, score)
    return best[0]


def _runs(mask_1d, min_len):
    """1次元占有列から、DENS_GAPまでのギャップを許容したランを列挙"""
    idx = np.nonzero(mask_1d)[0]
    if len(idx) == 0:
        return []
    runs = []
    start = prev = idx[0]
    for i in idx[1:]:
        if i - prev > DENS_GAP + 1:
            runs.append((start, prev))
            start = i
        prev = i
    runs.append((start, prev))
    return [(a, b) for a, b in runs if b - a + 1 >= min_len]


def _wall_lines(img, axis):
    """占有ラスタから壁「線」を抽出: 行(axis=0)/列(axis=1)の占有数の
    局所ピークを線位置とし、±DENS_BAND帯を潰した1D占有からラン(線分)を得る。
    返り値: list of (line_pos, a, b)"""
    occ = img.sum(axis=1 - axis)
    lines = []
    for i in range(len(occ)):
        if occ[i] < DENS_MIN_ARM:
            continue
        lo, hi = max(0, i - DENS_BAND), min(len(occ), i + DENS_BAND + 1)
        if occ[i] < occ[lo:hi].max():
            continue
        band = (img[lo:hi, :] if axis == 0 else img[:, lo:hi].T).any(axis=0)
        for a, b in _runs(band, DENS_MIN_ARM):
            lines.append((i, a, b))
    # 近接する平行線 (壁厚の両面など) は長い方を残す
    lines.sort(key=lambda t: -(t[2] - t[1]))
    kept = []
    for p, a, b in lines:
        dup = any(abs(p - p2) <= DENS_BAND * 2
                  and min(b, b2) - max(a, a2) > 0.6 * min(b - a, b2 - a2)
                  for p2, a2, b2 in kept)
        if not dup:
            kept.append((p, a, b))
    return kept


def density_corners(pts):
    """密度の壁点 (row,col) からコーナーを検出。
    返り値: (corners list of dict{r, c, kind, quad, arms}, 支配的方位deg)
    r,c は入力と同じ座標系 (回転を戻して返す)"""
    deg = dominant_angle(pts)
    th = math.radians(deg)
    R = np.array([[math.cos(th), -math.sin(th)],
                  [math.sin(th), math.cos(th)]])
    q = pts @ R.T
    o = q.min(axis=0) - 2
    ij = np.round(q - o).astype(int)
    H = ij.max(axis=0) + 3
    img = np.zeros(H, bool)
    img[ij[:, 0], ij[:, 1]] = True

    h_lines = _wall_lines(img, axis=0)   # (row, col_a, col_b)
    v_lines = _wall_lines(img, axis=1)   # (col, row_a, row_b)
    out = []
    for r, a, b in h_lines:
        for c, va, vb in v_lines:
            if not (a - DENS_TOL <= c <= b + DENS_TOL
                    and va - DENS_TOL <= r <= vb + DENS_TOL):
                continue
            h_end = min(abs(c - a), abs(c - b)) <= DENS_TOL
            v_end = min(abs(r - va), abs(r - vb)) <= DENS_TOL
            sx = 0 if not h_end else (1 if abs(c - a) <= DENS_TOL else -1)
            sy = 0 if not v_end else (1 if abs(r - va) <= DENS_TOL else -1)
            if h_end and v_end:
                kind = "L"
            elif h_end or v_end:
                kind = "T"
            else:
                continue
            arms = (min(b - a, vb - va),)
            out.append({"rc_rot": (float(r), float(c)), "kind": kind,
                        "quad": (int(sx), int(sy)), "arm": int(arms[0])})
    # 重複マージ (rot座標のまま、L優先→腕の長い順)
    kept = []
    for cn in sorted(out, key=lambda c: (c["kind"] != "L", -c["arm"])):
        if all(math.dist(cn["rc_rot"], k["rc_rot"]) > DENS_TOL * 2
               for k in kept):
            kept.append(cn)
    # 元座標へ戻す
    for cn in kept:
        rc = (np.array(cn.pop("rc_rot")) + o) @ R
        cn["r"], cn["c"] = float(rc[0]), float(rc[1])
    return kept, deg


# ---------------------------------------------------------------- 照合

MATCH_TOL_M = 0.30        # インライア判定距離
PAIR_ANG_TOL = math.radians(4)
MIN_PAIR_D = 20           # RANSAC対の最小距離 (セル=1m、近すぎる対はスケールが不安定)
MAX_CAND = 900            # 互換対応の上限 (超過はランダム間引き。後期工程の高速化)
REPS_PER_BUCKET = 3       # 並進バケツごとに数え上げる代表仮説数


def _rot2(th):
    return np.array([[math.cos(th), -math.sin(th)],
                     [math.sin(th), math.cos(th)]])


def _quad_rot(quad, k):
    """象限シグネチャ (scol,srow) を90°×k回転 (plan cell系の回転と同じ規約)"""
    v = np.array([quad[1], quad[0]], float)   # (row,col)ベクトル
    v = _rot2(math.radians(90 * k)) @ v
    return (int(round(v[1])), int(round(v[0])))


def match_corners(dens_cs, deg, plan_cs_cell, s_prior=None, s_rel_tol=0.4,
                  top_k=12):
    """密度コーナー↔図面コーナーのRANSAC照合

    dens_cs: density_cornersの出力 (r,c=density cell系, quadは軸整列系)
    deg: density_cornersの支配方位。plan_cs_cell: {rc:(row,col), kind, quad}
    top_k: 返す候補数の上限 (インライア数の降順)。s_rel_tolを広げるときは
    合わせて増やすこと — 固定のままだと偽仮説が増えて正解が押し出される
    (2026-10-13実測: tol 0.4→0.6 で正解が候補から消えた)。
    返り値: 90°分岐kごとの最良候補リスト [{s,th,t,pairs,n,k}] (n降順)。
    Manhattanエイリアス(90/180°偽解)があるため候補は呼び出し側で
    カメラ妥当性などにより検証すること。変換は density cell → plan cell。
    """
    R0 = _rot2(math.radians(deg))
    d_rot = np.array([[c["r"], c["c"]] for c in dens_cs]) @ R0.T
    p_rc = np.array([c["rc"] for c in plan_cs_cell])
    # (k, 並進バケツ) ごとに最良仮説を保持 — 910mmモジュールの並進エイリアスは
    # コーナーのインライア数では同点になりうるため、複数仮説を呼び出し側で
    # 壁被覆率により検証する
    hyps = {}
    rng = np.random.default_rng(0)
    for k in range(4):
        phi = math.radians(90 * k)
        # 象限の互換対応 (L同士は象限一致、Tはどちらか片腕一致で許容)
        compat = np.zeros((len(dens_cs), len(plan_cs_cell)), bool)
        for i, dc in enumerate(dens_cs):
            dq = _quad_rot(dc["quad"], k)
            for j, pc in enumerate(plan_cs_cell):
                compat[i, j] = not (dc["kind"] == "L" and pc["kind"] == "L"
                                    and dq != tuple(pc["quad"]))
        ii, jj = np.nonzero(compat)
        if len(ii) < 2:
            continue
        if len(ii) > MAX_CAND:          # 後期工程はコーナー多 → RANSAC的に間引く
            sel = rng.choice(len(ii), MAX_CAND, replace=False)
            ii, jj = ii[sel], jj[sel]
        dk = d_rot @ _rot2(phi).T
        # ペア仮説の列挙とフィルタをベクトル化 (旧実装のPython二重ループ置換)
        a, b = np.triu_indices(len(ii), k=1)
        m = (ii[a] != ii[b]) & (jj[a] != jj[b])
        a, b = a[m], b[m]
        v_d = dk[ii[b]] - dk[ii[a]]
        v_p = p_rc[jj[b]] - p_rc[jj[a]]
        nd = np.linalg.norm(v_d, axis=1)
        npn = np.linalg.norm(v_p, axis=1)
        m = (nd >= MIN_PAIR_D) & (npn >= MIN_PAIR_D)
        s_all = npn / np.maximum(nd, 1e-9)
        if s_prior is not None:
            m &= (s_all >= s_prior * (1 - s_rel_tol)) \
                & (s_all <= s_prior * (1 + s_rel_tol))
        dang = np.arctan2(v_p[:, 1], v_p[:, 0]) \
            - np.arctan2(v_d[:, 1], v_d[:, 0])
        dang = np.arctan2(np.sin(dang), np.cos(dang))
        m &= np.abs(dang) <= PAIR_ANG_TOL
        idx = np.nonzero(m)[0]
        if not len(idx):
            continue
        cos_, sin_ = np.cos(dang[idx]), np.sin(dang[idx])
        p1 = dk[ii[a[idx]]]
        rot_p1 = np.stack([cos_ * p1[:, 0] - sin_ * p1[:, 1],
                           sin_ * p1[:, 0] + cos_ * p1[:, 1]], axis=1)
        t_all = p_rc[jj[a[idx]]] - s_all[idx, None] * rot_p1
        # 並進バケツごとに代表仮説だけ高コストなインライア数え上げへ
        keys = np.round(t_all / 8).astype(int)
        counted, reps = {}, []
        for o in range(len(idx)):
            key = (int(keys[o, 0]), int(keys[o, 1]))
            if counted.get(key, 0) >= REPS_PER_BUCKET:
                continue
            counted[key] = counted.get(key, 0) + 1
            reps.append(o)
        for o in reps:
            si, da = float(s_all[idx[o]]), float(dang[idx[o]])
            proj = (si * (dk @ _rot2(da).T)) + t_all[o]
            dist = np.linalg.norm(proj[:, None, :] - p_rc[None], axis=2) * 0.05
            dist[~compat] = np.inf
            pairs = []
            avail = np.ones(len(plan_cs_cell), bool)
            for i in range(len(dens_cs)):
                row = np.where(avail, dist[i], np.inf)
                j = int(np.argmin(row))
                if row[j] < MATCH_TOL_M:
                    avail[j] = False
                    pairs.append((i, j))
            score = len(pairs)
            if score < 3:
                continue
            key = (k, int(keys[o, 0]), int(keys[o, 1]))
            if key not in hyps or score > hyps[key]["n"]:
                hyps[key] = {"n": score, "k": k, "pairs": pairs}
    out = []
    from recon3d.postproc.plan_icp import fit_similarity
    for cnd in sorted(hyps.values(), key=lambda b: -b["n"])[:top_k]:
        src = np.array([[dens_cs[i]["r"], dens_cs[i]["c"]]
                        for i, _ in cnd["pairs"]])
        dst = np.array([p_rc[j] for _, j in cnd["pairs"]])
        s, th, t = fit_similarity(src, dst)
        out.append({"s": s, "th": th, "t": t, "pairs": cnd["pairs"],
                    "n": cnd["n"], "k": cnd["k"]})
    return out


# ---------------------------------------------------------------- 検証

def _plot_plan(floor, out_dir):
    corners = plan_corners(floor)
    bg = np.asarray(Image.open(ZUMEN_DIR / f"{floor}.jpg").convert("L"))
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.imshow(bg, cmap="gray")
    for c in corners:
        color = "red" if c["kind"] == "L" else "orange"
        ax.plot(c["x"], c["y"], "o", ms=7, mfc="none", mew=2, color=color)
        ax.annotate(f"{c['quad'][0]:+d}{c['quad'][1]:+d}", (c["x"], c["y"]),
                    fontsize=5, color=color)
    nL = sum(1 for c in corners if c["kind"] == "L")
    ax.set_title(f"{floor}: plan corners  L(red)={nL} T(orange)={len(corners)-nL}")
    ax.set_axis_off()
    fig.savefig(out_dir / f"corners_plan_{floor}.jpg", dpi=130,
                bbox_inches="tight")
    plt.close(fig)
    print(f"{floor}: {nL} L + {len(corners) - nL} T corners")
    return corners


def _plot_island(tag, floor, plan_cs, out_dir):
    """島の密度コーナーを検出し、raysアンカーで図面に載せて重なりを確認"""
    from recon3d.config import OUT_DIR
    from recon3d.postproc.anchor import prepare_island
    from recon3d.core.plan_grid import PlanGrid
    from recon3d.postproc.plan_icp import cells_to_px, transform_pts
    anchor_p = OUT_DIR / f"anchor_{tag}_rays.json"
    if not anchor_p.exists():
        anchor_p = OUT_DIR / f"anchor_{tag}.json"
    a = json.loads(anchor_p.read_text())
    d = prepare_island(tag)
    if a["mirror_col"] is not None:
        d["density"] = np.ascontiguousarray(d["density"][:, ::-1])
    zz, xx = np.nonzero(d["density"] > DENS_THR)
    pts = np.stack([zz, xx], axis=1).astype(np.float64)
    corners, deg = density_corners(pts)
    print(f"{tag}: {len(corners)} density corners (dominant angle {deg:.0f}deg)")

    grid = PlanGrid(floor)
    rc = np.array([[c["r"], c["c"]] for c in corners])
    px = cells_to_px(transform_pts(rc, a["s"], a["th"], np.array(a["t"])), grid)
    wall_px = cells_to_px(
        transform_pts(pts[:: max(1, len(pts) // 40000)],
                      a["s"], a["th"], np.array(a["t"])), grid)

    bg = np.asarray(Image.open(ZUMEN_DIR / f"{floor}.jpg").convert("L"))
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.imshow(bg, cmap="gray")
    ax.scatter(wall_px[:, 0], wall_px[:, 1], s=0.5, c="tomato", alpha=0.4)
    for c in plan_cs:
        color = "blue" if c["kind"] == "L" else "deepskyblue"
        ax.plot(c["x"], c["y"], "s", ms=8, mfc="none", mew=1.6, color=color)
    for (x, y), c in zip(px, corners):
        color = "red" if c["kind"] == "L" else "orange"
        ax.plot(x, y, "o", ms=7, mfc="none", mew=2, color=color)
    ax.set_title(f"{tag}: density corners (red/orange) vs plan corners (blue)")
    ax.set_axis_off()
    fig.savefig(out_dir / f"corners_island_{tag}.jpg", dpi=130,
                bbox_inches="tight")
    plt.close(fig)


def plan_corners_cell(floor, grid):
    """図面コーナーを plan cell 座標へ"""
    out = []
    for c in plan_corners(floor):
        out.append({"rc": ((c["y"] - grid.y0_px) / grid.cell_px,
                           (c["x"] - grid.x0_px) / grid.cell_px),
                    "kind": c["kind"], "quad": c["quad"]})
    return out


# 候補生成の既定 (2026-10-13に実測で決めた。詳細は results/REPORT.md)
# スケール事前には誤差があるので1点に賭けず倍率を掃引し、各倍率の近傍だけを
# 細く探索する。スコア関数が信頼できるようになったので候補を増やしてよい
S_SWEEP = (0.7, 0.85, 1.0, 1.18, 1.4)
S_REL_TOL = 0.25
TOP_K = 24
# ICPは1候補1.43秒かかる一方、採点は0.009秒・cam_feasは0.00002秒 (2026-10-14実測)。
# 全120候補にICPをかけると170秒になるので、**ICP前の素の変換で採点**して
# 上位だけICPにかける。ICP前後でスコアの順位はほぼ保たれる (eval/anchor_oracle で検証)
ICP_TOP_K = 32
# 各候補の90°回転変種も候補に入れる。Manhattan構造では壁の重なりが同じになる
# 分岐が複数あり、RANSACの対応付けだけでは正しい分岐が生成されないことがある
# (2026-10-15実測: 0616_s01m は正解姿勢を入れればスコアで勝てるのに候補に無かった)
ADD_ROT_BRANCHES = True
# 候補の順位付けに使うスコア名 (anchor_score.SCORERS のいずれか)
SCORE_KEY = "f1_feas_scale"


# 継続性の許容距離。歩行速度の上限から出すと緩すぎる (直線移動の仮定)ので、
# **隣接島の端点間距離をGTで実測**して決めた (2026-10-15、全5セッション25対):
# m/フレームの中央値 0.04〜0.20、最大 0.38。安全率1.5をかけて 0.57。
CONT_SPEED = 0.38
CONT_MARGIN = 1.5
CONT_MIN_M = 4.0          # 隙間ゼロでもこれだけは許す (島内の端点ずれを吸収)


def _cont_bound(gap_frames):
    """フレーム差から許容距離[m]を決める (人は瞬間移動しない)"""
    return max(CONT_MIN_M, gap_frames * CONT_SPEED * CONT_MARGIN)


def _nearest_anchored(tag, frames, suffix=""):
    """アンカー済みの他島から、この島に時間的に最も近い基準点を拾う

    返り値 (この島のindex, 基準の図面px, 基準フレーム, フレーム差)。
    無ければ None。`corner_anchor_<tag>.json` を直接読むので、
    先に別の島をアンカーしてあれば自動で効く。
    """
    from recon3d.config import OUT_DIR
    known = {}
    for p in OUT_DIR.glob("corner_anchor_*.json"):
        if p.stem == f"corner_anchor_{tag}":
            continue
        try:
            for n, xy in json.loads(p.read_text()).items():
                f = int(n.replace("frame_", ""))
                if f not in frames:
                    known[f] = tuple(xy)
        except Exception:
            continue
    if not known:
        return None
    j = min(range(len(frames)),
            key=lambda i: min(abs(frames[i] - k) for k in known))
    kf = min(known, key=lambda k: abs(k - frames[j]))
    return j, known[kf], kf, abs(kf - frames[j])


def _rot_branches(cands, cam_rc):
    """各候補を島の重心まわりに90/180/270°回した変種 (重心は動かさない)"""
    c0 = cam_rc.mean(axis=0)
    out = []
    for c in cands:
        for k in (1, 2, 3):
            th = c["th"] + math.radians(90 * k)
            R = _rot2(th)
            # 回転後も元の候補と同じ位置に重心が来るよう t を解き直す
            center = c["s"] * (_rot2(c["th"]) @ c0) + c["t"]
            out.append({"s": c["s"], "th": th,
                        "t": center - c["s"] * (R @ c0),
                        "pairs": c["pairs"], "n": c["n"], "k": c["k"]})
    return out


def corner_anchor(tag, floor, ref_tag=None, cont_max_m=4.0, mpu=None,
                  s_prior=None, suffix="_corners", return_info=False,
                  mpu_source="mapany", sweep=S_SWEEP, continuity_ref=False):
    """島をコーナーRANSAC照合でアンカーし、壁ICPで精密化して保存する

    ref_tag: スケール事前用の配置済み島 (スラブ推定の比。無ければ事前なし)
    cont_max_m: 時系列連続性フィルタ。既存trajectory.jsonの島フレームのうち
    時間的に最近のものとの距離がこれ超の候補を棄却する
    (階段吹き抜けは上下階で同位置のため、階段直後の島はきつく縛れる)
    mpu/s_prior: メトリック出力のフロントエンド (MapAnything条件付け等) は
    mpu=1.0, s_prior≈1.0 を直接指定できる (ref_tag指定時はref優先)
    continuity_ref: 他島のアンカー結果を時系列の基準に使うか。**既定OFF**。
    ONにすると `corner_anchor_*.json` の**残っているファイル**を拾うので、
    並列実行では順序で、再実行では前回の残骸で結果が変わる。
    精度面でも実測で有害だった (2026-10-15: 0616が1/3→0/3。誤った基準に
    引きずられて、正しかった島まで寄ってしまう)
    mpu_source: mpu未指定時のスケール事前の供給源 (postproc/scale_prior)。
    既定の床スラブ推定は2〜5倍ずれるため mapany を既定にしている
    sweep: 事前スケールの倍率掃引。Noneで無効 (従来どおり1点のみ)
    """
    from recon3d.config import OUT_DIR
    from recon3d.postproc.anchor import prepare_island
    from recon3d.core.plan_grid import PlanGrid
    from recon3d.postproc.plan_icp import cells_to_px, icp_to_walls, segs_to_cells, transform_pts
    if mpu is None and mpu_source:
        from recon3d.postproc.scale_prior import m_per_unit
        mpu, minfo = m_per_unit(tag, source=mpu_source)
        print(f"scale prior: m_per_unit={mpu:.3f} ({minfo})")
    grid = PlanGrid(floor)
    d = prepare_island(tag, mpu=mpu)
    zz, xx = np.nonzero(d["density"] > DENS_THR)
    pts = np.stack([zz, xx], axis=1).astype(np.float64)
    dens_cs, deg = density_corners(pts)
    plan_cs = plan_corners_cell(floor, grid)
    if ref_tag:
        ra = json.loads((OUT_DIR / f"anchor_{ref_tag}.json").read_text())
        s_prior = ra["s"] * ra["m_per_unit"] / d["m_per_unit"]
        print(f"scale prior {s_prior:.3f} (from {ref_tag})")
    cam_rc = np.stack(
        [(d["cam_pos_aligned"][:, 2] - d["origin"][1]) / d["cell_units"],
         (d["cam_pos_aligned"][:, 0] - d["origin"][0]) / d["cell_units"]],
        axis=1)
    base = s_prior if s_prior else 1.0
    cands = []
    for mult in (sweep or [1.0]):
        cands += match_corners(dens_cs, deg, plan_cs, s_prior=base * mult,
                               s_rel_tol=S_REL_TOL if sweep else 0.4,
                               top_k=TOP_K if sweep else 12)
    if not cands:
        raise RuntimeError(f"{tag}: corner match failed")
    if ADD_ROT_BRANCHES:
        cands += _rot_branches(cands, cam_rc)

    # 時系列連続性: アンカー済みの隣接島から「この島はここにあるはず」を得る
    frames = [int(str(n).replace("frame_", "")) for n in d["names"]]
    cont_ref = _nearest_anchored(tag, frames) if continuity_ref else None
    if cont_ref:
        j, xy, kf, gap = cont_ref
        print(f"continuity: f{frames[j]} は既知のf{kf}から{gap}フレーム "
              f"(許容 {_cont_bound(gap):.1f}m)")

    def continuity(s_, th_, t_):
        """参照フレームとの距離[m]。人は連続して歩くので離れすぎた解は偽"""
        if cont_ref is None:
            return 0.0
        j, xy, _, _ = cont_ref
        px = cells_to_px(transform_pts(cam_rc[j][None], s_, th_, t_), grid)[0]
        from recon3d.eval.gt_traj import mm_per_px
        return math.dist(px, xy) * mm_per_px() / 1000

    zz2, xx2 = np.nonzero(d["density"] > 0.15)
    wall_pts = np.stack([zz2, xx2], axis=1).astype(np.float64)
    wall_pts = wall_pts[:: max(1, len(wall_pts) // 20000)]
    segs = segs_to_cells(wall_segments_px(floor), grid)
    from recon3d.postproc.anchor_score import AnchorScorer
    scorer = AnchorScorer(grid, wall_pts, segs, cam_rc, s_prior=base)

    # 順位付けスコアの名前 (anchor_score.SCORERS)。A/Bはここを差し替える
    key = SCORE_KEY
    # 順位付けは f1_feas_scale (対称被覆 × カメラ妥当性 × スケール逸脱ペナルティ)。
    # 従来の wall_cover (片方向被覆) は縮小退化のため正解より偽解を好み、
    # 全周島8本で0/8しか当てられなかった (2026-10-13実測)
    # ICP前に素の変換で足切りする (ICPは1本1.43秒、採点は0.009秒)
    pre = []
    for cnd in cands:
        cont = continuity(cnd["s"], cnd["th"], cnd["t"])
        if cont_ref is not None and cont > _cont_bound(cont_ref[3]):
            continue
        v0 = scorer(cnd["s"], cnd["th"], cnd["t"])
        if v0["cam_feas"] < 0.3:     # 明らかに屋外/壁内の解だけ落とす
            continue
        pre.append((v0[key], cnd))
    pre.sort(key=lambda x: -x[0])
    print(f"候補 {len(cands)} -> 足切り後 {len(pre)} -> ICP {min(len(pre), ICP_TOP_K)}本")

    best, scores = None, []
    for _, cnd in pre[:ICP_TOP_K]:
        s2, th2, t2, resid = icp_to_walls(wall_pts, segs,
                                          cnd["s"], cnd["th"], cnd["t"])
        v = scorer(s2, th2, t2)
        if v["cam_feas"] < 0.3:
            continue
        scores.append(v[key])
        if best is None or v[key] > best[0]:
            best = (v[key], (s2, th2, t2, resid, cnd["n"]), v)
    if best is None:
        raise RuntimeError(f"{tag}: 全候補がカメラ妥当性/連続性で棄却された")
    s2, th2, t2, resid, n_in = best[1]
    v = best[2]
    # 1位と2位の差。正解が候補に無いと候補同士が拮抗して小さくなる (信頼度の手掛かり)
    top = sorted(scores, reverse=True)
    margin = float(top[0] - top[1]) if len(top) > 1 else float(top[0])
    print(f"selected: {n_in} inliers, score {best[0]:.4f} (margin {margin:.4f}) "
          f"(P {v['precision']:.3f} / R {v['recall']:.3f} / "
          f"feas {v['cam_feas']:.2f} / Δs {v['scale_dev']:.2f}), "
          f"resid {resid:.3f}m, th={math.degrees(th2) % 360:.1f} s={s2:.3f}")
    px = cells_to_px(transform_pts(cam_rc, s2, th2, t2), grid)
    out = {
        "tag": tag, "floor": floor,
        "R_align": np.asarray(d["R_align"]).tolist(),
        "floor_y_units": d["floor_y_units"],
        # 物理床 (点群書き出しの高さ基準)。欠けると export_aligned2 が
        # floor_y_units=天井 にフォールバックし高さが誤る
        "floor_phys_y_units": d["floor_phys_y_units"],
        "m_per_unit": d["m_per_unit"],
        "origin": np.asarray(d["origin"]).tolist(),
        "cell_units": float(d["cell_units"]),
        "mirror_col": None,
        "s": float(s2), "th": float(th2), "t": [float(v) for v in t2],
        "wall_resid_m": resid,
        "score": round(float(best[0]), 4),
        "margin": round(margin, 4),
        "wall_cover": round(float(v["precision"]), 4),
        "wall_recall": round(float(v["recall"]), 4),
        "cam_feas": round(float(v["cam_feas"]), 4),
        "n_inliers": int(n_in),
        "anchored_by": f"corners({n_in} inliers)",
    }
    (OUT_DIR / f"anchor_{tag}{suffix}.json").write_text(
        json.dumps(out, indent=1))
    names = [str(n) for n in d["names"]]
    track = {n: (float(x), float(y)) for n, (x, y) in zip(names, px)}
    (OUT_DIR / f"corner_anchor_{tag}.json").write_text(
        json.dumps(track, indent=1))
    print(f"-> anchor_{tag}{suffix}.json / corner_anchor_{tag}.json")
    if return_info:
        return track, {"score": float(best[0]), "margin": margin,
                       "wall_cover": float(v["precision"]),
                       "wall_recall": float(v["recall"]),
                       "wall_resid_m": resid,
                       "n_inliers": int(n_in), "floor": floor, "anchor": out}
    return track


def main():
    import argparse
    from recon3d.config import OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", default=None, help="'tag:floor' コーナーアンカー実行")
    ap.add_argument("--ref", default=None, help="スケール事前用の配置済み島tag")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.anchor:
        tag, floor = args.anchor.split(":")
        corner_anchor(tag, floor, ref_tag=args.ref)
        return
    plan_cs = {f: _plot_plan(f, OUT_DIR) for f in ("1F", "2F")}
    for tag, floor in (("panovggt_A_yawfix", "1F"),
                       ("panovggt_Bcore_yawfix", "2F"),
                       ("panovggt_C_full", "1F")):
        _plot_island(tag, floor, plan_cs[floor], OUT_DIR)
    print(f"-> {OUT_DIR}/corners_*.jpg")


if __name__ == "__main__":
    main()
