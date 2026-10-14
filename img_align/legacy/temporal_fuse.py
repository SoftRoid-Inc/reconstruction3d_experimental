"""時系列統合: 仮想スキャン尤度 × PanoVGGTオドメトリ のパーティクルフィルタ

状態 = (階, x, y, θ)。遷移 = チャンク姿勢由来の相対移動(オドメトリ)、
観測 = 仮想スキャンの壁/開口カーネル一致。階の遷移はオドメトリの鉛直変位が
大きいステップ(=階段昇降)でのみ許可する。

方位規約C: スキャンの方位原点とカメラ姿勢のヨー原点の差(全体で1定数)。
同一チャンク内にあるアンカーフレーム対(独立2入力で位置が一致したフレーム)から校正する。

出力: results/fuse/trajectory_{1F,2F}.jpg / fused.json / eval_anchors.json
"""
import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from img_align.core.plan_grid import RES_M, ZUMEN_DIR, PlanGrid
from img_align.legacy.localize import MAX_DIST_M, frame_points
from img_align.core.panovggt import PCD_DIR, local_floor_height, select_entries
from img_align.core.chunking import chunk_windows
from img_align.legacy.stitch_chunks import gravity_rotation
from img_align.config import RESULTS_DIR

FUSE_DIR = RESULTS_DIR / "fuse"
CHUNKS_DIR = PCD_DIR / "chunks"

N_PARTICLES = 6000          # 1階あたり初期粒子数
THETA_NOISE_DEG = 4.0
POS_NOISE_BASE_M = 0.06
POS_NOISE_REL = 0.12
OBS_TEMP = 0.06             # 観測尤度の温度
STAIR_SEG_FRAC = 0.02       # セグのstairs被覆率がこれを超えるフレームで階遷移を許可
                            # (PanoVGGTの鉛直オドメトリは階段区間で壊れるため使わない)
FLOOR_SWITCH_P = 0.35       # 階段中の階遷移確率
FLOOR_SWITCH_LOGPEN = 0.5   # 階遷移した粒子への対数重みペナルティ (弱め)
MODE_RADIUS_CELL = 30       # 推定値のモード探索半径 (30セル=1.5m)
FLOOR_SMOOTH_WIN = 9        # 階ラベルの事後メディアン平滑化窓
CAM_HEIGHT_M = 1.5
ISLAND_PRIOR_SIGMA_M = 4.0  # 島(SceneAligner-lite)由来の位置プライアの幅
                            # (島の相似変換は粗いので位置は弱く、階判定は強く効かせる)
ISLAND_FLOOR_LOGPEN = 2.0   # 島の階と不一致な粒子への対数重みペナルティ
ISLAND_TAGS = ("islandA", "islandB", "islandC")

# 独立2入力(GT/Mask2Former)で位置が一致したアンカー (loc_gt の値を正とする)
ANCHOR_FRAMES = [30, 121, 181, 211, 302, 362, 393]


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class ChunkOdometry:
    """チャンクごとの重力整列+z反転+メートル化した位置・向きを提供する"""

    def __init__(self, entries):
        self.frame_to_entry = {e["index"]: j for j, e in enumerate(entries)}
        self.windows = chunk_windows(len(entries))
        self.chunks = []
        for k, (a, b) in enumerate(self.windows):
            cdir = CHUNKS_DIR / f"chunk_{k:02d}"
            cams = np.load(cdir / "cameras.npz")
            names = [str(n) for n in cams["names"]]
            G = gravity_rotation(cams["poses"])
            pos = cams["positions"] @ G.T
            fwd = cams["poses"][:, :3, 2] @ G.T
            pos[:, 2] *= -1  # 図面と掌性を合わせる (density_mapと同じz反転)
            fwd[:, 2] *= -1
            heights = [local_floor_height(
                cdir / "panovggt_out" / "pointclouds" / "per_frame" / f"{n}.ply")
                for n in names]
            heights = [h for h in heights if h is not None]
            mpu = CAM_HEIGHT_M / float(np.median(heights))
            self.chunks.append({
                "range": (a, b), "names": names,
                "pos": pos * mpu,                      # メートル
                "heading": np.arctan2(fwd[:, 2], fwd[:, 0]),
                "mpu": mpu,
            })

    def chunk_for_pair(self, j):
        """entry j と j+1 を両方含み、窓の中央に最も近いチャンク"""
        best, margin = None, -1
        for c in self.chunks:
            a, b = c["range"]
            if a <= j and j + 1 < b:
                m = min(j - a, b - 1 - (j + 1))
                if m > margin:
                    best, margin = c, m
        return best

    def lookup(self, chunk, frame_idx):
        i = chunk["names"].index(f"frame_{frame_idx:06d}")
        return chunk["pos"][i], chunk["heading"][i]

    def step(self, entries, j):
        """entry j → j+1 の相対移動 (r[m], δ, Δh, Δy[m])"""
        c = self.chunk_for_pair(j)
        p1, h1 = self.lookup(c, entries[j]["index"])
        p2, h2 = self.lookup(c, entries[j + 1]["index"])
        d = p2 - p1
        return {
            "r": float(np.hypot(d[0], d[2])),
            "delta": float(math.atan2(d[2], d[0])),
            "dh": float(wrap(h2 - h1)),
            "dy": float(d[1]),
            "h1": float(h1),
        }


def calibrate_C(odo, entries, loc_gt):
    """同一チャンク内アンカー対から方位規約Cを推定する

    γ(図面上の変位方向) = θ1 + C + (δ_total − h1)
    アンカーは3刻み系列に無いことがあるため最近傍の系列フレームで近似する。
    """
    def nearest_j(frame):
        return min(range(len(entries)),
                   key=lambda j: abs(entries[j]["index"] - frame))

    pairs = [(181, 211), (362, 393)]
    ests = []
    for fa, fb in pairs:
        ja, jb = nearest_j(fa), nearest_j(fb)
        c = None
        for cand in odo.chunks:
            a, b = cand["range"]
            if a <= ja < b and a <= jb < b:
                c = cand
                break
        if c is None:
            continue
        p1, h1 = odo.lookup(c, entries[ja]["index"])
        p2, _ = odo.lookup(c, entries[jb]["index"])
        delta = math.atan2(p2[2] - p1[2], p2[0] - p1[0])
        ga = loc_gt[f"frame_{fa:06d}"]
        gb = loc_gt[f"frame_{fb:06d}"]
        gamma = math.atan2(gb["y_px"] - ga["y_px"], gb["x_px"] - ga["x_px"])
        theta1 = math.radians(ga["theta_deg"])
        C = wrap(gamma - theta1 - delta + h1)
        ests.append(C)
        print(f"C from pair {fa}->{fb}: {math.degrees(C):.1f} deg "
              f"(odo dist {np.hypot(p2[0]-p1[0], p2[2]-p1[2]):.2f}m)")
    C = math.atan2(np.mean([math.sin(c) for c in ests]),
                   np.mean([math.cos(c) for c in ests]))
    print(f"C = {math.degrees(C):.1f} deg (n={len(ests)})")
    return C


class Observation:
    """粒子集合に対する仮想スキャン尤度 (壁/開口カーネル + 壁透過ペナルティ)"""

    def __init__(self, grids, scans):
        self.grids = grids
        self.scans = scans

    def score(self, frame_name, floor, x, y, theta):
        """x,y: グリッドセル(float array), theta: rad array → スコア array"""
        g = self.grids[floor]
        pts = frame_points(self.scans, frame_name)  # az, dist, is_open, w
        if len(pts) < 20:
            return np.zeros_like(x)
        # レイを間引いて高速化
        if len(pts) > 120:
            pts = pts[:: len(pts) // 120 + 1]
        az, dist, is_open, w = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
        ok = dist < MAX_DIST_M
        az, dist, is_open, w = az[ok], dist[ok], is_open[ok], w[ok]
        ang = theta[:, None] + az[None, :]
        ex = x[:, None] + dist[None, :] * np.cos(ang) / RES_M
        ey = y[:, None] + dist[None, :] * np.sin(ang) / RES_M
        exi = np.clip(ex.round().astype(int), 0, g.nx - 1)
        eyi = np.clip(ey.round().astype(int), 0, g.ny - 1)
        k = np.where(is_open[None, :] > 0.5,
                     g.k_open[eyi, exi], g.k_wall[eyi, exi])
        # 壁透過ペナルティ (中間点1つ)
        mxi = np.clip((x[:, None] + 0.55 * (ex - x[:, None])).round().astype(int),
                      0, g.nx - 1)
        myi = np.clip((y[:, None] + 0.55 * (ey - y[:, None])).round().astype(int),
                      0, g.ny - 1)
        k = k - 0.8 * g.solid[myi, mxi]
        return (k * w[None, :]).sum(axis=1) / w.sum()


def load_island_priors(floors):
    """島ごとの図面照合結果 (align_summary.json) をフレーム→(階,px)の事前分布にする"""
    priors = {}
    for tag in ISLAND_TAGS:
        p = PCD_DIR / tag / "align_summary.json"
        if not p.exists():
            continue
        summ = json.loads(p.read_text())
        floor = summ["selected_floor"]
        res = summ["results"][floor]
        for name, (px, py) in zip(summ["names"], res["cam_px"]):
            priors[name] = {"floor": floors.index(floor),
                            "x_px": px, "y_px": py}
    print(f"island priors: {len(priors)} frames")
    return priors


def run_filter(entries, odo, obs, grids, C, rng, priors):
    floors = list(grids)
    n0 = N_PARTICLES
    parts = []
    for f in floors:
        ys, xs = np.nonzero(grids[f].valid_pose)
        sel = rng.integers(0, len(ys), n0)
        parts.append({
            "floor": np.full(n0, floors.index(f), dtype=np.int8),
            "x": xs[sel].astype(float), "y": ys[sel].astype(float),
            "theta": rng.uniform(-math.pi, math.pi, n0),
        })
    P = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    n = len(P["x"])
    logw = np.zeros(n)

    steps = [odo.step(entries, j) for j in range(len(entries) - 1)]

    # IMU由来の相対ヨーによるΔh置換はオプトイン (実測: 同期±0.5s×高速回転で
    # 数十度の誤差が乗り、チャンク内は視覚ヨーの方が正確。anchor err 2.3→4.1mに悪化)
    import os
    imu_yaw = RESULTS_DIR / "imu" / "yaw_odometry.npz"
    if os.environ.get("USE_IMU_YAW") and imu_yaw.exists():
        pairs = np.load(imu_yaw)["pairs_step3"]
        dyaw = {(int(a), int(b)): float(d) for a, b, d in pairs}
        n_rep = 0
        for j in range(len(entries) - 1):
            key = (entries[j]["index"], entries[j + 1]["index"])
            if key in dyaw:
                steps[j]["dh"] = dyaw[key]
                n_rep += 1
        print(f"IMU yaw odometry: {n_rep}/{len(steps)} steps replaced")

    # 階段ステップの事前判定: IMU階段検出があれば優先、なければstairsセグ
    imu_stairs = RESULTS_DIR / "imu" / "stairs.json"
    if imu_stairs.exists():
        ivs = json.loads(imu_stairs.read_text())["stairs"]
        def in_stairs(f):
            return any(iv["frame0"] - 3 <= f <= iv["frame1"] + 3 for iv in ivs)
        stair = np.array([
            in_stairs(entries[j]["index"]) or in_stairs(entries[j + 1]["index"])
            for j in range(len(entries) - 1)])
        print(f"stair steps (IMU): {int(stair.sum())}/{len(stair)}")
    else:
        seg_stats = json.loads((RESULTS_DIR / "seg" / "stats.json").read_text())

        def stairs_frac(idx):
            return seg_stats.get(f"frame_{idx:06d}", {}).get("stairs", 0.0)

        stair = np.array([
            max(stairs_frac(entries[j]["index"]),
                stairs_frac(entries[j + 1]["index"])) > STAIR_SEG_FRAC
            for j in range(len(entries) - 1)])
        print(f"stair steps (seg): {int(stair.sum())}/{len(stair)}")

    track = []
    states = []      # 各ステップの粒子状態 (系譜バックトラック用)
    resamples = []   # 各ステップのリサンプル添字 (無ければNone)
    final_w = None
    for j, entry in enumerate(entries):
        name = f"frame_{entry['index']:06d}"
        if j > 0:
            o = steps[j - 1]
            direction = P["theta"] + C + (o["delta"] - o["h1"])
            r = o["r"] + rng.normal(0, POS_NOISE_BASE_M + POS_NOISE_REL * o["r"], n)
            P["x"] += r * np.cos(direction) / RES_M
            P["y"] += r * np.sin(direction) / RES_M
            P["theta"] = (P["theta"] + o["dh"]
                          + rng.normal(0, math.radians(THETA_NOISE_DEG), n))
            if stair[j - 1]:
                flip = rng.random(n) < FLOOR_SWITCH_P
                P["floor"] = np.where(flip, 1 - P["floor"], P["floor"])
                logw[flip] -= FLOOR_SWITCH_LOGPEN

        # 観測更新 (階ごとにまとめて) + 島プライア
        # 注: 階段可視中に観測を切る案は試したが悪化した (「可視」は通過中以外の
        # フレームも多く、切りすぎて接地を失う)。観測は常時オンが現状のベスト
        s = np.zeros(n)
        inb = np.zeros(n, dtype=bool)
        prior = priors.get(name)
        for fi, f in enumerate(floors):
            m = P["floor"] == fi
            if not m.any():
                continue
            g = grids[f]
            xi = np.clip(P["x"][m].round().astype(int), 0, g.nx - 1)
            yi = np.clip(P["y"][m].round().astype(int), 0, g.ny - 1)
            inb[m] = g.valid_pose[yi, xi]
            s[m] = obs.score(name, f, P["x"][m], P["y"][m], P["theta"][m])
            if prior is not None:
                px = g.x0_px + P["x"][m] * g.cell_px
                py = g.y0_px + P["y"][m] * g.cell_px
                d_m = np.hypot(px - prior["x_px"], py - prior["y_px"]) \
                    * g.m_per_px
                logw[m] += -0.5 * (d_m / ISLAND_PRIOR_SIGMA_M) ** 2
                if fi != prior["floor"]:
                    logw[m] -= ISLAND_FLOOR_LOGPEN
        logw += s / OBS_TEMP
        logw[~inb] -= 12.0  # 屋外・壁内の粒子を強く減点

        w = np.exp(logw - logw.max())
        w /= w.sum()
        ess = 1.0 / (w ** 2).sum()
        states.append({k: P[k].copy() for k in P})
        final_w = w
        if ess < n / 2:
            idx = np.searchsorted(np.cumsum(w), (rng.random() + np.arange(n)) / n)
            idx = np.clip(idx, 0, n - 1)
            for k in P:
                P[k] = P[k][idx]
            logw = np.zeros(n)
            w = np.full(n, 1.0 / n)
            resamples.append(idx)
        else:
            resamples.append(None)

        # 推定値: 階ごとにモード探索 (最大重み粒子の近傍平均。多峰分布の混合を防ぐ)
        floor_w = [w[P["floor"] == fi].sum() for fi in range(len(floors))]
        ests = {}
        for fi, f in enumerate(floors):
            m = P["floor"] == fi
            if w[m].sum() < 1e-9:
                continue
            wi = np.where(m)[0]
            best_i = wi[np.argmax(w[wi])]
            near = m & (np.hypot(P["x"] - P["x"][best_i],
                                 P["y"] - P["y"][best_i]) < MODE_RADIUS_CELL)
            wm = w[near] / w[near].sum()
            cx = float((P["x"][near] * wm).sum())
            cy = float((P["y"][near] * wm).sum())
            px, py = grids[f].cell_to_px(cx, cy)
            theta = math.degrees(math.atan2(
                (np.sin(P["theta"][near]) * wm).sum(),
                (np.cos(P["theta"][near]) * wm).sum())) % 360
            ests[f] = {"x_px": round(float(px), 1), "y_px": round(float(py), 1),
                       "theta_deg": round(theta, 1),
                       "conf": round(float(floor_w[fi]), 3)}
        fi = int(np.argmax(floor_w))
        track.append({"frame": entry["index"], "floor": floors[fi],
                      "floor_conf": round(float(floor_w[fi]), 3),
                      "ests": ests, "ess": round(float(ess), 0)})
        if j % 20 == 0:
            e = ests[floors[fi]]
            print(f"[{j:3d}] {name} floor={floors[fi]}({floor_w[fi]:.2f}) "
                  f"px=({e['x_px']:.0f},{e['y_px']:.0f}) ess={ess:.0f}")

    # 系譜バックトラック: 最終ステップの最良粒子から先祖を遡り、
    # 一貫した1本の軌跡を取り出す (モード切替による点推定のテレポートを排除)
    T = len(states)
    chain = [0] * T
    chain[T - 1] = int(np.argmax(final_w))
    for j in range(T - 1, 0, -1):
        r = resamples[j - 1]
        chain[j - 1] = int(r[chain[j]]) if r is not None else chain[j]
    final = []
    for j, entry in enumerate(entries):
        st, i = states[j], chain[j]
        f = floors[int(st["floor"][i])]
        px, py = grids[f].cell_to_px(st["x"][i], st["y"][i])
        final.append({"frame": entry["index"], "floor": f,
                      "floor_conf": track[j]["floor_conf"],
                      "x_px": round(float(px), 1), "y_px": round(float(py), 1),
                      "theta_deg": round(math.degrees(st["theta"][i]) % 360, 1),
                      "ess": track[j]["ess"]})
    return final


def evaluate(track, loc_gt, mm_per_px):
    by_frame = {t["frame"]: t for t in track}
    rows = []
    for fa in ANCHOR_FRAMES:
        j = min(by_frame, key=lambda f: abs(f - fa))
        t = by_frame[j]
        a = loc_gt[f"frame_{fa:06d}"]
        err_px = math.dist((t["x_px"], t["y_px"]), (a["x_px"], a["y_px"]))
        rows.append({
            "anchor": fa, "nearest_seq_frame": j,
            "floor_ok": t["floor"] == a["floor"],
            "err_m": round(err_px * mm_per_px / 1000, 2),
        })
    return rows


def draw_track(track, grids, bgs):
    for floor in grids:
        pts = [t for t in track if t["floor"] == floor]
        if not pts:
            continue
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(bgs[floor], cmap="gray")
        xs = [t["x_px"] for t in pts]
        ys = [t["y_px"] for t in pts]
        order = np.linspace(0, 1, len(pts))
        ax.plot(xs, ys, "-", color="red", linewidth=0.9, alpha=0.6)
        ax.scatter(xs, ys, c=order, cmap="viridis", s=22, zorder=3)
        for t in pts[::5]:
            ax.annotate(str(t["frame"]), (t["x_px"], t["y_px"]),
                        fontsize=6, color="blue")
        ax.set_title(f"fused trajectory on {floor} ({len(pts)} frames)")
        ax.set_axis_off()
        fig.savefig(FUSE_DIR / f"trajectory_{floor}.jpg", dpi=120,
                    bbox_inches="tight")
        plt.close(fig)


def main():
    FUSE_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    entries = select_entries("0:454:3")
    grids = {f: PlanGrid(f) for f in ("1F", "2F")}
    bgs = {f: np.asarray(Image.open(ZUMEN_DIR / f"{f}.jpg").convert("L"))
           for f in grids}
    scans = np.load(RESULTS_DIR / "scan" / "scans.npz")
    loc_gt = json.loads((RESULTS_DIR / "loc_gt" / "summary.json").read_text())

    odo = ChunkOdometry(entries)
    C = calibrate_C(odo, entries, loc_gt)
    obs = Observation(grids, scans)
    priors = load_island_priors(list(grids))
    track = run_filter(entries, odo, obs, grids, C, rng, priors)

    mm_per_px = grids["1F"].m_per_px * 1000
    rows = evaluate(track, loc_gt, mm_per_px)
    (FUSE_DIR / "fused.json").write_text(json.dumps(
        {"C_deg": round(math.degrees(C), 1), "track": track}, indent=1))
    (FUSE_DIR / "eval_anchors.json").write_text(json.dumps(rows, indent=1))
    draw_track(track, grids, bgs)
    print(json.dumps(rows, indent=1))
    errs = [r["err_m"] for r in rows]
    print(f"anchor err median {np.median(errs):.2f}m / "
          f"floor正解 {sum(r['floor_ok'] for r in rows)}/{len(rows)}")
    print(f"done -> {FUSE_DIR}")


if __name__ == "__main__":
    main()
