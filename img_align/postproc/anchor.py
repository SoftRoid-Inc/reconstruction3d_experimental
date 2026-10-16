"""後段: 連結済み島(chain_chunks出力)を図面にアンカーし、IMUで階段を橋渡しして
全編軌跡を組み立てる

構成 (前段対決で確定):
  幾何 = panovggt_A (チャンク0-4, 1F) + panovggt_Bcore (チャンク7-10, 2F)
  階段チャンク(5,6,11,12)は幾何から除外し、島端アンカー+IMU階段タイムラインで補間

手順:
  1. 島ごとに 重力整列→z反転→スケール推定(床-カメラ高)→壁密度v2
  2. align_floorplan.match_floor でグリッド照合 → refine_align.icp_to_walls で連続精密化
  3. 全フレームの図面px軌跡を組み立て (島区間=ICP解 / ギャップ=端点補間+IMU階割当)
  4. GT評価 (階段内外を分けて報告)

使い方: python -m img_align.postproc.anchor
出力: results/backend/{trajectory.json, trajectory_{1F,2F}.jpg, eval.json}
"""
import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from img_align.postproc.plan_match import match_floor
from img_align.config import BENCH_DIR
from img_align.core.geometry import rotation_to_y
from img_align.postproc.density_map import slab_peaks
from img_align.eval.gt_traj import MM_PER_PX, load_gt
from img_align.core.plan_grid import ZUMEN_DIR, PlanGrid
from img_align.core.io_ply import read_ply_xyzrgb
from img_align.postproc.plan_icp import cells_to_px, icp_to_walls, segs_to_cells, similarity_from_grid, transform_pts, wall_segments_px

from img_align.config import (CAM_HEIGHT_M, CFG, OUT_DIR, SESSION,
                              STAIR_ASCEND, STAIR_DESCEND)

CELL_M = 0.05
Y_BIN_M = 0.20
MIN_Y_BINS = 4
# **壁の高さ帯**: カメラ高さ ±WALL_BAND_M だけを密度地図の素材にする。
# 完成物件ではソファ・カウンター・棚が密度を占めて図面の壁と一致せず、
# 偽解の方が被覆(f1)で勝ってしまう (2026-10-16 `eval/anchor_blame`:
# 0925の失敗島5本中4本で 正解f1 < 偽解f1、比2.7〜5.1倍)。
# カメラは約1.4mなので ±0.7m = 床上0.7〜2.1m。家具の天面(〜1.1m)と
# 天井設備を外し、**壁だけ**が残る (ユーザーがビューアで確認)。
# "auto" = 島ごとに実測する (postproc/density_band)。壁は鉛直なので真上から見た
# 形が高さによらず同じ、という性質だけで決まる。床/天井の検出に依存しないので
# 床の点群が欠けた島 (0925_s003r2) でも測れる。
# 数値を入れるとカメラ高さ±その値の対称帯、None で無効 (従来どおり全高)
WALL_BAND_M = 0.7
SLAB_MARGIN_M = 0.20

# 既定 = yaw_fix済みチャンク(方位ロックでフレーム回転修復)の連結島
# A=chunk0-5(1F, 階段直前まで) / B=chunk6-10(2F, 昇り後半込み) / C=chunk12(1F帰還)
# ポーズ全壊インスタンスは各チャンクのuntrusted.jsonに基づきchain_chunksが除外済み
ISLANDS = [("panovggt_A_yawfix", "1F"), ("panovggt_Bcore_yawfix", "2F"),
           ("panovggt_C_yawfix", "1F")]
# 旧方式のフレーム番号グローバル除外 (既定OFF: チャンク単位除外に移行済み)
UNTRUSTED_JSON = ""



def prepare_island(tag, mpu=None):
    """連結島 → 重力整列・z反転・スケール推定・密度v2 (照合入力の辞書を返す)

    mpu指定時はスケール推定(カメラ高さ仮定)をスキップして固定する
    (メトリック出力のフロントエンド: MapAnything条件付け等は mpu=1.0)。"""
    d = np.load(BENCH_DIR / tag / "global_cameras.npz")
    xyz, _ = read_ply_xyzrgb(BENCH_DIR / tag / "global_points.ply")
    names = [str(n) for n in d["names"]]
    ups = d["poses"][:, :3, 1]
    up = np.median(ups / np.linalg.norm(ups, axis=1, keepdims=True), axis=0)
    R = rotation_to_y(up / np.linalg.norm(up))
    pts = xyz @ R.T
    cam = d["positions"] @ R.T
    # 注意: PanoVGGTのカメラY軸は物理的に「下」(equirect画像座標系由来)のため、
    # この整列フレームは上下逆さ (y+が物理下向き。2026-09-11に照明の高さ分布で実証)。
    # 水平地図・アンカーはこの規約で検証済みなので維持し、鉛直は出力段で
    # floor_phys_y_units (物理床=カメラよりy大側の最大質量ピーク) を基準に反転解釈する
    pts[:, 2] *= -1
    cam = cam.copy()
    cam[:, 2] *= -1

    # スケール: 従来通り「カメラよりy小側の直近ピーク」(実は物理天井) を基準にする
    # (この推定に依存する水平パイプライン全体が検証済みのため変更しない)
    peaks, hist, centers = slab_peaks(pts[:, 1])
    cam_y = float(np.median(cam[:, 1]))
    below = peaks[peaks < cam_y]
    floor_y = float(below.max()) if len(below) else cam_y - 1.5
    if mpu is None:
        mpu = CAM_HEIGHT_M / (cam_y - floor_y)
    # 物理床 = カメラよりy大側 (物理的に下) の最大質量ピーク。出力の高さ基準用
    above = peaks[peaks > cam_y]
    if len(above):
        mass = [hist[np.argmin(np.abs(centers - p))] for p in above]
        floor_phys_y = float(above[int(np.argmax(mass))])
    else:
        floor_phys_y = cam_y + (cam_y - floor_y)
    print(f"{tag}: m_per_unit={mpu:.3f} (floor_y={floor_y:.2f}, cam_y={cam_y:.2f})")
    extras = {"R_align": R, "floor_y_units": floor_y,
              "floor_phys_y_units": floor_phys_y}

    # スラブ除去 + (任意で壁の高さ帯) + 空間バウンディング + 高さスパン密度
    keep = np.ones(len(pts), bool)
    for p in peaks:
        keep &= np.abs(pts[:, 1] - p) > SLAB_MARGIN_M / mpu
    if WALL_BAND_M:
        from img_align.postproc.density_band import band_mask
        band, (blo, bhi) = band_mask(pts[keep], cam_y, mpu, WALL_BAND_M)
        full = np.zeros(len(pts), bool)
        full[np.flatnonzero(keep)[band]] = True
        print(f"  wall band {blo:+.2f}〜{bhi:+.2f}m (カメラ基準): "
              f"{band.mean():.1%} の点を採用")
        keep = full
        extras["wall_band_m"] = [blo, bhi]
    w = pts[keep]
    lo_x, hi_x = np.percentile(w[:, 0], [2.5, 97.5])
    lo_z, hi_z = np.percentile(w[:, 2], [2.5, 97.5])
    w = w[(w[:, 0] >= lo_x) & (w[:, 0] <= hi_x)
          & (w[:, 2] >= lo_z) & (w[:, 2] <= hi_z)]
    cell_u = CELL_M / mpu
    ox, oz = w[:, 0].min() - 10 * cell_u, w[:, 2].min() - 10 * cell_u
    ix = ((w[:, 0] - ox) / cell_u).astype(np.int32)
    iz = ((w[:, 2] - oz) / cell_u).astype(np.int32)
    iy = ((w[:, 1] - w[:, 1].min()) / (Y_BIN_M / mpu)).astype(np.int32)
    nx, nz, ny = ix.max() + 10, iz.max() + 10, iy.max() + 1
    flat = (iz.astype(np.int64) * nx + ix) * ny + iy
    span = np.bincount(np.unique(flat) // ny, minlength=nz * nx).reshape(nz, nx)
    density = np.where(span >= MIN_Y_BINS, span, 0).astype(np.float32)
    density /= max(density.max(), 1)

    return {"density": density, "origin": np.array([ox, oz]),
            "cell_units": cell_u, "m_per_unit": mpu,
            "cam_pos_aligned": cam, "names": np.array(names),
            # density_pts: スラブ除去+バウンディング後の整列点群 (density の素材)。
            # align_tool がチャンク帰属でチャンク別密度を作るのに使う
            "density_pts": w, **extras}


def island_map_yaw(tag):
    """島のカメラ群の密度(row,col)系ヨー中央値 (島間の相対ヨー導出用)"""
    from img_align.core.chaining import med_yaw
    d = np.load(BENCH_DIR / tag / "global_cameras.npz")
    med, _ = med_yaw(d["poses"][:, :3, :3])
    return med


def _match_relaxed(grid, density, cam_rc, **kw):
    """カメラ妥当性ゲートを段階的に緩めながら照合 (密なフレーム列は壁際・
    階段のカメラ比率が高く、固定0.8では正解候補ごと弾かれることがある)"""
    import img_align.postproc.plan_match as af
    for feas_min in (0.8, 0.6, 0.4):
        af.CAM_FEAS_MIN = feas_min
        best = match_floor(grid, density, cam_rc, 1.0, **kw)
        if best is not None:
            if feas_min < 0.8:
                print(f"  (cam_feas gate relaxed to {feas_min})")
            return best
    return None


def anchor_island(tag, floor, gt, use_openings=False, ref=None,
                  anchored_so_far=None, allow_mirror=True, thetas=None,
                  scales=None, mpu=None):
    """島を図面へ: グリッド照合 → ICP精密化 → フレームpx辞書

    ref={'th_deg':…, 'psi':…} を渡すと、同フロアの既照合島との相対ヨー
    (方位ロックで既知) から回転候補を絞り、mirrorも禁止する(小島の照合偽解対策)。
    相対ヨーの±2分岐は、anchored_so_far(既知のフレーム→px)との時系列連続性
    (島の端フレームが直前の既知位置の近くにあるはず)で選ぶ。
    """
    d = prepare_island(tag, mpu=mpu)
    grid = PlanGrid(floor)
    cam_rc = np.stack(
        [(d["cam_pos_aligned"][:, 2] - d["origin"][1]) / d["cell_units"],
         (d["cam_pos_aligned"][:, 0] - d["origin"][0]) / d["cell_units"]], axis=1)
    if ref is None:
        kw = {"allow_mirror": allow_mirror}
        if thetas is not None:
            kw["thetas"] = thetas
            kw["scales"] = np.geomspace(0.3, 1.6, 24)
        if scales is not None:
            kw["scales"] = scales
        best = _match_relaxed(grid, d["density"], cam_rc, **kw)
    else:
        dpsi = math.degrees(ref["psi"] - island_map_yaw(tag))
        scales = np.geomspace(0.5, 1.5, 18)
        branches = []
        # sgn=0: PanoVGGTは先頭フレーム基準の座標系のため、方位ロック下では
        # 島間の地図ゲージがほぼ共通 → θ_ref自体が最有力候補 (実測で確認)。
        # ±Δψ はカメラヨー中央値がpose推定ドリフトで汚れている場合の保険。
        for sgn in (0, 1, -1):
            cands = sorted({int(round(ref["th_deg"] + sgn * dpsi + off)) % 360
                            for off in (-3, 0, 3)})
            b = _match_relaxed(grid, d["density"], cam_rc,
                               allow_mirror=False, thetas=cands, scales=scales)
            if b is not None:
                branches.append(b)
        best = None
        if len(branches) >= 2 and anchored_so_far:
            # 連続性 = 島の端フレームと時間的に最近の既知フレームの距離。
            # 明らかな偽解(遠すぎ)の除外にのみ使い、最終選択はスコア
            # (僅差の連続性で正解を落とした実績があるため)
            CONT_MAX_M = 8.0
            frames = [int(str(n).replace("frame_", "")) for n in d["names"]]
            known = sorted(anchored_so_far)
            j = min(range(len(frames)),
                    key=lambda i: min(abs(frames[i] - k) for k in known))
            k_near = min(known, key=lambda k: abs(k - frames[j]))
            ok = []
            for b in branches:
                s0, th0, t0 = similarity_from_grid(b, d, grid)
                px = cells_to_px(transform_pts(cam_rc[j][None], s0, th0, t0),
                                 grid)[0]
                dist_m = math.dist(px, anchored_so_far[k_near]) \
                    * MM_PER_PX / 1000
                print(f"  branch theta={b['theta']} score={b['score']:.3f} "
                      f"continuity {dist_m:.2f}m (frame {frames[j]} vs "
                      f"{k_near})")
                if dist_m < CONT_MAX_M:
                    ok.append(b)
            best = max(ok or branches, key=lambda b: b["score"])
        elif branches:
            best = max(branches, key=lambda b: b["score"])
        print(f"{tag}: constrained (dpsi {dpsi:.1f}) -> "
              f"theta {best['theta'] if best else None}")
    if best is None:
        raise RuntimeError(f"{tag}: グリッド照合が解を見つけられない")
    print(f"{tag} grid: score={best['score']:.3f} theta={best['theta']} "
          f"s={best['s']:.2f} mirror={best.get('mirror')}")
    mirror_col = None
    if best.get("mirror"):
        mirror_col = d["density"].shape[1] - 1
        d["density"] = np.ascontiguousarray(d["density"][:, ::-1])
        cam_rc[:, 1] = mirror_col - cam_rc[:, 1]
    s, th, t = similarity_from_grid(best, d, grid)
    zz, xx = np.nonzero(d["density"] > 0.15)
    pts_full = np.stack([zz, xx], axis=1).astype(np.float64)
    pts = pts_full
    if len(pts) > 20000:
        pts = pts[:: len(pts) // 20000 + 1]
    segs = segs_to_cells(wall_segments_px(floor), grid)
    s2, th2, t2, resid = icp_to_walls(pts, segs, s, th, t)
    print(f"{tag} ICP: wall resid {resid:.3f}m")
    if use_openings:
        from img_align.legacy.opening_icp import icp_with_openings, opening_axes_cells
        axes = opening_axes_cells(floor, grid)
        s2, th2, t2, resid, n_open = icp_with_openings(
            pts_full, segs, axes, s2, th2, t2)
        print(f"{tag} opening-ICP: wall resid {resid:.3f}m "
              f"({n_open}/{len(axes)} openings matched)")
    px = cells_to_px(transform_pts(cam_rc, s2, th2, t2), grid)
    # 点群書き出し用にアンカー変換を保存
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"anchor_{tag}.json").write_text(json.dumps({
        "tag": tag, "floor": floor,
        "R_align": d["R_align"].tolist(),
        "floor_y_units": d["floor_y_units"],
        "floor_phys_y_units": d["floor_phys_y_units"],
        "m_per_unit": d["m_per_unit"],
        "origin": d["origin"].tolist(),
        "cell_units": float(d["cell_units"]),
        "mirror_col": mirror_col,
        "s": s2, "th": th2, "t": list(map(float, t2)),
        "wall_resid_m": resid}, indent=1))
    info = {"th_deg": math.degrees(th2) % 360, "psi": island_map_yaw(tag)}
    return {f"frame_{n}".replace("frame_frame_", "frame_"): p
            for n, p in zip(d["names"], px)}, resid, info


def assemble_and_eval(anchored, floors, gt, label=""):
    """全編組み立て(ギャップ補間)→GT評価→軌跡図。labelは出力ファイルの接尾辞"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # ギャップ補間用のIMU PDR軌跡 — 既定OFF (実測でbridge誤差1.9→3.3mに悪化。
    # 階段の歩みは水平ストライドでないためPDRが階段区間でオーバーシュートする)
    import os
    pdr = None
    pdr_path = OUT_DIR.parent / "imu" / "pdr.npz"
    if os.environ.get("USE_PDR_BRIDGE") and pdr_path.exists():
        z = np.load(pdr_path)
        pdr = (z["frames"], z["pos"] * 1000 / MM_PER_PX)  # m→図面px

    def pdr_at(f):
        fr, pos = pdr
        return np.array([np.interp(f, fr, pos[:, 0]),
                         np.interp(f, fr, pos[:, 1])])

    def bridge_pos(f, lo, hi):
        """PDR形状を両端アンカーに2点相似フィットして補間 (無ければ線形)"""
        a, b = np.array(anchored[lo]), np.array(anchored[hi])
        if pdr is not None:
            p1, p2 = pdr_at(lo), pdr_at(hi)
            v_p, v_a = p2 - p1, b - a
            n = np.linalg.norm(v_p)
            if n > 1e-6:
                s = np.linalg.norm(v_a) / n
                ang = math.atan2(v_a[1], v_a[0]) - math.atan2(v_p[1], v_p[0])
                R = np.array([[math.cos(ang), -math.sin(ang)],
                              [math.sin(ang), math.cos(ang)]])
                return a + s * (R @ (pdr_at(f) - p1))
        w = (f - lo) / (hi - lo)
        return a * (1 - w) + b * w

    # 全編組み立て: 島区間はICP解、ギャップはPDR相似フィット補間
    all_frames = list(CFG["frames"])
    known = sorted(anchored)
    track = []
    for f in all_frames:
        if f in anchored:
            x, y = anchored[f]
            src = "island"
            floor = floors[f]
        else:
            lo = max([k for k in known if k < f], default=None)
            hi = min([k for k in known if k > f], default=None)
            if lo is None or hi is None:
                a = lo if lo is not None else hi
                x, y = anchored[a]
            else:
                x, y = bridge_pos(f, lo, hi)
            src = "bridge"
            # 階割当はIMU階段タイムライン
            if f < STAIR_ASCEND[0] + (STAIR_ASCEND[1] - STAIR_ASCEND[0]) / 2:
                floor = "1F" if f < STAIR_ASCEND[0] else "1F"
            if f < (STAIR_ASCEND[0] + STAIR_ASCEND[1]) / 2:
                floor = "1F"
            elif f < (STAIR_DESCEND[0] + STAIR_DESCEND[1]) / 2:
                floor = "2F"
            else:
                floor = "1F"
        track.append({"frame": f, "floor": floor, "x_px": round(x, 1),
                      "y_px": round(y, 1), "source": src})

    # 評価 (階段内外を分ける)
    stair_set = set()
    for a, b in (STAIR_ASCEND, STAIR_DESCEND):
        stair_set.update(range(a - 6, b + 7))
    rows = {"island": [], "bridge": [], "stair": []}
    floor_ok = 0
    n_eval = 0
    for tkr in track:
        g = min(gt, key=lambda k: abs(k - tkr["frame"]))
        if abs(g - tkr["frame"]) > 3:
            continue
        n_eval += 1
        e = math.dist((tkr["x_px"], tkr["y_px"]),
                      (gt[g]["x_px"], gt[g]["y_px"])) * MM_PER_PX / 1000
        if tkr["floor"] == gt[g]["floor"]:
            floor_ok += 1
        key = ("stair" if tkr["frame"] in stair_set
               else tkr["source"])
        rows[key].append(e)
    summary = {"floor_acc": round(floor_ok / n_eval, 3), "n": n_eval}
    for k, v in rows.items():
        if v:
            summary[k] = {"n": len(v), "median_m": round(float(np.median(v)), 3),
                          "p90_m": round(float(np.percentile(v, 90)), 3)}
    print(json.dumps(summary, indent=1))
    (OUT_DIR / f"trajectory{label}.json").write_text(json.dumps(
        {"summary": summary, "track": track}, indent=1))

    # 軌跡図
    for floor in ("1F", "2F"):
        bg = np.asarray(Image.open(ZUMEN_DIR / f"{floor}.jpg").convert("L"))
        fig, ax = plt.subplots(figsize=(11, 8))
        ax.imshow(bg, cmap="gray")
        g = [(f, v) for f, v in sorted(gt.items()) if v["floor"] == floor]
        ax.plot([v["x_px"] for _, v in g], [v["y_px"] for _, v in g],
                "-", color="lime", lw=2.5, label="GT")
        t_isl = [t for t in track if t["floor"] == floor and t["source"] == "island"]
        t_br = [t for t in track if t["floor"] == floor and t["source"] == "bridge"]
        ax.scatter([t["x_px"] for t in t_isl], [t["y_px"] for t in t_isl],
                   c="red", s=16, label="island (anchored)")
        ax.scatter([t["x_px"] for t in t_br], [t["y_px"] for t in t_br],
                   c="deepskyblue", s=16, marker="^", label="bridge (interp)")
        ax.legend()
        ax.set_title(f"{floor}: backend trajectory")
        ax.set_axis_off()
        fig.savefig(OUT_DIR / f"trajectory{label}_{floor}.jpg", dpi=120,
                    bbox_inches="tight")
        plt.close(fig)
    print(f"-> {OUT_DIR}")
    return summary


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--islands", nargs="*", default=None,
                    help="'tag:floor' の列で既定のISLANDSを差し替え")
    ap.add_argument("--label", default="", help="出力ファイルの接尾辞")
    ap.add_argument("--exclude", default=UNTRUSTED_JSON,
                    help="ポーズ不信フレームのjson (yaw_fixのpose_untrusted.json)。"
                         "該当フレームは島から外し橋補間にする。'' で無効化")
    ap.add_argument("--openings", action="store_true",
                    help="開口ランドマークICP (壁沿い零空間の拘束) を有効化")
    args = ap.parse_args()
    islands = ([tuple(v.split(":")) for v in args.islands]
               if args.islands else ISLANDS)
    exclude = set()
    if args.exclude:
        from pathlib import Path

        from img_align.config import IMG_ALIGN_DIR
        p = Path(args.exclude)
        if not p.is_absolute():
            p = IMG_ALIGN_DIR / p
        if p.exists():
            exclude = {int(n.replace("frame_", ""))
                       for n in json.loads(p.read_text())}
            print(f"exclude (pose untrusted): {sorted(exclude)}")

    gt = load_gt()
    anchored = {}
    floors = {}
    ref_by_floor = {}   # 同フロアの最初(=最大)の島の解を後続の小島の拘束に使う
    for tag, floor in islands:
        px_map, _, info = anchor_island(tag, floor, gt,
                                        use_openings=args.openings,
                                        ref=ref_by_floor.get(floor),
                                        anchored_so_far=dict(anchored))
        ref_by_floor.setdefault(floor, info)
        for name, p in px_map.items():
            fr = int(name.replace("frame_", ""))
            if fr in exclude:
                continue
            anchored[fr] = (float(p[0]), float(p[1]))
            floors[fr] = floor
    assemble_and_eval(anchored, floors, gt, label=args.label)


if __name__ == "__main__":
    main()
