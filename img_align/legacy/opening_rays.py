"""開口ランドマーク第2版: 視線横断(ray crossing)による開口検出 + ICP精密化

第1版(opening_icp: 密度ギャップ検出)はPanoVGGT点群の構造的性質で全滅した
(カーテン・ガラスが面として再構成 + 深度不連続のフライングピクセルが開口を橋渡し)。

第2版は物理的に確実な信号を使う: カメラはドアを通り窓越しに隣室・屋外を見ている。
「カメラ→点」の視線が図面の壁線を横切る位置の分布は、開口区間だけで高くなる。
フライングピクセルは視線上に乗るため横断位置を汚さない(むしろ開口で横断する)。

検出した横断区間の端 ↔ 図面開口の端 を対応にし、壁point-to-line ICPに加えて
相似変換を再推定する(壁沿い零空間の拘束)。

使い方: python -m img_align.legacy.opening_rays   (backend_alignのアンカー解を初期値に
        島ごとに精密化 → trajectory_rays.json を出力・GT評価)
"""
import json
import math

import numpy as np

from img_align.config import OUT_DIR
from img_align.postproc.anchor import UNTRUSTED_JSON, assemble_and_eval, prepare_island
from img_align.core.chaining import chain
from img_align.core.geometry import apply
from img_align.config import IMG_ALIGN_DIR, RESULTS_DIR
from img_align.eval.gt_traj import load_gt
from img_align.core.plan_grid import RES_M, PlanGrid
from img_align.legacy.opening_icp import fit_similarity_w, opening_axes_cells
from img_align.core.io_ply import read_ply_xyzrgb
from img_align.postproc.plan_icp import cells_to_px, segs_to_cells, transform_pts, wall_segments_px

# mode: "rays"=開口ランドマーク精密化 / "corner"=コーナーアンカー軌跡を
#        そのまま採用 / "corner_rays"=コーナーアンカー初期値+開口精密化
#        (コーナー系は corners.py --anchor で事前生成しておく)
ISLANDS = [
    ("panovggt_A_yawfix", "1F", "pcd/chunks_yawfix", (0, 6), "rays"),
    ("panovggt_Bcore_yawfix", "2F", "pcd/chunks_yawfix", (6, 11),
     "rays"),
    ("panovggt_C_full", "1F", "pcd/chunks_full_yawfix", (35, 38),
     "corner"),
]
# 工事中セッション0312 (IMG_SESSION=0312で実行)
# 教訓: 工事中は図面開口と実開口が不一致(建具未施工・仮塞ぎ)のため、
# 開口精密化は裸壁のグリッド+壁ICPより悪い (A: 0.089→0.185mで実証)
ISLANDS_0312 = [
    ("panovggt_0312_A", "1F", "pcd/chunks", (0, 10), "anchor"),
    ("panovggt_0312_B", "2F", "pcd/chunks", (11, 19), "rays"),
    ("panovggt_0312_C", "1F", "pcd/chunks", (20, 22), "corner"),
]
# 実験用: 島Aを全フレーム短島3枚(内部ドリフト小)に置換した構成
ISLANDS_SHORT_A = [
    ("panovggt_A1_full", "1F", "pcd/chunks_full_yawfix", (0, 7),
     "corner_rays"),
    ("panovggt_A2_full", "1F", "pcd/chunks_full_yawfix", (6, 13),
     "corner_rays"),
    ("panovggt_A3_full", "1F", "pcd/chunks_full_yawfix", (12, 19),
     "corner_rays"),
    ("panovggt_Bcore_yawfix", "2F", "pcd/chunks_yawfix", (6, 11),
     "rays"),
    ("panovggt_C_full", "1F", "pcd/chunks_full_yawfix", (35, 38),
     "corner"),
]
PT_STRIDE = 40        # per-frame点群の間引き
LOCAL_BAND = (-1.1, 0.4)   # カメラ基準の高さ帯 [m相当units] (床上0.4〜1.9m)
MAX_RANGE_U = 6.0     # 視線の最大長 (units)
EXT_M = 1.2
BEYOND_M = 0.30       # 壁の向こう側に最低これだけ抜けた点のみ「見通し」と数える
MIN_CROSS = 8         # 横断ビンの有効最小本数
SLACK_M = 0.8
WIDTH_RATIO = (0.4, 2.0)
ICP_ROUNDS = 4
ICP_ITERS_PER = 6
CUTOFF_CELL = 3.0
W_CAP = 200.0


def load_frame_scans(tag, chunks_root, chunk_range):
    """島の各フレームの (カメラ島座標, 点群島座標) を per-frame ply から作る"""
    root = RESULTS_DIR / chunks_root
    cdirs = sorted(root.glob("chunk_*"), key=lambda p: int(p.name.split("_")[1]))[chunk_range[0]:chunk_range[1]]
    chunks, T, _ = chain(cdirs)
    scans = {}
    for k, (names, pos, poses, _, untrusted) in enumerate(chunks):
        cams = np.load(cdirs[k] / "cameras.npz")
        for i, n in enumerate(names):
            if n in scans or n in untrusted:  # 重複は先勝ち・不信は除外
                continue
            ply = (cdirs[k] / "panovggt_out" / "pointclouds" / "per_frame"
                   / f"{n}.ply")
            xyz, _ = read_ply_xyzrgb(ply)
            xyz = xyz[::PT_STRIDE]
            m = ((xyz[:, 1] > LOCAL_BAND[0]) & (xyz[:, 1] < LOCAL_BAND[1])
                 & (np.linalg.norm(xyz, axis=1) < MAX_RANGE_U))
            pose = cams["poses"][i]
            w = xyz[m] @ pose[:3, :3].T + pose[:3, 3]
            scans[n] = (apply(T[k], pos[i][None])[0], apply(T[k], w))
    return scans


def island_to_src_cells(xyz, d, mirror_col):
    """島world座標 → density source cell座標 (row,col)"""
    p = xyz @ np.asarray(d["R_align"]).T
    p = p.copy()
    p[:, 2] *= -1
    rc = np.stack([(p[:, 2] - d["origin"][1]) / d["cell_units"],
                   (p[:, 0] - d["origin"][0]) / d["cell_units"]], axis=1)
    if mirror_col is not None:
        rc[:, 1] = mirror_col - rc[:, 1]
    return rc


def detect_crossings(cam_t, pts_t, axes):
    """変換済み (カメラ, 点群) の視線が各開口軸線を横切る位置のプロファイル →
    (src合成点リスト作成用の (axis, shift) 情報, 診断)"""
    ext = EXT_M / RES_M
    beyond = BEYOND_M / RES_M
    slack = SLACK_M / RES_M
    marks, diags = [], []
    for a, b in axes:
        d = b - a
        L = float(np.linalg.norm(d))
        u_dir = d / L
        n_dir = np.array([-u_dir[1], u_dir[0]])
        lo = -ext
        nb = int(math.ceil(L + 2 * ext))
        cnt = np.zeros(nb)
        for q, P in zip(cam_t, pts_t):
            sa = (q - a) @ n_dir
            sb = (P - a) @ n_dir
            m = (sa * sb < 0) & (np.abs(sb) > beyond)
            if not m.any():
                continue
            w = sa / (sa - sb[m])
            x = q + (P[m] - q) * w[:, None]
            u = (x - a) @ u_dir
            u = u[(u >= lo) & (u < L + ext)]
            cnt += np.bincount((u - lo).astype(int), minlength=nb)[:nb]
        occ = cnt >= MIN_CROSS
        if not occ.any():
            diags.append({"L_m": round(L * RES_M, 2), "status": "no_crossing"})
            continue
        # 横断区間 (連続run) のうち図面開口 [0,L] と最も重なるもの
        runs, i = [], 0
        while i < nb:
            if occ[i]:
                j = i
                while j < nb and occ[j]:
                    j += 1
                runs.append((lo + i, lo + j))
                i = j
            else:
                i += 1
        best = None
        for ga, gb in runs:
            ov = min(gb, L) - max(ga, 0)
            center_off = (ga + gb) / 2 - L / 2
            if abs(center_off) <= slack and \
                    (best is None or ov > best[2]):
                best = (ga, gb, ov)
        if best is None:
            diags.append({"L_m": round(L * RES_M, 2), "status": "no_overlap"})
            continue
        ga, gb, _ = best
        ratio = (gb - ga) / L
        if not (WIDTH_RATIO[0] <= ratio <= WIDTH_RATIO[1]):
            diags.append({"L_m": round(L * RES_M, 2),
                          "status": f"width_ratio {ratio:.2f}"})
            continue
        marks.append((a, u_dir, L, ga, gb))
        diags.append({"L_m": round(L * RES_M, 2),
                      "cross_m": round((gb - ga) * RES_M, 2),
                      "shift_m": (round(-ga * RES_M, 2),
                                  round((L - gb) * RES_M, 2)),
                      "status": "matched"})
    return marks, diags


def inv_transform(pts, s, th, t):
    R = np.array([[math.cos(th), -math.sin(th)],
                  [math.sin(th), math.cos(th)]])
    return ((pts - t) @ R) / s


def refine_with_rays(pts_cell, segs_cell, axes, cam_src, pts_src, s, th, t):
    """壁ICP + 視線横断ランドマークの交互精密化"""
    a = segs_cell[:, :2]
    b = segs_cell[:, 2:]
    ab = b - a
    ab2 = (ab ** 2).sum(axis=1).clip(min=1e-9)
    n_matched = 0
    for rnd in range(ICP_ROUNDS):
        cam_t = transform_pts(cam_src, s, th, t)
        pts_t = [transform_pts(p, s, th, t) for p in pts_src]
        marks, diags = detect_crossings(cam_t, pts_t, axes)
        n_matched = sum(1 for d in diags if d["status"] == "matched")
        if rnd == 0:
            for d in diags:
                print(f"    opening {d}")
        if not marks:
            print("    rays: no landmarks matched")
            break
        # ランドマーク対応: 横断区間の両端 → 図面開口の両端 (線上の合成点)
        src_l, dst_l = [], []
        for ax_a, u_dir, L, ga, gb in marks:
            for u_from, u_to in ((ga, 0.0), (gb, L)):
                p_now = ax_a + u_from * u_dir
                src_l.append(inv_transform(p_now[None], s, th, t)[0])
                dst_l.append(ax_a + u_to * u_dir)
        src_l = np.stack(src_l)
        dst_l = np.stack(dst_l)
        w_land = min(W_CAP, 0.3 * len(pts_cell) / len(src_l))
        for _ in range(ICP_ITERS_PER):
            p = transform_pts(pts_cell, s, th, t)
            diff = p[:, None, :] - a[None, :, :]
            tt = np.clip((diff * ab[None, :, :]).sum(-1) / ab2[None, :], 0, 1)
            foot = a[None, :, :] + tt[..., None] * ab[None, :, :]
            dist = np.linalg.norm(p[:, None, :] - foot, axis=-1)
            j = dist.argmin(axis=1)
            dmin = dist[np.arange(len(p)), j]
            m = dmin < CUTOFF_CELL
            src = np.concatenate([pts_cell[m], src_l])
            dst = np.concatenate([foot[np.arange(len(p)), j][m], dst_l])
            w = np.concatenate([np.ones(int(m.sum())),
                                np.full(len(src_l), w_land)])
            s, th, t = fit_similarity_w(src, dst, w)
    return s, th, t, n_matched


def main():
    gt = load_gt()
    exclude = set()
    p = IMG_ALIGN_DIR / UNTRUSTED_JSON if UNTRUSTED_JSON else None
    if p is not None and p.is_file():
        exclude = {int(n.replace("frame_", ""))
                   for n in json.loads(p.read_text())}
    anchored, floors = {}, {}
    for tag, floor, chunks_root, chunk_range, mode in ISLANDS:
        if mode == "corner":
            track = json.loads(
                (OUT_DIR / f"corner_anchor_{tag}.json").read_text())
            for n, (x, y) in track.items():
                fr = int(n.replace("frame_", ""))
                if fr in exclude:
                    continue
                anchored[fr] = (float(x), float(y))
                floors[fr] = floor
            print(f"--- {tag}: corner anchor track ({len(track)} frames)")
            continue
        # mode "anchor": 既存アンカー変換をそのまま採用 (精密化なし。
        #   工事中など開口が図面と不一致の場合に使う)
        # mode "rays": 既存の図面アンカーを初期値に開口精密化
        # mode "corner_rays": コーナーアンカーを初期値に開口精密化 (短島向け:
        #   配置はコーナーが頑健、精度は開口が上)
        anc_name = (f"anchor_{tag}_corners.json" if mode == "corner_rays"
                    else f"anchor_{tag}.json")
        anc = json.loads((OUT_DIR / anc_name).read_text())
        if mode == "anchor":
            d = prepare_island(tag)
            cam_rc = np.stack(
                [(d["cam_pos_aligned"][:, 2] - d["origin"][1])
                 / d["cell_units"],
                 (d["cam_pos_aligned"][:, 0] - d["origin"][0])
                 / d["cell_units"]], axis=1)
            if anc["mirror_col"] is not None:
                cam_rc[:, 1] = anc["mirror_col"] - cam_rc[:, 1]
            grid = PlanGrid(floor)
            px = cells_to_px(transform_pts(cam_rc, anc["s"], anc["th"],
                                           np.array(anc["t"])), grid)
            for n, (x, y) in zip([str(v) for v in d["names"]], px):
                fr = int(n.replace("frame_", ""))
                if fr in exclude:
                    continue
                anchored[fr] = (float(x), float(y))
                floors[fr] = floor
            print(f"--- {tag}: anchor track ({len(px)} frames, no refine)")
            continue
        grid = PlanGrid(floor)
        d = prepare_island(tag)
        mirror_col = anc["mirror_col"]
        if mirror_col is not None:
            d["density"] = np.ascontiguousarray(d["density"][:, ::-1])
        zz, xx = np.nonzero(d["density"] > 0.15)
        pts_cell = np.stack([zz, xx], axis=1).astype(np.float64)
        segs = segs_to_cells(wall_segments_px(floor), grid)
        axes = opening_axes_cells(floor, grid)

        print(f"--- {tag}: per-frame scans ...")
        scans = load_frame_scans(tag, chunks_root, chunk_range)
        names = sorted(scans)
        anc_d = {"R_align": anc["R_align"], "origin": anc["origin"],
                 "cell_units": anc["cell_units"]}
        cam_src = island_to_src_cells(
            np.stack([scans[n][0] for n in names]), anc_d, mirror_col)
        pts_src = [island_to_src_cells(scans[n][1], anc_d, mirror_col)
                   for n in names]

        s, th, t = anc["s"], anc["th"], np.array(anc["t"])
        s, th, t, n_open = refine_with_rays(
            pts_cell, segs, axes, cam_src, pts_src, s, th, t)
        print(f"{tag}: {n_open}/{len(axes)} openings matched")
        # 精密化済みアンカーを保存 (export_aligned2 --suffix _rays 用)
        anc2 = dict(anc)
        anc2.update({"s": s, "th": th, "t": [float(v) for v in t],
                     "n_openings_matched": n_open})
        (OUT_DIR / f"anchor_{tag}_rays.json").write_text(
            json.dumps(anc2, indent=1))

        px = cells_to_px(transform_pts(cam_src, s, th, t), grid)
        for n, q in zip(names, px):
            fr = int(n.replace("frame_", ""))
            if fr in exclude:
                continue
            anchored[fr] = (float(q[0]), float(q[1]))
            floors[fr] = floor
    assemble_and_eval(anchored, floors, gt, label=LABEL)


LABEL = "_rays"

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="_rays")
    ap.add_argument("--preset", default="canon",
                    choices=["canon", "short_a", "s0312"])
    args = ap.parse_args()
    LABEL = args.label
    if args.preset == "short_a":
        ISLANDS = ISLANDS_SHORT_A
    elif args.preset == "s0312":
        ISLANDS = ISLANDS_0312
    main()
