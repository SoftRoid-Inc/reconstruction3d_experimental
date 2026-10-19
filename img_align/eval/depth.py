"""深度源の壁面品質比較: PanoVGGT vs MapAnything (ポーズ条件付け/深度のみ)

各ソースの軌跡を重力整列+2D相似でGT軌跡へフィットし、同じ変換で点群を
図面座標(メートル)へ載せ、図面GT壁の面ライン (refine_align.wall_face_segments_px)
への距離で比較する。ポーズのゲージをGTで固定するため、純粋に
「点群(深度)の幾何品質」の横並び比較になる。

ソース指定:
  --tags <tag> ...            frontend_bench/<tag>/global_points.ply をそのタグの
                              軌跡フィットで載せる (PanoVGGT chain / MapAnything出力)
  --local <ptag>:<posetag> ...  <ptag>/local_points.npz (リグ座標・メートル) を
                              <posetag> の軌跡・姿勢で配置する (深度のみ利用=案2)

使い方 (ホスト img_align env):
  IMG_SESSION=0616 python -m img_align.eval.depth --floor 2F \
      --tags pv0616_2F mapany0616_2F macond0616_2F --local mapany0616_2F:pv0616_2F
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from img_align.config import BENCH_DIR
from img_align.core.geometry import rotation_to_y
from img_align.eval.frontend import fit2d
from img_align.eval.gt_traj import mm_per_px, load_gt
from img_align.core.io_ply import read_ply_xyzrgb
from img_align.postproc.plan_icp import wall_face_segments_px

BAND_HALF_M = 0.5       # 壁帯スライス: カメラ中央高さ ±0.5m
DIST_CAP_M = 1.0        # 壁面距離の打ち切り (外れ点対策)


def load_traj(tag):
    d = np.load(BENCH_DIR / tag / "global_cameras.npz")
    names = [str(n) for n in d["names"]]
    frames = np.array([int(n.replace("frame_", "")) for n in names])
    return names, frames, d["positions"], d["poses"]


def fit_to_gt(frames, positions, poses, gt):
    """重力整列+鏡映2択の2D相似フィット。返り値:
    (A: 3x3線形(回転+鏡映, 正規直交), s: スケール, t2: XZ並進, ate_median)
    図面メートル座標 = s * A@p の[XZ] + t2、高さ = (s*A@p)[Y]"""
    ups = poses[:, :3, 1]
    up = np.median(ups / np.linalg.norm(ups, axis=1, keepdims=True), axis=0)
    rg = rotation_to_y(up / np.linalg.norm(up))
    p = positions @ rg.T
    gt_xy, used = [], []
    for i, f in enumerate(frames):
        g = min(gt, key=lambda k: abs(k - f))
        if abs(g - f) <= 3:
            gt_xy.append([gt[g]["x_px"] * mm_per_px() / 1000,
                          gt[g]["y_px"] * mm_per_px() / 1000])
            used.append(i)
    gt_xy = np.array(gt_xy)
    best = None
    for mirror in (False, True):
        mir = np.diag([1.0, 1.0, -1.0 if mirror else 1.0])
        q = p @ mir.T
        xz = q[:, [0, 2]]
        t2d = fit2d(xz[used], gt_xy)
        err = np.linalg.norm(t2d(xz[used]) - gt_xy, axis=1)
        if best is None or np.median(err) < best[0]:
            # fit2dのクロージャからs,R,tを数値で取り出す (基底ベクトルを通す)
            o = t2d(np.zeros((1, 2)))[0]
            ex = t2d(np.array([[1.0, 0]]))[0] - o
            ez = t2d(np.array([[0, 1.0]]))[0] - o
            s = float(np.linalg.norm(ex))
            r2 = np.stack([ex / s, ez / s], axis=1)     # 列=像
            a = np.eye(3)
            a[np.ix_([0, 2], [0, 2])] = r2
            best = (float(np.median(err)), a @ mir @ rg, s, o,
                    bool(mirror))
    ate, A, s, t2, mirror = best
    return A, s, t2, ate, mirror


def to_drawing(pts, A, s, t2):
    """点群→(図面X[m], 図面Y[m], 高さ[m])"""
    q = (s * (pts @ A.T))
    return np.stack([q[:, 0] + t2[0], q[:, 2] + t2[1], q[:, 1]], axis=1)


def wall_metrics(draw_pts, cam_h, segs_px, label):
    """カメラ高さ帯の点の壁面距離統計"""
    band = (draw_pts[:, 2] > cam_h - BAND_HALF_M) & \
           (draw_pts[:, 2] < cam_h + BAND_HALF_M)
    pts = draw_pts[band][:, :2] * 1000 / mm_per_px()          # px
    # 面ラインを2px間隔でサンプルしてKDTree
    samples = []
    for x1, y1, x2, y2 in segs_px:
        n = max(2, int(np.hypot(x2 - x1, y2 - y1) / 2))
        samples.append(np.stack([np.linspace(x1, x2, n),
                                 np.linspace(y1, y2, n)], axis=1))
    tree = cKDTree(np.concatenate(samples))
    d_px, _ = tree.query(pts, k=1)
    d_m = np.minimum(d_px * mm_per_px() / 1000, DIST_CAP_M)
    return {
        "source": label,
        "n_band_pts": int(len(pts)),
        "wall_med_m": round(float(np.median(d_m)), 4),
        "wall_p90_m": round(float(np.percentile(d_m, 90)), 4),
        "within10cm": round(float((d_m <= 0.10).mean()), 3),
        "within20cm": round(float((d_m <= 0.20).mean()), 3),
    }, pts


def overlay_png(path, pts_px, segs_px, title):
    fig, ax = plt.subplots(figsize=(10, 7))
    for x1, y1, x2, y2 in segs_px:
        ax.plot([x1, x2], [y1, y2], color="c", lw=1)
    sub = pts_px[:: max(1, len(pts_px) // 60000)]
    ax.scatter(sub[:, 0], sub[:, 1], s=0.3, c="r", alpha=0.3)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_title(title)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", required=True)
    ap.add_argument("--tags", nargs="*", default=[])
    ap.add_argument("--local", nargs="*", default=[],
                    help="'点群タグ:ポーズタグ' の列 (local_points.npzを配置)")
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    gt = {f: v for f, v in load_gt().items() if v["floor"] == args.floor}
    segs = wall_face_segments_px(args.floor)
    results = []

    for tag in args.tags:
        names, frames, pos, poses = load_traj(tag)
        A, s, t2, ate, mirror = fit_to_gt(frames, pos, poses, gt)
        xyz, _ = read_ply_xyzrgb(BENCH_DIR / tag / "global_points.ply")
        draw = to_drawing(xyz, A, s, t2)
        cam_h = float(np.median(to_drawing(pos, A, s, t2)[:, 2]))
        r, pts_px = wall_metrics(draw, cam_h, segs, tag)
        r.update(traj_ate_m=round(ate, 3), mirror=mirror, fit_scale=round(s, 4))
        results.append(r)
        overlay_png(BENCH_DIR / f"depth_eval_{tag}.png", pts_px, segs,
                    f"{tag}: wall med {r['wall_med_m']}m "
                    f"/ ≤10cm {r['within10cm']:.0%}")
        print(json.dumps(r, ensure_ascii=False))

    for spec in args.local:
        ptag, posetag = spec.split(":")
        names, frames, pos, poses = load_traj(posetag)
        A, s, t2, ate, mirror = fit_to_gt(frames, pos, poses, gt)
        d = np.load(BENCH_DIR / ptag / "local_points.npz")
        name_to_i = {n: i for i, n in enumerate(names)}
        placed = []
        for key in d.files:
            if key not in name_to_i:
                continue
            i = name_to_i[key]
            loc = d[key][:, :3]                       # リグ座標・メートル
            c_draw = to_drawing(pos[i][None], A, s, t2)[0]
            rot = A @ poses[i][:3, :3]                # 図面系でのリグ回転
            q = loc @ rot.T                           # メートルのまま回す (sは掛けない)
            placed.append(np.stack([q[:, 0] + c_draw[0], q[:, 2] + c_draw[1],
                                    q[:, 1] + c_draw[2]], axis=1))
        draw = np.concatenate(placed)
        cam_h = float(np.median(to_drawing(pos, A, s, t2)[:, 2]))
        label = f"{ptag}@{posetag}"
        r, pts_px = wall_metrics(draw, cam_h, segs, label)
        r.update(traj_ate_m=round(ate, 3), mirror=mirror, fit_scale=round(s, 4),
                 n_frames_placed=len(placed))
        results.append(r)
        overlay_png(BENCH_DIR / f"depth_eval_{ptag}_at_{posetag}.png", pts_px,
                    segs, f"{label}: wall med {r['wall_med_m']}m "
                    f"/ ≤10cm {r['within10cm']:.0%}")
        print(json.dumps(r, ensure_ascii=False))

    out = BENCH_DIR / f"depth_eval{('_' + args.label) if args.label else ''}.json"
    out.write_text(json.dumps(results, indent=1, ensure_ascii=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
