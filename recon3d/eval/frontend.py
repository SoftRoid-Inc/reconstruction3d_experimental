"""前段対決の評価: 連結後のグローバル軌跡・点群をGTと比較する

評価は前段に対して公平になるよう「図面照合なし」で行う:
  重力整列 → (鏡映2択 × 2D相似変換) をGT軌跡に全フレームでフィット → ATE。
これは「前段の相対幾何がどれだけ正確か」を測る (後段の図面ICPの性能とは独立)。

使い方: python main.py tool frontend --input /data/x --tags panovggt da3base
出力: results/frontend_bench/compare.json / compare_<tag>.png (上面図)
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from recon3d.core.geometry import rotation_to_y
from recon3d.eval.gt_traj import mm_per_px, load_gt


def stair_ranges(paths):
    """階段区間。IMU検出 (imu/transitions.json) から取る"""
    from recon3d.core.stair_check import imu_stair_ranges
    return imu_stair_ranges(paths)


def fit2d(src, dst):
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    sc, dc = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd(dc.T @ sc / len(sc))
    S = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[1, 1] = -1
    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / (sc ** 2).sum() * len(sc))
    return lambda p: (s * (p @ R.T)) + (mu_d - s * R @ mu_s)


def evaluate(paths, tag, gt, frame_range=None):
    d = np.load(paths.bench / tag / "global_cameras.npz")
    names = [str(n) for n in d["names"]]
    frames = np.array([int(n.replace("frame_", "")) for n in names])
    pos = d["positions"]
    poses_all = d["poses"]
    if frame_range is not None:
        m = (frames >= frame_range[0]) & (frames <= frame_range[1])
        frames, pos, poses_all = frames[m], pos[m], poses_all[m]
        tag = f"{tag}[{frame_range[0]}-{frame_range[1]}]"
    ups = poses_all[:, :3, 1]
    up = np.median(ups / np.linalg.norm(ups, axis=1, keepdims=True), axis=0)
    R = rotation_to_y(up / np.linalg.norm(up))
    p = pos @ R.T

    # GTは線形補間する。最近傍スナップ(旧実装)だと1フレームのずれが
    # 0.5s×0.6m/s = 約0.3mの誤差として乗り、GT欠損区間でATEを大きく水増しした
    from recon3d.eval.chunk_cmp import gt_interp
    mpx = mm_per_px(paths)
    if mpx is None:
        # ATEはメートルで出す約束なので、px→m が決まらない入力では評価しない。
        # 壁注釈だけでは足りず**図面の実寸 (known_width_mm) も要る**
        raise ValueError(
            "mm/px が無いので ATE を測れない (図面の labelme 注釈と "
            "floors/meta.json の known_width_mm の両方が必要)。"
            "評価を飛ばすなら eval ステップを外してください")
    gt_xy, used = gt_interp(gt, frames, mpx)

    best = None
    for mirror in (False, True):
        xz = p[:, [0, 2]].copy()
        if mirror:
            xz[:, 1] *= -1
        T = fit2d(xz[used], gt_xy)
        e = np.linalg.norm(T(xz[used]) - gt_xy, axis=1)
        if best is None or np.median(e) < best["ate_median"]:
            best = {"mirror": mirror, "ate_median": float(np.median(e)),
                    "ate_p90": float(np.percentile(e, 90)),
                    "ate_mean": float(np.mean(e)),
                    "xy": T(xz), "err": e}
    # 階段区間をまたぐ連続性 (隣接フレーム間の移動距離)
    step = np.linalg.norm(np.diff(best["xy"], axis=0), axis=1)
    stair_steps = [s for i, s in enumerate(step)
                   if any(a <= frames[i] <= b for a, b in stair_ranges(paths))]
    chain_path = paths.bench / tag.split("[")[0] / "chain_log.json"
    chain = json.loads(chain_path.read_text()) if chain_path.exists() else []
    # chain_log の "scale" は連結ログではスカラー、posegraph では [min,max] リスト
    scales = []
    for c in chain:
        v = c.get("scale")
        if isinstance(v, (int, float)):
            scales.append(float(v))
        elif isinstance(v, (list, tuple)):
            scales.extend(float(x) for x in v)
        for k in ("scale_min", "scale_max"):
            if isinstance(c.get(k), (int, float)):
                scales.append(float(c[k]))
    scales = scales or [1.0]
    res = {
        "tag": tag,
        "n_frames": len(frames),
        "ate_median_m": round(best["ate_median"], 3),
        "ate_p90_m": round(best["ate_p90"], 3),
        "ate_mean_m": round(best["ate_mean"], 3),
        "mirror": best["mirror"],
        "step_median_m": round(float(np.median(step)), 3),
        "step_max_m": round(float(step.max()), 3),
        "stair_step_max_m": round(float(max(stair_steps)), 3) if stair_steps else None,
        "chain_scale_min_max": [round(min(scales), 3), round(max(scales), 3)],
        "chain_resid_median": round(float(np.median(
            [c["icp_resid"] for c in chain if "icp_resid" in c] or [0.0])), 4),
    }
    print(json.dumps(res, indent=1))

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.plot(gt_xy[:, 0], gt_xy[:, 1], "-", color="lime", lw=2, label="GT")
    sc = ax.scatter(best["xy"][:, 0], best["xy"][:, 1],
                    c=np.linspace(0, 1, len(best["xy"])), cmap="autumn", s=14,
                    label=tag)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_title(f"{tag}: ATE median {res['ate_median_m']}m / "
                 f"p90 {res['ate_p90_m']}m")
    ax.legend()
    fig.savefig(paths.bench / f"compare_{tag}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return res


def main(paths, argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    args = ap.parse_args(argv)
    gt = load_gt(paths)
    results = [evaluate(paths, t, gt) for t in args.tags]
    (paths.bench / "compare.json").write_text(json.dumps(results, indent=1))
    print(f"-> {paths.bench / 'compare.json'}")

