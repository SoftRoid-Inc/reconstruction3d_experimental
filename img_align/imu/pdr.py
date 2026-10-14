"""IMU単独の歩行者デッドレコニング(PDR)軌跡 — 参考用の可視化

歩数検出(鉛直加速度のピーク)×ジャイロ方位×固定歩幅で2D軌跡を作る。
IMU単独ではジャイロバイアスによる方位ドリフトと歩幅誤差が蓄積するため、
位置合わせには使わず「IMUがどこまで形状を語れるか」の参考図とする。
右図はGT軌跡へ相似変換フィットした形状比較。

出力: results/imu/pdr_trajectory.png, pdr.npz
"""
import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

from img_align.config import IMU_DIR
from img_align.eval.gt_traj import MM_PER_PX, load_gt

STEP_LEN_M = 0.65
STEP_PROM = 0.8      # 歩数ピークのprominence [m/s^2]
STEP_MIN_DT = 0.35   # 歩の最小間隔 [s]


def fit_similarity_2d(src, dst):
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    sc, dc = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd(dc.T @ sc / len(sc))
    S = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[1, 1] = -1
    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / (sc ** 2).sum() * len(sc))
    return s, R, mu_d - s * R @ mu_s


def main():
    att = np.load(IMU_DIR / "attitude.npz")
    sync = json.loads((IMU_DIR / "sync.json").read_text())
    stairs = json.loads((IMU_DIR / "stairs.json").read_text())["stairs"]
    t, yaw_rate, a_vert = att["t"], att["yaw_rate"] * sync["sign"], att["a_vert"]

    heading = np.concatenate([[0], np.cumsum(yaw_rate[:-1] * np.diff(t))])
    rate = len(t) / (t[-1] - t[0])
    peaks, _ = find_peaks(a_vert, prominence=STEP_PROM,
                          distance=int(STEP_MIN_DT * rate))
    print(f"steps detected: {len(peaks)} "
          f"(cadence {len(peaks) / (t[-1] - t[0]):.2f} /s)")

    pos = np.zeros((len(peaks) + 1, 2))
    for i, p in enumerate(peaks):
        h = heading[p]
        pos[i + 1] = pos[i] + STEP_LEN_M * np.array([math.cos(h), math.sin(h)])
    t_steps = np.concatenate([[t[0]], t[peaks]])
    frames = (t_steps - sync["offset_s"]) * sync["fps"]

    # GT (図面px→m、1F/2F連結) と形状比較のための相似フィット
    gt = load_gt()
    gt_f = np.array(sorted(gt))
    gt_xy = np.array([[gt[f]["x_px"], gt[f]["y_px"]] for f in gt_f]) \
        * MM_PER_PX / 1000
    pdr_at_gt = np.stack([np.interp(gt_f, frames, pos[:, 0]),
                          np.interp(gt_f, frames, pos[:, 1])], axis=1)
    s, R, tr = fit_similarity_2d(pdr_at_gt, gt_xy)
    pdr_fit = pdr_at_gt @ (s * R).T + tr
    resid = np.linalg.norm(pdr_fit - gt_xy, axis=1)
    print(f"PDR vs GT (相似フィット後): median {np.median(resid):.2f}m / "
          f"p90 {np.percentile(resid, 90):.2f}m / scale {s:.2f}")

    np.savez_compressed(IMU_DIR / "pdr.npz", frames=frames, pos=pos,
                        t_steps=t_steps)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    o = np.linspace(0, 1, len(pos))
    axes[0].scatter(pos[:, 0], pos[:, 1], c=o, cmap="viridis", s=8)
    for iv in stairs:
        m = (frames >= iv["frame0"]) & (frames <= iv["frame1"])
        axes[0].plot(pos[m, 0], pos[m, 1], "r.", ms=8)
    axes[0].set_title("IMU-only PDR path (rel. coords, red=stairs)")
    axes[0].set_aspect("equal")
    axes[1].plot(gt_xy[:, 0], gt_xy[:, 1], "-", color="lime", lw=2, label="GT")
    axes[1].scatter(pdr_fit[:, 0], pdr_fit[:, 1],
                    c=np.linspace(0, 1, len(pdr_fit)), cmap="autumn", s=8,
                    label="PDR (similarity-fitted)")
    axes[1].invert_yaxis()
    axes[1].set_title(f"shape vs GT: median {np.median(resid):.2f}m")
    axes[1].set_aspect("equal")
    axes[1].legend()
    fig.savefig(IMU_DIR / "pdr_trajectory.png", dpi=110, bbox_inches="tight")
    print(f"-> {IMU_DIR / 'pdr_trajectory.png'}")


if __name__ == "__main__":
    main()
