"""鉛直加速度から階段昇降イベントを検出し、GTの階タイムラインと照合する

二重積分のドリフトは「歩行の平均鉛直速度≈0」を仮定した移動平均除去で抑える
(絶対高さではなくイベント検出が目的)。

出力: results/imu/stairs.json (検出区間+フレーム範囲+推定Δz)
      results/imu/vertical.png (v_z, z, GT階境界の可視化)
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from img_align.config import IMU_DIR

V_TH = 0.12          # 階段判定の鉛直速度しきい値 [m/s]
MIN_DUR = 2.0        # 最小持続 [s]
EDGE_TRIM_S = 12.0   # 録画開始/終了付近はカメラ上げ下ろしのアーティファクト
DETREND_WIN_S = 30.0  # 速度ドリフト除去の移動平均窓
SMOOTH_S = 1.2
GT_FLOOR_BOUNDS = [(233.5, "1F->2F"), (411.5, "2F->1F")]  # フレーム境界 (GT)


def rolling_mean(x, n):
    n = max(int(n), 1)
    pad = np.pad(x, (n // 2, n - n // 2 - 1), mode="edge")
    return np.convolve(pad, np.ones(n) / n, mode="valid")


def main():
    att = np.load(IMU_DIR / "attitude.npz")
    sync = json.loads((IMU_DIR / "sync.json").read_text())
    t, a_vert = att["t"], att["a_vert"]
    rate = len(t) / (t[-1] - t[0])

    dt = np.gradient(t)
    v = np.cumsum(a_vert * dt)
    v = v - rolling_mean(v, DETREND_WIN_S * rate)   # 歩行≈平均0仮定のドリフト除去
    v_s = rolling_mean(v, SMOOTH_S * rate)
    z = np.cumsum(v_s * dt)

    active = np.abs(v_s) > V_TH
    intervals = []
    i = 0
    while i < len(active):
        if active[i]:
            j = i
            while j < len(active) and active[j]:
                j += 1
            if t[j - 1] - t[i] >= MIN_DUR:
                dz = float(np.sum(v_s[i:j] * dt[i:j]))
                f0 = (t[i] - sync["offset_s"]) * sync["fps"]
                f1 = (t[j - 1] - sync["offset_s"]) * sync["fps"]
                intervals.append({"t0": round(float(t[i]), 2),
                                  "t1": round(float(t[j - 1]), 2),
                                  "frame0": round(f0, 1),
                                  "frame1": round(f1, 1),
                                  "dz_m": round(dz, 2)})
            i = j
        else:
            i += 1
    big = [iv for iv in intervals if abs(iv["dz_m"]) > 1.0]
    for iv in big:
        iv["edge_artifact"] = bool(iv["t0"] < t[0] + EDGE_TRIM_S
                                   or iv["t1"] > t[-1] - EDGE_TRIM_S)
    stairs = [iv for iv in big if not iv["edge_artifact"]]
    (IMU_DIR / "stairs.json").write_text(json.dumps(
        {"intervals": intervals, "stair_candidates": big,
         "stairs": stairs}, indent=1))
    print(json.dumps(stairs, indent=1))
    print("GT境界(フレーム):", GT_FLOOR_BOUNDS)

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fr = (t - sync["offset_s"]) * sync["fps"]
    axes[0].plot(fr, v_s, lw=0.7)
    axes[0].axhline(V_TH, color="r", ls=":")
    axes[0].axhline(-V_TH, color="r", ls=":")
    axes[0].set_ylabel("v_z [m/s]")
    axes[1].plot(fr, z, lw=0.9)
    axes[1].set_ylabel("z (drift-removed) [m]")
    axes[1].set_xlabel("frame")
    for ax in axes:
        for fb, name in GT_FLOOR_BOUNDS:
            ax.axvline(fb, color="g", ls="--")
        for iv in big:
            ax.axvspan(iv["frame0"], iv["frame1"], color="orange", alpha=0.3)
    fig.savefig(IMU_DIR / "vertical.png", dpi=110, bbox_inches="tight")
    print(f"-> {IMU_DIR / 'vertical.png'}")


if __name__ == "__main__":
    main()
