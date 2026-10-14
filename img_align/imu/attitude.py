"""相補フィルタによる姿勢推定と派生量 (重力方向・ヨーレート・鉛直加速度)

出力: results/imu/attitude.npz
  t[s], g_body(N,3 単位ベクトル: 身体座標系での重力方向),
  yaw_rate[rad/s] (重力軸まわりの回転レート), a_vert[m/s^2] (重力除去済み鉛直加速度)
"""
import json

import numpy as np

from img_align.config import IMU_DIR

G = 9.80665
ACC_GAIN = 0.02          # 相補フィルタの補正ゲイン (|a|≈g のとき)
ACC_TOL = 1.5            # 補正を有効にする |a|-g の許容 [m/s^2]


def load_imu():
    d = np.load(IMU_DIR / "imu_raw.npz")
    return d["t"], d["gyro"], d["accel"]


def run_filter(t, gyro, accel):
    """g_body(t): 重力方向(身体座標)をジャイロで伝播し加速度で補正"""
    n = len(t)
    g_body = np.empty((n, 3))
    a0 = accel[: max(int(len(t) * 0.005), 10)].mean(axis=0)
    g_body[0] = a0 / np.linalg.norm(a0)
    for i in range(1, n):
        dt = t[i] - t[i - 1]
        if not (0 < dt < 0.1):
            dt = 0.0
        # 重力ベクトルは身体系では -ω × g で回る
        g = g_body[i - 1] - np.cross(gyro[i], g_body[i - 1]) * dt
        an = np.linalg.norm(accel[i])
        if abs(an - G) < ACC_TOL:
            g = (1 - ACC_GAIN) * g + ACC_GAIN * (accel[i] / an)
        g_body[i] = g / np.linalg.norm(g)
    yaw_rate = np.einsum("ij,ij->i", gyro, g_body)
    a_vert = np.einsum("ij,ij->i", accel, g_body) - G
    return g_body, yaw_rate, a_vert


def main():
    t, gyro, accel = load_imu()
    g_body, yaw_rate, a_vert = run_filter(t, gyro, accel)
    # 重力の実測ノルム(静止窓)で定常バイアスを除去 (センサスケール誤差対策)
    meta = json.loads((IMU_DIR / "meta.json").read_text())
    a_vert = a_vert - (meta["gravity_norm_still"] - G)
    np.savez_compressed(IMU_DIR / "attitude.npz", t=t, g_body=g_body,
                        yaw_rate=yaw_rate, a_vert=a_vert)
    # 静止窓での鉛直加速度残差 (フィルタ品質の目安)
    meta = json.loads((IMU_DIR / "meta.json").read_text())
    i0, i1 = meta["notes"]["still_window_idx"]
    print(json.dumps({
        "a_vert_still_bias": round(float(np.median(a_vert[i0:i1])), 4),
        "a_vert_std": round(float(a_vert.std()), 3),
        "yaw_rate_p99_deg_s": round(float(np.rad2deg(
            np.percentile(np.abs(yaw_rate), 99))), 1),
    }, indent=1))
    print(f"-> {IMU_DIR / 'attitude.npz'}")


if __name__ == "__main__":
    main()
