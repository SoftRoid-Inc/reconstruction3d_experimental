"""ジャイロ積分によるフレーム間相対ヨー (temporal_fuse / joint_refine 用の高精度事前)

出力: results/imu/yaw_odometry.npz
  pairs_step1: (453,3) [frame_a, frame_b, dyaw_rad] 全フレーム1刻み
  pairs_step3: (151,3) 3刻み系列 (temporal_fuseのオドメトリ置換用)
"""
import json

import numpy as np

from img_align.config import IMU_DIR



def main():
    att = np.load(IMU_DIR / "attitude.npz")
    sync = json.loads((IMU_DIR / "sync.json").read_text())
    t, yaw_rate = att["t"], att["yaw_rate"] * sync["sign"]
    cum = np.concatenate([[0], np.cumsum(yaw_rate[:-1] * np.diff(t))])

    def integral(f0, f1):
        t0 = f0 / sync["fps"] + sync["offset_s"]
        t1 = f1 / sync["fps"] + sync["offset_s"]
        i0, i1 = np.clip(np.searchsorted(t, [t0, t1]), 0, len(cum) - 1)
        return cum[i1] - cum[i0]

    out = {}
    for step, key in ((1, "pairs_step1"), (3, "pairs_step3")):
        rows = [(a, a + step, integral(a, a + step))
                for a in range(0, 454 - step, step)]
        out[key] = np.array(rows)
    np.savez_compressed(IMU_DIR / "yaw_odometry.npz", **out)
    d3 = np.rad2deg(np.abs(out["pairs_step3"][:, 2]))
    print(f"step3 |dyaw|: median {np.median(d3):.1f} deg / p95 "
          f"{np.percentile(d3, 95):.1f} deg")
    print(f"-> {IMU_DIR / 'yaw_odometry.npz'}")


if __name__ == "__main__":
    main()
