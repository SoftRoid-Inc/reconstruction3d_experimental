"""フレーム番号↔IMU時刻の同期推定

仮説: t_frame = frame / FPS + offset。
「フレーム間ヨー差(PanoVGGTチャンク姿勢由来)」と「IMUヨーレート積分」の
一致度(cos誤差)を offset のグリッドで探索する。

出力: results/imu/sync.json {offset_s, fps, score}
"""
import json
import math

import numpy as np

from img_align.config import IMU_DIR, SESSION

FPS = 2.0
OFFSET_GRID = np.arange(-8.0, 8.0, 0.05)


def frame_yaw_diffs():
    """隣接フレーム間ヨー差 (チャンク姿勢由来、rad)。

    チャンクごとに重力整列した前方ベクトルの方位角の差を、チャンク内の
    隣接フレームペアについて集める (チャンク境界はまたがない。重複ペアは先勝ち)
    """
    from img_align.core.panovggt import PCD_DIR
    from img_align.legacy.stitch_chunks import gravity_rotation
    root = PCD_DIR / ("chunks" if SESSION == "0828" else f"chunks_{SESSION}")
    frames, dyaws, seen = [], [], set()
    for cdir in sorted(root.glob("chunk_*")):
        p = cdir / "cameras.npz"
        if not p.exists():
            continue
        cams = np.load(p)
        names = [str(n) for n in cams["names"]]
        G = gravity_rotation(cams["poses"])
        fwd = cams["poses"][:, :3, 2] @ G.T
        fwd[:, 2] *= -1  # density_mapと同じz反転
        heading = np.arctan2(fwd[:, 2], fwd[:, 0])
        nums = [int(n.replace("frame_", "")) for n in names]
        for i in range(len(nums) - 1):
            pair = (nums[i], nums[i + 1])
            if pair in seen:
                continue
            seen.add(pair)
            frames.append(pair)
            dh = math.remainder(heading[i + 1] - heading[i], 2 * math.pi)
            dyaws.append(dh)
    if not frames:
        raise FileNotFoundError(f"チャンクが見つからない: {root}")
    return frames, np.array(dyaws)


def imu_yaw_integral(t, yaw_rate):
    cum = np.concatenate([[0], np.cumsum(
        yaw_rate[:-1] * np.diff(t))])

    def integral(t0, t1):
        i0, i1 = np.searchsorted(t, [t0, t1])
        i0, i1 = np.clip([i0, i1], 0, len(cum) - 1)
        return cum[i1] - cum[i0]
    return integral


def main():
    att = np.load(IMU_DIR / "attitude.npz")
    t, yaw_rate = att["t"], att["yaw_rate"]
    frames, dyaw_f = frame_yaw_diffs()
    integral = imu_yaw_integral(t, yaw_rate)

    best = None
    for off in OFFSET_GRID:
        d_imu = np.array([integral(a / FPS + off, b / FPS + off)
                          for a, b in frames])
        # ヨーの符号規約が逆の可能性もあるので両方試す
        for sign in (1, -1):
            err = np.angle(np.exp(1j * (sign * d_imu - dyaw_f)))
            score = float(np.cos(err).mean())
            if best is None or score > best["score"]:
                best = {"offset_s": round(float(off), 3), "sign": sign,
                        "score": round(score, 4),
                        "median_err_deg": round(float(np.median(
                            np.abs(np.rad2deg(err)))), 2)}
    best["fps"] = FPS
    (IMU_DIR / "sync.json").write_text(json.dumps(best, indent=1))
    print(json.dumps(best, indent=1))


if __name__ == "__main__":
    main()
