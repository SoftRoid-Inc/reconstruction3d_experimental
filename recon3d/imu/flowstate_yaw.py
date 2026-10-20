"""FlowState の安定化クォータニオンから、フレーム毎の世界ヨーを取り出す

**注意 (この2点を外すと無意味な値になる)**:
1. insv の生IMU (gyro/accel) は**カメラ本体の回転**を測るが、画像は FlowState で
   世界固定に安定化されている。両者は別物で、画像側の向きを知りたいなら
   **安定化クォータニオン (トレーラー id=24)** を使う。生IMUからは出ない。
2. クォータニオンは約3ms刻み、画像は0.5秒刻み。`imu/timeline` の実測フレーム
   時刻で補間して揃える。両者とも**フレーム0の露光 = 0秒**基準
   (0912で実測: 動画長の差0.011秒)。

これが効く理由: PanoVGGTのヨーゲージは島ごとに任意 (チャンク間でstd 34〜47°)
だが、FlowState は画像側の**絶対基準**なので島をまたいで共通。島ごとの
「ゲージと世界のヨー差」を取れば、島同士の相対回転が分かる。

使い方 (imuコンテナ内。telemetry_parser が要る):
  docker compose exec -e IMG_SESSION=0912 imu \\
      python -m recon3d.imu.flowstate_yaw
出力: results/<s>/imu/flowstate_yaw.npz (frames, yaw_deg)
"""
import argparse
import sys
from pathlib import Path

import numpy as np


IMU_TRAJ_PATHS = (
    "/mnt/data/git/polish-pipeline/submodules/imu_experiments_code",
    "/polish-pipeline/submodules/imu_experiments_code",
)


def _imu_traj():
    for p in IMU_TRAJ_PATHS:
        if Path(p).exists():
            if p not in sys.path:
                sys.path.insert(0, p)
            from imu_traj.gpmf_quat import load_video_quaternions
            from imu_traj.pdr import _flowstate_yaw
            return load_video_quaternions, _flowstate_yaw
    raise RuntimeError(f"imu_traj が見つかりません: {IMU_TRAJ_PATHS}")


def insv_path(paths):
    d = paths.root
    cands = sorted(d.glob("*.insv"))
    if not cands:
        raise FileNotFoundError(f"{d} に .insv がありません")
    return cands[0]


def compute(paths, out_name="flowstate_yaw.npz"):
    """フレーム毎の世界ヨー [deg] を出す"""
    from recon3d.imu.timeline import frame_times, n_frames
    load_quat, flow_yaw = _imu_traj()
    stab = load_quat(insv_path(paths))
    t_y, yaw = flow_yaw(stab)
    n = n_frames(paths)
    t, src = frame_times(paths, n)
    t = t - t[0]                      # クォータニオン側と同じ「フレーム0=0」へ
    # ヨーは角度なので複素数にしてから補間する (±180°の折返しを跨がないため)
    z = np.interp(t, t_y, np.cos(np.radians(yaw))) + \
        1j * np.interp(t, t_y, np.sin(np.radians(yaw)))
    deg = np.degrees(np.angle(z))
    paths.imu.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(paths.imu / out_name,
                        frames=np.arange(n), yaw_deg=deg,
                        n_quat=len(stab.q), frame_time_src=src)
    print(f"[flowstate] {n}フレーム (quat {len(stab.q)}個, 時刻={src}) "
          f"ヨー {deg.min():.1f}〜{deg.max():.1f}° -> {paths.imu / out_name}")
    return deg


def load(paths, out_name="flowstate_yaw.npz"):
    """保存済みのフレーム毎ヨー [deg]。無ければ None"""
    p = paths.imu / out_name
    return np.load(p)["yaw_deg"] if p.exists() else None



def main(paths, argv=None):
    """`.insv` の手ぶれ補正クォータニオンからフレーム毎ヨーを作る"""
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="flowstate_yaw.npz")
    args = ap.parse_args(argv)
    compute(paths, out_name=args.out)
