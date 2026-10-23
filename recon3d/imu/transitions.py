"""IMUフロア遷移(階段)検出: imu_traj の学習済みstairnet CNNのラッパー

本番用の階段区間検出。GT軌跡に依存せず、imu_raw.npz だけから昇降イベント
(up/down、フレーム区間、信頼度)を出す。島計画(core/chunking)のGTフリー区間ソース。

実装本体は polish-pipeline の imu_traj パッケージ (本番と同一コード):
  ZUPT→相補フィルタ→世界系6ch@200Hz→1秒窓CNN(flat/up/down)→平滑→イベント抽出。
  156動画・1,276遷移イベントで学習済み。
本モジュールは (1) npzスキーマ変換 (gyro/accel→gyr/acc)、(2) フレーム時刻の供給、
(3) IMU_DIRへの保存、のみを担う。

イベント時刻→フレーム番号の変換には **insvテレメトリ由来の実測フレーム時刻**
(imu/frame_times.py → imu/timeline.py) を渡す。等間隔仮定 (duration_s/N) は
末尾で約1フレームずれる (全セッションで実測)。

telemetry_parserとimu_trajはimuコンテナにのみあるため、コンテナ内で実行する:
  docker compose exec imu python main.py tool transitions --input /data/x
出力: results/<s>/imu/transitions.json
"""
import json
import sys

import numpy as np

from recon3d.imu.timeline import frame_times, n_frames

IMU_TRAJ = "/polish-pipeline/submodules/imu_experiments_code"


def _imu_traj():
    if IMU_TRAJ not in sys.path:
        sys.path.insert(0, IMU_TRAJ)
    from imu_traj.transitions import detect_floor_transitions
    return detect_floor_transitions


def detect(paths, out_name="transitions.json", **kw):
    """imu_raw.npz → transitions.json。kwはdetect_floor_transitionsへ透過

    offset_sec=0.0 (教師ラベル遅れ補正は本データでは過補正で、GT比10〜15フレーム
    早くずれることを0828/0616/0912で実測)。**imu_traj 側の既定も 0.0 になった**ので
    この setdefault は通常 no-op。旧版の imu_traj を指した場合の保険として残す
    (145撮影・GT境界230本の再採点で境界再現率 79.6%→93.9%。
     ~/.claude/jobs/imu-transitions/)。
    既知の限界: 短い階段(2F→3F、数秒)を見逃すことがある (0616/0925で実測)。
    見逃しは core/stair_check の幾何検算が引き受ける。
    """
    kw.setdefault("offset_sec", 0.0)
    raw = np.load(paths.imu / "imu_raw.npz")
    conv = paths.imu / "imu_raw_traj.npz"      # imu_trajスキーマへ変換
    np.savez_compressed(conv, t=raw["t"], gyr=raw["gyro"], acc=raw["accel"],
                        scaled_from_g=np.array(False))
    n = n_frames(paths)
    t, src = frame_times(paths, n)
    kw.setdefault("frame_times", t)
    kw.setdefault("n_frames", n)
    kw.setdefault("frame_dt", float(np.median(np.diff(t))))
    print(f"[transitions] frame times: {src} (n={n}, "
          f"median dt={kw['frame_dt']:.4f}s)")
    res = _imu_traj()(imu_npz=conv, out_json=paths.imu / out_name, **kw)
    for e in res["events"]:
        print(f"  {e['direction']:4s} frames {e['frame_start']:4.0f}-"
              f"{e['frame_end']:4.0f}  conf {e.get('confidence', 0):.2f}")
    return res



def main(paths, argv=None):
    """IMUから階移動を検出して transitions.json に保存する"""
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="transitions.json")
    args = ap.parse_args(argv)
    detect(paths, out_name=args.out)
