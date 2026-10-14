"""静止区間の検出: 基線退化による局所破綻を防ぐ (imu_traj idle.py のラッパー)

島内誤差の診断 (2026-10-08) で、軌跡の局所破綻の**最大の原因はカメラがほぼ
静止していること**と判明した (0828 f300-312 は撮影者が電話中でフレーム間差分
2.9〜9.9、健全区間は19〜60)。並進ゼロでは三角測量が原理的に成立せず、
その区間を含むチャンク接続が島全体のフィットを壊す。

検出は本番パイプライン (gpu_prepro/imu_pdr_trajectory の idle_spans.json) と
同じ `imu_traj.idle.detect_idle_spans` を使う。判定はZUPTと同じ
「1秒窓の角速度・加速度 rolling std が両方とも閾値未満」。
フレーム変換には imu/timeline.py の実測フレーム時刻を渡す。

依存はnumpyのみ (torch不要) なのでホストの img_align env で動く。

使い方:
  IMG_SESSION=0828 python -m img_align.imu.idle
  IMG_SESSION=0828 python -m img_align.imu.idle --min-duration-sec 1.0 --gyr 0.08
出力: results/<s>/imu/idle_spans.json
"""
import json
import sys
from pathlib import Path

import numpy as np

from img_align.config import IMU_DIR
from img_align.imu.timeline import frame_times, n_frames

# ホスト / コンテナ 両方のパスを試す
IMU_TRAJ_PATHS = (
    "/mnt/data/git/polish-pipeline/submodules/imu_experiments_code",
    "/polish-pipeline/submodules/imu_experiments_code",
)


def _idle_mod():
    for p in IMU_TRAJ_PATHS:
        if Path(p).exists():
            if p not in sys.path:
                sys.path.insert(0, p)
            from imu_traj.idle import detect_idle_spans
            return detect_idle_spans
    raise RuntimeError(f"imu_traj が見つかりません: {IMU_TRAJ_PATHS}")


def _traj_npz():
    """imu_trajスキーマ (t, gyr, acc) のnpzを用意する"""
    conv = IMU_DIR / "imu_raw_traj.npz"
    if not conv.exists():
        raw = np.load(IMU_DIR / "imu_raw.npz")
        np.savez_compressed(conv, t=raw["t"], gyr=raw["gyro"], acc=raw["accel"],
                            scaled_from_g=np.array(False))
    return conv


def detect(out_name="idle_spans.json", **kw):
    """imu_raw.npz → idle_spans.json。kwはdetect_idle_spansへ透過"""
    n = n_frames()
    t, src = frame_times(n)
    kw.setdefault("frame_times", t)
    kw.setdefault("n_frames", n)
    kw.setdefault("frame_dt", float(np.median(np.diff(t))))
    res = _idle_mod()(imu_npz=_traj_npz(), out_json=IMU_DIR / out_name, **kw)
    m = res["meta"]
    print(f"[idle] frame times={src} spans={m['n_spans']} "
          f"静止フレーム {m['idle_frame_count']}/{n} "
          f"({m['idle_frame_frac']:.1%})")
    for s in res["spans"]:
        print(f"  frames {s['frame_start']:5d}-{s['frame_end']:5d} "
              f"({s['duration_sec']:.1f}s)")
    return res


def spans(out_name="idle_spans.json"):
    """保存済み静止区間を [(frame_start, frame_end)] で返す (無ければ空)"""
    p = IMU_DIR / out_name
    if not p.exists():
        return []
    return [(int(s["frame_start"]), int(s["frame_end"]))
            for s in json.loads(p.read_text())["spans"]]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-duration-sec", type=float, default=1.5)
    ap.add_argument("--window-sec", type=float, default=1.0)
    ap.add_argument("--gyr", type=float, default=0.05, help="rad/s")
    ap.add_argument("--acc", type=float, default=0.3, help="m/s^2")
    ap.add_argument("--out", default="idle_spans.json")
    args = ap.parse_args()
    detect(args.out, min_duration_sec=args.min_duration_sec,
           window_sec=args.window_sec, gyr_thresh=args.gyr,
           acc_thresh=args.acc)
