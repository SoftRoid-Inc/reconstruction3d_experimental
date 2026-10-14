"""Insta360 .insv からIMU(ジャイロ+加速度)を抽出して imu_raw.npz に保存する

実行 (imuコンテナ):
  docker compose exec imu python -m img_align.imu.extract \
      --insv /code2/reconstruction_3d/28-AAAAA/video_data/0828/*.insv

出力: results/imu/imu_raw.npz (t[s], gyro[rad/s](N,3), accel[m/s^2](N,3)) + meta.json
受け入れテスト: サンプルレート / 収録長 / 静止時の重力ノルム / ジャイロバイアス
"""
import argparse
import glob
import json
import math

import numpy as np

from img_align.config import IMU_DIR



def parse_one(path):
    import telemetry_parser
    tp = telemetry_parser.Parser(path)
    meta = {"camera": getattr(tp, "camera", None),
            "model": getattr(tp, "model", None), "file": path}
    imu = tp.normalized_imu()
    # 実装差に備えて最初のサンプルの形を確認
    s0 = imu[0]
    keys = list(s0.keys()) if isinstance(s0, dict) else None
    print(f"{path}: {len(imu)} samples, first sample keys/repr: {keys or s0}")
    t = np.array([s["timestamp_ms"] for s in imu], dtype=np.float64) / 1000.0
    gyro = np.array([s["gyro"] for s in imu], dtype=np.float64)
    accel = np.array([s["accl"] for s in imu], dtype=np.float64)
    return t, gyro, accel, meta


def autoscale(gyro, accel):
    """単位の自動判定: gyroはdeg/s→rad/s、accelはg→m/s^2 の可能性に対応"""
    notes = {}
    # 静止に近い窓 (ジャイロノルム最小の2秒) を探す
    n = len(gyro)
    rate = n / 1.0  # 仮
    gnorm = np.linalg.norm(gyro, axis=1)
    win = max(int(len(gnorm) * 2.0 / max((gnorm.size), 1)), 200)
    # 単純に移動平均最小の窓
    k = np.convolve(gnorm, np.ones(win) / win, mode="valid")
    i0 = int(k.argmin())
    sl = slice(i0, i0 + win)
    anorm = float(np.median(np.linalg.norm(accel[sl], axis=1)))
    if 0.5 < anorm < 2.0:
        accel = accel * 9.80665
        notes["accel_unit"] = "g -> m/s^2 (x9.80665)"
    else:
        notes["accel_unit"] = f"m/s^2 (norm={anorm:.2f})"
    # gyro: 歩行撮影でrad/sなら典型ピーク<10、deg/sなら数百
    gp = float(np.percentile(np.abs(gyro), 99))
    if gp > 50:
        gyro = np.deg2rad(gyro)
        notes["gyro_unit"] = "deg/s -> rad/s"
    else:
        notes["gyro_unit"] = f"rad/s (p99={gp:.2f})"
    notes["still_window_idx"] = [int(i0), int(i0 + win)]
    return gyro, accel, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--insv", nargs="+", required=True)
    args = ap.parse_args()
    paths = sorted(sum([glob.glob(p) for p in args.insv], []))
    if not paths:
        raise SystemExit("insvが見つからない")

    ts, gs, as_, meta = [], [], [], []
    t_end = 0.0
    for p in paths:
        t, g, a, m = parse_one(p)
        ts.append(t + t_end)  # セグメント連結 (連番録画対応)
        gs.append(g)
        as_.append(a)
        meta.append(m)
        t_end = ts[-1][-1]
    t = np.concatenate(ts)
    gyro = np.concatenate(gs)
    accel = np.concatenate(as_)
    gyro, accel, notes = autoscale(gyro, accel)

    IMU_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(IMU_DIR / "imu_raw.npz", t=t, gyro=gyro, accel=accel)
    rate = len(t) / (t[-1] - t[0])
    i0, i1 = notes["still_window_idx"]
    g_norm = float(np.median(np.linalg.norm(accel[i0:i1], axis=1)))
    bias = np.median(gyro[i0:i1], axis=0)
    report = {
        "files": meta, "notes": notes,
        "n_samples": int(len(t)),
        "rate_hz": round(rate, 1),
        "duration_s": round(float(t[-1] - t[0]), 2),
        "gravity_norm_still": round(g_norm, 3),
        "gyro_bias_still_rad_s": [round(float(b), 5) for b in bias],
    }
    (IMU_DIR / "meta.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))

    ok = (50 <= rate <= 2000 and abs(g_norm - 9.81) < 0.5
          and all(abs(b) < 0.05 for b in bias))
    print("受け入れテスト:", "PASS" if ok else "CHECK NEEDED")


if __name__ == "__main__":
    main()
