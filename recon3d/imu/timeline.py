"""フレーム番号 ↔ IMU時刻 の唯一の変換窓口

優先順:
  1. results/<s>/imu/frame_times.npz (insvテレメトリ由来の実測時刻。imu/frame_times.py が生成)
  2. meta.json の duration_s / N による等間隔
時刻はIMUと同一タイムライン (先頭フレーム = 0 秒の相対秒)。

従来の `sync.py` (t = frame/2.0 + offset、ヨー相互相関で推定) は0828とexp19176にしか
存在せず誤差も大きい (0828: median_err 30°)。本モジュールはそれに依存しない。

使い方:
  from recon3d.imu.timeline import frame_times, t_of_frame, frame_of_t
  python main.py tool timeline --input /data/x --report     # 等間隔仮定とのずれを表示
"""
import json

import numpy as np


_CACHE = {}


def n_frames(paths):
    return len(paths.frame_indices)


def frame_times(paths, n=None):
    """(時刻配列[秒], source) を返す。sourceは "telemetry" | "uniform" """
    n = n or n_frames(paths)
    key = (str(paths.imu), n)
    if key not in _CACHE:
        p = paths.imu / "frame_times.npz"
        if p.exists():
            d = np.load(p, allow_pickle=False)
            t, src = np.asarray(d["t"], float), str(d["source"])
            if len(t) >= n:
                _CACHE[key] = (t[:n], src)
        if key not in _CACHE:
            meta = json.loads((paths.imu / "meta.json").read_text())
            dt = meta["duration_s"] / n
            _CACHE[key] = (np.arange(n) * dt, "uniform")
    return _CACHE[key]


def t_of_frame(paths, fr):
    """フレーム番号(小数可) → 秒。範囲外は端の間隔で外挿"""
    t, _ = frame_times(paths)
    return float(np.interp(fr, np.arange(len(t)), t))


def frame_of_t(paths, sec):
    """秒 → フレーム番号(小数)。非等間隔に対応"""
    t, _ = frame_times(paths)
    return float(np.interp(sec, t, np.arange(len(t))))


def report(paths):
    """テレメトリ時刻 vs 等間隔仮定のずれを定量化する"""
    n = n_frames(paths)
    meta = json.loads((paths.imu / "meta.json").read_text())
    t, src = frame_times(paths, n)
    dt_uni = meta["duration_s"] / n
    uni = np.arange(n) * dt_uni
    d = np.diff(t)
    diff = t - uni
    med_dt = float(np.median(d))
    out = {
        "n_frames": n, "source": src,
        "imu_duration_s": round(meta["duration_s"], 2),
        "telemetry_end_s": round(float(t[-1]), 2),
        "uniform_dt_s": round(dt_uni, 6),
        "telemetry_dt_min_med_max": [round(float(d.min()), 3),
                                     round(med_dt, 3), round(float(d.max()), 3)],
        "n_irregular_intervals": int((np.abs(d - med_dt) > 0.01).sum()),
        "vs_uniform_max_s": round(float(np.abs(diff).max()), 3),
        "vs_uniform_tail_s": round(float(diff[-1]), 3),
        "vs_uniform_tail_frames": round(float(diff[-1] / med_dt), 2),
        "vs_uniform_rms_s": round(float(np.sqrt((diff ** 2).mean())), 3),
    }
    print(json.dumps(out, ensure_ascii=False))
    return out



def main(paths, argv=None):
    """テレメトリ時刻と等間隔仮定のずれを表示する"""
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    report(paths)
