"""フレーム毎の正確な撮影時刻を .insv テレメトリから抽出する (imuコンテナ内で実行)

Insta360のタイムラプスは公称間隔(500ms)を守らず実測0.50〜0.80秒で変動するため、
等間隔仮定 (frame/2.0 や duration_s/N) は時間比例でずれる。
.insv の `Default/Timestamps` にフレーム毎の撮影時刻が入っており、これはIMUと
同一タイムライン(先頭フレーム=0の相対秒)。本番パイプラインの
gpu_prepro/imu_pdr_trajectory と同じ情報源を使う。

frame_number は .insv のフレーム索引と1:1・0始まり (全セッションで画像枚数==nb_framesを確認)。

telemetry_parser はホストに無くimuコンテナにのみ入っているため、抽出はコンテナ内で行い、
結果の npz だけをホスト側 (timeline.py) が読む。

使い方 (ホストから):
  docker compose exec -e IMG_SESSION=0616 imu python -m img_align.imu.frame_times
出力: results/<s>/imu/frame_times.npz {t, source, insv, n_frames}
"""
import json
import sys
from pathlib import Path

import numpy as np

from img_align.config import FRAMES_DIR, IMU_DIR, PROP, SESSION

IMU_TRAJ = "/polish-pipeline/submodules/imu_experiments_code"


def _frames_mod():
    if IMU_TRAJ not in sys.path:
        sys.path.insert(0, IMU_TRAJ)
    from imu_traj import frames
    return frames


def find_insv():
    """meta.json の記録 (旧コンテナパス) → 物件ディレクトリの glob の順で探す"""
    meta_p = IMU_DIR / "meta.json"
    if meta_p.exists():
        for f in json.loads(meta_p.read_text()).get("files", []):
            name = Path(f.get("file", "")).name
            if name:
                hits = sorted(PROP.root.rglob(name))
                if hits:
                    return hits[0]
    for pat in (f"{SESSION}/VID_*.insv", f"video_data/{SESSION}/VID_*.insv",
                "VID_*.insv"):
        hits = sorted(PROP.root.glob(pat))
        if hits:
            return hits[0]
    return None


def extract(out_name="frame_times.npz"):
    n = len(sorted(FRAMES_DIR.glob("frame_number_*.jpg")))
    insv = find_insv()
    meta = json.loads((IMU_DIR / "meta.json").read_text())
    t = None
    if insv is not None:
        t = _frames_mod().load_frame_times(insv, n_frames=n)
    if t is None:
        dt = meta["duration_s"] / n
        t = np.arange(n) * dt
        source = "uniform"
        print(f"[frame_times] テレメトリ取得不可 -> 等間隔 dt={dt:.6f}s")
    else:
        source = "telemetry"
        d = np.diff(t)
        print(f"[frame_times] telemetry: n={len(t)} 末尾={t[-1]:.2f}s "
              f"(IMU duration {meta['duration_s']:.2f}s) "
              f"間隔 min/med/max = {d.min():.3f}/{np.median(d):.3f}/{d.max():.3f}s")
    out = IMU_DIR / out_name
    np.savez(out, t=np.asarray(t, float), source=source,
             insv=insv.name if insv else "", n_frames=n)
    print(f"-> {out} ({source}, {n} frames)")
    return t, source


if __name__ == "__main__":
    extract()
