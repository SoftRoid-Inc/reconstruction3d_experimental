"""`imu_segments` の録画端フィルタが、録画長を meta から取ること

かつては `fdt = max(t_end) / n_frames` と逆算していたため、端判定の
`t_start > n_frames * fdt - IMU_EDGE_SEC` が恒等変形で
`t_start > max(t_end) - IMU_EDGE_SEC` に潰れていた。つまり**最終イベントは
長さが IMU_EDGE_SEC 未満なら必ず捨てられる**。検出イベントの 64% が 12 秒未満で、
実データ 487 件の 18% がこれで消えていた。人手入力では区間が 1 つの撮影の
切断点が丸ごと消え、「IMU 無し」と区別がつかなくなる。
"""
import json

import pytest

from recon3d.core.chunking import IMU_EDGE_SEC, IMU_STAIR_PAD, imu_segments


class _Paths:
    """`imu_segments` が触るのは `paths.imu` だけ。"""

    def __init__(self, imu_dir):
        self.imu = imu_dir


def _write(tmp_path, events, *, frame_dt=None, n_frames=None):
    imu = tmp_path / "imu"
    imu.mkdir(exist_ok=True)
    meta = {}
    if frame_dt is not None:
        meta["frame_dt"] = frame_dt
    if n_frames is not None:
        meta["n_frames"] = n_frames
    (imu / "transitions.json").write_text(json.dumps({"events": events, "meta": meta}))
    return _Paths(imu)


def _event(a, b, frame_dt):
    return {"t_start": a * frame_dt, "t_end": b * frame_dt,
            "direction": "up", "confidence": 1.0, "frame_start": a, "frame_end": b}


def test_録画の真ん中にある単独イベントを捨てない(tmp_path):
    """区間 1 つだけの撮影。人手入力の典型で、旧実装では必ず消えていた。"""
    dt, n = 0.5, 900
    paths = _write(tmp_path, [_event(358, 367, dt)], frame_dt=dt, n_frames=n)
    segs, cuts = imu_segments(paths, n, use_idle=False)
    assert cuts == [(358 - IMU_STAIR_PAD, 367 + IMU_STAIR_PAD)]
    assert [(a, b) for _, a, b in segs] == [(0, 353), (372, n - 1)]


def test_本当に録画端のイベントは捨てる(tmp_path):
    """端フィルタ自体は生きている（カメラ上げ下ろしのアーチファクト対策）。"""
    dt, n = 0.5, 900
    tail = int((n * dt - IMU_EDGE_SEC / 2) / dt)        # 末尾 6 秒の位置
    paths = _write(tmp_path, [_event(2, 10, dt), _event(tail, tail + 5, dt)],
                   frame_dt=dt, n_frames=n)
    segs, cuts = imu_segments(paths, n, use_idle=False)
    assert cuts == []                                   # 両端とも落ちる
    assert [(a, b) for _, a, b in segs] == [(0, n - 1)]


def test_meta_が無い旧ファイルは従来どおり逆算する(tmp_path):
    """既存の成果物で挙動を変えないこと。逆算では単独イベントは落ちる。"""
    dt, n = 0.5, 900
    paths = _write(tmp_path, [_event(358, 367, dt)])    # meta 無し
    segs, cuts = imu_segments(paths, n, use_idle=False)
    assert cuts == []
    assert [(a, b) for _, a, b in segs] == [(0, n - 1)]


@pytest.mark.parametrize("n_events", [1, 2, 3])
def test_イベント数によらず中央のイベントは残る(tmp_path, n_events):
    """旧実装は「最後の 1 件」が落ちたので、件数を変えると結果が変わっていた。"""
    dt, n = 0.5, 1200
    ev = [_event(200 + i * 300, 209 + i * 300, dt) for i in range(n_events)]
    paths = _write(tmp_path, ev, frame_dt=dt, n_frames=n)
    _, cuts = imu_segments(paths, n, use_idle=False)
    assert len(cuts) == n_events
