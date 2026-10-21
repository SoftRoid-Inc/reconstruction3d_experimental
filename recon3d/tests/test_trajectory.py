"""`recon3d.out.trajectory` の frame_poses が npz を正しく素の値へ直すこと"""
import numpy as np
import pytest

from recon3d.out.trajectory import frame_poses


def test_frame_poses_reads_names_positions_poses(tmp_path):
    npz = tmp_path / "global_cameras.npz"
    np.savez(npz,
             names=np.array(["frame_000002", "frame_000000"]),
             positions=np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]),
             poses=np.stack([np.eye(4), np.eye(4)]))
    out = frame_poses(npz)
    assert [p.frame for p in out] == [0, 2]          # 昇順に並べ替える
    assert out[1].pos == (1.0, 2.0, 3.0)
    assert out[0].quat == (0.0, 0.0, 0.0, 1.0)       # 単位回転・(x,y,z,w)順


def test_frame_poses_rejects_missing_file(tmp_path):
    with pytest.raises(SystemExit):
        frame_poses(tmp_path / "nope.npz")
