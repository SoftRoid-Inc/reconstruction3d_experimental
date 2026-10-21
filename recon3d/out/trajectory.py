"""チェーン出力 (global_cameras.npz) を pipeline 非依存の素のフレーム姿勢に直す

pipeline (別リポジトリ) 側は recon3d の npz レイアウト (`names`/`positions`/
`poses` の3配列、chunk単位の補正の有無など) を知らずに済むよう、ここで
frame番号・位置・回転だけの単純な値に変換して渡す。pipeline 側の contract
型には依存しない (recon3d は pipeline を知らない一方通行)。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from recon3d.types import Mat3


@dataclass(frozen=True)
class FramePose:
    frame: int
    pos: tuple[float, float, float]
    quat: tuple[float, float, float, float]   # (x, y, z, w)


def _rot_to_quat_xyzw(R: Mat3) -> tuple[float, float, float, float]:
    """回転行列 → quaternion (x, y, z, w)

    `geometry.quat_to_rot` の逆変換。あちらは w が先頭 (COLMAP/PanoVGGT 規約)
    だが、こちらが返す `FramePose.quat` は pipeline 側の規約に合わせて
    x,y,z,w 順 — 変換してから並べ替える。scipy.spatial.transform.Rotation は
    使わず、既存依存の numpy だけで書く (トレースによる標準的な変換)。
    """
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return (float(x), float(y), float(z), float(w))


def frame_poses(cameras_npz: Path) -> list[FramePose]:
    """連結済み島の `global_cameras.npz` → フレーム昇順の `FramePose` 列

    位置は `positions` をそのまま使う (`poses[:, :3, 3]` ではない) —
    `chain_and_save` が複数chunkにまたがる同一フレームの重心などで補正した
    値が `positions` に入っており、`poses` の並進はその補正を経ていない
    ことがあるため。
    """
    if not cameras_npz.exists():
        raise SystemExit(f"カメラ軌跡が無い: {cameras_npz}")
    d = np.load(cameras_npz)
    names, positions, poses = d["names"], d["positions"], d["poses"]
    out = [
        FramePose(frame=int(str(n).replace("frame_", "")),
                  pos=(float(p[0]), float(p[1]), float(p[2])),
                  quat=_rot_to_quat_xyzw(po[:3, :3]))
        for n, p, po in zip(names, positions, poses)
    ]
    return sorted(out, key=lambda fp: fp.frame)
