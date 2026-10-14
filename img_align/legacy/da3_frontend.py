"""DA3前段: キューブマップ4面×チャンクをDA3に一括入力し、PanoVGGT互換形式で保存

da3コンテナ内で実行する (依存: torch/numpy/PIL のみ。scipy等は使わない):
  docker compose exec da3 python -m img_align.legacy.da3_frontend \
      --model depth-anything/DA3-BASE --out chunks_da3base

出力 (チャンクごと results/da3/<out>/chunk_XX/):
  points.ply     — チャンクworldの色付き点群
  cameras.npz    — poses(フレーム単位cam2world 4x4, y-up規約に変換), positions,
                   names, view_spread(リグ品質: 同一フレーム4視点の中心のばらつき)
  per_frame.npz  — フレームごとの間引き点群 (IRLS連結用)
"""
import argparse
import json
import math

import numpy as np

from img_align.core.io_ply import write_ply
from img_align.config import RESULTS_DIR

DA3_DIR = RESULTS_DIR / "da3"
CROPS_DIR = DA3_DIR / "crops"
YAWS = [0, 90, 180, 270]
CHUNK_SIZE, CHUNK_STRIDE = 16, 12
FRAMES = list(range(0, 454, 3))
SUBSAMPLE_PLY = 6
SUBSAMPLE_FRAME = 12
CONF_PCT = 30  # 信頼度下位30%の点を捨てる (confidenceが出る場合)

# OpenCV(y down, z fwd) → 既存規約(y up, z fwd) の変換
FLIP = np.diag([1.0, -1.0, -1.0])


def chunk_windows(n):
    starts = list(range(0, max(n - CHUNK_SIZE, 0) + 1, CHUNK_STRIDE))
    if starts and starts[-1] + CHUNK_SIZE < n:
        starts.append(n - CHUNK_SIZE)
    return [(s, min(s + CHUNK_SIZE, n)) for s in starts] or [(0, n)]


def yaw_rot(yaw_deg):
    """リグ内回転: yaw0視点系→yawθ視点系 (y-up規約, ヨーはy軸回り)"""
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])




def run_chunk(model, names, out_dir):
    paths = [str(CROPS_DIR / f"{n}_y{y:03d}.jpg") for n in names for y in YAWS]
    pred = model.inference(paths)
    depth = np.asarray(pred.depth)                    # (N,H,W)
    ext = np.asarray(pred.extrinsics)                 # (N,3,4) w2c opencv
    K = np.asarray(pred.intrinsics)                   # (N,3,3)
    imgs = np.asarray(pred.processed_images)          # (N,H,W,3)
    conf = getattr(pred, "conf", None)
    n_views = len(paths)
    h, w = depth.shape[1:]

    # w2c → c2w
    R_c2w = ext[:, :3, :3].transpose(0, 2, 1)
    centers = -np.einsum("nij,nj->ni", R_c2w, ext[:, :3, 3])

    all_xyz, all_rgb, per_frame = [], [], {}
    poses, positions, spreads = [], [], []
    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    for fi, name in enumerate(names):
        vidx = [fi * len(YAWS) + k for k in range(len(YAWS))]
        c = centers[vidx]
        positions.append(c.mean(axis=0))
        spreads.append(float(np.linalg.norm(c - c.mean(axis=0), axis=1).max()))
        # フレーム姿勢: yaw0視点のc2wをy-up規約へ (R' = R_c2w @ FLIP)
        pose = np.eye(4)
        pose[:3, :3] = R_c2w[vidx[0]] @ FLIP
        pose[:3, 3] = positions[-1]
        poses.append(pose)

        pts_f = []
        for k in vidx:
            z = depth[k]
            m = np.isfinite(z) & (z > 1e-4)
            if conf is not None:
                th = np.percentile(np.asarray(conf[k])[m], CONF_PCT)
                m &= np.asarray(conf[k]) >= th
            fx, fy = K[k][0, 0], K[k][1, 1]
            cx, cy = K[k][0, 2], K[k][1, 2]
            x = (uu[m] - cx) / fx * z[m]
            y = (vv[m] - cy) / fy * z[m]
            pc = np.stack([x, y, z[m]], axis=1)
            pw = pc @ R_c2w[k].T + centers[k]
            col = imgs[k][m]
            if col.max() <= 1.0:
                col = (col * 255)
            all_xyz.append(pw[::SUBSAMPLE_PLY])
            all_rgb.append(col[::SUBSAMPLE_PLY].astype(np.uint8))
            pts_f.append(pw[::SUBSAMPLE_FRAME])
        per_frame[name] = np.concatenate(pts_f).astype(np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)
    xyz = np.concatenate(all_xyz)
    # world座標もy-up規約へ (視点変換と整合: world' = world @ FLIP)
    xyz = xyz @ FLIP
    for name in per_frame:
        per_frame[name] = per_frame[name] @ FLIP
    positions = np.array(positions) @ FLIP
    for p in poses:
        p[:3, :3] = FLIP @ p[:3, :3]
        p[:3, 3] = FLIP @ p[:3, 3]

    write_ply(out_dir / "points.ply", xyz, np.concatenate(all_rgb))
    np.savez_compressed(out_dir / "cameras.npz",
                        poses=np.stack(poses), positions=positions,
                        names=np.array(names),
                        view_spread=np.array(spreads))
    np.savez_compressed(out_dir / "per_frame.npz", **per_frame)
    print(f"  {out_dir.name}: {len(xyz):,} pts, "
          f"view_spread median {np.median(spreads):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="depth-anything/DA3-BASE")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunks", default=None, help="例 '0:3' (省略時は全チャンク)")
    args = ap.parse_args()

    import torch
    from depth_anything_3.api import DepthAnything3
    model = DepthAnything3.from_pretrained(args.model)
    model = model.to(device=torch.device("cuda"))

    windows = chunk_windows(len(FRAMES))
    sel = range(len(windows))
    if args.chunks:
        a, b = (int(v) for v in args.chunks.split(":"))
        sel = range(a, b)
    for k in sel:
        a, b = windows[k]
        names = [f"frame_{FRAMES[i]:06d}" for i in range(a, b)]
        out_dir = DA3_DIR / args.out / f"chunk_{k:02d}"
        if (out_dir / "cameras.npz").exists():
            continue
        print(f"chunk {k:02d} ({names[0]}..{names[-1]})")
        run_chunk(model, names, out_dir)
    print("done")


if __name__ == "__main__":
    main()
