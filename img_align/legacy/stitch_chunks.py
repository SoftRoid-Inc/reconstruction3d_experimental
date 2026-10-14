"""全フレームをチャンク分割してPanoVGGT処理し、共有フレームで縫合する

低重なりフレームの一括推論では姿勢が破綻するため、連続フレームのチャンク
(重なりあり)ごとに推論し、隣接チャンク間で共有するカメラ中心の対応から
Umeyama相似変換(スケール+回転+並進)を推定して chunk0 座標系へ統合する。

出力: results/pcd/full/points.ply + cameras.npz (以降は density_map --tag full,
align_floorplan --tag full で処理)
"""
import argparse
import json

import numpy as np

from img_align.core.geometry import rotation_to_y
from img_align.core.io_ply import read_ply_xyzrgb
from img_align.core.panovggt import PCD_DIR, local_floor_height, run_panovggt, select_entries

from img_align.core.chunking import chunk_windows  # noqa: E402

FULL_DIR = PCD_DIR / "full"
CHUNK_SIZE = 16
CHUNK_STRIDE = 12          # 重なり = CHUNK_SIZE - CHUNK_STRIDE = 4フレーム
POINT_SUBSAMPLE = 3        # 統合時の点の間引き


def gravity_rotation(poses):
    """チャンク内カメラup(cam2world規約: R[:,1])の中央値を+yへ回す回転"""
    ups = poses[:, :3, 1]
    u = ups / np.linalg.norm(ups, axis=1, keepdims=True)
    med = np.median(u, axis=0)
    return rotation_to_y(med / np.linalg.norm(med))


def umeyama_yaw(src, dst):
    """重力整列済みの点列間で y軸回り回転のみの相似変換を推定する

    共有カメラは歩行軌跡上でほぼ一直線に並ぶため、3D Umeyamaでは進行方向
    まわりの回転が不定になる。重力で roll/pitch を固定しヨーだけ解く。
    dst ≈ s * R @ src + t
    """
    s2, d2 = src[:, [0, 2]], dst[:, [0, 2]]
    mu_s, mu_d = s2.mean(axis=0), d2.mean(axis=0)
    sc, dc = s2 - mu_s, d2 - mu_d
    cov = dc.T @ sc / len(sc)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[1, 1] = -1
    R2 = U @ S @ Vt
    var = (sc ** 2).sum() / len(sc)
    s = float(np.trace(np.diag(D) @ S) / var)
    t2 = mu_d - s * R2 @ mu_s
    ty = float((dst[:, 1] - s * src[:, 1]).mean())
    R = np.eye(3)
    R[0, 0], R[0, 2] = R2[0, 0], R2[0, 1]
    R[2, 0], R[2, 2] = R2[1, 0], R2[1, 1]
    t = np.array([t2[0], ty, t2[1]])
    return s, R, t




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="0:454:3",
                    help="'start:end:step' (既定: 全454フレームを3刻み)")
    ap.add_argument("--chunk-range", default=None,
                    help="'a:b' 使用するチャンク番号範囲 (階段区間で分割した島を作る用)")
    ap.add_argument("--out", default="full",
                    help="出力サブディレクトリ名 (results/pcd/<out>/)")
    ap.add_argument("--use-mask", action="store_true",
                    help="personマスク(seg由来)を使って撮影者の点を除外する")
    ap.add_argument("--chunks-dir", default="chunks",
                    help="チャンク出力ディレクトリ名 (results/pcd/<chunks-dir>/)")
    args = ap.parse_args()
    out_dir = PCD_DIR / args.out

    entries = select_entries(args.frames)
    windows = chunk_windows(len(entries))
    print(f"{len(entries)} frames -> {len(windows)} chunks")

    # 1) 各チャンクを推論 (既に出力があればスキップ)
    chunk_dirs = []
    for k, (a, b) in enumerate(windows):
        cdir = PCD_DIR / args.chunks_dir / f"chunk_{k:02d}"
        chunk_dirs.append(cdir)
        if (cdir / "cameras.npz").exists():
            continue
        print(f"chunk {k}: frames[{a}:{b}]")
        cdir.mkdir(parents=True, exist_ok=True)
        run_panovggt(entries[a:b], cdir, use_mask=args.use_mask)

    if args.chunk_range:
        a, b = (int(v) for v in args.chunk_range.split(":"))
        chunk_dirs = chunk_dirs[a:b]
        print(f"using chunks [{a}:{b}] -> {out_dir}")

    # 2) 各チャンクを重力整列してから、隣接チャンクの共有カメラで縫合
    #    (chunk0の重力整列系をグローバルとし、ヨー回転のみの相似変換で接続)
    cams0 = np.load(chunk_dirs[0] / "cameras.npz")
    transforms = [(1.0, gravity_rotation(cams0["poses"]), np.zeros(3))]
    stitch_log = []
    for k in range(1, len(chunk_dirs)):
        prev = np.load(chunk_dirs[k - 1] / "cameras.npz")
        curr = np.load(chunk_dirs[k] / "cameras.npz")
        prev_names = [str(n) for n in prev["names"]]
        curr_names = [str(n) for n in curr["names"]]
        shared = [n for n in curr_names if n in prev_names]
        if len(shared) < 2:
            raise RuntimeError(f"chunk {k}: 共有フレーム不足 ({len(shared)})")
        src = np.stack([curr["positions"][curr_names.index(n)] for n in shared])
        dst = np.stack([prev["positions"][prev_names.index(n)] for n in shared])
        # prev側はグローバル系へ、curr側は自前の重力整列系へ
        sp, Rp, tp = transforms[k - 1]
        dst_g = (sp * (Rp @ dst.T)).T + tp
        Gk = gravity_rotation(curr["poses"])
        src_g = (Gk @ src.T).T
        s, Ry, t = umeyama_yaw(src_g, dst_g)
        R = Ry @ Gk
        res = np.linalg.norm((s * (R @ src.T)).T + t - dst_g, axis=1)
        transforms.append((s, R, t))
        stitch_log.append({"chunk": k, "shared": len(shared),
                           "scale": round(s, 4),
                           "residual_mean": round(float(res.mean()), 4)})
        print(f"chunk {k}: shared={len(shared)} scale={s:.3f} "
              f"resid={res.mean():.3f}")

    # 3) 点群・カメラをグローバル座標へ統合
    all_pts, all_names, all_pos, all_poses, all_fh = [], [], [], [], []
    seen = set()
    for k, cdir in enumerate(chunk_dirs):
        s, R, t = transforms[k]
        xyz, _ = read_ply_xyzrgb(cdir / "points.ply")
        xyz = xyz[::POINT_SUBSAMPLE]
        all_pts.append((s * (R @ xyz.T)).T + t)
        cams = np.load(cdir / "cameras.npz")
        for name, pos, pose in zip(cams["names"], cams["positions"],
                                   cams["poses"]):
            if str(name) in seen:
                continue
            seen.add(str(name))
            all_names.append(str(name))
            all_pos.append(s * (R @ pos) + t)
            g = np.eye(4)
            g[:3, :3] = R @ pose[:3, :3]
            g[:3, 3] = s * (R @ pose[:3, 3]) + t
            all_poses.append(g)
            # カメラ直下床までの高さ (チャンクスケール補正済み、グローバルunits)
            fh = local_floor_height(
                cdir / "panovggt_out" / "pointclouds" / "per_frame"
                / f"{name}.ply")
            all_fh.append(s * fh if fh is not None else np.nan)

    pts = np.concatenate(all_pts)
    order = np.argsort(all_names)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_ply(out_dir / "points.ply", pts)
    np.savez_compressed(
        out_dir / "cameras.npz",
        poses=np.stack(all_poses)[order],
        positions=np.stack(all_pos)[order],
        names=np.array(all_names)[order],
        floor_heights=np.array(all_fh)[order])
    (out_dir / "stitch_log.json").write_text(json.dumps(stitch_log, indent=2))
    print(f"total {len(pts):,} points, {len(all_names)} cameras -> {out_dir}")


def _write_ply(path, xyz):
    n = len(xyz)
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {n}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "end_header\n")
    data = np.empty(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                              ("r", "u1"), ("g", "u1"), ("b", "u1")])
    data["x"], data["y"], data["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    data["r"] = data["g"] = data["b"] = 128
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())


if __name__ == "__main__":
    main()
