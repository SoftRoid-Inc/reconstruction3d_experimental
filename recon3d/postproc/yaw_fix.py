"""前段修正: 方位ロック拘束によるチャンク内フレーム回転の頑健平均化

Insta360のFlowState方位ロックにより、全フレームの向きは世界基準で一定
(実測: 健全チャンクでヨーstd 0.4〜1.6°)。よってチャンク内でヨーが
頑健中央値から大きく外れるフレームはPanoVGGTのポーズ回転エラーと断定でき、
カメラ中心まわりの鉛直軸回転で中央値に戻せる(真値=相対回転0)。

処理: cameras.npz のpose回転を修正し、per-frameローカル点群を修正後poseで
world合成して points.ply を作り直す。修正不要チャンクはsymlinkで共有。

使い方: python -m recon3d.postproc.yaw_fix [--chunks 0,1,2,3,4]
出力: results/pcd/chunks_yawfix/chunk_XX/
"""
import argparse
import json
import math

import numpy as np

from recon3d.core.io_ply import write_ply
from recon3d.core.geometry import (gravity_from_cameras,
                                     robust_yaw_median, rot_y,
                                     rotation_to_y)
from recon3d.core.io_ply import read_ply_xyzrgb

CHUNKS_SUB = "chunks_masked"   # --src で差し替え可
OUT_SUB = "chunks_yawfix"      # --out で差し替え可
FIX_DEG = 3.0     # 中央値からの逸脱がこれ超のフレームを修正
DROP_DEG = 30.0   # これ超はポーズ全壊とみなし位置も信頼しない (実測: 逸脱62/82°の
                  # フレームは位置も1.5/2.6mずれる。≦16°なら位置は正常)




def chunk_yaw(poses):
    """重力整列後のヨー((row,col)=(−z,x)規約) と 重力回転Rg"""
    _, up = gravity_from_cameras(poses)
    Rg = rotation_to_y(up / np.linalg.norm(up))
    fwd = poses[:, :3, 2] @ Rg.T
    phi = np.arctan2(-fwd[:, 2], fwd[:, 0])
    return phi, Rg


def fix_chunk(name, src_root, out_root):
    """1チャンクを修正して out_root/name に書く。修正不要ならFalse"""
    src = src_root / name
    cams = np.load(src / "cameras.npz")
    poses = cams["poses"].copy()
    names = [str(n) for n in cams["names"]]

    phi, Rg = chunk_yaw(poses)
    med, dev = robust_yaw_median(phi)
    bad = np.abs(dev) > FIX_DEG
    print(f"{name}: yaw med {math.degrees(med):6.1f}deg, "
          f"core std {dev[~bad].std():.2f}deg, fix {int(bad.sum())}/{len(dev)}")
    untrusted = {names[i]: round(float(dev[i]), 1)
                 for i in np.nonzero(np.abs(dev) > DROP_DEG)[0]}
    if not bad.any():
        return False, untrusted
    dst = out_root / name
    dst.mkdir(parents=True, exist_ok=True)
    # (チャンク,フレーム)単位の不信リスト: 同じフレームでも別チャンクでは
    # 健全なことがある (例: 186/189はchunk_04で全壊だがchunk_05では健全)
    (dst / "untrusted.json").write_text(json.dumps(untrusted, indent=1))

    # 鉛直軸(world up)まわりの補正回転をカメラ中心不動で適用
    for i in np.nonzero(bad)[0]:
        delta = math.atan2(math.sin(med - phi[i]), math.cos(med - phi[i]))
        R_fix = Rg.T @ rot_y(delta) @ Rg
        poses[i, :3, :3] = R_fix @ poses[i, :3, :3]
        print(f"  {names[i]}: dev {dev[i]:+7.1f}deg -> corrected")
    phi2, _ = chunk_yaw(poses)
    _, dev2 = robust_yaw_median(phi2)
    print(f"  after fix: yaw std {dev2.std():.2f}deg (max |dev| "
          f"{np.abs(dev2).max():.2f})")

    # per-frameローカル点群を修正後poseでworld合成
    # (ポーズ全壊フレームは位置も壊れており、360°スキャンがフロア全体の
    #  「ずれた複製」ゴーストを描くため点群ごと除外する)
    pf_dir = src / "panovggt_out" / "pointclouds" / "per_frame"
    all_pts, all_rgb = [], []
    for i, n in enumerate(names):
        if n in untrusted:
            print(f"  {n}: points dropped (pose untrusted)")
            continue
        xyz, rgb = read_ply_xyzrgb(pf_dir / f"{n}.ply")
        w = xyz @ poses[i, :3, :3].T + poses[i, :3, 3]
        all_pts.append(w.astype(np.float32))
        all_rgb.append(rgb)
    pts = np.concatenate(all_pts)
    n_orig, _ = read_ply_xyzrgb(src / "points.ply")
    print(f"  rebuilt points: {len(pts):,} (orig merged {len(n_orig):,})")
    write_ply(dst / "points.ply", pts, np.concatenate(all_rgb))
    np.savez_compressed(dst / "cameras.npz", poses=poses,
                        positions=poses[:, :3, 3], names=cams["names"])
    link = dst / "panovggt_out"
    if not link.exists():
        link.symlink_to(src / "panovggt_out")
    return True, untrusted


def main(paths, argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="0,1,2,3,4",
                    help="'0,1,2' か 'all' (srcの全chunk_*)")
    ap.add_argument("--src", default=None, help="入力チャンク群 (results/pcd 相対)")
    ap.add_argument("--out", default=None, help="出力先 (results/pcd 相対)")
    args = ap.parse_args(argv)
    src_root = paths.pcd / (args.src or CHUNKS_SUB)
    out_root = paths.pcd / (args.out or OUT_SUB)
    out_root.mkdir(parents=True, exist_ok=True)
    if args.chunks == "all":
        ids = [int(p.name.split("_")[1]) for p in sorted(src_root.glob("chunk_*"))]
    else:
        ids = [int(v) for v in args.chunks.split(",")]
    bad_path = out_root / "pose_untrusted.json"
    all_bad = json.loads(bad_path.read_text()) if bad_path.exists() else {}
    for k in ids:
        name = f"chunk_{k:02d}"
        fixed, untrusted = fix_chunk(name, src_root, out_root)
        all_bad.update(untrusted)
        for n, d in untrusted.items():
            print(f"  {n}: |dev| {abs(d):.0f}deg > {DROP_DEG:.0f} "
                  f"-> pose untrusted (位置も除外推奨)")
        if not fixed:
            link = out_root / name
            if not link.exists():
                link.symlink_to(src_root / name)
            print(f"  -> symlink (no fix needed)")
    bad_path.write_text(json.dumps(all_bad, indent=1))
    print(f"-> {out_root} (untrusted: {sorted(all_bad)})")

