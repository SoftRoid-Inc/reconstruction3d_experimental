"""COLMAP軌跡プライア: チャンク配置をCOLMAP大域解にSim(3)フィットする

チャンク逐次連結 (chain_chunks) はドリフトが累積する (島B 0.24-0.3mの主因)。
COLMAPは大域BA済みで軌跡が大域整合しているため、各チャンクのカメラ中心を
COLMAPカメラ中心へ個別にSim(3)フィットすれば、チャンク間の連結を介さずに
全チャンクが共通ゲージに載る (=ドリフトなし)。点群はPanoVGGT密度を使う。

出力: chain_chunksと同形式 (frontend_bench/<tag>/global_{cameras.npz,points.ply})
なので、既存の図面アンカー・開口/コーナー精密化・評価がそのまま使える。

使い方: python main.py tool colmap_prior --input /data/x --model <sparse/0> \
          --chunks-root results/pcd/chunks_0312 --islands A:0:10 B:11:19 C:20:22
"""
import argparse
import json
import math
import struct
from pathlib import Path

import numpy as np

from recon3d.core.chaining import load_chunk
from recon3d.core.geometry import sim3_umeyama, apply
from recon3d.core.io_ply import read_ply_xyzrgb


def read_colmap_images_bin(path):
    """images.bin → {image_name: (qvec(4), tvec(3))} (COLMAP: world→cam)"""
    out = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            _img_id = struct.unpack("<I", f.read(4))[0]
            q = struct.unpack("<4d", f.read(32))
            t = struct.unpack("<3d", f.read(24))
            _cam_id = struct.unpack("<I", f.read(4))[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            n2d = struct.unpack("<Q", f.read(8))[0]
            f.seek(n2d * 24, 1)   # (x,y,point3D_id) を読み飛ばす
            out[name.decode()] = (np.array(q), np.array(t))
    return out


def qvec_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def colmap_centers(model_dir):
    """{frame番号: カメラ中心(world)} と {frame番号: R(world→cam)}"""
    imgs = read_colmap_images_bin(Path(model_dir) / "images.bin")
    centers, rots = {}, {}
    for name, (q, t) in imgs.items():
        stem = Path(name).stem
        digits = "".join(ch for ch in stem if ch.isdigit())
        if not digits:
            continue
        fr = int(digits)
        R = qvec_to_R(q)
        centers[fr] = -R.T @ t
        rots[fr] = R
    return centers, rots


def fit_chunk_to_colmap(names, positions, centers, trim=0.2):
    """チャンクカメラ中心→COLMAP中心のSim(3) (残差上位trim割合を捨てて再フィット)"""
    frames = [int(n.replace("frame_", "")) for n in names]
    m = [i for i, fr in enumerate(frames) if fr in centers]
    if len(m) < 4:
        return None, None, len(m)
    src = positions[m]
    dst = np.stack([centers[frames[i]] for i in m])
    T = sim3_umeyama(src, dst)
    r = np.linalg.norm(apply(T, src) - dst, axis=1)
    keep = r <= np.quantile(r, 1 - trim)
    if keep.sum() >= 4:
        T = sim3_umeyama(src[keep], dst[keep])
        r = np.linalg.norm(apply(T, src[keep]) - dst[keep], axis=1)
    return T, float(np.median(r)), len(m)


def main(paths, argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="COLMAP sparseモデル dir")
    ap.add_argument("--chunks-root", required=True)
    ap.add_argument("--islands", nargs="+", required=True,
                    help="'名前:開始チャンク:終了チャンク' の列")
    ap.add_argument("--tag-prefix", default="cm_")
    ap.add_argument("--metric-rescale", action="store_true",
                    help="COLMAPのスケールドリフトをPanoVGGT実尺で補正する: "
                         "チャンク毎fitスケールs_k(=COLMAP単位/m)を物差しに、"
                         "軌跡の各ステップを実寸へ直して再積分 (長尺セッション用)")
    args = ap.parse_args(argv)

    centers, _ = colmap_centers(args.model)
    print(f"COLMAP registered frames: {len(centers)}")
    root = Path(args.chunks_root)
    if not root.is_absolute():
        root = paths.results / args.chunks_root
    cdirs_all = sorted(root.glob("chunk_*"), key=lambda p: int(p.name.split("_")[1]))  # 数値ソート (chunk_100問題)

    # 'name:a:b' の繰り返し。同名は結合 (共有ゲージなので離れたレンジも1島にできる)
    groups = {}
    for spec in args.islands:
        name, a, b = spec.split(":")
        groups.setdefault(name, []).extend(cdirs_all[int(a):int(b)])
    for name, cdirs in groups.items():
        print(f"--- island {name}: {[c.name for c in cdirs]}")
        frame_pos, frame_pose, frame_scale = {}, {}, {}
        chunk_data = []
        for cdir in cdirs:
            names, pos, poses, _, untrusted = load_chunk(cdir)
            T, resid, n_match = fit_chunk_to_colmap(names, pos, centers)
            if T is None:
                print(f"  {cdir.name}: COLMAP対応不足 ({n_match}) -> skip")
                continue
            print(f"  {cdir.name}: fit resid {resid:.3f} (colmap units, "
                  f"{n_match} frames, scale {T[0]:.3f})")
            s, R, t = T
            chunk_data.append((cdir, names, pos, poses, untrusted, T))
            for n, p, po in zip(names, pos, poses):
                if n in untrusted:
                    continue
                fr = int(n.replace("frame_", ""))
                # 位置はCOLMAP中心を直接使う (プライアを最大限信頼)
                pos_g = centers[fr] if fr in centers else apply(T, p[None])[0]
                frame_pos.setdefault(n, []).append(pos_g)
                g = np.eye(4)
                g[:3, :3] = R @ po[:3, :3]
                g[:3, 3] = pos_g
                frame_pose.setdefault(n, []).append(g)
                frame_scale.setdefault(n, []).append(s)

        out = paths.bench / f"{args.tag_prefix}{name}"
        out.mkdir(parents=True, exist_ok=True)
        fnames = sorted(frame_pos)
        positions = np.stack([np.mean(frame_pos[n], axis=0) for n in fnames])
        poses = np.stack([frame_pose[n][0] for n in fnames])

        if args.metric_rescale:
            # 各フレームの局所スケール (COLMAP単位/m) を幾何平均→移動平均で平滑化
            ls = np.array([np.log(frame_scale[n]).mean() for n in fnames])
            k = 9
            pad = np.pad(ls, (k // 2, k // 2), mode="edge")
            ls = np.convolve(pad, np.ones(k) / k, mode="valid")
            s_f = np.exp(ls)
            # ステップを実寸(メートル)へ直して再積分 (形と回転はCOLMAP、長さは実尺)
            cpos = np.zeros_like(positions)
            for i in range(1, len(fnames)):
                s_mid = math.sqrt(s_f[i - 1] * s_f[i])
                cpos[i] = cpos[i - 1] + (positions[i] - positions[i - 1]) / s_mid
            print(f"  metric-rescale: s {s_f.min():.3f}〜{s_f.max():.3f} → "
                  f"軌跡を再積分 ({len(fnames)}f)")
            positions = cpos
            poses = poses.copy()
            poses[:, :3, 3] = positions

        # 点群: 補正後(または素の)カメラ位置へ各チャンクをSim(3)で載せ直す
        all_pts, all_rgb = [], []
        name_to_i = {n: i for i, n in enumerate(fnames)}
        for cdir, names, pos, poses_c, untrusted, T in chunk_data:
            ok = [i for i, n in enumerate(names)
                  if n in name_to_i and n not in untrusted]
            if len(ok) >= 4:
                Tc = sim3_umeyama(pos[ok],
                                  positions[[name_to_i[names[i]] for i in ok]])
            else:
                Tc = T
            xyz, rgb = read_ply_xyzrgb(cdir / "points.ply")
            all_pts.append(apply(Tc, xyz[::8]).astype(np.float32))
            all_rgb.append(rgb[::8])

        np.savez_compressed(out / "global_cameras.npz",
                            names=np.array(fnames), positions=positions,
                            poses=poses)
        from recon3d.core.io_ply import write_ply
        write_ply(out / "global_points.ply",
                  np.concatenate(all_pts), np.concatenate(all_rgb))
        (out / "chain_log.json").write_text(json.dumps(
            [{"mode": "colmap_prior", "model": str(args.model),
              "metric_rescale": bool(args.metric_rescale)}], indent=1))
        print(f"-> {out} ({len(fnames)} frames)")

