"""島の曲がり直し (unbend): 方位ロックで測るフレームヨー誤差の除去

方位ロック(Insta360 FlowState)により島内の全フレームのヨーは一定のはずだが、
実測では単調に~10°ドリフトする(島A) = PanoVGGT地図の内部曲がり。
グローバル相似アンカーでは吸収できず、島端0.4-0.5mの系統誤差の主因になる。

各フレームのヨー誤差 ε = φ − φ_ref は方位ロックにより直接測れる(GT・図面不要)ので、
SLAMの回転ドリフト補正と同型の処理で島を伸ばす:
  軌跡:   c'_{f+1} = c'_f + R_v(−ε_mid)·(c_{f+1} − c_f)   (増分を回して再積分)
  点群:   p' = c'_f + R_v(−ε_i)·(p − c_i)                  (カメラ中心まわり回転)
重複フレームの2インスタンスは同じ c'_f に収束するため、チャンク間の食い違いも縮む。

使い方: python -m img_align.legacy.unbend --chunks-root results/pcd/chunks_yawfix \
          --chunk-range 0:5 --tag-out panovggt_A_unbend
出力: results/frontend_bench/<tag-out>/ (chain_chunks互換: global_cameras.npz 等)
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from img_align.config import BENCH_DIR
from img_align.core.chaining import chain
from img_align.core.geometry import apply
from img_align.config import RESULTS_DIR
from img_align.core.geometry import rotation_to_y
from img_align.core.io_ply import read_ply_xyzrgb
from img_align.core.geometry import robust_yaw_median, rot_y

PLY_STRIDE = 8


def circ_mean(angles):
    c = np.exp(1j * np.asarray(angles))
    return math.atan2(np.imag(c).mean(), np.real(c).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks-root", required=True)
    ap.add_argument("--tag-out", required=True)
    ap.add_argument("--chunk-range", default=None)
    args = ap.parse_args()
    root = Path(args.chunks_root)
    if not root.is_absolute():
        root = RESULTS_DIR / args.chunks_root
    cdirs = sorted(root.glob("chunk_*"), key=lambda p: int(p.name.split("_")[1]))
    if args.chunk_range:
        a, b = (int(v) for v in args.chunk_range.split(":"))
        cdirs = cdirs[a:b]
    chunks, T, log = chain(cdirs)

    # 全インスタンス (チャンクk × フレームn) のglobal回転・中心・ヨー
    inst = []   # (frame_name, k, i, R_g, c_g)
    for k, (names, pos, poses, _, _) in enumerate(chunks):
        for i, n in enumerate(names):
            R_g = T[k][1] @ poses[i, :3, :3]
            c_g = apply(T[k], pos[i][None])[0]
            inst.append((n, k, i, R_g, c_g))

    # 島の重力とヨー基準
    rots = np.stack([r for _, _, _, r, _ in inst])
    ups = rots[:, :, 1]
    up = np.median(ups / np.linalg.norm(ups, axis=1, keepdims=True), axis=0)
    Rg = rotation_to_y(up / np.linalg.norm(up))
    fwd = rots[:, :, 2] @ Rg.T
    phi = np.arctan2(-fwd[:, 2], fwd[:, 0])
    phi_ref, dev_all = robust_yaw_median(phi)
    print(f"phi_ref {math.degrees(phi_ref):.1f}deg, "
          f"inst yaw spread std {dev_all.std():.1f}deg")

    def rv(theta):
        """world系の鉛直軸まわり回転 (ヨーに+thetaが乗る向き)"""
        return Rg.T @ rot_y(theta) @ Rg

    # フレーム単位のヨー誤差 ε_f と平均中心
    names = sorted({n for n, *_ in inst})
    eps_f, c_f = {}, {}
    for n in names:
        rows = [j for j, e in enumerate(inst) if e[0] == n]
        eps_f[n] = math.atan2(math.sin(circ_mean(phi[rows]) - phi_ref),
                              math.cos(circ_mean(phi[rows]) - phi_ref))
        c_f[n] = np.mean([inst[j][4] for j in rows], axis=0)
    print(f"frame eps: min {math.degrees(min(eps_f.values())):.1f} "
          f"max {math.degrees(max(eps_f.values())):.1f}deg")

    # 軌跡の再積分 (増分をヨー誤差の中点で回す)
    c_new = {names[0]: c_f[names[0]]}
    for n0, n1 in zip(names, names[1:]):
        e_mid = (eps_f[n0] + eps_f[n1]) / 2
        c_new[n1] = c_new[n0] + rv(-e_mid) @ (c_f[n1] - c_f[n0])

    # 点群・poseの補正 (インスタンス単位のεで回し、c'_fへ平行移動)
    out = BENCH_DIR / args.tag_out
    out.mkdir(parents=True, exist_ok=True)
    all_pts, all_rgb = [], []
    frame_pose = {}
    cams_cache = {k: np.load(cdirs[k] / "cameras.npz") for k in range(len(cdirs))}
    for j, (n, k, i, R_g, c_g) in enumerate(inst):
        e_i = math.atan2(math.sin(phi[j] - phi_ref), math.cos(phi[j] - phi_ref))
        Rc = rv(-e_i)
        ply = cdirs[k] / "panovggt_out" / "pointclouds" / "per_frame" / f"{n}.ply"
        xyz, rgb = read_ply_xyzrgb(ply)
        xyz = xyz[::PLY_STRIDE]
        rgb = rgb[::PLY_STRIDE]
        pose = cams_cache[k]["poses"][i]
        w = xyz @ pose[:3, :3].T + pose[:3, 3]
        p_g = apply(T[k], w)
        all_pts.append((c_new[n] + (p_g - c_g) @ Rc.T).astype(np.float32))
        all_rgb.append(rgb)
        if n not in frame_pose:
            g = np.eye(4)
            g[:3, :3] = Rc @ R_g
            g[:3, 3] = c_new[n]
            frame_pose[n] = g

    positions = np.stack([c_new[n] for n in names])
    poses_out = np.stack([frame_pose[n] for n in names])
    np.savez_compressed(out / "global_cameras.npz",
                        names=np.array(names), positions=positions,
                        poses=poses_out)
    from img_align.core.io_ply import write_ply
    write_ply(out / "global_points.ply",
              np.concatenate(all_pts), np.concatenate(all_rgb))
    (out / "chain_log.json").write_text(json.dumps(log, indent=1))
    (out / "unbend_eps.json").write_text(json.dumps(
        {n: round(math.degrees(e), 2) for n, e in eps_f.items()}, indent=1))
    print(f"-> {out} ({sum(len(p) for p in all_pts):,} pts, {len(names)} frames)")


if __name__ == "__main__":
    main()
