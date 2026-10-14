"""複数パノラマの PanoRecon 3DGS を PanoVGGT の相対姿勢+スケールで1つに統合する

各 PanoRecon 出力はパノラマごとのカメラ中心座標系+相対スケール。
同じフレーム集合を PanoVGGT に通した結果(相対姿勢+ローカル点群)を基準に、
フレームごとに (座標規約の回転差, スケール) を推定してワールドへ変換し、
ガウシアン属性(位置・回転四元数・対数スケール)を正しく変換して結合する。

使い方: python -m img_align.legacy.merge_gs --frames 90,91,92 --tag triplet
"""
import argparse
import math

import numpy as np
from scipy.spatial import cKDTree

from img_align.core.io_ply import read_ply_xyzrgb
from img_align.core.panovggt import PCD_DIR
from img_align.config import RESULTS_DIR

GS_DIR = RESULTS_DIR / "geniesim_world"

# 座標規約差の候補 (180°回転族。det=+1)
ROT_CANDIDATES = {
    "identity": np.eye(3),
    "Rx180": np.diag([1.0, -1.0, -1.0]),
    "Ry180": np.diag([-1.0, 1.0, -1.0]),
    "Rz180": np.diag([-1.0, -1.0, 1.0]),
}
# DA360の画像毎スケールは0.5〜2超までばらつく (境界に張り付くと規約推定まで壊れる)
SCALE_SEARCH = np.geomspace(0.25, 2.8, 45)


def read_gaussians(path):
    """3DGS PLY の vertex ブロック (float 14プロパティ) を読む"""
    with open(path, "rb") as f:
        header = b""
        while not header.endswith(b"end_header\n"):
            header += f.readline()
        lines = header.decode().splitlines()
        n = int([l for l in lines if l.startswith("element vertex")][0].split()[-1])
        i = [k for k, l in enumerate(lines) if l.startswith("element vertex")][0]
        props = []
        for l in lines[i + 1:]:
            if l.startswith("element") or l.startswith("end_header"):
                break
            props.append(l.split()[-1])
        dtype = np.dtype([(p, "<f4") for p in props])
        data = np.frombuffer(f.read(n * dtype.itemsize), dtype=dtype).copy()
    return data, props


def write_gaussians(path, data, props):
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(data)}\n"
              + "".join(f"property float {p}\n" for p in props)
              + "end_header\n")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())


def estimate_conv_and_scale(gs_xyz, ref_xyz, rng):
    """PanoRecon点群→PanoVGGTローカル点群 の座標規約回転とスケールをNN距離で推定"""
    gs = gs_xyz[rng.choice(len(gs_xyz), 15000, replace=False)]
    ref = ref_xyz[rng.choice(len(ref_xyz), 60000, replace=False)]
    tree = cKDTree(ref)
    best = None
    for name, R in ROT_CANDIDATES.items():
        p = gs @ R.T
        for s in SCALE_SEARCH:
            d, _ = tree.query(s * p[::3], k=1)
            err = float(np.median(d))
            if best is None or err < best[0]:
                best = (err, name, R, float(s))
    err, name, R, s = best
    print(f"  conv={name} scale={s:.3f} median NN dist={err:.3f}")
    return R, s


def rot_to_quat(R):
    w = math.sqrt(max(0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    if w > 1e-6:
        x = (R[2, 1] - R[1, 2]) / (4 * w)
        y = (R[0, 2] - R[2, 0]) / (4 * w)
        z = (R[1, 0] - R[0, 1]) / (4 * w)
    else:
        # 180°回転: 対角から復元
        x = math.sqrt(max(0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
        y = math.sqrt(max(0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2
        z = math.sqrt(max(0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2
        x = math.copysign(x, R[2, 1] - R[1, 2]) if abs(R[2, 1] - R[1, 2]) > 1e-9 else x
        y = math.copysign(y, R[0, 2] - R[2, 0]) if abs(R[0, 2] - R[2, 0]) > 1e-9 else y
        z = math.copysign(z, R[1, 0] - R[0, 1]) if abs(R[1, 0] - R[0, 1]) > 1e-9 else z
    return np.array([w, x, y, z])


def quat_mul(q1, q2):
    """q1 ⊗ q2 (q2は(N,4)の配列, w,x,y,z順)"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="例 '90,91,92'")
    ap.add_argument("--tag", required=True, help="PanoVGGT実行のタグ (results/pcd/<tag>)")
    ap.add_argument("--out", default=None, help="出力PLY (既定: geniesim_world/merged_<tag>.ply)")
    args = ap.parse_args()
    frames = [int(v) for v in args.frames.split(",")]
    out_path = args.out or (GS_DIR / f"merged_{args.tag}.ply")

    pv_dir = PCD_DIR / args.tag
    cams = np.load(pv_dir / "cameras.npz")
    names = [str(n) for n in cams["names"]]
    rng = np.random.default_rng(0)

    merged = []
    props_ref = None
    for fr in frames:
        name = f"frame_{fr:06d}"
        print(name)
        data, props = read_gaussians(GS_DIR / name / "merged_gaussians.ply")
        props_ref = props
        ref_xyz, _ = read_ply_xyzrgb(
            pv_dir / "panovggt_out" / "pointclouds" / "per_frame" / f"{name}.ply")
        xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)

        R_conv, s = estimate_conv_and_scale(xyz, ref_xyz, rng)
        pose = cams["poses"][names.index(name)]  # cam2world
        R_total = pose[:3, :3] @ R_conv
        t = pose[:3, 3]

        w = (s * xyz) @ R_total.T + t
        data["x"], data["y"], data["z"] = (w[:, 0].astype(np.float32),
                                           w[:, 1].astype(np.float32),
                                           w[:, 2].astype(np.float32))
        logs = math.log(s)
        for k in ("scale_0", "scale_1", "scale_2"):
            data[k] += logs
        q = np.stack([data[f"rot_{i}"] for i in range(4)], axis=1).astype(np.float64)
        q_new = quat_mul(rot_to_quat(R_total), q)
        for i in range(4):
            data[f"rot_{i}"] = q_new[:, i].astype(np.float32)
        merged.append(data)

    out = np.concatenate(merged)
    write_gaussians(out_path, out, props_ref)
    print(f"merged {len(out):,} gaussians -> {out_path}")


if __name__ == "__main__":
    main()
