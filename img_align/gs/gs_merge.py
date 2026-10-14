"""島ごとの学習済みGSを図面メートル座標へ変換し1つに合併する

図面座標への写像は反射(det<0)を含むため、共分散を A·Σ·Aᵀ で写してから
固有分解で quaternion+scale に復元する。色はSHのDC成分をRGB化して持つ
(.splat形式も閲覧もDC色のみで足りる。反射下の高次SH回転は行わない)。

使い方 (env=panovggt):
  IMG_SESSION=0312 conda run -n panovggt python -m img_align.gs.gs_merge [--selftest|--check]
出力: results/backend/gaussian_house{SFX}.ply (3DGS標準ply)
"""
import argparse
import json

import numpy as np

from img_align.config import SESSION
from img_align.core.plan_grid import PlanGrid
from img_align.mesh.meshing import ISLANDS, OUT_DIR, SFX, drawing_matrix, load_anchor

GS_DIR = OUT_DIR.parent / "gs"
SH_C0 = 0.28209479


def quat_to_rot(q):
    """(N,4) wxyz → (N,3,3)"""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], axis=-1).reshape(-1, 3, 3)


def rot_to_quat(R):
    """(N,3,3) → (N,4) wxyz (数値安定版)"""
    m00, m11, m22 = R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]
    q = np.zeros((len(R), 4))
    t = m00 + m11 + m22
    # 4通りの分岐を全計算して最大対角で選ぶ
    cand = np.stack([t, m00, m11, m22], axis=1)
    case = cand.argmax(axis=1)
    for c in range(4):
        sel = case == c
        if not sel.any():
            continue
        r = R[sel]
        if c == 0:
            s = np.sqrt(1 + t[sel]) * 2
            q[sel] = np.stack([s / 4, (r[:, 2, 1] - r[:, 1, 2]) / s,
                               (r[:, 0, 2] - r[:, 2, 0]) / s,
                               (r[:, 1, 0] - r[:, 0, 1]) / s], 1)
        elif c == 1:
            s = np.sqrt(1 + r[:, 0, 0] - r[:, 1, 1] - r[:, 2, 2]) * 2
            q[sel] = np.stack([(r[:, 2, 1] - r[:, 1, 2]) / s, s / 4,
                               (r[:, 0, 1] + r[:, 1, 0]) / s,
                               (r[:, 0, 2] + r[:, 2, 0]) / s], 1)
        elif c == 2:
            s = np.sqrt(1 + r[:, 1, 1] - r[:, 0, 0] - r[:, 2, 2]) * 2
            q[sel] = np.stack([(r[:, 0, 2] - r[:, 2, 0]) / s,
                               (r[:, 0, 1] + r[:, 1, 0]) / s, s / 4,
                               (r[:, 1, 2] + r[:, 2, 1]) / s], 1)
        else:
            s = np.sqrt(1 + r[:, 2, 2] - r[:, 0, 0] - r[:, 1, 1]) * 2
            q[sel] = np.stack([(r[:, 1, 0] - r[:, 0, 1]) / s,
                               (r[:, 0, 2] + r[:, 2, 0]) / s,
                               (r[:, 1, 2] + r[:, 2, 1]) / s, s / 4], 1)
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def transform_gaussians(means, quats, log_scales, M):
    """4x4相似(反射可)でガウシアンを写す。返り値は同形式"""
    L = M[:3, :3]
    means2 = means @ L.T + M[:3, 3]
    R = quat_to_rot(quats)
    S = np.exp(log_scales)
    A = (L[None] @ R) * S[:, None, :]              # L·R·S  (N,3,3)
    Sigma = A @ A.transpose(0, 2, 1)               # L·Σ·Lᵀ
    w, V = np.linalg.eigh(Sigma)
    w = np.clip(w, 1e-12, None)
    scales2 = np.log(np.sqrt(w))
    det = np.linalg.det(V)
    V[det < 0, :, 2] *= -1                         # 右手系に直す
    quats2 = rot_to_quat(V)
    return means2, quats2, scales2


def selftest():
    rng = np.random.default_rng(0)
    n = 100
    means = rng.normal(size=(n, 3))
    q = rng.normal(size=(n, 4))
    quats = q / np.linalg.norm(q, axis=1, keepdims=True)
    log_scales = rng.normal(scale=0.5, size=(n, 3)) - 3
    isl = ISLANDS[SESSION][0]
    M, info = drawing_matrix(load_anchor(isl), PlanGrid(isl["floor"]),
                             isl["z_off"], metric_input=True)
    flip = np.diag([-1.0, 1, 1, 1])
    for label, Mt in (("det<0", M), ("det>0", flip @ M)):
        d = np.linalg.det(Mt[:3, :3])
        assert (d < 0) == (label == "det<0"), (label, d)
        m2, q2, s2 = transform_gaussians(means, quats, log_scales, Mt)
        R0 = quat_to_rot(quats)
        S0 = np.exp(log_scales)
        Sig0 = (R0 * S0[:, None, :] ** 2) @ R0.transpose(0, 2, 1)
        R2 = quat_to_rot(q2)
        S2 = np.exp(s2)
        Sig2 = (R2 * S2[:, None, :] ** 2) @ R2.transpose(0, 2, 1)
        L = Mt[:3, :3]
        err = np.abs(L[None] @ Sig0 @ L.T[None] - Sig2).max()
        assert err < 1e-9, (label, err)
    print("selftest OK (det<0, det>0)")


def load_island_gaussians(isl):
    ckpt = GS_DIR / isl["tag"] / "ckpt.npz"
    if not ckpt.exists():
        return None
    d = np.load(ckpt)
    M, info = drawing_matrix(load_anchor(isl), PlanGrid(isl["floor"]),
                             isl["z_off"], metric_input=True)
    means, quats, log_scales = transform_gaussians(
        d["means"], d["quats"], d["log_scales"], M)
    rgb = np.clip(0.5 + SH_C0 * d["sh0"][:, 0], 0, 1)
    opac = 1 / (1 + np.exp(-d["logit_opacities"]))
    keep = ((means[:, 2] > isl["z_off"] - 0.3)
            & (means[:, 2] < isl["z_off"] + 3.0))
    return (means[keep], quats[keep], log_scales[keep],
            rgb[keep], opac[keep])


def write_3dgs_ply(path, means, quats, log_scales, rgb, opac):
    """3DGS標準ply (f_dc + opacity + scale + rot)"""
    n = len(means)
    props = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
             "scale_0", "scale_1", "scale_2",
             "rot_0", "rot_1", "rot_2", "rot_3"]
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {n}\n"
              + "".join(f"property float {p}\n" for p in props)
              + "end_header\n")
    f_dc = (rgb - 0.5) / SH_C0
    logit = np.log(np.clip(opac, 1e-6, 1 - 1e-6)
                   / (1 - np.clip(opac, 1e-6, 1 - 1e-6)))
    data = np.hstack([means, f_dc, logit[:, None], log_scales,
                      quats]).astype(np.float32)
    with open(path, "wb") as fp:
        fp.write(header.encode())
        fp.write(np.ascontiguousarray(data).tobytes())


def merge():
    missing = [isl["tag"] for isl in ISLANDS[SESSION]
               if not (GS_DIR / isl["tag"] / "ckpt.npz").exists()]
    if missing:
        raise SystemExit(f"ckpt無しの島: {missing} (gs_trainを先に実行)")
    parts = []
    for isl in ISLANDS[SESSION]:
        g = load_island_gaussians(isl)
        parts.append(g)
        print(f"{isl['tag']}: {len(g[0]):,} gaussians (クロップ後)")
    merged = [np.concatenate([p[i] for p in parts]) for i in range(5)]
    out = OUT_DIR / f"gaussian_house{SFX}.ply"
    write_3dgs_ply(out, *merged)
    print(f"-> {out} ({len(merged[0]):,} gaussians)")


def check(ply=None, suffix=""):
    """GS中心点の壁整合を点群評価と同じ物差しで見る + 真上ビュー"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from img_align.eval.gt_traj import MM_PER_PX
    from img_align.mesh.meshing import wall_distance_stats
    from img_align.postproc.plan_icp import wall_face_segments_px
    d = read_3dgs_centers(ply or OUT_DIR / f"gaussian_house{SFX}.ply")
    res = {}
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    for ax, (floor, z_off) in zip(axes, (("1F", 0.0), ("2F", 2.9))):
        band = d[(d[:, 2] > z_off + 0.3) & (d[:, 2] < z_off + 2.2)]
        if len(band) > 1000:
            res[floor] = wall_distance_stats(band, floor)
        ax.scatter(band[::5, 0], band[::5, 1], s=0.3, c="orange", alpha=0.5)
        for x1, y1, x2, y2 in wall_face_segments_px(floor) * MM_PER_PX / 1000:
            ax.plot([x1, x2], [y1, y2], "-", color="cyan", lw=1.0)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_facecolor("#101014")
        t = res.get(floor, {})
        ax.set_title(f"GS {floor}: med {t.get('median_m', '-')}m "
                     f"cover {t.get('wall_coverage', '-')}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"gs_check{suffix}{SFX}.png", dpi=100,
                facecolor="#181820")
    print(json.dumps(res, indent=1))
    print(f"-> {OUT_DIR / f'gs_check{suffix}{SFX}.png'}")


def read_3dgs_centers(path):
    with open(path, "rb") as fp:
        header = b""
        while not header.endswith(b"end_header\n"):
            header += fp.readline()
        n = int([ln for ln in header.decode().splitlines()
                 if ln.startswith("element vertex")][0].split()[-1])
        data = np.frombuffer(fp.read(n * 14 * 4), dtype=np.float32)
    return data.reshape(n, 14)[:, :3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--ply", default=None)
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()
    if args.selftest:
        selftest()
    elif args.check:
        check(ply=args.ply, suffix=args.suffix)
    else:
        merge()


if __name__ == "__main__":
    main()
