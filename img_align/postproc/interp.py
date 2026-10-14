"""macond制御点 × pvチェーン形状 → 全フレーム精錬軌跡 (mafullタグ)

MapAnythingポーズ条件付け (macond) はVRAM制約で間引きフレームしか精錬できない。
そこでmacondポーズを制御点、pvチェーンの局所形状を事前として全フレームを補間する:
  1. 各制御点kの近傍窓 (±3制御点) で pv→macond のSim(3) T_k をUmeyamaフィット
  2. 中間フレームは隣接するT_k/T_{k+1}で写した位置の線形ブレンド
     (回転はslerp相当のブレンド)
  3. 制御点残差を線形補間して加算し、制御点では macondポーズと厳密一致させる
長波長のチェーンドリフトは制御点間隔で切られ、局所形状は保存される。
出力はメートル実寸 (macondゲージ)。

使い方 (ホスト img_align env):
  IMG_SESSION=0616 python -m img_align.postproc.interp --pv pv0616_2F --ma macond0616_2F
  IMG_SESSION=0912 python -m img_align.postproc.interp --all   # pv_manifest.jsonの全島
出力: frontend_bench/maf<島名>/ (chain_chunks契約。点群はmacondのものを流用)
"""
import argparse
import json
import shutil

import numpy as np

from img_align.config import BENCH_DIR
from img_align.core.geometry import apply, sim3_umeyama

CTRL_WIN = 3      # Sim3フィットに使う近傍制御点数 (片側)


def _load(tag):
    d = np.load(BENCH_DIR / tag / "global_cameras.npz")
    names = [str(n) for n in d["names"]]
    frames = np.array([int(n.replace("frame_", "")) for n in names])
    order = np.argsort(frames)
    return ([names[i] for i in order], frames[order],
            d["positions"][order], d["poses"][order])


def _blend_rot(Ra, Rb, w):
    """2回転の球面補間 (クォータニオン正規化で十分な近似)"""
    qa, qb = _quat(Ra), _quat(Rb)
    if np.dot(qa, qb) < 0:
        qb = -qb
    q = (1 - w) * qa + w * qb
    q /= np.linalg.norm(q)
    return _quat_to_r(q)


def _quat(R):
    w = np.sqrt(max(0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    if w < 1e-6:
        i = int(np.argmax(np.diag(R)))
        q = np.zeros(4)
        q[1 + i] = 1.0
        return q
    return np.array([w, (R[2, 1] - R[1, 2]) / (4 * w),
                     (R[0, 2] - R[2, 0]) / (4 * w),
                     (R[1, 0] - R[0, 1]) / (4 * w)])


def _quat_to_r(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def interp(pv_tag, ma_tag, out_tag):
    names_p, fr_p, P, pose_p = _load(pv_tag)
    names_m, fr_m, Q, pose_m = _load(ma_tag)
    name_to_i = {n: i for i, n in enumerate(names_p)}
    ctrl = [(name_to_i[n], j) for j, n in enumerate(names_m)
            if n in name_to_i]           # (pv index, macond index)
    if len(ctrl) < 3:
        raise RuntimeError(f"{ma_tag}: 制御点不足 ({len(ctrl)})")
    ci = np.array([c[0] for c in ctrl])  # pv側index (フレーム昇順)
    cj = np.array([c[1] for c in ctrl])

    # 各制御点の局所Sim3 (pv座標→macond座標)
    Ts = []
    for k in range(len(ctrl)):
        lo, hi = max(0, k - CTRL_WIN), min(len(ctrl), k + CTRL_WIN + 1)
        Ts.append(sim3_umeyama(P[ci[lo:hi]], Q[cj[lo:hi]]))

    # フレームごとの区間ブレンド
    n = len(names_p)
    seg = np.clip(np.searchsorted(fr_p[ci], fr_p, side="right") - 1,
                  0, len(ctrl) - 2)
    x = np.zeros((n, 3))
    rot = np.zeros((n, 3, 3))
    for i in range(n):
        k = seg[i]
        f0, f1 = fr_p[ci[k]], fr_p[ci[k + 1]]
        w = float(np.clip((fr_p[i] - f0) / max(f1 - f0, 1), 0.0, 1.0))
        xa = apply(Ts[k], P[i][None])[0]
        xb = apply(Ts[k + 1], P[i][None])[0]
        x[i] = (1 - w) * xa + w * xb
        Rb = _blend_rot(Ts[k][1], Ts[k + 1][1], w)
        rot[i] = Rb @ pose_p[i][:3, :3]

    # 制御点スナップ: 残差を線形補間して加算 (制御点でmacondと厳密一致)
    resid = Q[cj] - x[ci]
    for d in range(3):
        x[:, d] += np.interp(fr_p, fr_p[ci], resid[:, d])
    # 制御点の回転はmacondそのもの
    for (ip, jm) in ctrl:
        rot[ip] = pose_m[jm][:3, :3]

    poses = np.tile(np.eye(4), (n, 1, 1))
    poses[:, :3, :3] = rot
    poses[:, :3, 3] = x
    out = BENCH_DIR / out_tag
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "global_cameras.npz",
                        names=np.array(names_p), positions=x, poses=poses)
    shutil.copy(BENCH_DIR / ma_tag / "global_points.ply",
                out / "global_points.ply")
    (out / "chain_log.json").write_text(json.dumps([{
        "mode": "macond_interp", "pv": pv_tag, "ma": ma_tag,
        "n_ctrl": len(ctrl), "unit": "metric_m"}], indent=1))
    snap = np.linalg.norm(resid, axis=1)
    print(f"-> {out} ({n} frames, {len(ctrl)} ctrl, "
          f"ctrl resid pre-snap med {np.median(snap):.3f}m)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pv")
    ap.add_argument("--ma")
    ap.add_argument("--out", default=None)
    ap.add_argument("--all", action="store_true",
                    help="pv_manifest.jsonの全島を pv→macond→maf で処理")
    args = ap.parse_args()
    if args.all:
        manifest = json.loads((BENCH_DIR / "pv_manifest.json").read_text())
        for m in manifest:
            pv = m["tag"]
            ma = pv.replace("pv", "macond", 1)
            if not (BENCH_DIR / ma / "global_cameras.npz").exists():
                print(f"{ma}: なし -> skip")
                continue
            interp(pv, ma, pv.replace("pv", "maf", 1))
    else:
        interp(args.pv, args.ma, args.out or
               args.pv.replace("pv", "maf", 1))


if __name__ == "__main__":
    main()
