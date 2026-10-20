"""チャンク連結 (VGGT-Long式): 重複フレームの点群同士を trimmed Sim(3) ICP で整合

従来の「共有カメラ中心4点(ほぼ共線)のUmeyama」を置き換える共通後処理。
PanoVGGT形式 (panovggt_out/pointclouds/per_frame/*.ply) と
DA3形式 (per_frame.npz) の両方のチャンクレイアウトに対応する。

使い方:
  python main.py tool chaining --input /data/x --chunks-root results/pcd/chunks_masked --tag panovggt
  python main.py tool chaining --input /data/x --chunks-root results/da3/chunks_da3base --tag da3base

出力: results/frontend_bench/<tag>/ {global_cameras.npz, global_points.ply, chain_log.json}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.spatial import cKDTree  # type: ignore[attr-defined]

from recon3d.core.geometry import apply, compose, sim3_umeyama
from recon3d.core.geometry import rotation_to_y
from recon3d.core.io_ply import read_ply_xyzrgb
from recon3d.core.geometry import robust_yaw_median, rot_y
from recon3d.paths import Paths
from recon3d.types import F64, Mat3, Points3, Sim3

ICP_ITERS = 15
TRIM = 0.7            # 対応の下位30%(距離大)を捨てる
MAX_PTS = 20000


def trimmed_icp_sim3(src: Points3, dst: Points3, T0: Sim3,
                     fix_R: bool = False) -> tuple[Sim3, float]:
    """src(→dst)の trimmed Sim(3) ICP。src/dst: (N,3)

    スケールは初期値(カメラ中心Umeyama)の±25%にクランプする。
    対応が悪いときにSim(3) ICPが縮小退化(重心へ潰れて残差最小化)するのを防ぐ。
    fix_R=True では回転をT0の値に固定し、スケール+並進のみ解く
    (方位ロック: 重力+一定ヨーから回転が既知の場合)。
    """
    if len(src) > MAX_PTS:
        src = src[:: len(src) // MAX_PTS + 1]
    if len(dst) > MAX_PTS * 3:
        dst = dst[:: len(dst) // (MAX_PTS * 3) + 1]
    tree = cKDTree(dst)
    T = T0
    s0 = T0[0]
    resid = float("nan")
    for _ in range(ICP_ITERS):
        p = apply(T, src)
        d, j = tree.query(p, k=1)
        keep = d <= np.quantile(d, TRIM)
        if fix_R:
            R = T0[1]
            rs = src[keep] @ R.T
            mu_s, mu_d = rs.mean(axis=0), dst[j[keep]].mean(axis=0)
            num = float(((dst[j[keep]] - mu_d) * (rs - mu_s)).sum())
            den = float(((rs - mu_s) ** 2).sum())
            s = float(np.clip(num / max(den, 1e-12), 0.75 * s0, 1.25 * s0))
            t = mu_d - s * mu_s
        else:
            s, R, t = sim3_umeyama(src[keep], dst[j[keep]])
            if not (0.75 * s0 <= s <= 1.25 * s0):
                s = float(np.clip(s, 0.75 * s0, 1.25 * s0))
                # スケールを固定して並進を取り直す
                mu_s = src[keep].mean(axis=0)
                mu_d = dst[j[keep]].mean(axis=0)
                t = mu_d - s * R @ mu_s
        T = (s, R, t)
        resid = float(np.median(d[keep]))
    return T, resid


def med_yaw(rots: F64) -> tuple[float, Mat3]:
    """cam2world回転群の重力整列ヨー中央値と重力回転Rg"""
    ups = rots[:, :, 1]
    up = np.median(ups / np.linalg.norm(ups, axis=1, keepdims=True), axis=0)
    Rg = rotation_to_y(up / np.linalg.norm(up))
    fwd = rots[:, :, 2] @ Rg.T
    med, _ = robust_yaw_median(np.arctan2(-fwd[:, 2], fwd[:, 0]))
    return med, Rg


def lock_rotation(g_rots: Sequence[Mat3], cur_rots: F64) -> Mat3:
    """方位ロック: 回転 = 重力整列の一致 + ヨー中央値の一致 で一意に決まる

    「撮影中ヨーは一定」という仮定を置く。FlowStateの実測ヨーが使えるなら
    `lock_rotation_flowstate` の方が正確 (実測ではヨーは一定でなく±12°動く)。
    """
    m_g, Rg_g = med_yaw(np.stack(list(g_rots)))
    m_c, Rg_c = med_yaw(cur_rots)
    return Rg_g.T @ rot_y(m_g - m_c) @ Rg_c


def _gauge_offset(names: Sequence[str], rots: F64,
                  fs_yaw: F64) -> tuple[float | None, Mat3]:
    """このチャンクの「ゲージ ↔ 世界」のヨー差 [rad]。FlowStateの実測ヨーを使う

    ヨーは角度なので複素平均で畳む (±180°の折返しを跨がないため)。
    """
    _, Rg = med_yaw(rots)
    fwd = rots[:, :, 2] @ Rg.T
    vy = np.arctan2(-fwd[:, 2], fwd[:, 0])
    fr = np.array([int(str(n).replace("frame_", "")) for n in names])
    ok = fr < len(fs_yaw)
    if ok.sum() < 3:
        return None, Rg
    d = np.radians(fs_yaw[fr[ok]]) - vy[ok]
    return float(np.angle(np.exp(1j * d).mean())), Rg


def lock_rotation_flowstate(g_rots: Sequence[Mat3], g_names: Sequence[str],
                            cur_rots: F64, cur_names: Sequence[str],
                            fs_yaw: F64) -> Mat3 | None:
    """FlowStateの実測ヨーで連結回転を決める (累積回転を断つ)

    従来の `lock_rotation` は「ヨー一定」を仮定するため、仮定からのずれが
    連結のたびに積み上がる。2026-10-15の実測では0912の全6島で**同符号の
    ドリフト** (−9.7〜−27.4°、−1.5〜−3.7°/チャンク) が出ていた。
    FlowStateは画像側の**絶対基準**なので、各チャンクを独立に世界へ合わせれば
    累積が原理的に起きない。

    返り値: 回転行列。FlowStateが使えなければ None (呼び出し側が従来版へ)
    """
    o_g, Rg_g = _gauge_offset(g_names, np.stack(list(g_rots)), fs_yaw)
    o_c, Rg_c = _gauge_offset(cur_names, cur_rots, fs_yaw)
    if o_g is None or o_c is None:
        return None
    # 世界ヨーを一致させるのに必要な、重力軸まわりの回転
    return Rg_g.T @ rot_y(o_c - o_g) @ Rg_c


def chain(paths: Paths, cdirs: Sequence[Path],
          yaw_lock: bool = False) -> tuple[list, list[Sim3], list[dict]]:
    """チャンク列を順に連結し (chunks, 変換列T, ログ) を返す

    yaw_lock: False=ICPが回転も解く (既定) / True=ヨー一定を仮定してロック /
    "flowstate"=FlowStateの実測ヨーでロック (累積回転を断つ。要 imu/flowstate_yaw)
    """
    chunks = [load_chunk(c) for c in cdirs]
    T = [(1.0, np.eye(3), np.zeros(3))]
    g_rots = list(chunks[0][2][:, :3, :3])   # 連結済みフレームのglobal回転
    g_names = list(chunks[0][0])
    fs_yaw = None
    if yaw_lock == "flowstate":
        from recon3d.imu.flowstate_yaw import load as load_fs
        fs_yaw = load_fs(paths)
        if fs_yaw is None:
            print("[chain] flowstate_yaw.npz が無い -> 従来のヨーロックへ")
            yaw_lock = True
    log = []
    for k in range(1, len(chunks)):
        n_prev, _, _, pf_prev, _ = chunks[k - 1]
        n_cur, pos_cur, poses_cur, pf_cur, _ = chunks[k]
        shared = [n for n in n_cur if n in n_prev and n in pf_prev and n in pf_cur]
        src = np.concatenate([pf_cur[n] for n in shared])
        c_cur = np.stack([pos_cur[n_cur.index(n)] for n in shared])
        c_prev_g = apply(T[k - 1], np.stack(
            [chunks[k - 1][1][n_prev.index(n)] for n in shared]))
        if len(shared) >= 4:
            dst = apply(T[k - 1], np.concatenate([pf_prev[n] for n in shared]))
            # 初期値: 共有カメラ中心のUmeyama (弱いが初期値としては十分)
            T0 = sim3_umeyama(c_cur, c_prev_g)
        else:
            # 共有カメラが少なくUmeyamaの回転が縮退 → 回転初期値は方位ロック、
            # ICP参照は前チャンクの清掃済み全体点群 (global配置)
            xyz_prev, _ = read_ply_xyzrgb(cdirs[k - 1] / "points.ply")
            dst = apply(T[k - 1], xyz_prev[::20])
            R0 = lock_rotation(g_rots, poses_cur[:, :3, :3])
            s0 = T[k - 1][0]
            t0 = c_prev_g.mean(axis=0) - s0 * R0 @ c_cur.mean(axis=0)
            T0 = (s0, R0, t0)
            print(f"chunk {k:02d}: few shared ({len(shared)}) -> "
                  f"lock-rotation init + full prev map as ICP target")
        if yaw_lock:
            R_lock = None
            if fs_yaw is not None:
                R_lock = lock_rotation_flowstate(
                    g_rots, g_names, poses_cur[:, :3, :3], n_cur, fs_yaw)
            if R_lock is None:
                R_lock = lock_rotation(g_rots, poses_cur[:, :3, :3])
            t0 = c_prev_g.mean(axis=0) - T0[0] * R_lock @ c_cur.mean(axis=0)
            T0 = (T0[0], R_lock, t0)
        Tk, resid = trimmed_icp_sim3(src, dst, T0, fix_R=bool(yaw_lock))
        T.append(Tk)
        g_rots.extend(Tk[1] @ po[:3, :3] for po in poses_cur)
        g_names.extend(n_cur)
        log.append({"chunk": k, "shared": len(shared),
                    "scale": round(Tk[0], 4), "icp_resid": round(resid, 4)})
        print(f"chunk {k:02d}: shared={len(shared)} scale={Tk[0]:.3f} "
              f"resid={resid:.4f}")
    return chunks, T, log


# **警告**: `panovggt_out/pointclouds/per_frame/` は削除してはいけない。
# chain() のチャンク間ICPはこのフレーム別点群を使う (共有カメラ4点の
# Umeyamaだけでは共線で弱いため)。消すとチャンクが使用不能になり、
# 既存のチェーンは残るが**再生成できなくなる** (2026-10-16に外部データで
# 実際に踏み、再推論し直した)。`posegraph` も同じものを使う。


def load_chunk(cdir: Path) -> tuple:
    """チャンクの (names, cam_positions, poses, per-frame点群dict, 不信集合) を読む

    untrusted.json (yaw_fixが出力) にあるフレームはこのチャンク内でポーズ全壊
    → per-frame点群(重なりICP用)から除外する。
    """
    cams = np.load(cdir / "cameras.npz")
    names = [str(n) for n in cams["names"]]
    bad_path = cdir / "untrusted.json"
    untrusted = set(json.loads(bad_path.read_text())) if bad_path.exists() \
        else set()
    pf = {}
    npz = cdir / "per_frame.npz"
    if npz.exists():                     # DA3形式
        d = np.load(npz)
        pf = {n: d[n] for n in d.files if n not in untrusted}
    else:                                # PanoVGGT形式
        for n in names:
            if n in untrusted:
                continue
            ply = cdir / "panovggt_out" / "pointclouds" / "per_frame" / f"{n}.ply"
            if ply.exists():
                xyz, _ = read_ply_xyzrgb(ply)
                # ローカル(カメラ)座標 → チャンクworldへ (cam2world)
                pose = cams["poses"][names.index(n)]
                pf[n] = (xyz[::15] @ pose[:3, :3].T + pose[:3, 3]).astype(
                    np.float32)
    return names, cams["positions"], cams["poses"], pf, untrusted


def _sim3_mul(A: Sim3, B: Sim3) -> Sim3:
    """相似変換の合成 (Bを適用してからA)"""
    sa, Ra, ta = A
    sb, Rb, tb = B
    return (sa * sb, Ra @ Rb, sa * (Ra @ tb) + ta)


def _sim3_inv(T: Sim3) -> Sim3:
    s, R, t = T
    return (1.0 / s, R.T, -(R.T @ t) / s)


def drift_correct(chunks: Sequence, T: list[Sim3], fs_yaw: F64,
                  min_links: int = 3) -> tuple[list[Sim3], float]:
    """連結で積み上がったヨーの**系統バイアス**だけを除く

    `lock_rotation_flowstate` のようにICPの回転をFlowStateで置き換えると、
    ICPの方が正確な分まで壊れる (2026-10-15実測: 形状残差 0.21→0.25m 悪化)。
    そこでICPの相対回転は信じたまま、**1連結あたり一定量のバイアス**
    (実測で0912の全6島が同符号、−1.5〜−3.7°/チャンク) だけを引く。
    FlowStateは島全体の傾きを1つ推定するのにしか使わないので、
    フレーム単位の揺れ (±6°) は平均されて効かない。

    返り値: 補正後の変換列 T'。推定できなければ元の T。
    """
    if fs_yaw is None or len(T) < min_links + 1:
        return T, 0.0
    # 各チャンクの「ゲージ↔世界」オフセットを、連結後のglobal姿勢で測る
    xs, ys = [], []
    for k, (names, _, poses, _, _) in enumerate(chunks):
        g_rot = np.stack([T[k][1] @ po[:3, :3] for po in poses])
        o, _ = _gauge_offset(names, g_rot, fs_yaw)
        if o is not None:
            xs.append(k)
            ys.append(o)
    if len(xs) < min_links:
        return T, 0.0
    # 折返しを跨がないよう unwrap してから直線を当てる
    slope = float(np.polyfit(np.array(xs), np.unwrap(np.array(ys)), 1)[0])
    if abs(np.degrees(slope)) < 0.3:
        return T, slope          # バイアスが小さければ触らない
    # 相対変換を取り出し、回転だけ -slope 補正して積み直す
    out = [T[0]]
    for k in range(1, len(T)):
        rel = _sim3_mul(_sim3_inv(T[k - 1]), T[k])
        _, Rg = med_yaw(np.stack([T[k - 1][1] @ po[:3, :3]
                                  for po in chunks[k - 1][2]]))
        fix = Rg.T @ rot_y(slope) @ Rg
        out.append(_sim3_mul(out[k - 1], (rel[0], fix @ rel[1], rel[2])))
    return out, slope


def chain_and_save(paths: Paths, cdirs: Sequence[Path], tag: str,
                   yaw_lock: bool = False,
                   drift_fix: bool = False) -> Path:
    """チャンク列を連結して frontend_bench/<tag>/ に書き出す (mainの本体)

    drift_fix: 連結後にヨーの系統バイアスを除く (要 imu/flowstate_yaw)
    """
    chunks, T, log = chain(paths, cdirs, yaw_lock=yaw_lock)
    if drift_fix:
        from recon3d.imu.flowstate_yaw import load as load_fs
        fs = load_fs(paths)
        if fs is None:
            raise SystemExit(
                f"drift_fix には FlowState ヨーが要る: {paths.imu}/"
                "flowstate_yaw.npz\n  docker compose exec imu "
                "python main.py tool flowstate_yaw --input "
                f"{paths.root}")
        T, sl = drift_correct(chunks, T, fs)
        print(f"[drift_fix] ヨーバイアス {np.degrees(sl):+.2f}°/chunk を除去")
    # グローバル軌跡と点群
    out = paths.bench / tag
    out.mkdir(parents=True, exist_ok=True)
    frame_pos, frame_pose = {}, {}
    all_pts, all_rgb = [], []
    n_skipped = 0
    for k, cdir in enumerate(cdirs):
        names, pos, poses, _, untrusted = chunks[k]
        s, R, t = T[k]
        for n, p, po in zip(names, pos, poses):
            if n in untrusted:    # ポーズ全壊インスタンスは軌跡平均に入れない
                n_skipped += 1
                continue
            frame_pos.setdefault(n, []).append(apply(T[k], p[None])[0])
            g = np.eye(4)
            g[:3, :3] = R @ po[:3, :3]
            g[:3, 3] = apply(T[k], po[None, :3, 3])[0]
            frame_pose.setdefault(n, []).append(g)
        xyz, rgb = read_ply_xyzrgb(cdir / "points.ply")
        # 間引きは `[::8]`。ボクセル化は実測で有害だった (README「試して捨てた
        # 実装」)。点の多さ自体が信頼度情報なので一様化してはいけない
        all_pts.append(apply(T[k], xyz[::8]).astype(np.float32))
        all_rgb.append(rgb[::8])
    if n_skipped:
        print(f"untrusted instances skipped: {n_skipped}")
    names = sorted(frame_pos)
    positions = np.stack([np.mean(frame_pos[n], axis=0) for n in names])
    poses = np.stack([frame_pose[n][0] for n in names])
    np.savez_compressed(out / "global_cameras.npz",
                        names=np.array(names), positions=positions,
                        poses=poses)
    pts = np.concatenate(all_pts)
    rgb = np.concatenate(all_rgb)
    from recon3d.core.io_ply import write_ply
    write_ply(out / "global_points.ply", pts, rgb)
    (out / "chain_log.json").write_text(json.dumps(log, indent=1))
    print(f"-> {out} ({len(pts):,} pts, {len(names)} frames)")
    return out


def main(paths: Paths, argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks-root", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--chunk-range", default=None,
                    help="'a:b' 使用チャンク範囲 (階段を挟まない区間評価用)")
    ap.add_argument("--yaw-lock", action="store_true",
                    help="方位ロック: 連結回転を重力+一定ヨーに固定 (s,tのみICP)")
    args = ap.parse_args(argv)
    root = Path(args.chunks_root)
    if not root.is_absolute():
        root = paths.results / args.chunks_root
    cdirs = sorted(root.glob("chunk_*"), key=lambda p: int(p.name.split("_")[1]))
    if args.chunk_range:
        a, b = (int(v) for v in args.chunk_range.split(":"))
        cdirs = cdirs[a:b]
    print(f"{len(cdirs)} chunks from {root}")
    chain_and_save(paths, cdirs, args.tag, yaw_lock=args.yaw_lock)

