"""チャンク単位ポーズグラフ (ループ閉合つき): 逐次連結のドリフトを再訪拘束で畳む

逐次連結 (core/chaining) は誤差を一本道に累積する (実測: 130フレーム以下で
0.03〜0.27m、664フレームで1.3m)。一方で**各エッジ単体は非常に良い**:
  - チャンク単体のATE 0.009m
  - 隣接ICP残差 3〜10cm
  - 共有フレームから測る相対ヨーのばらつき 0.13〜0.17° (ICP解とも0.2°で一致)
つまり悪いのは「積む」構造だけ。

重要: **逐次エッジだけのポーズグラフは逐次連結と同じ解になる** (木構造=冗長性ゼロ)。
改善は再訪 (ループ) 拘束が入って初めて生じる。本モジュールはそれを入れる:

  変数 (チャンクkごと5自由度): log scale a_k, 重力軸まわりヨー ψ_k, 並進 t_k ∈ R³
    (各チャンクを重力整列し回転はヨーのみに縮退。方位ロック撮影と整合)
  残差:
    ① 逐次エッジ: 共有フレームのカメラ位置一致 + 共有フレーム由来の相対ヨー
    ② ループエッジ: 時間的に離れた (|j−k|≧LOOP_MIN_GAP) が空間的に近いチャンク対を
       現在解から候補に取り、点群ICPで検証して採用した拘束
  ゲージ: チャンク0を固定

なおIMUヨーは使えない: 撮影がFlowState方位ロックのため**画像のヨーは世界基準で固定**
(チャンク内ヨーstd 0.4〜1.3°で実証) され、IMUが測る体の物理回転とは別物
(両者の差 median 12〜14° ≒ 実際の体回転量そのもの)。方位ロックの恩恵は
①の視覚相対ヨーとして既に取り込まれている。

使い方:
  IMG_SESSION=0925 python -m recon3d.core.posegraph --from-tag pv0925_s002
  IMG_SESSION=0616 python -m recon3d.core.posegraph --chunks 0 1 2 3 --tag pg_test
出力: frontend_bench/<tag>/ (chaining と同契約)
"""
import argparse
import json

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from recon3d.config import BENCH_DIR, RESULTS_DIR
from recon3d.core.chaining import load_chunk, trimmed_icp_sim3
from recon3d.core.geometry import rot_y, rotation_to_y, sim3_umeyama
from recon3d.core.io_ply import read_ply_xyzrgb, write_ply

PLY_STRIDE = 8
YAW_W = 20.0          # ヨー残差[rad]→位置残差[m]相当の重み
SCALE_W = 5.0         # スケール連続性 log(s_k/s_j) の重み (Sim(3)縮小退化の防止)
LOG_S_BOUND = 0.35    # |log s| の上限 (s ∈ [0.70, 1.42])
LOOP_MIN_GAP = 3      # これ以上チャンク番号が離れた対のみループ候補
LOOP_RADIUS = 2.0     # 候補判定: チャンク重心距離 (カメラ高=1とした相対単位)
LOOP_MAX_RESID = 0.12 # ICP残差がこれ以下なら採用 (隣接の実測3〜10cmが目安)
LOOP_W = 0.7          # ループエッジの重み (逐次エッジより控えめ)


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def chunk_gravity_yaw(poses):
    """チャンクを重力整列し、各フレームのヨー [rad] と重力回転Rgを返す"""
    ups = poses[:, :3, 1]
    up = np.median(ups / np.linalg.norm(ups, axis=1, keepdims=True), axis=0)
    rg = rotation_to_y(up / np.linalg.norm(up))
    fwd = poses[:, :3, 2] @ rg.T
    return np.arctan2(-fwd[:, 2], fwd[:, 0]), rg


def prepare(chunks):
    meta = []
    for names, pos, poses, pf, unt in chunks:
        phi, rg = chunk_gravity_yaw(poses)
        ok = np.array([n not in unt for n in names])
        meta.append({"names": names, "phi": phi, "rg": rg,
                     "pos_g": pos @ rg.T, "ok": ok,
                     "pf_g": {n: v @ rg.T for n, v in pf.items()}})
    return meta


def _yaw_of(R):
    """回転行列の重力軸(y)まわり成分"""
    return float(np.arctan2(R[0, 2], R[0, 0]))


def icp_edge(meta, j, k, kind="seq", w=1.0, T0=None):
    """チャンク j,k の相対Sim(3)を点群ICPで測りエッジ化する (重力整列系)

    返す測定値 (s_jk, yaw_jk, t_jk) は「chunk k の点を chunk j 系へ写す」変換。
    共有カメラ中心4点のUmeyamaだけでは共線で弱いため、chaining と同じく
    per-frame点群の trimmed ICP を使う。
    """
    a, b = meta[j], meta[k]
    sh = [(ia, b["names"].index(n)) for ia, n in enumerate(a["names"])
          if n in b["names"] and a["ok"][ia] and b["ok"][b["names"].index(n)]]
    if T0 is None:
        if len(sh) < 3:
            return None
        ca = np.stack([a["pos_g"][ia] for ia, _ in sh])
        cb = np.stack([b["pos_g"][ib] for _, ib in sh])
        T0 = sim3_umeyama(cb, ca)
    if sh:   # 共有フレームがあればその点群同士 (chainingと同条件)
        src = np.concatenate([b["pf_g"][b["names"][ib]] for _, ib in sh
                              if b["names"][ib] in b["pf_g"]])
        dst = np.concatenate([a["pf_g"][a["names"][ia]] for ia, _ in sh
                              if a["names"][ia] in a["pf_g"]])
    else:    # ループ: チャンク全体の点群
        if not a["pf_g"] or not b["pf_g"]:
            return None
        src = np.concatenate(list(b["pf_g"].values()))
        dst = np.concatenate(list(a["pf_g"].values()))
    if len(src) < 100 or len(dst) < 100:
        return None
    T, resid = trimmed_icp_sim3(src, dst, T0)
    return {"j": j, "k": k, "kind": kind, "w": w,
            "s": float(T[0]), "dpsi": _yaw_of(T[1]), "t": np.asarray(T[2]),
            "resid": round(float(resid), 4)}


def seq_edges(meta):
    """隣接チャンクのICP相対変換をエッジにする"""
    edges = []
    for j in range(len(meta) - 1):
        e = icp_edge(meta, j, j + 1)
        if e is not None:
            edges.append(e)
    return edges


def initial_solution(meta, edges):
    """逐次エッジを順に合成した初期値 (= 従来のチェーン相当)"""
    n = len(meta)
    s, psi, t = np.ones(n), np.zeros(n), np.zeros((n, 3))
    by_k = {e["k"]: e for e in edges if e["kind"] == "seq"}
    for k in range(1, n):
        e = by_k.get(k)
        if e is None:
            s[k], psi[k], t[k] = s[k - 1], psi[k - 1], t[k - 1]
            continue
        j = e["j"]
        s[k] = s[j] * e["s"]
        psi[k] = psi[j] + e["dpsi"]
        t[k] = s[j] * (rot_y(psi[j]) @ e["t"]) + t[j]
    return s, psi, t


def place(meta, s, psi, t, k, pts=None):
    p = meta[k]["pos_g"] if pts is None else pts
    return s[k] * (p @ rot_y(psi[k]).T) + t[k]


def find_loops(meta, s, psi, t, radius=LOOP_RADIUS,
               min_gap=LOOP_MIN_GAP, max_resid=LOOP_MAX_RESID):
    """現在解で空間的に近い非隣接チャンク対をICP検証してループエッジにする"""
    cen = np.stack([place(meta, s, psi, t, k).mean(axis=0)
                    for k in range(len(meta))])
    step = float(np.median([
        np.linalg.norm(np.diff(place(meta, s, psi, t, k), axis=0),
                       axis=1).mean() for k in range(len(meta))]))
    r = radius * max(step, 1e-6) * 6
    tree = cKDTree(cen)
    loops, tried = [], 0
    for j, k in sorted(tree.query_pairs(r)):
        if k - j < min_gap:
            continue
        tried += 1
        # 現在解を初期値に (chunk k 系 → chunk j 系)
        T0 = (s[k] / s[j], rot_y(psi[k] - psi[j]),
              (rot_y(-psi[j]) @ (t[k] - t[j])) / s[j])
        e = icp_edge(meta, j, k, kind="loop", w=LOOP_W, T0=T0)
        if e is None or e["resid"] > max_resid:
            continue
        loops.append(e)
    print(f"loop: 候補 {tried} -> 採用 {len(loops)}"
          + ("  " + ", ".join(f"{e['j']}-{e['k']}({e['resid']:.3f})"
                              for e in loops[:8]) if loops else ""))
    return loops


def solve(meta, edges, yaw_w=YAW_W):
    n = len(meta)

    def unpack(x):
        p = np.concatenate([np.zeros(5), x]).reshape(n, 5)
        return np.exp(p[:, 0]), p[:, 1], p[:, 2:5]

    def resid(x):
        s, psi, t = unpack(x)
        out = []
        for e in edges:
            j, k, w = e["j"], e["k"], e["w"]
            # 整合条件: s_k = s_j·s_jk, ψ_k = ψ_j + Δψ, t_k = s_j·R(ψ_j)·t_jk + t_j
            out.append([w * SCALE_W * (np.log(s[k]) - np.log(s[j])
                                       - np.log(e["s"]))])
            out.append([w * yaw_w * wrap(psi[k] - psi[j] - e["dpsi"])])
            pred = s[j] * (rot_y(psi[j]) @ e["t"]) + t[j]
            out.append(w * (t[k] - pred))
        return np.concatenate(out)

    s0, psi0, t0 = initial_solution(meta, [e for e in edges if e["kind"] == "seq"])
    x0 = np.stack([np.log(s0), psi0, t0[:, 0], t0[:, 1], t0[:, 2]],
                  axis=1)[1:].ravel()
    r0 = float(np.sqrt((resid(x0) ** 2).mean()))
    lo = np.tile([-LOG_S_BOUND, -np.inf, -np.inf, -np.inf, -np.inf], n - 1)
    hi = np.tile([LOG_S_BOUND, np.inf, np.inf, np.inf, np.inf], n - 1)
    x0 = np.clip(x0, lo + 1e-6, hi - 1e-6)
    sol = least_squares(resid, x0, loss="soft_l1", f_scale=0.05, max_nfev=400,
                        bounds=(lo, hi))
    r1 = float(np.sqrt((sol.fun ** 2).mean()))
    print(f"posegraph: {n} chunks, {len(edges)} edges "
          f"(loop {sum(e['kind'] == 'loop' for e in edges)}), "
          f"RMS {r0:.4f} -> {r1:.4f}")
    return unpack(sol.x), r1


def run(cdirs, tag, yaw_w=YAW_W, loops=True):
    chunks = [load_chunk(c) for c in cdirs]
    meta = prepare(chunks)
    edges = seq_edges(meta)
    rs = [e["resid"] for e in edges]
    print(f"seq edges {len(edges)}, ICP残差 median "
          f"{np.median(rs) if rs else float('nan'):.4f}")
    s, psi, t = initial_solution(meta, edges)
    if loops:
        edges += find_loops(meta, s, psi, t)
    (s, psi, t), rms = solve(meta, edges, yaw_w)

    frame_pos, frame_pose = {}, {}
    all_pts, all_rgb = [], []
    for k, cdir in enumerate(cdirs):
        names, _, poses, _, unt = chunks[k]
        R = rot_y(psi[k]) @ meta[k]["rg"]
        g_pos = place(meta, s, psi, t, k)
        for i, nm in enumerate(names):
            if nm in unt:
                continue
            frame_pos.setdefault(nm, []).append(g_pos[i])
            g = np.eye(4)
            g[:3, :3] = R @ poses[i][:3, :3]
            g[:3, 3] = g_pos[i]
            frame_pose.setdefault(nm, []).append(g)
        xyz, rgb = read_ply_xyzrgb(cdir / "points.ply")
        xyz = xyz[::PLY_STRIDE]
        all_pts.append(place(meta, s, psi, t, k,
                             xyz @ meta[k]["rg"].T).astype(np.float32))
        all_rgb.append(rgb[::PLY_STRIDE])

    names = sorted(frame_pos)
    out = BENCH_DIR / tag
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "global_cameras.npz", names=np.array(names),
        positions=np.stack([np.mean(frame_pos[n], axis=0) for n in names]),
        poses=np.stack([frame_pose[n][0] for n in names]))
    write_ply(out / "global_points.ply", np.concatenate(all_pts),
              np.concatenate(all_rgb))
    (out / "chain_log.json").write_text(json.dumps([{
        "mode": "posegraph", "n_chunks": len(cdirs), "n_edges": len(edges),
        "n_loops": sum(e["kind"] == "loop" for e in edges),
        "rms": round(rms, 5), "yaw_w": yaw_w,
        "scale_min": round(float(s.min()), 4), "scale_max": round(float(s.max()), 4),
    }], indent=1))
    print(f"-> {out} ({len(names)} frames, scale {s.min():.3f}〜{s.max():.3f})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag")
    ap.add_argument("--chunks", nargs="*", type=int)
    ap.add_argument("--from-tag", default=None,
                    help="pv_manifest.json の島タグからチャンク列を取る")
    ap.add_argument("--chunks-dir", default="chunks")
    ap.add_argument("--manifest", default="pv_manifest.json",
                    help="--from-tag が読むマニフェスト。--chunks-dir と"
                         "必ず対で指定する (チャンク番号→フレームの対応が"
                         "stride/chunk-size で変わるため)")
    ap.add_argument("--yaw-w", type=float, default=YAW_W)
    ap.add_argument("--no-loops", action="store_true",
                    help="ループ閉合なし (逐次のみ=従来相当の対照実験)")
    args = ap.parse_args()
    pcd = RESULTS_DIR / "pcd" / args.chunks_dir
    ks = args.chunks
    if args.from_tag:
        man = json.loads((BENCH_DIR / args.manifest).read_text())
        ks = next(m["chunks"] for m in man if m["tag"] == args.from_tag)
    cdirs = [pcd / f"chunk_{k:02d}" for k in ks]
    tag = args.tag or (f"pg_{args.from_tag}" if args.from_tag else "pg_test")
    run(cdirs, tag, args.yaw_w, loops=not args.no_loops)


if __name__ == "__main__":
    main()
