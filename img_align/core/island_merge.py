"""島間の点群ICP接続: 共通重力で5自由度に落とし、ヨーを総当たりして繋ぐ

島は階段や破綻点で切るため共有フレームを持たず、従来は幾何的に完全独立だった
(拘束ゼロ)。そのため各島はフロアの一部しか覆わず、図面アンカーが部分一致の
偽解を拾う主因になっていた (2026-10-08の診断)。

FlowState撮影の性質を測った結果:
  - 水平化は保たれ、**重力軸はチャンク間で共通** (実測 0.7〜2.6°)
  - ヨーのゲージはチャンク/島ごとに任意 (std 34〜47°)
したがって島間の未知数は (ヨー, 並進3, スケール) の **5自由度**に厳密に落ちる。
7自由度の盲目ICPは初期値なしでは収束しないが、5自由度ならヨーを総当たりできる。

手順: セッション共通重力で全島を整列 → 島ペアごとにヨー36方向×粗ICP →
最良をtrimmed ICPで精密化 → 残差とインライア率で採否 → 連結成分を同一階グループに。

使い方:
  IMG_SESSION=0912 python -m img_align.core.island_merge            # 全島ペア
  IMG_SESSION=0912 python -m img_align.core.island_merge --check    # 判定のみ
出力: frontend_bench/island_links.json (ペアごとの残差と採否、グループ分け)
"""
import argparse
import itertools
import json

import numpy as np
from scipy.spatial import cKDTree

from img_align.config import BENCH_DIR
from img_align.core.chaining import trimmed_icp_sim3
from img_align.core.geometry import rot_y, rotation_to_y
from img_align.core.io_ply import read_ply_xyzrgb

YAW_STEPS = 36        # ヨー総当たりの分割数 (10°刻み)
SUB = 4000            # ICPに使う点数 (粗探索用のサブサンプル)
MAX_RESID = 0.15      # 採用する残差 (島内チャンク接続の実測3〜10cmが目安)
MIN_INLIER = 0.35     # 対応点のうち残差2倍以内に入る割合
MIN_PTS = 2000


def session_gravity(tags):
    """全島のカメラupから1つの重力軸を決める (チャンク間で共通なことを実証済み)"""
    ups = []
    for t in tags:
        d = np.load(BENCH_DIR / t / "global_cameras.npz")
        u = d["poses"][:, :3, 1]
        ups.append(np.median(u / np.linalg.norm(u, axis=1, keepdims=True),
                             axis=0))
    g = np.median(np.stack(ups), axis=0)
    return rotation_to_y(g / np.linalg.norm(g))


def island_cloud(tag, rg, sub=SUB):
    """重力整列した島の点群 (間引き) とカメラ位置"""
    xyz, _ = read_ply_xyzrgb(BENCH_DIR / tag / "global_points.ply")
    d = np.load(BENCH_DIR / tag / "global_cameras.npz")
    p = xyz @ rg.T
    if len(p) > sub:
        p = p[:: len(p) // sub + 1]
    return p, d["positions"] @ rg.T


def inlier_frac(src, dst, T, tol):
    s, R, t = T
    q = (s * (src @ R.T)) + t
    d, _ = cKDTree(dst).query(q, k=1)
    return float((d < tol).mean())


def try_link(a_pts, b_pts, yaw_steps=YAW_STEPS):
    """b を a へ合わせる5自由度ICP。ヨー総当たり→最良を精密化"""
    if len(a_pts) < MIN_PTS or len(b_pts) < MIN_PTS:
        return None
    ca, cb = a_pts.mean(axis=0), b_pts.mean(axis=0)
    best = None
    for i in range(yaw_steps):
        th = 2 * np.pi * i / yaw_steps
        R = rot_y(th)
        T0 = (1.0, R, ca - (R @ cb))
        T, resid = trimmed_icp_sim3(b_pts, a_pts, T0)
        if best is None or resid < best[1]:
            best = (T, resid)
    T, resid = best
    frac = inlier_frac(b_pts, a_pts, T, max(2 * resid, 0.05))
    return {"s": float(T[0]), "yaw_deg": float(np.degrees(
        np.arctan2(T[1][0, 2], T[1][0, 0]))), "t": [float(v) for v in T[2]],
        "resid": round(float(resid), 4), "inlier": round(frac, 3)}


def groups_from_links(tags, links):
    """採用リンクの連結成分 = 同一階グループ"""
    parent = {t: t for t in tags}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for l in links:
        if l["accept"]:
            ra, rb = find(l["a"]), find(l["b"])
            if ra != rb:
                parent[ra] = rb
    g = {}
    for t in tags:
        g.setdefault(find(t), []).append(t)
    return list(g.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="pv_manifest.json")
    ap.add_argument("--tags", nargs="*", default=None)
    ap.add_argument("--max-resid", type=float, default=MAX_RESID)
    ap.add_argument("--yaw-steps", type=int, default=YAW_STEPS)
    args = ap.parse_args()
    tags = args.tags
    if not tags:
        man = json.loads((BENCH_DIR / args.manifest).read_text())
        tags = [m["tag"] for m in man]
    rg = session_gravity(tags)
    clouds = {t: island_cloud(t, rg) for t in tags}
    links = []
    for a, b in itertools.combinations(tags, 2):
        r = try_link(clouds[a][0], clouds[b][0], args.yaw_steps)
        if r is None:
            continue
        r["a"], r["b"] = a, b
        r["accept"] = bool(r["resid"] <= args.max_resid
                           and r["inlier"] >= MIN_INLIER)
        links.append(r)
        print(f"  {a:20s} - {b:20s} resid {r['resid']:.4f} "
              f"inlier {r['inlier']:.2f} yaw {r['yaw_deg']:+7.1f}° "
              f"s {r['s']:.3f} {'採用' if r['accept'] else ''}")
    grp = groups_from_links(tags, links)
    print("グループ:")
    for g in grp:
        print("  " + ", ".join(sorted(g)))
    out = {"links": links, "groups": grp}
    (BENCH_DIR / "island_links.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False))
    print(f"-> {BENCH_DIR / 'island_links.json'}")


if __name__ == "__main__":
    main()
