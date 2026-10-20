"""セッション俯瞰: パイプラインが出した全島軌跡を色分けしてGTに重ね、実性能を集計する

各島を独立に相似フィット (eval/frontend と同条件: 重力整列→鏡映2択×2D相似) して
1枚の俯瞰図に描く。島=1本の軌跡を色で見分けられるようにし、凡例に島ごとの
フィットATE medianを添える。GTは灰色の実線。

集計 (islands_overview.json):
  - 島ごとの median / p90 / フレーム数
  - セッション合算 (全島の誤差をプール) の median / p90
  - カバレッジ = 島がカバーするGTフレームの割合

使い方: python main.py tool overview --input /data/x [--manifest pv_manifest.json]
出力: frontend_bench/islands_overview.png / islands_overview.json
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from recon3d.core.geometry import rotation_to_y
from recon3d.eval.frontend import fit2d
from recon3d.eval.gt_traj import mm_per_px, load_gt


def fit_island(paths, tag, gt):
    """島をGTへ相似フィット。(fit後xy[used], gt_xy, err, frames[used]) を返す"""
    d = np.load(paths.bench / tag / "global_cameras.npz")
    names = [str(n) for n in d["names"]]
    frames = np.array([int(n.replace("frame_", "")) for n in names])
    pos, poses = d["positions"], d["poses"]
    ups = poses[:, :3, 1]
    up = np.median(ups / np.linalg.norm(ups, axis=1, keepdims=True), axis=0)
    p = pos @ rotation_to_y(up / np.linalg.norm(up)).T
    # GTは線形補間 (最近傍スナップはGT欠損区間でATEを水増しする)
    from recon3d.eval.chunk_cmp import gt_interp
    gt_xy, used = gt_interp(gt, frames, mm_per_px(paths))
    if len(used) < 4:
        return None
    best = None
    for mirror in (False, True):
        xz = p[:, [0, 2]].copy()
        if mirror:
            xz[:, 1] *= -1
        T = fit2d(xz[used], gt_xy)
        e = np.linalg.norm(T(xz[used]) - gt_xy, axis=1)
        if best is None or np.median(e) < best[0]:
            best = (float(np.median(e)), T(xz[used]), e)
    med, xy, err = best
    return xy, gt_xy, err, frames[used]


def main(paths, argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="pv_manifest.json")
    ap.add_argument("--out", default="islands_overview")
    args = ap.parse_args(argv)
    gt = load_gt(paths)
    manifest = json.loads((paths.bench / args.manifest).read_text())

    gt_sorted = np.array([[gt[f]["x_px"] * mm_per_px(paths) / 1000,
                           gt[f]["y_px"] * mm_per_px(paths) / 1000]
                          for f in sorted(gt)])
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.plot(gt_sorted[:, 0], gt_sorted[:, 1], "-", color="0.65", lw=3,
            alpha=0.8, label="GT (manual)", zorder=1)

    cmap = plt.cm.tab10 if len(manifest) <= 10 else plt.cm.tab20
    rows, pooled, covered = [], [], set()
    for i, m in enumerate(sorted(manifest, key=lambda m: m["a"])):
        r = fit_island(paths, m["tag"], gt)
        if r is None:
            rows.append({"tag": m["tag"], "note": "GT対応フレーム不足"})
            continue
        xy, _, err, fr = r
        med, p90 = float(np.median(err)), float(np.percentile(err, 90))
        pooled.extend(err.tolist())
        covered.update(int(f) for f in fr)
        c = cmap(i % cmap.N)
        ax.plot(xy[:, 0], xy[:, 1], "-", color=c, lw=1.0, alpha=0.7, zorder=2)
        ax.scatter(xy[:, 0], xy[:, 1], s=10, color=c, zorder=3,
                   label=f"{m['tag']} ({m['a']}-{m['b']})  {med:.2f}m")
        rows.append({"tag": m["tag"], "a": m["a"], "b": m["b"],
                     "n": int(len(err)), "ate_median_m": round(med, 3),
                     "ate_p90_m": round(p90, 3)})

    # GTフレームのうち、島の±3フレーム以内でカバーされた割合
    cov = (sum(1 for f in gt if any(abs(f - c) <= 3 for c in covered))
           / max(len(gt), 1)) if covered else 0.0
    pooled = np.array(pooled)
    summary = {
        "session": paths.name, "n_islands": len(manifest),
        "pooled_median_m": round(float(np.median(pooled)), 3) if len(pooled) else None,
        "pooled_p90_m": round(float(np.percentile(pooled, 90)), 3) if len(pooled) else None,
        "gt_coverage": round(cov, 3), "islands": rows,
    }
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_title(f"session {paths.name}: {len(manifest)} islands  "
                 f"pooled median {summary['pooled_median_m']}m / "
                 f"p90 {summary['pooled_p90_m']}m / GT coverage {cov:.0%}")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)
    fig.savefig(paths.bench / f"{args.out}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    (paths.bench / f"{args.out}.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=1, ensure_ascii=False))

