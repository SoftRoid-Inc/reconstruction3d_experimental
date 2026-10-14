"""PanoVGGT / Pi3 / Pi3-FT の軌跡を窓ごとに重ねて描く (モデル比較の可視化)

各窓で3モデルをGTへSim(2)フィットし、GTと3本を同じ図に重ねる。
「3モデルは互いに一致しGTとだけ食い違う」ことが一目で分かるのが狙い。
GT点は**フレーム間速度**で色分けし、物理的に無理な区間 (屋内歩行で1.5 m/s超) を
赤で示す — そこが残差の出どころ。

セッションは物件をまたぐ (0828=28-AAAAA / 他=28-BBBBB) ので、スケールは
物件ごとに持つ。configの凍結済み定数は使わない。

使い方:
  python -m img_align.eval.model_plot                  # 全窓を1枚に
  python -m img_align.eval.model_plot --session 0912   # セッション限定
出力: 28-BBBBB/results/model_bench/model_bench{,_pcd}.png
"""
import argparse

import numpy as np

from img_align.config import PROPERTY
from img_align.core.geometry import rotation_to_y
from img_align.eval.chunk_cmp import BREAKDOWN, CONTROL, gt_interp
from img_align.eval.frontend import fit2d
from img_align.eval.gt_traj import load_gt

DT = 0.5              # フレーム間隔[s] (実測で厳密に0.500s)
WALK_MAX = 1.5        # 屋内歩行の速度上限の目安 [m/s]
MM = {"28-AAAAA": 11.968, "28-BBBBB": 12.021969168662748}
COLORS = {"pv": "#1f77b4", "pi3": "#2ca02c", "holo": "#d62728"}
LABEL = {"pv": "PanoVGGT", "pi3": "Pi3", "holo": "Pi3-FT(Holo360D)"}


def horiz(sess, tag):
    """cameras.npz → (frames, 重力で水平化した平面座標)"""
    d = np.load(PROPERTY[sess].root / "results" / sess / "pcd" / tag
                / "cameras.npz")
    fr = np.array([int(str(n).replace("frame_", "")) for n in d["names"]])
    P = d["poses"]
    u = P[:, :3, 1]
    u = np.median(u / np.linalg.norm(u, axis=1, keepdims=True), 0)
    return fr, (P[:, :3, 3] @ rotation_to_y(u / np.linalg.norm(u)).T)[:, [0, 2]]


def fit_to_gt(p, g):
    """鏡像も試してGTへ相似フィット → (フィット後座標, 残差)"""
    best = None
    for m in (1, -1):
        x = p.copy()
        x[:, 1] *= m
        y = fit2d(x, g)(x)
        e = np.linalg.norm(y - g, axis=1)
        if best is None or np.median(e) < np.median(best[1]):
            best = (y, e)
    return best


def panel(ax, sess, a, note):
    mm = MM[PROPERTY[sess].root.name]
    gt = load_gt(sess)
    fr, _ = horiz(sess, f"pv_{sess}_c{a}")
    g, idx = gt_interp(gt, fr, mm)
    if len(idx) < 4:
        ax.set_title(f"{sess} c{a}: GT不足", fontsize=8)
        return None

    # GTの速度 (補間点の間隔はフレーム差で正規化)
    df = np.diff(fr[idx]) * DT
    vel = np.r_[np.nan, np.linalg.norm(np.diff(g, axis=0), axis=1) / df]
    ax.plot(g[:, 0], g[:, 1], "-", color="0.55", lw=3, zorder=1, label="GT")
    bad = vel > WALK_MAX
    ax.scatter(g[:, 0], g[:, 1], s=14, zorder=4,
               c=np.where(bad, "#e31a1c", "0.35"))

    meds = {}
    for m in ("pv", "pi3", "holo"):
        try:
            p = horiz(sess, f"{m}_{sess}_c{a}")[1][idx]
        except FileNotFoundError:
            continue
        y, e = fit_to_gt(p, g)
        meds[m] = float(np.median(e))
        ax.plot(y[:, 0], y[:, 1], "-o", ms=2.4, lw=1.1, color=COLORS[m],
                label=LABEL[m], zorder=3)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=6)
    ttl = " / ".join(f"{m} {v:.3f}" for m, v in meds.items())
    ax.set_title(f"{sess} f{a}-{a + 16}  {note}\n{ttl}  [m]", fontsize=7.5)
    if bad.any():
        ax.text(0.02, 0.02, f"GT速度 max {np.nanmax(vel):.1f} m/s",
                transform=ax.transAxes, fontsize=7, color="#e31a1c")
    return meds


def pcd_panel(ax, sess, a, model, name):
    """点群を真上から描く。スケールはモデル毎に非メトリックなので
    カメラ移動距離で正規化して形だけを比べられるようにする"""
    from img_align.core.io_ply import read_ply_xyzrgb
    cdir = PROPERTY[sess].root / "results" / sess / "pcd" / f"{model}_{sess}_c{a}"
    d = np.load(cdir / "cameras.npz")
    P = d["poses"]
    u = P[:, :3, 1]
    u = np.median(u / np.linalg.norm(u, axis=1, keepdims=True), 0)
    R = rotation_to_y(u / np.linalg.norm(u))
    cam = P[:, :3, 3] @ R.T
    span = float(np.linalg.norm(np.diff(cam[:, [0, 2]], axis=0), axis=1).sum())
    xyz, rgb = read_ply_xyzrgb(cdir / "points.ply")
    step = max(1, len(xyz) // 400_000)
    p, col = xyz[::step] @ R.T, rgb[::step] / 255.0
    h = p[:, 1] - np.median(cam[:, 1])       # 床〜天井だけ残して霞を落とす
    k = (h > -2.0 * span) & (h < 1.2 * span)
    c0 = cam[:, [0, 2]].mean(0)
    ax.scatter((p[k, 0] - c0[0]) / span, (p[k, 2] - c0[1]) / span,
               s=0.07, c=col[k], marker=".", lw=0)
    ax.plot((cam[:, 0] - c0[0]) / span, (cam[:, 2] - c0[1]) / span,
            "-o", ms=3, lw=1.2, color="#e31a1c")
    ax.set_aspect("equal")
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.tick_params(labelsize=6)
    ax.set_title(f"{sess} f{a}-{a + 16}  {name}\n{len(xyz) / 1e6:.1f}M点",
                 fontsize=8)


def plot_pcd(wins, out):
    import matplotlib.pyplot as plt
    models = [("pv", "PanoVGGT"), ("pi3", "Pi3"), ("holo", "Pi3-FT(Holo360D)")]
    fig, axes = plt.subplots(len(wins), 3, figsize=(13, 4.4 * len(wins)),
                             squeeze=False)
    for r, (sess, a, _, _) in enumerate(wins):
        for c, (m, name) in enumerate(models):
            try:
                pcd_panel(axes[r][c], sess, a, m, name)
            except FileNotFoundError:
                axes[r][c].axis("off")
    fig.suptitle("点群の見た目比較 (真上から、赤=カメラ軌跡)\n"
                 "スケールはカメラ移動距離で正規化 — 壁の二重化=姿勢の不整合",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=130)
    print(f"-> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default=None)
    ap.add_argument("--pcd", action="store_true", help="軌跡でなく点群を描く")
    ap.add_argument("--windows", nargs="*", type=int, default=None,
                    help="--pcd時に描く窓の開始フレーム (既定: 破綻窓の代表2本)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]

    wins = [w for w in BREAKDOWN + CONTROL
            if args.session is None or w[0] == args.session]
    wins.sort()
    from img_align.tools.model_bench import bench_dir
    root = bench_dir()
    if args.pcd:
        sel = args.windows or [472, 632]
        plot_pcd([w for w in wins if w[1] in sel],
                 args.out or root / "model_bench_pcd.png")
        return
    nc = 3 if len(wins) > 4 else len(wins)
    nr = -(-len(wins) // nc)
    fig, axes = plt.subplots(nr, nc, figsize=(4.2 * nc, 3.8 * nr))
    axes = np.atleast_1d(axes).ravel()
    for ax, (sess, a, _, note) in zip(axes, wins):
        panel(ax, sess, a, note)
    for ax in axes[len(wins):]:
        ax.axis("off")
    axes[0].legend(fontsize=7, loc="best")
    fig.suptitle("窓ごとの軌跡比較 — 3モデルは互いに一致し、GT(灰)とだけ食い違う"
                 "  / 赤点=GT速度が歩行上限1.5m/s超", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = args.out or root / ("model_bench.png" if not args.session
                              else f"model_bench_{args.session}.png")
    fig.savefig(out, dpi=130)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
