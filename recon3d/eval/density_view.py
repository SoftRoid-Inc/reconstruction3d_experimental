"""島の壁密度地図を描く (図面アンカーの入力そのものを目で見る)

`postproc/anchor.prepare_island` が作る density は、重力整列した点群を真上から
見たセル地図で、**各セルの高さ方向の占有スパン**を値にしている
(家具は高さスパンが浅いので落ち、壁だけが残る)。
`postproc/corners.density_corners` はこの地図からL字/T字コーナーを拾い、
図面側のコーナーとRANSAC照合する。つまりここが**アンカーの素材**。

アンカーが外れたとき、原因が「素材(密度地図)が痩せている」のか
「照合・選択」なのかを切り分けるために使う。

図の見方:
  グレースケール = 密度 (濃いほど壁)
  赤点 = 検出されたコーナー (L字/T字)
  青線 = カメラ軌跡 (同じセル座標)

使い方:
  IMG_SESSION=0912 python -m recon3d.eval.density_view
出力: <物件>/results/<sess>/backend/density_islands.png
"""
import argparse
import json

import numpy as np

from recon3d.config import BENCH_DIR, OUT_DIR, SESSION
from recon3d.postproc.anchor import prepare_island
from recon3d.postproc.corners import DENS_THR, density_corners


def island_density(tag):
    """(density地図, コーナー一覧, カメラ軌跡[セル], 支配方位deg)"""
    d = prepare_island(tag)
    zz, xx = np.nonzero(d["density"] > DENS_THR)
    cs, deg = density_corners(np.stack([zz, xx], axis=1).astype(np.float64))
    cam = np.stack(
        [(d["cam_pos_aligned"][:, 2] - d["origin"][1]) / d["cell_units"],
         (d["cam_pos_aligned"][:, 0] - d["origin"][0]) / d["cell_units"]],
        axis=1)
    return d["density"], cs, cam, deg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="pv_full_manifest.json")
    ap.add_argument("--islands", nargs="*", default=None)
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]

    man = json.loads((BENCH_DIR / args.manifest).read_text())
    tags = [m for m in man
            if not args.islands or m["tag"] in args.islands]
    n = len(tags)
    nc = min(3, n)
    nr = -(-n // nc)
    fig, axes = plt.subplots(nr, nc, figsize=(6.0 * nc, 5.4 * nr),
                             squeeze=False)
    for ax, m in zip(axes.ravel(), tags):
        try:
            dens, cs, cam, deg = island_density(m["tag"])
        except Exception as e:
            ax.set_title(f"{m['tag']}: 失敗 ({type(e).__name__})", fontsize=9)
            ax.axis("off")
            continue
        ax.imshow(dens, cmap="gray_r", origin="upper", vmin=0, vmax=1)
        if len(cs):
            r = [c["r"] for c in cs]
            c = [c["c"] for c in cs]
            ax.scatter(c, r, s=22, c="#e31a1c", zorder=4,
                       label=f"コーナー {len(cs)}")
        ax.plot(cam[:, 1], cam[:, 0], "-", lw=1.2, color="#1f77b4",
                alpha=0.85, zorder=3, label="カメラ軌跡")
        ax.legend(fontsize=8, loc="upper right", framealpha=0.85)
        ax.set_title(f"{m['tag']}  {m.get('floor', '?')}  "
                     f"{m['b'] - m['a'] + 1}f  支配方位 {deg:.0f}°",
                     fontsize=10)
        ax.tick_params(labelsize=6)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(f"{SESSION} 島ごとの壁密度地図 "
                 f"(濃い=壁 / 赤=検出コーナー / 青=カメラ軌跡)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = OUT_DIR / "density_islands.png"
    fig.savefig(out, dpi=130)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
