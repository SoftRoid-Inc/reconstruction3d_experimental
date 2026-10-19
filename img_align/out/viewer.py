"""aligned_house.ply の確認画像とブラウザビューアの生成

- aligned_house_check.png: 1F/2F の真上ビュー (点群RGB + 図面壁中心線=シアン)
- viewer.html: WebGL単体ファイルビューア (uint16量子化 + base64埋め込み)

使い方: python -m img_align.out.viewer   (export_aligned2 の後に実行)
"""
import base64
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from img_align.config import BENCH_DIR
from img_align.eval.gt_traj import mm_per_px
from img_align.core.io_ply import read_ply_xyzrgb
from img_align.postproc.plan_icp import wall_segments_px

from img_align.config import SESSION

OUT_DIR = BENCH_DIR.parent / "backend"
SFX = "" if SESSION == "0828" else f"_{SESSION}"   # セッション別の入出力名
VIEWER_MAX_PTS = 550_000
FLOOR_SPLIT_M = 2.5   # これ未満=1F表示 / 以上=2F表示 (2F床は+2.9m)


def render_check(xyz, rgb, sfx=SFX):
    fig, axes = plt.subplots(1, 2, figsize=(20, 11))
    for ax, floor, (h0, h1) in ((axes[0], "1F", (-0.3, FLOOR_SPLIT_M)),
                                (axes[1], "2F", (FLOOR_SPLIT_M, 6.0))):
        m = (xyz[:, 2] >= h0) & (xyz[:, 2] < h1)
        p = xyz[m][::2]
        c = rgb[m][::2] / 255.0
        ax.scatter(p[:, 0], p[:, 1], s=0.3, c=c, linewidths=0)
        segs = wall_segments_px(floor) * mm_per_px() / 1000
        for x1, y1, x2, y2 in segs:
            ax.plot([x1, x2], [y1, y2], "-", color="cyan", lw=1.5, alpha=0.9)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_title(f"{floor} top view  (cyan = GT wall centerlines)")
        ax.set_facecolor("#101014")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"aligned_house_check{sfx}.png", dpi=110,
                facecolor="#181820")
    plt.close(fig)
    print(f"-> {OUT_DIR / f'aligned_house_check{sfx}.png'}")


def build_viewer(xyz, rgb, sfx=SFX):
    if len(xyz) > VIEWER_MAX_PTS:
        step = len(xyz) // VIEWER_MAX_PTS + 1
        xyz, rgb = xyz[::step], rgb[::step]
    lo = xyz.min(axis=0)
    hi = xyz.max(axis=0)
    q = np.round((xyz - lo) / (hi - lo) * 65535).astype(np.uint16)
    blob = q.tobytes() + rgb.astype(np.uint8).tobytes()
    b64 = base64.b64encode(blob).decode()
    meta = {"n": len(xyz), "lo": lo.tolist(), "hi": hi.tolist()}
    tpl = (Path(__file__).resolve().parent / "viewer_template.html").read_text()
    html = tpl.replace(
        "/*DATA*/", f"const META={json.dumps(meta)};\nconst B64=\"{b64}\";")
    html = html.replace("508k", f"{len(xyz) // 1000}k")
    (OUT_DIR / f"viewer{sfx}.html").write_text(html)
    print(f"-> {OUT_DIR / f'viewer{sfx}.html'} ({len(xyz):,} pts)")


def main(sfx=SFX):
    xyz, rgb = read_ply_xyzrgb(OUT_DIR / f"aligned_house{sfx}.ply")
    render_check(xyz, rgb, sfx)
    build_viewer(xyz, rgb, sfx)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sfx", default=SFX,
                    help="入出力接尾辞 (aligned_house<sfx>.ply → viewer<sfx>.html)")
    args = ap.parse_args()
    main(args.sfx)
