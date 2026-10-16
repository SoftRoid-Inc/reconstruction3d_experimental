"""島の点群を重力整列して viewer HTML にする (鉛直被覆の目視確認用)

図面アンカーの密度地図は「鉛直に長いもの=壁」で作るが、0925では**セルの
spanの中央値が0.6〜1.0m**しかなく、天井まで届くセルは0〜7%だった
(2026-10-16実測)。壁が縦に再構成されていないと、この判定が成立しない。
素の点群を見て、どこまで伸びているかを目で確かめるためのツール。

`prepare_island` と同じ整列 (重力整列 + z反転) を通すので、**y が物理的な
上下**になる (y+が下。PanoVGGTのカメラY軸がequirect由来で下向きのため)。
色は既定で高さ (床=青 → 天井=赤)。--rgb で元の色に戻せる。

使い方:
  IMG_SESSION=0925 python -m img_align.eval.island_viewer --islands pv0925_s003r2
出力: <物件>/results/<sess>/backend/island_viewer_<tag>.html
"""
import argparse
import json

import numpy as np

from img_align.config import BENCH_DIR, OUT_DIR
from img_align.out.viewer import VIEWER_MAX_PTS, build_viewer
from img_align.postproc.anchor import prepare_island
from img_align.postproc.scale_prior import m_per_unit


def height_rgb(y):
    """高さ → 色 (低い=青 / 中間=緑 / 高い=赤)

    床/天井のy値は座標規約が紛らわしい (prepare_island の "floor_y" は
    実は物理天井側) ので、**点群自身の1〜99パーセンタイル**で正規化する。
    規約に依存せず、見たいもの (縦にどこまで伸びているか) がそのまま出る。
    """
    lo, hi = np.percentile(y, [1, 99])
    h = np.clip((y - lo) / max(hi - lo, 1e-6), 0, 1)
    rgb = np.zeros((len(y), 3), np.uint8)
    rgb[:, 0] = (255 * np.clip(h * 1.6 - 0.6, 0, 1)).astype(np.uint8)
    rgb[:, 1] = (255 * np.clip(1.6 - np.abs(h - 0.5) * 3.2, 0, 1)).astype(np.uint8)
    rgb[:, 2] = (255 * np.clip(1.0 - h * 1.6, 0, 1)).astype(np.uint8)
    return rgb


def export(tag, use_rgb=False, max_pts=VIEWER_MAX_PTS):
    mpu, _ = m_per_unit(tag)
    d = prepare_island(tag, mpu=mpu)
    pts = d["density_pts"]            # スラブ除去+バウンディング済み (密度の素材)
    full = np.load(BENCH_DIR / tag / "global_cameras.npz")
    k = max(1, len(pts) // max_pts)
    w = pts[::k]
    # メートルに直す (以降の寸法読みが素直になる)
    xyz = w * mpu
    cam_y = float(np.median(d["cam_pos_aligned"][:, 1])) * mpu
    floor_y = d["floor_y_units"] * mpu
    span_m = float(np.percentile(xyz[:, 1], 99) - np.percentile(xyz[:, 1], 1))
    print(f"{tag}: {len(w):,}点 (全{len(pts):,})  "
          f"カメラy={cam_y:.2f}m 床y={floor_y:.2f}m 点群の高さ幅={span_m:.2f}m")
    # **ビューアは Z上向き** (viewer_template.html は up=(0,0,1)、高さスライダーも
    # lo[2]/hi[2]、シェーダの高さ色も aP.z)。整列系は y+ が物理下向きなので
    # (x, z, -y) に並べ替える。ここを間違えると「真上から」が横から見える
    out = np.stack([xyz[:, 0], xyz[:, 2], -xyz[:, 1]], axis=1)
    rgb = (np.full((len(w), 3), 180, np.uint8) if use_rgb
           else height_rgb(out[:, 2]))
    build_viewer(out, rgb, sfx=f"_island_{tag}")
    src = OUT_DIR / f"viewer_island_{tag}.html"
    print(f"-> {src}")
    return src, {"tag": tag, "n_pts": int(len(pts)), "cam_y_m": cam_y,
                 "floor_y_m": floor_y, "height_span_m": span_m,
                 "n_frames": int(len(full["names"]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--islands", nargs="*", default=None)
    ap.add_argument("--manifest", default="pv_manifest.json")
    ap.add_argument("--rgb", action="store_true", help="高さ色でなく無彩色")
    args = ap.parse_args()
    man = json.loads((BENCH_DIR / args.manifest).read_text())
    info = []
    for m in man:
        if args.islands and m["tag"] not in args.islands:
            continue
        try:
            _, i = export(m["tag"], args.rgb)
            info.append(i)
        except Exception as e:
            print(f"{m['tag']}: 失敗 ({type(e).__name__}: {e})")
    if info:
        p = OUT_DIR / "island_viewer_summary.json"
        p.write_text(json.dumps(info, ensure_ascii=False, indent=1))
        print(f"-> {p}")


if __name__ == "__main__":
    main()
