"""手法Aのラベルマップから仮想2Dスキャンを作る (スパイク実装)

各方位ビンで壁-床境界の俯角を検出し、カメラ高さの仮定値から水平距離に変換する
(擬似LiDAR)。各レイには意味タグ(壁 / 開口=door∪window)を付け、
撮影者(person)が下半分に写る方位は無効化する。
"""
import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from img_align.config import GROUP_IDS, RESULTS_DIR
from img_align.legacy.sample_frames import load_selected

SCAN_DIR = RESULTS_DIR / "scan"

CAM_HEIGHT_M = 1.5        # カメラ高さの仮定値 (要検証パラメータ)
CEIL_HEIGHT_M = 2.4       # 天井高の仮定値 (recon3d と同じ。勾配天井では崩れる)
N_BINS = 256              # 方位ビン数
MIN_ANGLE_DEG = 4.0       # これ未満の俯角/仰角は遠すぎとして無効
MIN_RUN = 15              # 床/天井と認める最小連続ピクセル数
TAG_WALL, TAG_OPENING = 1, 2


def longest_run(col, gid):
    """列内で gid の最長連続ラン (start, end) を返す。無ければ None"""
    idx = np.flatnonzero(col == gid)
    if idx.size < MIN_RUN:
        return None
    runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    run = max(runs, key=len)
    if len(run) < MIN_RUN:
        return None
    return int(run[0]), int(run[-1])


def ray_tag(band):
    """境界近傍の帯の多数決で 壁/開口 を判定する"""
    opening = np.isin(band, [GROUP_IDS["window"], GROUP_IDS["door"]]).sum()
    wallish = np.isin(band, [GROUP_IDS["wall"], GROUP_IDS["column"]]).sum()
    if opening == 0 and wallish == 0:
        return TAG_WALL
    return TAG_OPENING if opening > wallish else TAG_WALL


def scan_frame(group_map):
    """ラベルマップ (H,W) → 床由来/天井由来の2系統のスキャン (無効はdist=nan)

    床由来: 壁-床境界の俯角 + カメラ高さ → 距離。家具・撮影者に遮られやすい
    天井由来: 天井-壁境界の仰角 + (天井高-カメラ高) → 距離。遮蔽に強いが
              レバーアームが短く角度誤差に敏感、勾配天井で崩れる
    """
    h, w = group_map.shape
    az = (np.arange(N_BINS) + 0.5) / N_BINS * 2 * math.pi
    dist_f = np.full(N_BINS, np.nan)
    tag_f = np.zeros(N_BINS, dtype=np.int8)
    dist_c = np.full(N_BINS, np.nan)
    tag_c = np.zeros(N_BINS, dtype=np.int8)

    for b in range(N_BINS):
        u = int((b + 0.5) / N_BINS * w)
        col = group_map[:, u]

        # 床由来 (下半分に撮影者が写る方位は捨てる)
        lower = col[h // 2:]
        if (lower == GROUP_IDS["person"]).sum() <= 0.08 * lower.size:
            run = longest_run(col, GROUP_IDS["floor"])
            if run is not None:
                vb = run[0]
                depression = math.pi * (vb + 0.5) / h - math.pi / 2
                if depression >= math.radians(MIN_ANGLE_DEG):
                    dist_f[b] = CAM_HEIGHT_M / math.tan(depression)
                    tag_f[b] = ray_tag(col[max(0, vb - 60): max(0, vb - 5)])

        # 天井由来
        run = longest_run(col, GROUP_IDS["ceiling"])
        if run is not None:
            vc = run[1]
            elevation = math.pi / 2 - math.pi * (vc + 0.5) / h
            if elevation >= math.radians(MIN_ANGLE_DEG):
                dist_c[b] = (CEIL_HEIGHT_M - CAM_HEIGHT_M) / math.tan(elevation)
                tag_c[b] = ray_tag(col[vc + 5: vc + 60])

    return az, dist_f, tag_f, dist_c, tag_c


def plot_scan(az, dist_f, tag_f, dist_c, tag_c, path):
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(projection="polar")
    ax.set_theta_zero_location("N")
    for dist, tag, marker, suffix in ((dist_f, tag_f, "o", "floor"),
                                      (dist_c, tag_c, "x", "ceil")):
        for t, color, label in ((TAG_WALL, "tab:blue", "wall"),
                                (TAG_OPENING, "tab:red", "opening")):
            m = (tag == t) & ~np.isnan(dist)
            ax.scatter(az[m], dist[m], s=6, c=color, marker=marker,
                       label=f"{label}/{suffix}")
    ax.set_rmax(8)
    ax.legend(loc="lower left", fontsize=6)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="",
                    help="例 '_gt': seg_gt を読み scan_gt へ出力")
    ap.add_argument("--frames", default=None,
                    help="'start:end:step' の連続レンジ。省略時は代表16フレーム")
    args = ap.parse_args()
    seg_dir = RESULTS_DIR / f"seg{args.variant}"
    scan_dir = RESULTS_DIR / f"scan{args.variant}"
    if args.frames:
        from img_align.core.panovggt import select_entries
        entries = select_entries(args.frames)
    else:
        entries = load_selected()

    scan_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    stats = {}
    for entry in entries:
        name = f"frame_{entry['index']:06d}"
        group_map = np.load(seg_dir / f"{name}_labels.npz")["labels"]
        az, dist_f, tag_f, dist_c, tag_c = scan_frame(group_map)
        out[f"{name}_az"] = az
        out[f"{name}_dist_floor"] = dist_f
        out[f"{name}_tag_floor"] = tag_f
        out[f"{name}_dist_ceil"] = dist_c
        out[f"{name}_tag_ceil"] = tag_c
        plot_scan(az, dist_f, tag_f, dist_c, tag_c, scan_dir / f"{name}_scan.png")
        vf, vc = ~np.isnan(dist_f), ~np.isnan(dist_c)
        stats[name] = {
            "floor_rays": int(vf.sum()),
            "ceil_rays": int(vc.sum()),
            "median_floor_m": round(float(np.median(dist_f[vf])), 2) if vf.any() else None,
            "median_ceil_m": round(float(np.median(dist_c[vc])), 2) if vc.any() else None,
            "opening_rays": int(((tag_f == TAG_OPENING) & vf).sum()
                                + ((tag_c == TAG_OPENING) & vc).sum()),
        }
    np.savez_compressed(scan_dir / "scans.npz", **out)
    (scan_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
