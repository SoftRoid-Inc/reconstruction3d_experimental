"""チャンク単体をGT軌跡と突き合わせてモデルを比較する (PanoVGGT vs Pi3 vs Pi3-FT)

島の連結を挟まず `results/pcd/<tag>/cameras.npz` をそのまま採点するので、
**推論そのものの良し悪し**だけを見られる。採点は eval/frontend と同じ手順:
重力軸で水平化 → GT(図面mm)との2D Sim(2)フィット → 残差のmedian/p90。
鏡像はPanoVGGT/Pi3ともに系の取り方で反転しうるので両方試して良い方を採る。

既知の局所破綻区間 (2026-10-08の島内診断) を `--preset breakdown` で引ける。
対照として健全区間 `--preset control` も同じ長さで用意してある。

使い方:
  IMG_SESSION=0925 python -m recon3d.eval.chunk_cmp --tags pv_c29 pi3_c29 holo_c29
  python -m recon3d.eval.chunk_cmp --preset breakdown        # 窓の一覧を表示
"""
import argparse
import json

import numpy as np

from recon3d.config import RESULTS_DIR
from recon3d.core.geometry import rotation_to_y

# (session, 開始, 終了, ラベル) — 診断で局所破綻と判定された区間を含む16フレーム窓
BREAKDOWN = [
    ("0925", 472, 488, "テクスチャレス+鏡(洗面)"),
    ("0925", 492, 508, "磨き大理石+狭所"),
    ("0912", 632, 648, "白壁への180°旋回+ガラス"),
    ("0828", 240, 256, "ブランク壁+ドア通過"),
    ("0616", 160, 176, "不均一な動き+反復構造"),
]
# 同一セッションの健全区間 (同じ島内から選んだ対照)
CONTROL = [
    ("0925", 520, 536, "健全(居室)"),
    ("0912", 600, 616, "健全"),
    ("0828", 200, 216, "健全"),
    ("0616", 120, 136, "健全"),
]


MAX_GAP = 4      # これより広いGT欠損を跨ぐフレームは採点に使わない


def gt_interp(gt, frames, mm_per_px):
    """GTを線形補間して各フレームの位置[m]を返す (広い欠損は除外)

    GTは基本1フレーム刻みだが最大45フレームの欠損がある。最近傍スナップだと
    0.5s×0.6m/s → 1フレームずれで0.3mの誤差が入るため、必ず補間する。
    """
    keys = np.array(sorted(gt))
    xy = np.array([[gt[k]["x_px"], gt[k]["y_px"]] for k in keys]) * mm_per_px / 1000
    j = np.searchsorted(keys, frames)
    ok, out = [], []
    for i, f in enumerate(frames):
        r = min(max(j[i], 1), len(keys) - 1)
        lo, hi = keys[r - 1], keys[r]
        if f == lo or f == hi:
            out.append(xy[r - 1] if f == lo else xy[r])
        elif lo < f < hi and hi - lo <= MAX_GAP:
            w = (f - lo) / (hi - lo)
            out.append(xy[r - 1] * (1 - w) + xy[r] * w)
        else:
            continue
        ok.append(i)
    return np.array(out), ok


def score(tag, gt, mm_per_px):
    """pcd/<tag>/cameras.npz を GT と照合 → 指標dict"""
    from recon3d.eval.frontend import fit2d
    d = np.load(RESULTS_DIR / "pcd" / tag / "cameras.npz")
    frames = np.array([int(str(n).replace("frame_", "")) for n in d["names"]])
    poses = d["poses"]
    ups = poses[:, :3, 1]
    up = np.median(ups / np.linalg.norm(ups, axis=1, keepdims=True), axis=0)
    p = poses[:, :3, 3] @ rotation_to_y(up / np.linalg.norm(up)).T

    gt_xy, used = gt_interp(gt, frames, mm_per_px)
    if len(used) < 4:
        return {"tag": tag, "error": f"GT対応 {len(used)}点 (不足)"}

    best = None
    for mirror in (False, True):
        xz = p[:, [0, 2]].copy()
        if mirror:
            xz[:, 1] *= -1
        e = np.linalg.norm(fit2d(xz[used], gt_xy)(xz[used]) - gt_xy, axis=1)
        if best is None or np.median(e) < best["ate_median_m"]:
            best = {"mirror": mirror, "ate_median_m": float(np.median(e)),
                    "ate_p90_m": float(np.percentile(e, 90)),
                    "ate_max_m": float(e.max())}
    st = RESULTS_DIR / "pcd" / tag / "pcd_stats.json"
    if st.exists():
        s = json.loads(st.read_text())
        best["n_points"] = s.get("n_points")
        for k in ("pose_spread_t_median", "pose_spread_r_deg_median"):
            if k in s:
                best[k] = round(s[k], 4)
    return {"tag": tag, "n_frames": len(frames), "n_gt": len(used),
            **{k: (round(v, 3) if isinstance(v, float) else v)
               for k, v in best.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="*", default=None)
    ap.add_argument("--preset", choices=("breakdown", "control"), default=None)
    ap.add_argument("--out", default=None, help="結果JSONの保存先")
    args = ap.parse_args()

    if args.preset and not args.tags:
        for s, a, b, note in (BREAKDOWN if args.preset == "breakdown" else CONTROL):
            print(f"{s}  {a}:{b + 1}:1   {note}")
        return

    from recon3d.eval.gt_traj import mm_per_px, load_gt
    gt = load_gt()
    rows = [score(t, gt, mm_per_px()) for t in args.tags]
    w = max(len(r["tag"]) for r in rows)
    print(f"{'tag':{w}}  {'n':>3} {'median':>7} {'p90':>7} {'max':>7}  spread_t/r")
    for r in rows:
        if "error" in r:
            print(f"{r['tag']:{w}}  {r['error']}")
            continue
        sp = (f"{r.get('pose_spread_t_median', float('nan')):.3f}/"
              f"{r.get('pose_spread_r_deg_median', float('nan')):.2f}°"
              if "pose_spread_t_median" in r else "-")
        print(f"{r['tag']:{w}}  {r['n_frames']:3d} {r['ate_median_m']:7.3f} "
              f"{r['ate_p90_m']:7.3f} {r['ate_max_m']:7.3f}  {sp}")
    if args.out:
        from pathlib import Path
        Path(args.out).write_text(json.dumps(rows, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
