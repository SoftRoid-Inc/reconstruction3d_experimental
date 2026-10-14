"""島内誤差の診断: 誤差が「どこで・どう」生じているかをフレーム単位で分解する

島の精度が悪いとき、原因は大きく3つに分かれる:
  (a) 累積ドリフト   … 誤差が単調に増える。接続の回転/スケール誤差の蓄積
  (b) 局所破綻       … 特定チャンク境界で誤差が階段状に跳ぶ。1本の接続失敗
  (c) 全体の歪み     … 誤差が往復で対称。地図自体の曲がり (unbend系の問題)

判別材料を一枚にまとめる:
  1. フレーム毎のGT誤差 (相似フィット後) とチャンク境界
  2. 区間ごとの局所スケール誤差 = |推定の移動距離| / |GTの移動距離|
  3. 区間ごとの方位誤差 = 推定の進行方向 − GTの進行方向
  4. チャンク境界をまたぐ誤差ジャンプ (= その接続が持ち込んだ誤差)

使い方:
  IMG_SESSION=0912 python -m img_align.eval.island_diag --tag pv0912_s02m
出力: frontend_bench/diag_<tag>.png / diag_<tag>.json
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from img_align.config import BENCH_DIR, RESULTS_DIR
from img_align.core.geometry import rotation_to_y
from img_align.eval.frontend import fit2d
from img_align.eval.gt_traj import MM_PER_PX, load_gt

WIN = 10          # 局所スケール/方位を測る窓 (フレーム)


def load_fit(tag, gt):
    """島をGTへ相似フィットし、(frames, 推定xy, GT xy, 誤差) を返す"""
    d = np.load(BENCH_DIR / tag / "global_cameras.npz")
    names = [str(n) for n in d["names"]]
    fr = np.array([int(n.replace("frame_", "")) for n in names])
    o = np.argsort(fr)
    fr, pos, poses = fr[o], d["positions"][o], d["poses"][o]
    ups = poses[:, :3, 1]
    up = np.median(ups / np.linalg.norm(ups, axis=1, keepdims=True), axis=0)
    p = pos @ rotation_to_y(up / np.linalg.norm(up)).T
    gxy, used = [], []
    for i, f in enumerate(fr):
        g = min(gt, key=lambda k: abs(k - f))
        if abs(g - f) <= 3:
            gxy.append([gt[g]["x_px"] * MM_PER_PX / 1000,
                        gt[g]["y_px"] * MM_PER_PX / 1000])
            used.append(i)
    gxy = np.array(gxy)
    best = None
    for mirror in (False, True):
        xz = p[:, [0, 2]].copy()
        if mirror:
            xz[:, 1] *= -1
        T = fit2d(xz[used], gxy)
        e = np.linalg.norm(T(xz[used]) - gxy, axis=1)
        if best is None or np.median(e) < best[0]:
            best = (float(np.median(e)), T(xz[used]), e, mirror)
    return fr[used], best[1], gxy, best[2]


def chunk_bounds(tag):
    """島が使ったチャンクのフレーム境界 (manifest + chunk_windows から)"""
    from img_align.core.chunking import chunk_windows
    from img_align.config import FRAMES_DIR
    man_p = BENCH_DIR / "pv_manifest.json"
    if not man_p.exists():
        return []
    man = json.loads(man_p.read_text())
    m = next((m for m in man if m["tag"] == tag), None)
    if m is None:
        return []
    n = len(sorted(FRAMES_DIR.glob("frame_number_*.jpg")))
    w = chunk_windows(n)
    return [w[k] for k in m["chunks"]]


def local_metrics(fr, est, gt_xy, win=WIN):
    """窓ごとの局所スケール比と方位差 (推定 vs GT)"""
    out = []
    for i in range(0, len(fr) - win):
        de, dg = est[i + win] - est[i], gt_xy[i + win] - gt_xy[i]
        ne, ng = np.linalg.norm(de), np.linalg.norm(dg)
        if ng < 0.3:          # ほぼ静止 -> 方位もスケールも不定
            continue
        ang = np.degrees(np.arctan2(de[1], de[0]) - np.arctan2(dg[1], dg[0]))
        out.append((fr[i], ne / ng, (ang + 180) % 360 - 180))
    return np.array(out) if out else np.zeros((0, 3))


SCALE_LO, SCALE_HI = 0.85, 1.18   # 局所スケール比の正常域
HEAD_TOL = 8.0                    # 局所方位差の正常域 [deg]


def anomalies(lm, min_len=2):
    """局所スケール/方位が正常域を外れる連続区間を拾う"""
    if not len(lm):
        return []
    bad = ((lm[:, 1] < SCALE_LO) | (lm[:, 1] > SCALE_HI)
           | (np.abs(lm[:, 2]) > HEAD_TOL))
    runs, cur = [], None
    for i, b in enumerate(bad):
        if b and cur is None:
            cur = i
        elif not b and cur is not None:
            if i - cur >= min_len:
                runs.append((cur, i - 1))
            cur = None
    if cur is not None and len(bad) - cur >= min_len:
        runs.append((cur, len(bad) - 1))
    return [{"frames": [int(lm[a, 0]), int(lm[b, 0])],
             "scale_max": round(float(np.max(lm[a:b + 1, 1])), 2),
             "scale_min": round(float(np.min(lm[a:b + 1, 1])), 2),
             "heading_abs_max_deg": round(float(np.max(np.abs(lm[a:b + 1, 2]))), 1)}
            for a, b in runs]


def classify(err, jumps, anom):
    """誤差の出方から主因を推定する"""
    n = len(err)
    trend = float(np.median(err[n // 2:]) - np.median(err[:n // 2]))
    severe = [a for a in anom
              if a["scale_max"] > 1.5 or a["scale_min"] < 0.6
              or a["heading_abs_max_deg"] > 15]
    if severe:
        return (f"局所破綻 ({len(severe)}箇所: "
                + ", ".join(f"{a['frames'][0]}-{a['frames'][1]}" for a in severe[:3])
                + ") — 島全体のフィットを歪めている")
    if anom:
        return f"軽度の局所異常 {len(anom)}箇所 (スケール/方位が正常域外)"
    big = [j for j in jumps if j[1] > max(0.15, 0.4 * float(np.median(err)))]
    if big and max(j[1] for j in big) > 0.5 * float(err.max()):
        return "接続起因 (チャンク境界で誤差が跳ぶ)"
    if trend > 0.5 * float(np.median(err)) and trend > 0.05:
        return "累積ドリフト (誤差が後半ほど増える)"
    return "健全 (特定箇所に集中せず、局所量も正常域)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--win", type=int, default=WIN)
    args = ap.parse_args()
    gt = load_gt()
    fr, est, gxy, err = load_fit(args.tag, gt)
    bounds = chunk_bounds(args.tag)
    starts = sorted({b[0] for b in bounds} | {b[1] for b in bounds})

    # チャンク境界をまたぐ誤差ジャンプ
    jumps = []
    for b in starts:
        i = int(np.searchsorted(fr, b))
        if 3 <= i < len(fr) - 3:
            jumps.append((int(b), float(abs(np.median(err[i:i + 3])
                                            - np.median(err[i - 3:i])))))
    lm = local_metrics(fr, est, gxy, args.win)
    anom = anomalies(lm)
    verdict = classify(err, jumps, anom)
    res = {
        "tag": args.tag, "n": int(len(fr)),
        "err_median_m": round(float(np.median(err)), 3),
        "err_max_m": round(float(err.max()), 3),
        "err_first_half": round(float(np.median(err[:len(err) // 2])), 3),
        "err_second_half": round(float(np.median(err[len(err) // 2:])), 3),
        "local_scale_median": round(float(np.median(lm[:, 1])), 3) if len(lm) else None,
        "local_scale_p10_p90": [round(float(np.percentile(lm[:, 1], 10)), 3),
                                round(float(np.percentile(lm[:, 1], 90)), 3)] if len(lm) else None,
        "local_heading_abs_median_deg": round(float(np.median(np.abs(lm[:, 2]))), 2) if len(lm) else None,
        "local_heading_abs_max_deg": round(float(np.abs(lm[:, 2]).max()), 2) if len(lm) else None,
        "worst_boundary_jumps": sorted(jumps, key=lambda j: -j[1])[:5],
        "anomalies": anom,
        "verdict": verdict,
    }
    print(json.dumps(res, ensure_ascii=False, indent=1))

    fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    ax[0].plot(fr, err, "-", color="crimson")
    for b in starts:
        ax[0].axvline(b, color="0.85", lw=0.8, zorder=0)
    ax[0].set_ylabel("GT誤差 [m]")
    ax[0].set_title(f"{args.tag}: {verdict}")
    if len(lm):
        ax[1].plot(lm[:, 0], lm[:, 1], "-", color="teal")
        ax[1].axhline(1.0, color="0.6", lw=0.8)
        ax[1].set_ylabel(f"局所スケール比 (窓{args.win}f)")
        ax[2].plot(lm[:, 0], lm[:, 2], "-", color="darkorange")
        ax[2].axhline(0.0, color="0.6", lw=0.8)
        ax[2].set_ylabel("局所方位差 [deg]")
    for a in ax[1:]:
        for b in starts:
            a.axvline(b, color="0.85", lw=0.8, zorder=0)
    ax[2].set_xlabel("frame")
    fig.tight_layout()
    fig.savefig(BENCH_DIR / f"diag_{args.tag}.png", dpi=110)
    plt.close(fig)
    (BENCH_DIR / f"diag_{args.tag}.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1))
    print(f"-> {BENCH_DIR / f'diag_{args.tag}.png'}")


if __name__ == "__main__":
    main()
