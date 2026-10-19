"""スコアの「犯人」を特定する: GT最適姿勢と採用姿勢を並べて採点する

オラクル (`eval/anchor_oracle`) は480候補すべてにICPを掛けるので1島12分かかる。
こちらは**GTから正解の姿勢を直接作って採点する**だけなので数十秒で済む。
知りたいのが「どの項が偽解に負けているか」なら、候補を探す必要はない。

読み方:
  正解のスコア > 採用のスコア → 候補生成の問題 (正解の姿勢を作れていない)
  正解のスコア < 採用のスコア → **スコアの問題**。差が出ている項が犯人

使い方:
  IMG_SESSION=0925 python -m img_align.eval.anchor_blame
  IMG_SESSION=0925 python -m img_align.eval.anchor_blame --islands pv0925_s02
"""
import argparse
import json
import math

import numpy as np

from img_align.config import BENCH_DIR, OUT_DIR
from img_align.core.plan_grid import PlanGrid
from img_align.eval.gt_traj import mm_per_px, load_gt
from img_align.postproc.anchor import prepare_island
from img_align.postproc.anchor_score import AnchorScorer
from img_align.postproc.plan_icp import (cells_to_px, segs_to_cells,
                                         transform_pts, wall_segments_px)
from img_align.postproc.scale_prior import m_per_unit

TERMS = ("precision", "recall", "f1", "cam_feas", "scale_dev", "wall_cross")


def px_to_cells(px, grid):
    """図面px → グリッドセル (row, col)。cells_to_px の逆"""
    return np.stack([(px[:, 1] - grid.y0_px) / grid.cell_px,
                     (px[:, 0] - grid.x0_px) / grid.cell_px], axis=1)


def sim2_umeyama(src, dst):
    """dst ≈ s R src + t (2D、鏡映なし)。transform_pts と同じ規約で返す

    鏡映を許さないのが要点。prepare_island は既に z 反転で掌性を図面に
    合わせてあるので、ここで鏡映を許すと「あり得ない姿勢」を正解にしてしまう。
    """
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    sc, dc = src - mu_s, dst - mu_d
    zs = sc[:, 0] + 1j * sc[:, 1]
    zd = dc[:, 0] + 1j * dc[:, 1]
    k = (zd * zs.conj()).sum() / max((zs * zs.conj()).sum().real, 1e-12)
    s, th = abs(k), math.atan2(k.imag, k.real)
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    return float(s), float(th), mu_d - s * (R @ mu_s)


def blame(tag, floor, gt, mpu_source="mapany"):
    """(正解のスコア一式, 採用のスコア一式, 正解のGT残差m, 採用のGT残差m)"""
    grid = PlanGrid(floor)
    mpu, _ = m_per_unit(tag, source=mpu_source)
    d = prepare_island(tag, mpu=mpu)
    cam_rc = np.stack(
        [(d["cam_pos_aligned"][:, 2] - d["origin"][1]) / d["cell_units"],
         (d["cam_pos_aligned"][:, 0] - d["origin"][0]) / d["cell_units"]],
        axis=1)
    frames = [int(str(n).replace("frame_", "")) for n in d["names"]]
    k = [i for i, f in enumerate(frames) if f in gt]
    if len(k) < 10:
        return None
    gt_px = np.array([[gt[frames[i]]["x_px"], gt[frames[i]]["y_px"]]
                      for i in k])
    gt_cells = px_to_cells(gt_px, grid)

    zz, xx = np.nonzero(d["density"] > 0.15)
    wall_pts = np.stack([zz, xx], axis=1).astype(np.float64)
    wall_pts = wall_pts[:: max(1, len(wall_pts) // 20000)]
    segs = segs_to_cells(wall_segments_px(floor), grid)
    scorer = AnchorScorer(grid, wall_pts, segs, cam_rc, s_prior=1.0)

    def err_of(s, th, t, idx=None):
        idx = k if idx is None else idx
        px = cells_to_px(transform_pts(cam_rc[idx], s, th, t), grid)
        ref = gt_px if idx is k else np.array(
            [[gt[frames[i]]["x_px"], gt[frames[i]]["y_px"]] for i in idx])
        return float(np.median(np.linalg.norm(px - ref, axis=1))) \
            * mm_per_px() / 1000

    # 正解: GTへの最適相似変換 (鏡映なし)
    s_t, th_t, t_t = sim2_umeyama(cam_rc[k], gt_cells)
    truth = scorer(s_t, th_t, t_t)
    e_truth = err_of(s_t, th_t, t_t)

    # 採用: corner_anchor が書いた図面px から変換を復元する
    p = OUT_DIR / f"corner_anchor_{tag}.json"
    if not p.exists():
        return truth, None, e_truth, None
    a = json.loads(p.read_text())
    sel_px = np.array([a[f"frame_{frames[i]:06d}"] for i in k
                       if f"frame_{frames[i]:06d}" in a])
    kk = [i for i in k if f"frame_{frames[i]:06d}" in a]
    if len(sel_px) < 10:
        return truth, None, e_truth, None
    s_s, th_s, t_s = sim2_umeyama(cam_rc[kk], px_to_cells(sel_px, grid))
    sel = scorer(s_s, th_s, t_s)
    return truth, sel, e_truth, err_of(s_s, th_s, t_s, kk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="pv_manifest.json")
    ap.add_argument("--islands", nargs="*", default=None)
    ap.add_argument("--key", default="f1_feas_scale", help="合成スコア名")
    ap.add_argument("--wall-band", type=float, default=None,
                    help="カメラ高さ±この値[m]だけを密度の素材にする "
                         "(家具と天井設備を外して壁だけ残す)")
    ap.add_argument("--min-y-bins", type=int, default=None,
                    help="密度地図の壁判定: 何個の高さビンに跨れば壁か "
                         "(既定4 = 0.8m。家具もこれを超えるので上げると壁選択的)")
    args = ap.parse_args()
    import img_align.postproc.anchor as _a
    if args.wall_band:
        _a.WALL_BAND_M = args.wall_band
        print(f"[density] WALL_BAND_M=±{args.wall_band}m")
    if args.min_y_bins:
        _a.MIN_Y_BINS = args.min_y_bins
        print(f"[density] MIN_Y_BINS={args.min_y_bins} "
              f"(鉛直 {args.min_y_bins * 0.20:.1f}m 以上を壁とみなす)")
    gt = load_gt()
    from img_align.core.floor_assign import apply as apply_floor
    man = apply_floor(json.loads(
        (BENCH_DIR / args.manifest).read_text()), quiet=True)
    print(f"{'island':20s} {'姿勢':>4s} {'GT':>6s} {'score':>8s} "
          + " ".join(f"{t:>9s}" for t in TERMS))
    for m in man:
        if not m.get("floor") or (args.islands and m["tag"] not in args.islands):
            continue
        try:
            r = blame(m["tag"], m["floor"], gt)
        except Exception as e:
            print(f"{m['tag']:20s} 失敗 ({type(e).__name__}: {e})")
            continue
        if r is None:
            print(f"{m['tag']:20s} GT不足")
            continue
        truth, sel, e_t, e_s = r
        for lab, v, e in (("正解", truth, e_t), ("採用", sel, e_s)):
            if v is None:
                continue
            print(f"{m['tag']:20s} {lab:>4s} {e:5.2f}m {v[args.key]:8.4f} "
                  + " ".join(f"{v.get(t, 0):9.4f}" for t in TERMS))
        if sel is not None:
            win = "正解" if truth[args.key] > sel[args.key] else "**偽解**"
            print(f"{'':20s} ---> スコアの勝者: {win}")


if __name__ == "__main__":
    main()
