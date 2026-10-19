"""図面アンカーの候補集合に正解が入っているかを測る (提案 vs 検証の切り分け)

アンカーが外す原因は2通りあり、必要な打ち手がまるで違う:
  (A) 正解が**候補に入っていない** → 提案側の問題。特徴量そのものを
      作り直す必要がある (学習済み記述子など、学習データも要る大工事)
  (B) 正解は**候補に入っているが選べていない** → 検証側の問題。
      スコア関数を直すだけで済む (現状 wall_cover は正解0.16と偽解0.11-0.25を
      区別できない)

正式GT (<物件>/trajectory_points.csv) は図面ピクセル座標そのものなので、
各候補の変換を島のカメラ軌跡に当てて GT と直接比べれば、どちらかが分かる。

出力: 島ごとに「最良候補のGT誤差(=オラクル)」「wall_coverが選んだ候補のGT誤差」
「オラクルの順位」。オラクル誤差が小さく順位が1でなければ (B)。

使い方:
  IMG_SESSION=0912 python -m img_align.eval.anchor_oracle \
      --manifest pv_full_manifest.json --mpu-source mapany
"""
import argparse
import json
import math

import numpy as np

from img_align.config import BENCH_DIR
from img_align.core.plan_grid import PlanGrid
from img_align.eval.gt_traj import mm_per_px, load_gt
from img_align.postproc.anchor import prepare_island
from img_align.postproc.anchor_score import SCORERS, AnchorScorer
from img_align.postproc.corners import (ADD_ROT_BRANCHES, DENS_THR,
                                        _rot_branches, density_corners,
                                        match_corners, plan_corners_cell)
from img_align.postproc.plan_icp import wall_segments_px
from img_align.postproc.plan_icp import (cells_to_px, icp_to_walls,
                                         segs_to_cells, transform_pts)  # noqa

OK_M = 1.0        # これ以内なら「正解候補」とみなす (島は数十mなので十分厳しい)


def candidates(tag, floor, s_prior=1.0, mpu=None, s_rel_tol=0.4,
               top_k=12, sweep=None):
    """島の全候補を (スコア一式, 変換後のカメラpx, 付随情報) で返す

    sweep: 事前スケールの倍率リスト。スケール事前に誤差があるときは
    1点に賭けず複数倍率で候補を出して混ぜる (スコア関数が信頼できる前提)。
    """
    grid = PlanGrid(floor)
    d = prepare_island(tag, mpu=mpu)
    zz, xx = np.nonzero(d["density"] > DENS_THR)
    dens_cs, deg = density_corners(
        np.stack([zz, xx], axis=1).astype(np.float64))
    plan_cs = plan_corners_cell(floor, grid)
    cands = []
    for mult in (sweep or [1.0]):
        cands += match_corners(dens_cs, deg, plan_cs, s_prior=s_prior * mult,
                               s_rel_tol=s_rel_tol, top_k=top_k)
    cam_rc = np.stack(
        [(d["cam_pos_aligned"][:, 2] - d["origin"][1]) / d["cell_units"],
         (d["cam_pos_aligned"][:, 0] - d["origin"][0]) / d["cell_units"]],
        axis=1)
    if ADD_ROT_BRANCHES:     # 本番と同じ候補集合にする
        cands += _rot_branches(cands, cam_rc)
    zz2, xx2 = np.nonzero(d["density"] > 0.15)
    wall_pts = np.stack([zz2, xx2], axis=1).astype(np.float64)
    wall_pts = wall_pts[:: max(1, len(wall_pts) // 20000)]
    segs = segs_to_cells(wall_segments_px(floor), grid)
    scorer = AnchorScorer(grid, wall_pts, segs, cam_rc, s_prior=s_prior)

    # pre_* = ICP前の素の変換で採点した値 (足切りの妥当性を測るため)
    raw = [scorer(c["s"], c["th"], c["t"])["f1_feas_scale"] for c in cands]
    order = sorted(range(len(cands)), key=lambda i: -raw[i])
    pre_rank = {i: r + 1 for r, i in enumerate(order)}
    out = []
    for i, c in enumerate(cands):
        s2, th2, t2, resid = icp_to_walls(wall_pts, segs, c["s"], c["th"],
                                          c["t"])
        px = cells_to_px(transform_pts(cam_rc, s2, th2, t2), grid)
        out.append({**scorer(s2, th2, t2), "px": px, "resid": resid,
                    "n": c["n"], "th_deg": math.degrees(th2) % 360,
                    "pre_score": raw[i], "pre_rank": pre_rank[i]})
    return out, [str(n) for n in d["names"]]


def gt_err(px, names, gt):
    """候補の軌跡とGTの残差 median [m] (GTのあるフレームのみ)"""
    e = []
    for p, nm in zip(px, names):
        f = int(nm.replace("frame_", ""))
        if f in gt:
            e.append(math.dist(p, (gt[f]["x_px"], gt[f]["y_px"])))
    return float(np.median(e)) * mm_per_px() / 1000 if len(e) >= 4 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="pv_gtseg_manifest.json")
    ap.add_argument("--islands", nargs="*", default=None)
    ap.add_argument("--mpu-source", default="slab",
                    choices=("slab", "mapany", "gt"),
                    help="スケール事前の供給源 (postproc/scale_prior)")
    ap.add_argument("--s-rel-tol", type=float, default=0.4,
                    help="事前スケールの許容相対幅 (既定0.4=±40%)")
    ap.add_argument("--top-k", type=int, default=12, help="候補数の上限")
    ap.add_argument("--s-sweep", default=None,
                    help="事前スケールの倍率を掃引 (例 0.7,0.85,1.0,1.2,1.4)")
    args = ap.parse_args()
    sweep = ([float(v) for v in args.s_sweep.split(",")]
             if args.s_sweep else None)
    gt = load_gt()
    man = json.loads((BENCH_DIR / args.manifest).read_text())
    rows = []
    for m in man:
        if args.islands and m["tag"] not in args.islands:
            continue
        if not m.get("floor"):
            continue
        mpu = None
        if args.mpu_source != "slab":
            from img_align.postproc.scale_prior import m_per_unit
            mpu, minfo = m_per_unit(m["tag"], source=args.mpu_source)
            print(f"{m['tag']}: m_per_unit={mpu:.3f} ({minfo})")
        try:
            cands, names = candidates(m["tag"], m["floor"], mpu=mpu,
                                      s_rel_tol=args.s_rel_tol,
                                      top_k=args.top_k, sweep=sweep)
        except Exception as e:
            print(f"{m['tag']}: 候補生成に失敗 ({type(e).__name__}: {e})")
            continue
        if not cands:
            print(f"{m['tag']}: 候補なし")
            continue
        for c in cands:
            c["err"] = gt_err(c["px"], names, gt)
        ok = [c for c in cands if c["err"] is not None]
        if not ok:
            print(f"{m['tag']}: GT対応なし")
            continue
        best = min(ok, key=lambda c: c["err"])
        row = {"tag": m["tag"], "floor": m["floor"], "n_cand": len(ok),
               "oracle_err_m": round(best["err"], 3)}
        for sc in SCORERS:
            sel = max(ok, key=lambda c: c[sc])
            row[f"{sc}_err_m"] = round(sel["err"], 3)
            row[f"{sc}_rank"] = 1 + sorted((c[sc] for c in ok),
                                           reverse=True).index(best[sc])
        row["oracle_pre_rank"] = best["pre_rank"]
        rows.append(row)
        mark = "OK " if row["f1_feas_err_m"] <= OK_M else "NG "
        print(f"{mark}{row['tag']:18} 候補{row['n_cand']:3d}  "
              f"オラクル {row['oracle_err_m']:6.2f}m  | "
              + "  ".join(f"{sc}: {row[f'{sc}_err_m']:5.2f}m"
                          f"(順位{row[f'{sc}_rank']:2d})" for sc in SCORERS)
              + f"  | ICP前順位 {row['oracle_pre_rank']:3d}")

    if rows:
        n_prop = sum(1 for r in rows if r["oracle_err_m"] <= OK_M)
        print(f"\n正解が候補にある島: {n_prop}/{len(rows)}")
        from img_align.postproc.corners import ICP_TOP_K
        kept = sum(1 for r in rows if r["oracle_pre_rank"] <= ICP_TOP_K)
        print(f"ICP前の足切り(上位{ICP_TOP_K}本)を正解が通過: "
              f"{kept}/{len(rows)}島  "
              f"(順位 {[r['oracle_pre_rank'] for r in rows]})")
        print(f"{'scorer':16} {'当てた島':>8} {'正解の平均順位':>14}")
        for sc in SCORERS:
            n = sum(1 for r in rows if r[f"{sc}_err_m"] <= OK_M)
            mr = np.mean([r[f"{sc}_rank"] for r in rows])
            print(f"{sc:16} {n:4d}/{len(rows):<4d} {mr:14.1f}")
    (BENCH_DIR / "anchor_oracle.json").write_text(
        json.dumps(rows, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
