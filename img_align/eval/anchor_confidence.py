"""アンカーが当たったかを、GTを使わずに見分ける (信頼度指標の検証)

本番ではアンカーの当たり外れが島ごとに出る (0912で6島中2島が外れ)。
**外れを事前に見分けて手動へ回せれば**、全島を人が確認するより作業量が桁違いに減る。

2026-10-14に試して**外れた**アプローチ: 島の品質 (壁のシャープさ・コーナー数・
占有率) から予測する — 失敗島 s001 と成功島 s002 が全指標でほぼ同一だった。
オラクル診断の結果、両失敗島とも**候補120本の中に正解が1本も無かった**。
つまり島の質ではなく「その島が図面に対して一意に決まるか」が問題。

そこで本モジュールは**アンカー結果そのものに現れる兆候**を測る:
| 指標 | 期待する振る舞い |
|---|---|
| `score` | 正解が候補に無ければ、どれを選んでも整合が悪く低くなる |
| `margin` | 1位と2位の差。拮抗していれば決め手が無い |
| `recall` | 図面側の壁をどれだけ説明できたか (precisionより素直) |
| `cam_feas` | カメラが屋内に収まるか |
| `wall_resid_m` | 壁ICPの残差 |
いずれも corner_anchor が既に計算しているので**追加コストはゼロ**。

使い方 (GTがある開発環境で閾値を決める):
  IMG_SESSION=0912 python -m img_align.eval.anchor_confidence
出力: backend/anchor_confidence.json (指標とGT残差の対応)
"""
import argparse
import json
import math

import numpy as np

from img_align.config import BENCH_DIR, OUT_DIR, SESSION
from img_align.eval.gt_traj import MM_PER_PX, load_gt


def gt_error(tag, gt):
    """アンカー軌跡とGTの残差 median [m]。採れなければ None"""
    p = OUT_DIR / f"corner_anchor_{tag}.json"
    if not p.exists():
        return None
    tr = json.loads(p.read_text())
    e = [math.dist(xy, (gt[f]["x_px"], gt[f]["y_px"]))
         for n, xy in tr.items()
         if (f := int(n.replace("frame_", ""))) in gt]
    return float(np.median(e)) * MM_PER_PX / 1000 if len(e) >= 4 else None


def row(tag, gt, suffix="_corners"):
    a = OUT_DIR / f"anchor_{tag}{suffix}.json"
    if not a.exists():
        return None
    d = json.loads(a.read_text())
    return {"tag": tag, "err_m": gt_error(tag, gt),
            "score": d.get("score"), "margin": d.get("margin"),
            "recall": d.get("wall_recall"), "cover": d.get("wall_cover"),
            "cam_feas": d.get("cam_feas"), "resid": d.get("wall_resid_m"),
            "n_inliers": d.get("n_inliers")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="pv_manifest.json")
    ap.add_argument("--ok-m", type=float, default=1.0)
    args = ap.parse_args()
    gt = load_gt()
    man = json.loads((BENCH_DIR / args.manifest).read_text())
    rows = [r for m in man if (r := row(m["tag"], gt)) and r["err_m"] is not None]
    if not rows:
        raise SystemExit("採点できるアンカーがありません")
    print(f"{'island':18} {'GT残差':>7} {'score':>7} {'margin':>7} "
          f"{'recall':>7} {'feas':>6} {'resid':>6}")
    for r in sorted(rows, key=lambda r: r["err_m"]):
        mark = "✓" if r["err_m"] <= args.ok_m else "✗"
        print(f"{mark}{r['tag']:17} {r['err_m']:7.2f} "
              f"{(r['score'] or 0):7.4f} {(r['margin'] or 0):7.4f} "
              f"{(r['recall'] or 0):7.3f} {(r['cam_feas'] or 0):6.2f} "
              f"{(r['resid'] or 0):6.3f}")
    ok = np.array([r["err_m"] <= args.ok_m for r in rows])
    print(f"\n成功 {ok.sum()}/{len(rows)}  各指標の判別力 (成功群 vs 失敗群の中央値):")
    for k in ("score", "margin", "recall", "cover", "cam_feas", "resid"):
        v = np.array([r[k] if r[k] is not None else np.nan for r in rows], float)
        if np.isnan(v).all() or ok.all() or not ok.any():
            continue
        a, b = np.nanmedian(v[ok]), np.nanmedian(v[~ok])
        # 分離度: 群間差を全体のばらつきで正規化
        sep = abs(a - b) / max(np.nanstd(v), 1e-9)
        print(f"  {k:10} 成功 {a:8.4f}  失敗 {b:8.4f}  分離度 {sep:.2f}")
    (OUT_DIR / "anchor_confidence.json").write_text(json.dumps(
        {"session": SESSION, "rows": rows}, indent=1, ensure_ascii=False))
    print(f"-> {OUT_DIR / 'anchor_confidence.json'}")


if __name__ == "__main__":
    main()
