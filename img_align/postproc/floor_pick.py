"""島の階を自動決定する: 全フロアの図面と照合しスコア最良を採る

GTフリー運用 (core/chunking --segments imu) では島に階ラベルが付かない
(IMUの昇降積算は工事現場のはしご過検出で壊れるため)。代わりに本モジュールが
**各島を1F/2F/3Fすべての図面に対してコーナーアンカーし、最良の階を採用**する。

選択指標は corner_anchor の総合スコア `score`
(= 対称被覆F1 × カメラ妥当性 × スケール逸脱ペナルティ、postproc/anchor_score)。
2026-10-13まで使っていた wall_cover は**片方向被覆**で縮小退化があり、
候補順位で正解を当てられなかった (全周島8本で0/8) ため置き換えた。

使い方:
  IMG_SESSION=0616 python -m img_align.postproc.floor_pick            # 全島
  IMG_SESSION=0616 python -m img_align.postproc.floor_pick --tags pv0616_s00r2
出力: backend/anchor_<tag>_auto.json (採用階のアンカー) + floor_pick.json (全候補)
"""
import argparse
import json

from img_align.config import BENCH_DIR, OUT_DIR, PROP
from img_align.postproc.corners import corner_anchor

# 正解島の実測 score は 0.046〜0.208 (0912全周島4本、2026-10-13)。
# 下限はその最小値のさらに下に置き、明らかな不成立だけを落とす
MIN_SCORE = 0.02     # これ未満は「どの階でもない」として棄却


def available_floors():
    return sorted(p.stem for p in PROP.zumen_dir.glob("*.json")
                  if not p.name.endswith(".json.raw") and len(p.stem) <= 3)


def pick_floor(tag, floors=None, mpu=None, s_prior=1.0):
    """全フロアで試し、総合スコア最良の階を採用する"""
    floors = floors or available_floors()
    cands = []
    for fl in floors:
        try:
            _, info = corner_anchor(tag, fl, mpu=mpu, s_prior=s_prior,
                                    suffix=f"_try{fl}", return_info=True)
            cands.append(info)
            print(f"  [{tag}] {fl}: score {info['score']:.4f} "
                  f"(P {info['wall_cover']:.3f} / R {info['wall_recall']:.3f}) "
                  f"resid {info['wall_resid_m']:.3f} inliers {info['n_inliers']}")
        except Exception as e:
            print(f"  [{tag}] {fl}: 解なし ({type(e).__name__})")
    if not cands:
        return None
    best = max(cands, key=lambda c: c["score"])
    if best["score"] < MIN_SCORE:
        print(f"  [{tag}] 最良 {best['floor']} でも score "
              f"{best['score']:.4f} < {MIN_SCORE} -> 棄却")
        return None
    second = sorted((c["score"] for c in cands), reverse=True)
    margin = (second[0] - second[1]) if len(second) > 1 else second[0]
    out = dict(best["anchor"])
    out["anchored_by"] = f"auto_floor({best['floor']}, margin {margin:.3f})"
    (OUT_DIR / f"anchor_{tag}_auto.json").write_text(json.dumps(out, indent=1))
    print(f"  [{tag}] -> 採用 {best['floor']} (score {best['score']:.4f}, "
          f"2位との差 {margin:.4f})")
    return {"tag": tag, "floor": best["floor"],
            "score": round(best["score"], 4),
            "wall_cover": round(best["wall_cover"], 4),
            "wall_recall": round(best["wall_recall"], 4),
            "margin": round(margin, 4),
            "wall_resid_m": round(best["wall_resid_m"], 4),
            "candidates": {c["floor"]: round(c["score"], 4) for c in cands}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="*", default=None)
    ap.add_argument("--manifest", default="pv_manifest.json")
    ap.add_argument("--floors", nargs="*", default=None)
    ap.add_argument("--mpu", type=float, default=None,
                    help="メトリック系タグ(macond等)は1.0。pvタグは省略(カメラ高から推定)")
    args = ap.parse_args()
    tags = args.tags
    if not tags:
        man = json.loads((BENCH_DIR / args.manifest).read_text())
        tags = [m["tag"] for m in man]
    res = []
    for t in tags:
        print(f"=== {t}")
        r = pick_floor(t, args.floors, mpu=args.mpu)
        if r:
            res.append(r)
    (OUT_DIR / "floor_pick.json").write_text(
        json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
