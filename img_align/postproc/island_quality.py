"""島がアンカーに耐えるかを、図面と照合する前に判定する

アンカーは島ごとに当たり外れがある (本番0912で6島中2島が外れ: 8.14m / 2.18m)。
外れを**事前に**見分けられれば、その島だけ手動に回せる。全島を人が確認するのと
比べて作業量が桁違いに減る。

密度地図を見て分かったこと (2026-10-13):
アンカーの成否は**コーナーの数ではなく壁のシャープさ**で決まる。
島内の整合が甘いと同じ壁が二重三重に描かれ、コーナーは拾えても位置が信用できない
(0828_1Fa はコーナー94個で1.58m、0828_1Fb は14個で0.17m)。

ここで測る指標 (いずれも図面を使わない = 本番で使える):
| 指標 | 意味 |
|---|---|
| `wall_sharp` | 壁線に直交する方向の密度プロファイルの鋭さ。二重壁だと鈍る |
| `corner_arm` | 検出コーナーの腕の長さの中央値。短い=壁が途切れている |
| `n_corners` | コーナー数 (参考。単体では当てにならない) |
| `extent_cells` | 島が覆うセル範囲。小さい島は指紋が足りない |
| `occupancy` | 強壁セルの密度。疎だと照合が不安定 |

使い方:
  IMG_SESSION=0912 python -m img_align.postproc.island_quality
出力: backend/island_quality.json
"""
import argparse
import json

import numpy as np

from img_align.config import BENCH_DIR, OUT_DIR
from img_align.postproc.anchor import prepare_island
from img_align.postproc.corners import DENS_THR, density_corners

PROFILE_HALF = 6      # 壁線の直交方向に見る幅 (セル)


def wall_sharpness(dens, deg):
    """壁線に直交する方向の密度プロファイルの鋭さ [0,1]

    支配方位に合わせて回し、行・列ごとの占有ヒストグラムのピーク近傍が
    どれだけ急峻かを測る。壁が二重三重になっていると裾が広がって鈍る。
    """
    from scipy import ndimage
    img = ndimage.rotate(dens, -deg, reshape=True, order=1, mode="constant")
    vals = []
    for axis in (0, 1):
        occ = (img > DENS_THR).sum(axis=1 - axis).astype(float)
        if occ.max() < 4:
            continue
        # 局所ピーク周りの「ピーク値 / 近傍平均」= 鋭さ
        for i in np.argsort(occ)[::-1][:24]:
            lo, hi = max(0, i - PROFILE_HALF), min(len(occ), i + PROFILE_HALF + 1)
            nb = occ[lo:hi]
            if occ[i] < 4 or len(nb) < 3:
                continue
            vals.append(float(occ[i] / max(nb.mean(), 1e-6)))
    return float(np.median(vals)) if vals else 0.0


def quality(tag):
    """島1本の品質指標 (図面を使わない)"""
    d = prepare_island(tag)
    dens = d["density"]
    zz, xx = np.nonzero(dens > DENS_THR)
    if len(zz) < 50:
        return {"tag": tag, "error": f"強壁セル {len(zz)}個"}
    cs, deg = density_corners(np.stack([zz, xx], 1).astype(float))
    arms = [c["arm"] for c in cs] if cs else [0]
    return {
        "tag": tag,
        "wall_sharp": round(wall_sharpness(dens, deg), 3),
        "corner_arm": float(np.median(arms)),
        "n_corners": len(cs),
        "extent_cells": int(max(zz.ptp(), xx.ptp())),
        "occupancy": round(float(len(zz) / dens.size), 4),
        "deg": round(float(deg), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="pv_manifest.json")
    args = ap.parse_args()
    man = json.loads((BENCH_DIR / args.manifest).read_text())
    rows = []
    print(f"{'island':18} {'鋭さ':>6} {'腕':>5} {'角数':>5} {'広さ':>6} {'占有':>7}")
    for m in man:
        try:
            q = quality(m["tag"])
        except Exception as e:
            print(f"{m['tag']:18} 失敗 ({type(e).__name__})")
            continue
        if "error" in q:
            print(f"{m['tag']:18} {q['error']}")
            continue
        q["frames"] = m["b"] - m["a"] + 1
        rows.append(q)
        print(f"{q['tag']:18} {q['wall_sharp']:6.2f} {q['corner_arm']:5.0f} "
              f"{q['n_corners']:5d} {q['extent_cells']:6d} {q['occupancy']:7.4f}")
    (OUT_DIR / "island_quality.json").write_text(
        json.dumps(rows, indent=1, ensure_ascii=False))
    print(f"-> {OUT_DIR / 'island_quality.json'}")


if __name__ == "__main__":
    main()
