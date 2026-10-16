"""壁の高さ帯をデータから測る (WALL_BAND_M を手で決めない)

**壁は鉛直**なので、真上から見た footprint が高さによらず同じになる。家具は
鉛直でないので、高さを変えると footprint が変わる。この性質だけで壁の帯を
決められる — 床や天井を検出する必要が無い (0925_s003r2 のように床の点群が
欠けている島があるので、床基準は当てにならない)。

手順:
  1. 点群を0.2mの高さビンに分ける
  2. 各ビンの真上ビュー占有マップを作る
  3. 各ビンと**全体の合意マップ**との一致度 (IoU) を測る
  4. 一致度が高いビンの連続区間 = 壁の帯

使い方:
  IMG_SESSION=0925 python -m img_align.eval.wall_band_probe
  IMG_SESSION=0925 python -m img_align.eval.wall_band_probe --islands pv0925_s004
"""
import argparse
import json

import numpy as np

from img_align.config import BENCH_DIR, OUT_DIR
from img_align.postproc.anchor import CELL_M, Y_BIN_M, prepare_island
from img_align.postproc.scale_prior import m_per_unit

MIN_BIN_PTS = 2000     # これ未満の点しかないビンは判定しない
IOU_FRAC = 0.75        # 最良ビンのこの割合以上の一致度を「壁らしい」とする


def _occupancy(pts_xz, cell_u, shape, origin):
    ox, oz = origin
    ix = np.clip(((pts_xz[:, 0] - ox) / cell_u).astype(np.int32), 0,
                 shape[1] - 1)
    iz = np.clip(((pts_xz[:, 1] - oz) / cell_u).astype(np.int32), 0,
                 shape[0] - 1)
    occ = np.zeros(shape, bool)
    occ[iz, ix] = True
    return occ


def probe(tag, bin_m=Y_BIN_M):
    """島の高さビンごとの「壁らしさ」。(bins, 結果dict)"""
    import img_align.postproc.anchor as _a
    saved, _a.WALL_BAND_M = _a.WALL_BAND_M, None   # 帯を測るので帯は掛けない
    try:
        mpu, _ = m_per_unit(tag)
        d = prepare_island(tag, mpu=mpu)
    finally:
        _a.WALL_BAND_M = saved
    w = d["density_pts"]
    cam_y = float(np.median(d["cam_pos_aligned"][:, 1]))
    cell_u = CELL_M / mpu
    bin_u = bin_m / mpu
    ox = w[:, 0].min() - 2 * cell_u
    oz = w[:, 2].min() - 2 * cell_u
    nx = int((w[:, 0].max() - ox) / cell_u) + 3
    nz = int((w[:, 2].max() - oz) / cell_u) + 3
    shape, origin = (nz, nx), (ox, oz)

    iy = np.floor((w[:, 1] - cam_y) / bin_u).astype(np.int64)
    occs, rows = {}, []
    for b in np.unique(iy):
        m = iy == b
        if m.sum() < MIN_BIN_PTS:
            continue
        occs[int(b)] = _occupancy(w[m][:, [0, 2]], cell_u, shape, origin)
    if len(occs) < 3:
        return None
    # 合意マップ: 過半数のビンが占有しているセル (= 高さによらず在るもの)
    stack = np.stack(list(occs.values()))
    consensus = stack.sum(axis=0) >= max(2, int(0.5 * len(occs)))
    for b, occ in sorted(occs.items()):
        inter = (occ & consensus).sum()
        union = (occ | consensus).sum()
        rows.append({"bin": b,
                     # カメラからの相対高さ [m] (上が正。整列系は y+ が下向き)
                     "h_m": -(b + 0.5) * bin_m,
                     "n": int((iy == b).sum()),
                     "iou": float(inter / max(union, 1))})
    return rows


def band_from(rows, frac=IOU_FRAC):
    """壁らしいビンの連続区間 → (下端 h, 上端 h, 片側幅)。カメラ基準[m]"""
    if not rows:
        return None
    best = max(r["iou"] for r in rows)
    ok = [r for r in rows if r["iou"] >= best * frac]
    if not ok:
        return None
    # カメラ高さ(h=0)を含む連続区間だけを採る (飛び地は家具の塊)
    bins = sorted(r["bin"] for r in ok)
    run = [b for b in bins]
    center = min(bins, key=abs)
    lo = hi = center
    s = set(bins)
    while lo - 1 in s:
        lo -= 1
    while hi + 1 in s:
        hi += 1
    hs = [r["h_m"] for r in rows if lo <= r["bin"] <= hi]
    return min(hs), max(hs), max(abs(min(hs)), abs(max(hs))), len(run)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--islands", nargs="*", default=None)
    ap.add_argument("--manifest", default="pv_manifest.json")
    ap.add_argument("--detail", action="store_true", help="ビンごとに出す")
    args = ap.parse_args()
    man = json.loads((BENCH_DIR / args.manifest).read_text())
    out = []
    print(f"{'island':20s} {'壁らしい帯 (カメラ基準)':>26s} {'片側幅':>7s} "
          f"{'最良IoU':>8s}")
    for m in man:
        if args.islands and m["tag"] not in args.islands:
            continue
        try:
            rows = probe(m["tag"])
        except Exception as e:
            print(f"{m['tag']:20s} 失敗 ({type(e).__name__}: {e})")
            continue
        if not rows:
            print(f"{m['tag']:20s} ビン不足")
            continue
        b = band_from(rows)
        best = max(r["iou"] for r in rows)
        print(f"{m['tag']:20s} {b[0]:+.2f}m 〜 {b[1]:+.2f}m"
              f"{'':9s} {b[2]:6.2f}m {best:8.3f}")
        if args.detail:
            for r in sorted(rows, key=lambda r: -r["h_m"]):
                bar = "#" * int(r["iou"] / max(best, 1e-9) * 40)
                print(f"      h={r['h_m']:+5.2f}m n={r['n']:8,d} "
                      f"iou={r['iou']:.3f} {bar}")
        out.append({"tag": m["tag"], "lo_m": b[0], "hi_m": b[1],
                    "half_width_m": b[2], "best_iou": best})
    if out:
        p = OUT_DIR / "wall_band_probe.json"
        p.write_text(json.dumps(out, ensure_ascii=False, indent=1))
        print(f"-> {p}")


if __name__ == "__main__":
    main()
