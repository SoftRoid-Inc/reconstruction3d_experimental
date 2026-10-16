"""壁の高さ帯をデータから測る (密度地図の素材を壁だけに絞るため)

図面には**壁しか描かれていない**のに、密度地図は従来すべての高さの点を使って
いた。完成物件ではソファ・カウンター・棚・カーテン・設備が入り、図面と一致
しない。結果、正しい位置より偽の位置の方が被覆(f1)で勝ってしまう
(2026-10-16 `eval/anchor_blame`: 0925の失敗島で 正解f1 < 偽解f1、比2.7〜5.1倍)。

**壁は鉛直**なので、真上から見た footprint が高さによらず同じになる。家具は
鉛直でないので高さを変えると形が変わる。この性質だけで帯を決められる —
床や天井の検出が要らないのが重要で、`0925_s003r2` のように**床の点群が欠けて
いる島**でも測れる (床基準の推定はそこで破綻する)。

帯は**カメラ高さ基準**で表す。カメラ高さは cam_pos の中央値として直接測れ、
FlowStateで水平が保たれ撮影者の身長も一定なので安定した基準になる。
なお壁はカメラの上下に対称には無い (カメラ1.4mなら上に約1.0m・下に約1.4m)
ので、**非対称な帯**になるのが正常。
"""
import numpy as np

MIN_BIN_PTS = 2000     # これ未満の点しかないビンは判定に使わない
IOU_FRAC = 0.75        # 最良ビンのこの割合以上の一致度を「壁らしい」とする
FALLBACK_HALF_M = 0.9  # 測れなかったときのカメラ基準片側幅
MIN_BAND_M = 0.8       # これより狭い帯は信用しない
MIN_KEEP_FRAC = 0.25   # 帯で点がこれ未満しか残らないなら信用しない


def _occupancy(xz, cell_u, shape, origin):
    ox, oz = origin
    ix = np.clip(((xz[:, 0] - ox) / cell_u).astype(np.int32), 0, shape[1] - 1)
    iz = np.clip(((xz[:, 1] - oz) / cell_u).astype(np.int32), 0, shape[0] - 1)
    occ = np.zeros(shape, bool)
    occ[iz, ix] = True
    return occ


def bin_profile(pts, cam_y, mpu, cell_m=0.05, bin_m=0.20):
    """高さビンごとの「壁らしさ」 [{h_m, n, iou}]。h_m はカメラ基準 (上が正)"""
    cell_u, bin_u = cell_m / mpu, bin_m / mpu
    ox, oz = pts[:, 0].min() - 2 * cell_u, pts[:, 2].min() - 2 * cell_u
    shape = (int((pts[:, 2].max() - oz) / cell_u) + 3,
             int((pts[:, 0].max() - ox) / cell_u) + 3)
    iy = np.floor((pts[:, 1] - cam_y) / bin_u).astype(np.int64)
    occs = {}
    for b in np.unique(iy):
        m = iy == b
        if m.sum() >= MIN_BIN_PTS:
            occs[int(b)] = _occupancy(pts[m][:, [0, 2]], cell_u, shape,
                                      (ox, oz))
    if len(occs) < 3:
        return []
    # 合意マップ = 過半数のビンが占有するセル (高さによらず在るもの = 壁)
    stack = np.stack(list(occs.values()))
    consensus = stack.sum(axis=0) >= max(2, int(0.5 * len(occs)))
    out = []
    for b, occ in sorted(occs.items()):
        union = (occ | consensus).sum()
        inter = (occ & consensus).sum()
        out.append({"bin": b, "h_m": -(b + 0.5) * bin_m,
                    "n": int((iy == b).sum()),
                    "cells": int(occ.sum()),
                    "iou": float(inter / max(union, 1)),
                    # precision = このスライスの占有のうち鉛直に伸びている割合。
                    # IoUと違い**面積の大きさに報酬が出ない**ので、点が疎でも
                    # 中身が壁なら高く出る (天井側が不当に沈まない)
                    "prec": float(inter / max(occ.sum(), 1))})
    return out


def measure_band(pts, cam_y, mpu, frac=IOU_FRAC, **kw):
    """壁の帯 (lo_m, hi_m) をカメラ基準[m]で返す。測れなければ None

    カメラ高さを含む**連続区間**だけを採る。飛び地は家具の塊なので入れない。
    """
    rows = bin_profile(pts, cam_y, mpu, **kw)
    if not rows:
        return None
    best = max(r["iou"] for r in rows)
    ok = {r["bin"] for r in rows if r["iou"] >= best * frac}
    # **カメラ高さを含むこと**が必須。含まない帯は天井付近の塊などを掴んで
    # いるだけで、0.4%しか点が残らないような退化解になる (2026-10-16実測)
    center = 0 if 0 in ok else (-1 if -1 in ok else None)
    if center is None:
        return None
    lo = hi = center
    while lo - 1 in ok:
        lo -= 1
    while hi + 1 in ok:
        hi += 1
    hs = [r["h_m"] for r in rows if lo <= r["bin"] <= hi]
    band = float(min(hs)), float(max(hs))
    if band[1] - band[0] < MIN_BAND_M:
        return None
    kept = sum(r["n"] for r in rows if lo <= r["bin"] <= hi)
    if kept < MIN_KEEP_FRAC * sum(r["n"] for r in rows):
        return None
    return band


def band_mask(pts, cam_y, mpu, band):
    """帯に入る点のマスク。band は (lo_m, hi_m) / 数値(片側幅) / "auto" """
    if band == "auto":
        b = measure_band(pts, cam_y, mpu)
        if b is None:
            print(f"  [band] 測定不能 -> ±{FALLBACK_HALF_M}m に退避")
            b = (-FALLBACK_HALF_M, FALLBACK_HALF_M)
    elif isinstance(band, (int, float)):
        b = (-float(band), float(band))
    else:
        b = (float(band[0]), float(band[1]))
    # 整列系は y+ が物理下向きなので、カメラ基準の高さは -(y - cam_y)
    h = -(pts[:, 1] - cam_y) * mpu
    return (h >= b[0]) & (h <= b[1]), b
