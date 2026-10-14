"""島のメートル尺度 (m_per_unit) をどこから取るか

図面アンカーの `match_corners` は事前スケールの **±40% (s_rel_tol)** しか探索
しないので、m_per_unit がずれていると**正解の仮説が生成されない**。
2026-10-13の実測では、既定の床スラブ推定は0912で2〜5倍ずれており、
全周島4本中3本で正解が候補に入っていなかった (`eval/anchor_oracle`)。

供給源 (精度は0912実測):
| source | 誤差 | 中身 |
|---|---|---|
| `slab` | **97〜383%** | 既定。カメラ高 / 床スラブピーク。誤ピークを掴むと破綻 |
| `mapany` | **5〜25%** | MapAnything条件付け出力(ほぼメートル)との相似フィット |
| `gt` | 0% | GT軌跡との相似フィット。上限確認用 (本番では使えない) |

`mapany` と `gt` は島のカメラ軌跡だけで決まるので、点群の床検出に依存しない。

## 較正 (2026-10-14追加)
`mapany` の誤差は物件によって**系統的**なことがある (0828は3島とも約1.27倍)。
アンカー後の相似スケール `s` は事前の誤りをそのまま表し、
**補正後 m_per_unit = 事前 × s** が実測1%以内で成立する (0912で検証)。
これを使えば**GT無しで較正できる**:
  1回目: mapany事前でアンカー → 各島の s を読む
  2回目: 島自身の s で補正した事前で打ち直す (失敗した島はセッション中央値を使う)

使い方:
  from img_align.postproc.scale_prior import m_per_unit, calibrate
  calibrate(tags)                       # アンカー結果から scale_bias.json を書く
  mpu, info = m_per_unit("pv0912_1F")   # 較正があれば自動適用
"""
import json

import numpy as np

from img_align.config import BENCH_DIR
from img_align.core.geometry import rotation_to_y


def island_xz(tag):
    """島の (frames, 重力で水平化した平面座標)"""
    d = np.load(BENCH_DIR / tag / "global_cameras.npz")
    fr = np.array([int(str(n).replace("frame_", "")) for n in d["names"]])
    poses = d["poses"]
    u = poses[:, :3, 1]
    u = np.median(u / np.linalg.norm(u, axis=1, keepdims=True), 0)
    p = d["positions"] @ rotation_to_y(u / np.linalg.norm(u)).T
    return fr, p[:, [0, 2]]


def _fit_scale(src_fr, src_xy, dst_fr, dst_xy, min_n=8):
    """共有フレームで相似フィット → src 1単位あたりの dst 単位。鏡像も試す"""
    from img_align.eval.frontend import fit2d
    common = np.intersect1d(src_fr, dst_fr)
    if len(common) < min_n:
        return None, len(common)
    a = src_xy[np.searchsorted(src_fr, common)]
    b = dst_xy[np.searchsorted(dst_fr, common)]
    best = None
    for sgn in (1, -1):
        x = a.copy()
        x[:, 1] *= sgn
        T = fit2d(x, b)
        e = float(np.median(np.linalg.norm(T(x) - b, axis=1)))
        s = float(np.linalg.norm(T(np.array([[1., 0.]]))
                                 - T(np.array([[0., 0.]]))))
        if best is None or e < best[1]:
            best = (s, e)
    return best[0], len(common)


def _mapany_tag(tag, fr):
    """この島のフレームを最もよく覆う macond タグを選ぶ"""
    best = (None, 0)
    for p in BENCH_DIR.glob("macond*"):
        if not (p / "global_cameras.npz").exists():
            continue
        mfr, _ = island_xz(p.name)
        n = len(np.intersect1d(fr, mfr))
        if n > best[1]:
            best = (p.name, n)
    return best


def from_mapany(tag):
    """MapAnything条件付け出力(ほぼメートル)との相似フィット"""
    fr, xy = island_xz(tag)
    mtag, n_ov = _mapany_tag(tag, fr)
    if mtag is None:
        raise RuntimeError(f"{tag}: macond出力が見つかりません")
    mfr, mxy = island_xz(mtag)
    s, n = _fit_scale(fr, xy, mfr, mxy)
    if s is None:
        raise RuntimeError(f"{tag}: {mtag} との共有フレームが{n}本しかない")
    return s, {"source": "mapany", "ref": mtag, "n_shared": n}


def from_gt(tag, session=None):
    """GT軌跡との相似フィット (上限確認用。本番では使えない)"""
    from img_align.eval.chunk_cmp import gt_interp
    from img_align.eval.gt_traj import MM_PER_PX, load_gt
    fr, xy = island_xz(tag)
    gt = load_gt(session)
    g, idx = gt_interp(gt, fr, MM_PER_PX)
    s, n = _fit_scale(fr[idx], xy[idx], fr[idx], g)
    if s is None:
        raise RuntimeError(f"{tag}: GT対応が{n}本しかない")
    return s, {"source": "gt", "n_shared": n}


def from_slab(tag):
    """既定: カメラ高 / 床スラブピーク (点群依存。誤ピークで破綻する)"""
    from img_align.postproc.anchor import prepare_island
    return prepare_island(tag)["m_per_unit"], {"source": "slab"}


SOURCES = {"slab": from_slab, "mapany": from_mapany, "gt": from_gt}

CALIB_NAME = "scale_bias.json"
MIN_SCORE = 0.02      # これ未満のアンカーは較正に使わない (誤解の s は信用できない)


def anchored_s(tag, suffix="_corners"):
    """アンカー済みなら (事前m_per_unit, 相似スケールs, score) を返す"""
    from img_align.config import OUT_DIR
    p = OUT_DIR / f"anchor_{tag}{suffix}.json"
    if not p.exists():
        return None
    a = json.loads(p.read_text())
    return a.get("m_per_unit"), a.get("s"), a.get("score", 0.0)


def calibrate(tags, suffix="_corners", out_name=CALIB_NAME):
    """アンカー結果から較正ファイルを書く (島ごとの補正値 + セッション中央値)

    セッション中央値はアンカーに失敗した島への保険。散らばり (MAD) も残し、
    小さければ「系統バイアス」、大きければ「島ごとのばらつき」と読める。
    """
    from img_align.config import OUT_DIR
    per, ss = {}, []
    for t in tags:
        r = anchored_s(t, suffix)
        if r is None or not r[0] or not r[1] or r[2] < MIN_SCORE:
            continue
        prior, s, score = r
        per[t] = {"m_per_unit": round(prior * s, 5), "s": round(s, 4),
                  "score": round(score, 4)}
        ss.append(s)
    if not ss:
        print("[scale_prior] 較正に使えるアンカーがありません")
        return None
    med = float(np.median(ss))
    mad = float(np.median(np.abs(np.array(ss) - med)))
    out = {"session_bias": round(med, 4), "mad": round(mad, 4),
           "n": len(ss), "islands": per}
    (OUT_DIR / out_name).write_text(json.dumps(out, indent=1,
                                               ensure_ascii=False))
    kind = "系統バイアス" if mad < 0.08 else "島ごとのばらつき"
    print(f"[scale_prior] 較正: bias={med:.3f} MAD={mad:.3f} ({kind}, n={len(ss)})"
          f" -> {OUT_DIR / out_name}")
    return out


def _calib(out_name=CALIB_NAME):
    from img_align.config import OUT_DIR
    p = OUT_DIR / out_name
    return json.loads(p.read_text()) if p.exists() else None


def m_per_unit(tag, source="mapany", fallback="slab", use_calib=True):
    """島のm_per_unitを返す。較正があれば適用、取れなければ fallback へ

    優先順: ①その島自身のアンカー由来の補正値 ②事前×セッション中央値 ③素の事前
    """
    cal = _calib() if use_calib else None
    if cal and tag in cal["islands"]:
        v = cal["islands"][tag]["m_per_unit"]
        return float(v), {"source": "calib(self)", "s": cal["islands"][tag]["s"]}
    try:
        v, info = SOURCES[source](tag)
    except Exception as e:
        if not fallback or fallback == source:
            raise
        print(f"[scale_prior] {tag}: {source} 不可 ({e}) -> {fallback}")
        v, info = SOURCES[fallback](tag)
        info = {**info, "fallback_from": source}
    if cal:
        v = v * cal["session_bias"]
        info = {**info, "bias": cal["session_bias"]}
    return float(v), info


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="pv_full_manifest.json")
    ap.add_argument("--sources", nargs="*", default=list(SOURCES))
    args = ap.parse_args()
    man = json.loads((BENCH_DIR / args.manifest).read_text())
    print(f"{'island':16} " + " ".join(f"{s:>10}" for s in args.sources)
          + "   (gt比)")
    for m in man:
        vals = {}
        for s in args.sources:
            try:
                vals[s] = SOURCES[s](m["tag"])[0]
            except Exception:
                vals[s] = float("nan")
        ref = vals.get("gt")
        rel = ("  " + " ".join(f"{vals[s] / ref:.2f}x" for s in args.sources)
               if ref else "")
        print(f"{m['tag']:16} "
              + " ".join(f"{vals[s]:10.3f}" for s in args.sources) + rel)
