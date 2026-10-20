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

## セッション内の外れ値抑制 (2026-10-16追加)
m_per_unit は**同一セッションではほぼ一定であるべき** (同じカメラ・同じ推論
設定なので)。0828は4島とも0.492〜0.510 = ±2%に収まる。ところが0925は
0.396〜1.430 と**3.6倍**も散っていた。原因は `mapany` の共有フレーム不足で
3島が `slab` にフォールバックしていたこと (slabは0912実測で誤差97〜383%)。

これがアンカーを直接殺す。スケール罰は exp(-2·|log(s/s_prior)|) なので、
事前が2.4倍ずれた島では正解のスコアが**16分の1**に潰れる
(2026-10-16 `eval/anchor_blame`: 0925_s005r2 の正解 scale_dev=1.376、
正解は候補の順位57位に沈み8.24mの偽解が採用された)。

→ `m_per_unit` はセッション全島の**中央値から OUTLIER_TOL 倍を超えて外れた
値を中央値で置き換える**。物理的にあり得ないばらつきなので、個別の推定より
セッション中央値の方が信頼できる。

使い方:
  from recon3d.postproc.scale_prior import m_per_unit, calibrate
  calibrate(tags)                       # アンカー結果から scale_bias.json を書く
  mpu, info = m_per_unit(paths, "pv0912_1F")   # 較正+外れ値抑制を自動適用
"""
import json

import numpy as np

from recon3d.core.geometry import rotation_to_y


def island_xz(paths, tag):
    """島の (frames, 重力で水平化した平面座標)"""
    d = np.load(paths.bench / tag / "global_cameras.npz")
    fr = np.array([int(str(n).replace("frame_", "")) for n in d["names"]])
    poses = d["poses"]
    u = poses[:, :3, 1]
    u = np.median(u / np.linalg.norm(u, axis=1, keepdims=True), 0)
    p = d["positions"] @ rotation_to_y(u / np.linalg.norm(u)).T
    return fr, p[:, [0, 2]]


def _fit_scale(paths, src_fr, src_xy, dst_fr, dst_xy, min_n=8):
    """共有フレームで相似フィット → src 1単位あたりの dst 単位。鏡像も試す"""
    from recon3d.eval.frontend import fit2d
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


def _mapany_tag(paths, tag, fr):
    """この島のフレームを最もよく覆う macond タグを選ぶ"""
    best = (None, 0)
    for p in paths.bench.glob("macond*"):
        if not (p / "global_cameras.npz").exists():
            continue
        mfr, _ = island_xz(paths, p.name)
        n = len(np.intersect1d(fr, mfr))
        if n > best[1]:
            best = (p.name, n)
    return best


def from_mapany(paths, tag):
    """MapAnything条件付け出力(ほぼメートル)との相似フィット"""
    fr, xy = island_xz(paths, tag)
    mtag, n_ov = _mapany_tag(paths, tag, fr)
    if mtag is None:
        raise RuntimeError(f"{tag}: macond出力が見つかりません")
    mfr, mxy = island_xz(paths, mtag)
    s, n = _fit_scale(paths, fr, xy, mfr, mxy)
    if s is None:
        raise RuntimeError(f"{tag}: {mtag} との共有フレームが{n}本しかない")
    return s, {"source": "mapany", "ref": mtag, "n_shared": n}


def from_gt(paths, tag):
    """GT軌跡との相似フィット (上限確認用。本番では使えない)"""
    from recon3d.eval.chunk_cmp import gt_interp
    from recon3d.eval.gt_traj import mm_per_px, load_gt
    fr, xy = island_xz(paths, tag)
    gt = load_gt(paths)
    g, idx = gt_interp(gt, fr, mm_per_px(paths))
    s, n = _fit_scale(paths, fr[idx], xy[idx], fr[idx], g)
    if s is None:
        raise RuntimeError(f"{tag}: GT対応が{n}本しかない")
    return s, {"source": "gt", "n_shared": n}


def from_slab(paths, tag):
    """既定: カメラ高 / 床スラブピーク (点群依存。誤ピークで破綻する)"""
    from recon3d.postproc.anchor import prepare_island
    return prepare_island(paths, tag)["m_per_unit"], {"source": "slab"}


SOURCES = {"slab": from_slab, "mapany": from_mapany, "gt": from_gt}

CALIB_NAME = "scale_bias.json"
MIN_SCORE = 0.02      # これ未満のアンカーは較正に使わない (誤解の s は信用できない)


def anchored_s(paths, tag, suffix="_corners"):
    """アンカー済みなら (事前m_per_unit, 相似スケールs, score) を返す"""
    p = paths.backend / f"anchor_{tag}{suffix}.json"
    if not p.exists():
        return None
    a = json.loads(p.read_text())
    return a.get("m_per_unit"), a.get("s"), a.get("score", 0.0)


def calibrate(paths, tags, suffix="_corners", out_name=CALIB_NAME):
    """アンカー結果から較正ファイルを書く (島ごとの補正値 + セッション中央値)

    セッション中央値はアンカーに失敗した島への保険。散らばり (MAD) も残し、
    小さければ「系統バイアス」、大きければ「島ごとのばらつき」と読める。
    """
    per, ss = {}, []
    for t in tags:
        r = anchored_s(paths, t, suffix)
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
    (paths.backend / out_name).write_text(json.dumps(out, indent=1,
                                               ensure_ascii=False))
    kind = "系統バイアス" if mad < 0.08 else "島ごとのばらつき"
    print(f"[scale_prior] 較正: bias={med:.3f} MAD={mad:.3f} ({kind}, n={len(ss)})"
          f" -> {paths.backend / out_name}")
    return out


def _calib(paths, out_name=CALIB_NAME):
    p = paths.backend / out_name
    return json.loads(p.read_text()) if p.exists() else None


OUTLIER_TOL = 1.5      # セッション中央値のこの倍率を超えたら外れ値とみなす
                       # (0925実測: 健全な島は中央比0.87〜1.36、壊れた島は
                       #  2.20〜3.15。1.5はこの明確な谷間に置いてある)
_SESSION_MED = {}      # (manifest, source) -> 中央値。島ごとに測り直さない


def session_median(paths, source="mapany", fallback="slab",
                   manifest="pv_manifest.json"):
    """セッション全島の m_per_unit の中央値 (外れ値抑制の基準)

    各島の推定は当たり外れがあるが、**中央値は頑健**。同一セッションで
    m_per_unit が変わる物理的理由が無いので、これを真値の代理に使う。
    """
    key = (manifest, source)
    if key in _SESSION_MED:
        return _SESSION_MED[key]
    try:
        man = json.loads((paths.bench / manifest).read_text())
    except Exception:
        _SESSION_MED[key] = None
        return None
    # **主供給源の値だけで中央値を取る**。fallback (slab) は誤差97〜383%で、
    # 混ぜると中央値ごと引っ張られる (0925: 0.454 -> 0.575)
    vals, weak = [], []
    for m in man:
        try:
            v, _ = SOURCES[source](paths, m["tag"])
            tgt = vals
        except Exception:
            try:
                v, _ = SOURCES[fallback](paths, m["tag"])
                tgt = weak
            except Exception:
                continue
        if v and np.isfinite(v) and v > 0:
            tgt.append(float(v))
    use = vals if len(vals) >= 3 else vals + weak
    med = float(np.median(use)) if len(use) >= 3 else None
    _SESSION_MED[key] = med
    return med


def m_per_unit(paths, tag, source="mapany", fallback="slab",
               use_calib=True, clamp_outlier=True):
    """島のm_per_unitを返す。較正があれば適用、取れなければ fallback へ

    優先順: ①その島自身のアンカー由来の補正値 ②事前×セッション中央値 ③素の事前
    """
    cal = _calib(paths) if use_calib else None
    if cal and tag in cal["islands"]:
        v = cal["islands"][tag]["m_per_unit"]
        return float(v), {"source": "calib(self)", "s": cal["islands"][tag]["s"]}
    try:
        v, info = SOURCES[source](paths, tag)
    except Exception as e:
        if not fallback or fallback == source:
            raise
        print(f"[scale_prior] {tag}: {source} 不可 ({e}) -> {fallback}")
        v, info = SOURCES[fallback](paths, tag)
        info = {**info, "fallback_from": source}
    if cal:
        v = v * cal["session_bias"]
        info = {**info, "bias": cal["session_bias"]}
    if clamp_outlier:
        med = session_median(paths, source, fallback)
        # fallbackした島は供給源そのものが信用できない (slabは誤差97〜383%)。
        # 比が許容内でも中央値に寄せた方が安全
        fell_back = "fallback_from" in info
        off = med and not (1 / OUTLIER_TOL <= v / med <= OUTLIER_TOL)
        if med and (fell_back or off):
            # 同一セッションでm_per_unitが変わる物理的理由は無い。
            # 外れた個別推定より中央値の方が信頼できる
            info = {**info, "clamped_from": float(v), "session_median": med,
                    "why": "fallback" if fell_back else "outlier"}
            v = med
    return float(v), info


def main(paths, argv=None):
    """供給源ごとの m_per_unit を島別に並べる / アンカー結果から較正する"""
    import argparse
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--manifest", default="pv_full_manifest.json")
    ap.add_argument("--sources", nargs="*", default=list(SOURCES))
    ap.add_argument("--calibrate", action="store_true",
                    help="アンカー済みの島から scale_bias.json を書く。"
                         "以降は MapAnything 無しでも良い事前が使える")
    ap.add_argument("--suffix", default="_corners",
                    help="--calibrate が読むアンカーjsonの接尾辞")
    args = ap.parse_args(argv)
    man = json.loads((paths.bench / args.manifest).read_text())
    if args.calibrate:
        calibrate(paths, [m["tag"] for m in man], suffix=args.suffix)
        return
    print(f"{'island':16} " + " ".join(f"{s:>10}" for s in args.sources)
          + "   (gt比)")
    for m in man:
        vals = {}
        for s in args.sources:
            try:
                vals[s] = SOURCES[s](paths, m["tag"])[0]
            except Exception:
                vals[s] = float("nan")
        ref = vals.get("gt")
        rel = ("  " + " ".join(f"{vals[s] / ref:.2f}x" for s in args.sources)
               if ref else "")
        print(f"{m['tag']:16} "
              + " ".join(f"{vals[s]:10.3f}" for s in args.sources) + rel)
