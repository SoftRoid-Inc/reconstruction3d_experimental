"""島をメートル尺度の単一座標系へ合流させる (図面も階ラベルも使わない)

島はそれぞれ独立したゲージ (位置・向き・尺度が任意) なので、そのままでは
並べられない。図面アンカーは**階が既知**でないと使えず、階の自動決定 (C1) は
未解決なので、ここでは**図面も階も使わない経路**を用意する。

手順:
  1. MapAnythingをセッション全体に1回かける (`macond<sess>_ALL`、stride間引き)。
     フィードフォワードなので階段を跨いでも1つの枠に収まり、ほぼメートル
  2. 各島をその全体枠へ相似フィット (共有フレームで解く)
  3. 全島が同一のメートル枠に載る

精度の内訳:
  - **島の内部**: PanoVGGTの精度そのまま (フィット残差 0.12〜0.50m)
  - **島と島の相対**: 全体枠の精度に依存 (MapAnythingのメートル誤差 5〜25%)
つまり「各部屋は正確、建物全体の骨格はMapAnything任せ」という性質になる。

## ブリッジ方式 (--mode bridge)
全体を1本の基準で合わせる方式は、基準自体が長尺で破綻すると全島が巻き添えになる
(2026-10-14実測: 0912はPDR基準でATE 4.05m、階にも島にも偏りが無く基準の破綻と判明)。
そこで**島と島の継ぎ目だけ**をMapAnythingで解く:

  島A ---[ブリッジ(両隣に24fずつ食い込む)]--- 島B

ブリッジは両隣と共有フレームを持つので、A→ブリッジ→B と相似変換を合成すれば
Bを Aの枠に載せられる。これを鎖状に繋ぐ。窓は33〜50ビューに収まり
MapAnythingの得意な密度。島内精度 (0.12〜0.50m) はそのまま活きる。

使い方:
  IMG_SESSION=0912 python -m img_align.postproc.metric_stitch --mode bridge --eval
  IMG_SESSION=0912 python -m img_align.postproc.metric_stitch --ref macond0912_ALL
出力: backend/metric_traj.json (frame -> [x, y] メートル) + metric_stitch.json (診断)
"""
import argparse
import json

import numpy as np

from img_align.config import BENCH_DIR, OUT_DIR, SESSION
from img_align.postproc.scale_prior import island_xz

MIN_SHARED = 6      # 全体枠と島の共有フレーム数の下限


def fit_sim2(src, dst):
    """src -> dst の相似変換 (鏡像も試す)。(関数, 尺度, 残差, 鏡像) を返す"""
    from img_align.eval.frontend import fit2d
    best = None
    for sgn in (1, -1):
        x = src.copy()
        x[:, 1] *= sgn
        T = fit2d(x, dst)
        e = float(np.median(np.linalg.norm(T(x) - dst, axis=1)))
        s = float(np.linalg.norm(T(np.array([[1., 0.]]))
                                 - T(np.array([[0., 0.]]))))
        if best is None or e < best[2]:
            best = (T, s, e, sgn)
    return best


def stitch(ref_tag, tags):
    """各島を ref_tag (メートル枠) へ合流 → {frame: (x, y)} と診断"""
    rfr, rxy = island_xz(ref_tag)
    traj, diag = {}, []
    for tag in tags:
        fr, xy = island_xz(tag)
        common = np.intersect1d(fr, rfr)
        if len(common) < MIN_SHARED:
            diag.append({"tag": tag, "error": f"共有フレーム {len(common)}本"})
            print(f"[stitch] {tag}: 共有 {len(common)}本 -> skip")
            continue
        T, s, e, sgn = fit_sim2(xy[np.searchsorted(fr, common)],
                                rxy[np.searchsorted(rfr, common)])
        x = xy.copy()
        x[:, 1] *= sgn
        out = T(x)
        for f, p in zip(fr, out):
            traj.setdefault(int(f), []).append(p)
        diag.append({"tag": tag, "n_shared": int(len(common)),
                     "m_per_unit": round(s, 4), "fit_resid_m": round(e, 3),
                     "mirror": sgn < 0, "frames": [int(fr[0]), int(fr[-1])]})
        print(f"[stitch] {tag}: 共有{len(common):3d}本  m/unit {s:.3f}  "
              f"合流残差 {e:.3f}m")
    # 島が重なるフレームは平均
    return {f: np.mean(v, axis=0).tolist() for f, v in traj.items()}, diag


def stitch_bridges(tags, bridge_fmt="bridge{sess}_{i:02d}"):
    """島をブリッジで鎖状に繋ぐ。最初の島の枠を基準にする"""
    traj, diag = {}, []
    place = {tags[0]: lambda p: p}      # 1本目の枠を基準にする
    for i in range(len(tags) - 1):
        btag = bridge_fmt.format(sess=SESSION, i=i)
        if not (BENCH_DIR / btag / "global_cameras.npz").exists():
            diag.append({"bridge": btag, "error": "出力なし"})
            print(f"[bridge] {btag}: 出力なし -> 鎖が切れる")
            continue
        bfr, bxy = island_xz(btag)
        A, B = tags[i], tags[i + 1]
        afr, axy = island_xz(A)
        bfr2, bxy2 = island_xz(B)
        ca = np.intersect1d(afr, bfr)
        cb = np.intersect1d(bfr2, bfr)
        if len(ca) < MIN_SHARED or len(cb) < MIN_SHARED:
            diag.append({"bridge": btag, "error": f"共有 A{len(ca)} B{len(cb)}"})
            print(f"[bridge] {btag}: 共有 A{len(ca)} B{len(cb)} -> skip")
            continue
        # A -> ブリッジ枠
        Ta, sa, ea, ga = fit_sim2(axy[np.searchsorted(afr, ca)],
                                  bxy[np.searchsorted(bfr, ca)])
        # ブリッジ枠 -> B ... の逆、つまり B -> ブリッジ枠
        Tb, sb, eb, gb = fit_sim2(bxy2[np.searchsorted(bfr2, cb)],
                                  bxy[np.searchsorted(bfr, cb)])
        diag.append({"bridge": btag, "A": A, "B": B,
                     "n_shared_A": int(len(ca)), "n_shared_B": int(len(cb)),
                     "resid_A_m": round(ea, 3), "resid_B_m": round(eb, 3),
                     "mpu_A": round(sa, 4), "mpu_B": round(sb, 4)})
        print(f"[bridge] {btag}: A共有{len(ca):3d}(残差{ea:.2f}m) "
              f"B共有{len(cb):3d}(残差{eb:.2f}m)  m/unit A{sa:.3f} B{sb:.3f}")
        # A はすでに基準枠にある。ブリッジ枠 -> 基準枠 は A の逆写像で作る
        pa = place[A]
        inv = _invert(Ta, ga, axy[np.searchsorted(afr, ca)])
        place[B] = (lambda p, pa=pa, inv=inv, Tb=Tb, gb=gb:
                    pa(inv(_apply_mirror(Tb, gb, p))))
    for t in tags:
        if t not in place:
            print(f"[bridge] {t}: 鎖から外れた -> skip")
            continue
        fr, xy = island_xz(t)
        out = place[t](xy)
        for f, p in zip(fr, out):
            traj.setdefault(int(f), []).append(p)
    return {f: np.mean(v, axis=0) for f, v in traj.items()}, diag


def _apply_mirror(T, g, p):
    x = np.asarray(p, dtype=float).copy()
    x[:, 1] *= g
    return T(x)


def _invert(T, g, src):
    """T∘mirror の逆写像を、対応点から相似フィットして作る"""
    y = _apply_mirror(T, g, src)
    Ti, _, _, gi = fit_sim2(y, src)
    return lambda p: _apply_mirror(Ti, gi, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=None,
                    help="メートル枠のタグ (既定 macond<sess>_ALL)")
    ap.add_argument("--manifest", default="pv_manifest.json")
    ap.add_argument("--eval", action="store_true", help="GTと比較する")
    ap.add_argument("--mode", default="ref", choices=("ref", "bridge"),
                    help="ref=単一基準に合わせる / bridge=継ぎ目ごとに繋ぐ")
    args = ap.parse_args()
    man = json.loads((BENCH_DIR / args.manifest).read_text())
    tags = [m["tag"] for m in sorted(man, key=lambda m: m["a"])]
    if args.mode == "bridge":
        traj, diag = stitch_bridges(tags)
        ref = "bridges"
    else:
        traj, diag = _ref_mode(args, tags)
        ref = args.ref or f"macond{SESSION}_ALL"
    _finish(traj, diag, ref, args)


def _ref_mode(args, tags):
    ref = args.ref or f"macond{SESSION}_ALL"
    if not (BENCH_DIR / ref / "global_cameras.npz").exists():
        raise SystemExit(
            f"{ref} がありません。先に全体枠を作ってください:\n"
            f"  docker compose exec -e IMG_SESSION={SESSION} mapanything \\\n"
            f"    python -m img_align.postproc.mapany infer --tag {ref} "
            f"--frames 0:<N> --stride <N/55> --minibatch 1")
    return stitch(ref, tags)


def _finish(traj, diag, ref, args):
    (OUT_DIR / "metric_traj.json").write_text(json.dumps(
        {str(k): list(map(float, v)) for k, v in sorted(traj.items())}, indent=1))
    res = {"ref": ref, "n_frames": len(traj), "islands": diag}

    if args.eval:
        from img_align.eval.gt_traj import MM_PER_PX, load_gt
        gt = load_gt()
        fr = np.array(sorted(traj))
        est = np.array([traj[f] for f in fr])
        m = np.array([f in gt for f in fr])
        g = np.array([[gt[f]["x_px"], gt[f]["y_px"]]
                      for f in fr[m]]) * MM_PER_PX / 1000
        if m.sum() >= 8:
            T, s, e, _ = fit_sim2(est[m], g)
            err = np.linalg.norm(T(est[m] * [1, 1]) - g, axis=1)
            res["eval"] = {"n": int(m.sum()),
                           "scale_vs_gt": round(s, 4),
                           "ate_median_m": round(float(np.median(err)), 3),
                           "ate_p90_m": round(float(np.percentile(err, 90)), 3)}
            print(f"\nGT比: スケール {s:.3f} (=メートル誤差 {abs(s-1)*100:.0f}%)  "
                  f"ATE median {np.median(err):.3f}m p90 "
                  f"{np.percentile(err, 90):.3f}m  ({int(m.sum())}点)")
    (OUT_DIR / "metric_stitch.json").write_text(
        json.dumps(res, indent=1, ensure_ascii=False))
    print(f"-> {OUT_DIR / 'metric_traj.json'} ({len(traj)} frames)")


if __name__ == "__main__":
    main()
