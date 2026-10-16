"""橋渡しチャンクが切れ目を正しく跨げているかをGTで直接測る

**--join が本命の評価**。橋単体をGTに当てる測り方は両方とも公平でない:
  全体フィット  … 誤差を両側に分配するので**甘く出る**
  運ぶ誤差      … 片側8枚 (≒4秒歩行) はほぼ直線で相似フィットが悪条件。
                   スケールが決まらないまま外挿するので**厳しく出る**
                   (実測で尺度比が0.46〜3.92に散った)
--join は橋で島Bを島Aの座標系へ移し、**結合した200フレーム**をGTに1回
フィットする。評価点が多く条件が良い上、パイプラインの実際の出力と一致する。

パイプライン(島チェーン→アンカー)を通さずに橋単体を評価する。GTは図面px
なので、橋のカメラ中心を水平投影してGTに相似フィット(スケール+回転+並進)
すれば、残差がそのまま「橋の形の正しさ」になる。

**両側それぞれのフィット**も出すのが要点。橋が破綻しているときは、
片側ずつなら合うのに全体では合わない (=両側の相対関係だけが壊れている)。
  側別が小さく全体が大きい → 跨ぎに失敗。橋としては使えない
  側別も全体も小さい       → 成功。島間の相対Sim(3)として採用できる

使い方:
  IMG_SESSION=0912 python -m img_align.eval.bridge_eval
  IMG_SESSION=0912 python -m img_align.eval.bridge_eval --modes span both
"""
import argparse

import numpy as np

from img_align.core.bridge import bridge_frames, junctions, run_bridge
from img_align.eval.gt_traj import MM_PER_PX, load_gt


def _sim2_fit(src, dst, refl=None):
    """2D相似フィット。返り値 (残差中央値[px], 変換後, 変換(k, cs, cd, refl))

    **鏡映を許す**のが要点。PanoVGGTのworld系は図面と掌性が逆で、
    density_map は z 反転で吸収しているが、チェーンの生の姿勢は未補正。
    鏡映なしで当てると全島が1〜3mになり、橋の良し悪しが見えなくなる。
    refl を指定すると固定 (結合評価では全部分で同じ掌性を使うため)。
    """
    if len(src) < 3:
        return None, None, None
    cs, cd = src.mean(axis=0), dst.mean(axis=0)
    s0, d0 = src - cs, dst - cd
    # 複素数にすると相似変換は1つの複素スカラー倍で書ける
    zd = (d0[:, 0] + 1j * d0[:, 1])
    best = None
    for r in ((False, True) if refl is None else (refl,)):
        zs = (s0[:, 0] - 1j * s0[:, 1]) if r else (s0[:, 0] + 1j * s0[:, 1])
        k = (zd * zs.conj()).sum() / max((zs * zs.conj()).sum().real, 1e-12)
        out = zs * k
        fit = np.stack([out.real, out.imag], axis=1) + cd
        e = float(np.median(np.linalg.norm(fit - dst, axis=1)))
        if best is None or e < best[0]:
            best = (e, fit, (k, cs, cd, r))
    return best


def _apply2(T, pts):
    k, cs, cd, r = T
    dx, dy = pts[:, 0] - cs[0], pts[:, 1] - cs[1]
    z = (dx - 1j * dy) if r else (dx + 1j * dy)
    w = z * k
    return np.stack([w.real, w.imag], axis=1) + cd


def evaluate(a_end, b_start, mode, gt, horiz=(0, 2)):
    """橋1本を評価。(全体残差m, A側残差m, B側残差m, 使用枚数) を返す"""
    pos = run_bridge(a_end, b_start, mode)
    frames = [f for f in sorted(pos) if f in gt]
    if len(frames) < 6:
        return None
    src = np.stack([[pos[f][horiz[0]], pos[f][horiz[1]]] for f in frames])
    dst = np.stack([[gt[f]["x_px"], gt[f]["y_px"]] for f in frames])
    mid = (a_end + b_start) / 2
    ia = [i for i, f in enumerate(frames) if f <= mid]
    ib = [i for i, f in enumerate(frames) if f > mid]
    r_all, _, Tall = _sim2_fit(src, dst)
    refl = Tall[3]
    r_a, _, Ta = _sim2_fit(src[ia], dst[ia], refl=refl)
    r_b, _, Tb = _sim2_fit(src[ib], dst[ib], refl=refl)
    m = MM_PER_PX / 1000
    # **運ぶ誤差**: A側を正しく合わせた変換でB側を運んだときのGTとのずれ。
    # 「島Aが正しく図面に乗っているとき、橋で運んだ島Bがどれだけずれるか」
    # = パイプラインで実際に効く量。全体フィットは両側に誤差を分配するので
    # これより小さく出てしまい、判断を誤らせる
    carry = scale_ratio = None
    if Ta is not None and len(ib) >= 3:
        carry = float(np.median(np.linalg.norm(
            _apply2(Ta, src[ib]) - dst[ib], axis=1))) * m
    if Ta is not None and Tb is not None:
        scale_ratio = float(abs(Tb[0]) / max(abs(Ta[0]), 1e-12))
    return (r_all * m, (r_a or 0) * m, (r_b or 0) * m, carry, scale_ratio,
            len(frames), len(ia), len(ib))


def _islands(a_end, b_start):
    """切れ目の両側の島タグ (A側=終端が a_end 寄り / B側=始端が b_start 寄り)"""
    import json
    from img_align.config import BENCH_DIR
    from img_align.core.floor_assign import apply as apply_floor
    man = apply_floor(json.loads(
        (BENCH_DIR / "pv_manifest.json").read_text()), quiet=True)
    a = [m for m in man if m["b"] <= a_end + 2]
    b = [m for m in man if m["a"] >= b_start - 2]
    if not a or not b:
        return None, None
    return max(a, key=lambda m: m["b"]), min(b, key=lambda m: m["a"])


def join_eval(a_end, b_start, mode, gt, horiz=(0, 2)):
    """橋で島Bを島Aの座標系へ移し、結合軌跡をGTと比べる (本命の評価)

    mode が "nNsM" 形式 (例 n8s2) ならチェーン端から選ぶ推奨経路を使う。
    """
    from img_align.core.bridge import (_chain_poses, fit_side,
                                       run_chain_bridge)
    from img_align.core.geometry import apply as apply3
    ma, mb = _islands(a_end, b_start)
    if ma is None:
        return None
    ca, cb = _chain_poses(ma["tag"]), _chain_poses(mb["tag"])
    if mode.startswith("n") and "s" in mode:
        n, st = (int(v) for v in mode[1:].split("s"))
        pos, af, bf_ = run_chain_bridge(ma["tag"], mb["tag"], n, st)
        fa = fit_side(pos, ca, af)
        fb = fit_side(pos, cb, bf_)
    else:
        pos = run_bridge(a_end, b_start, mode)
        bf = sorted(pos)
        mid = (a_end + b_start) / 2
        fa = fit_side(pos, ca, [f for f in bf if f <= mid])
        fb = fit_side(pos, cb, [f for f in bf if f > mid])
    if fa is None or fb is None:
        return None
    sa, Ra, ta, resid_a, na = fa
    sb, Rb, tb, resid_b, nb = fb
    # 島B (Bゲージ) -> 橋ゲージ -> Aゲージ
    kb = np.stack([cb[f] for f in sorted(cb)])
    inv = (1.0 / sb, Rb.T, -(Rb.T @ tb) / sb)
    joined = apply3((sa, Ra, ta), apply3(inv, kb))
    ka = np.stack([ca[f] for f in sorted(ca)])
    frames = sorted(ca) + sorted(cb)
    allpos = np.vstack([ka, joined])
    keep = [i for i, f in enumerate(frames) if f in gt]
    if len(keep) < 20:
        return None
    src = allpos[keep][:, [horiz[0], horiz[1]]]
    dst = np.stack([[gt[frames[i]]["x_px"], gt[frames[i]]["y_px"]]
                    for i in keep])
    na_k = [i for i in keep if i < len(ka)]
    nb_k = [i for i in keep if i >= len(ka)]
    if len(na_k) < 3 or len(nb_k) < 3:
        return None
    # 掌性は**島A単独**で決めて全部に固定する。結合全体で選ぶと、橋が
    # 壊れているときに鏡映の選択まで引きずられ、島単独の残差まで変わって
    # 比較にならない (2026-10-15: 同じ島が0.19mと2.28mに振れた)
    _, _, Ta0 = _sim2_fit(allpos[na_k][:, [horiz[0], horiz[1]]],
                          np.stack([[gt[frames[i]]["x_px"],
                                     gt[frames[i]]["y_px"]] for i in na_k]))
    refl = Ta0[3]
    r_join, _, _ = _sim2_fit(src, dst, refl=refl)
    r_a, _, _ = _sim2_fit(allpos[na_k][:, [horiz[0], horiz[1]]],
                          np.stack([[gt[frames[i]]["x_px"],
                                     gt[frames[i]]["y_px"]] for i in na_k]),
                          refl=refl)
    r_b, _, _ = _sim2_fit(allpos[nb_k][:, [horiz[0], horiz[1]]],
                          np.stack([[gt[frames[i]]["x_px"],
                                     gt[frames[i]]["y_px"]] for i in nb_k]),
                          refl=refl)
    m = MM_PER_PX / 1000
    return (r_join * m, (r_a or 0) * m, (r_b or 0) * m, resid_a, resid_b,
            ma["tag"], mb["tag"], len(keep))


def bridge_transform(a_end, b_start, mode):
    """島Bゲージ → 島Aゲージ の Sim(3)。(T, タグ対)。失敗時は (None, None)

    橋を鎖状に合成して「群れ」を作るために、位置でなく変換を返す。
    """
    from img_align.core.bridge import _chain_poses, fit_side, run_chain_bridge
    from img_align.core.geometry import compose
    ma, mb = _islands(a_end, b_start)
    if ma is None:
        return None, None
    ca, cb = _chain_poses(ma["tag"]), _chain_poses(mb["tag"])
    if mode.startswith("n") and "s" in mode:
        n, st = (int(v) for v in mode[1:].split("s"))
        pos, af, bf_ = run_chain_bridge(ma["tag"], mb["tag"], n, st)
        fa, fb = fit_side(pos, ca, af), fit_side(pos, cb, bf_)
    else:
        pos = run_bridge(a_end, b_start, mode)
        bf = sorted(pos)
        mid = (a_end + b_start) / 2
        fa = fit_side(pos, ca, [f for f in bf if f <= mid])
        fb = fit_side(pos, cb, [f for f in bf if f > mid])
    if fa is None or fb is None:
        return None, None
    sa, Ra, ta, _, _ = fa
    sb, Rb, tb, _, _ = fb
    inv = (1.0 / sb, Rb.T, -(Rb.T @ tb) / sb)
    return compose((sa, Ra, ta), inv), (ma["tag"], mb["tag"])


def transported(a_end, b_start, mode):
    """橋で島Bを島Aの座標系へ運んだ位置 (GT不要)。(位置(N,3), タグ対)"""
    from img_align.core.bridge import (_chain_poses, fit_side,
                                       run_chain_bridge)
    from img_align.core.geometry import apply as apply3
    ma, mb = _islands(a_end, b_start)
    if ma is None:
        return None, None
    ca, cb = _chain_poses(ma["tag"]), _chain_poses(mb["tag"])
    if mode.startswith("n") and "s" in mode:
        n, st = (int(v) for v in mode[1:].split("s"))
        pos, af, bf_ = run_chain_bridge(ma["tag"], mb["tag"], n, st)
        fa, fb = fit_side(pos, ca, af), fit_side(pos, cb, bf_)
    else:
        pos = run_bridge(a_end, b_start, mode)
        bf = sorted(pos)
        mid = (a_end + b_start) / 2
        fa = fit_side(pos, ca, [f for f in bf if f <= mid])
        fb = fit_side(pos, cb, [f for f in bf if f > mid])
    if fa is None or fb is None:
        return None, None
    sa, Ra, ta, _, _ = fa
    sb, Rb, tb, _, _ = fb
    kb = np.stack([cb[f] for f in sorted(cb)])
    inv = (1.0 / sb, Rb.T, -(Rb.T @ tb) / sb)
    return apply3((sa, Ra, ta), apply3(inv, kb)), (ma["tag"], mb["tag"])


def consensus(a_end, b_start, modes):
    """方式間で運び先がどれだけ一致するか [m]。GTを使わない採否判定の候補

    橋の残差 (両側チェーンへのフィット) は**当てにならない** (2026-10-15実測:
    残差0.128/0.032で結合誤差4.30m)。橋が「両側には合うが相対関係が誤り」
    の状態を残差は検出できないため。一致度なら別の情報になりうる。
    """
    outs = []
    for m in modes:
        t, tags = transported(a_end, b_start, m)
        if t is not None:
            outs.append(t)
    if len(outs) < 2:
        return None
    n = min(len(o) for o in outs)
    d = []
    for i in range(len(outs)):
        for j in range(i + 1, len(outs)):
            d.append(np.median(np.linalg.norm(outs[i][:n] - outs[j][:n],
                                              axis=1)))
    return float(np.median(d))


def junction_kind(a_end, b_start):
    """切れ目の種類を返す ("階段" / "静止" / "分割" / "不明")

    島は①IMUの階段イベント ②静止 (ZUPT) ③--max-seg の等分割 で切れる。
    ③は**連続したフレームの人為的な分割**なので視覚的な重なりがあり、
    ①②とは橋の難易度が本質的に違う。混ぜて集計すると傾向が見えない。
    """
    import json
    from img_align.config import IMU_DIR
    gap = b_start - a_end - 1
    if gap <= 1:
        return "分割"
    tr = IMU_DIR / "transitions.json"
    if tr.exists():
        for e in json.loads(tr.read_text())["events"]:
            if int(e["frame_start"]) <= b_start and int(e["frame_end"]) >= a_end:
                return "階段"
    il = IMU_DIR / "idle_spans.json"
    if il.exists():
        for sp in json.loads(il.read_text())["spans"]:
            if int(sp["frame_start"]) <= b_start and \
                    int(sp["frame_end"]) >= a_end:
                return "静止"
    return "不明"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="*", default=["ends", "span", "both"])
    ap.add_argument("--junction", help="a_end:b_start (省略時は全切れ目)")
    ap.add_argument("--join", action="store_true",
                    help="島を橋で結合して結合軌跡をGTと比べる (本命)")
    ap.add_argument("--consensus", action="store_true",
                    help="方式間の一致度とGT誤差を並べる (採否判定の検証)")
    args = ap.parse_args()
    gt = load_gt()
    if args.junction:
        js = [tuple(int(v) for v in args.junction.split(":"))]
    elif args.consensus:
        # グループ化が実際に見る切れ目 = **島の隣接ペア**。IMU区間の切れ目だけ
        # 見ると --max-seg の分割 (連続フレームの人為的な分割) が漏れる。
        # 分割は視覚的な重なりがあるので、橋の難易度の対照群になる
        from img_align.eval.bridge_view import _manifest_islands
        man = _manifest_islands()
        js = [(A["b"], B["a"]) for A, B in zip(man, man[1:])]
    else:
        js = junctions()
    if args.consensus:
        print(f"{'切れ目':>12s} {'種類':>5s} {'階':>8s} {'隙間':>5s} "
              f"{'一致度':>8s} {'最良GT':>8s} {'最悪GT':>8s}")
        for a, b in js:
            try:
                c = consensus(a, b, args.modes)
                es = []
                for m in args.modes:
                    r = join_eval(a, b, m, gt)
                    if r:
                        es.append(r[0])
            except Exception as e:
                print(f"{a:5d}->{b:<5d}  失敗 ({type(e).__name__}: {e})")
                continue
            ma, mb = _islands(a, b)
            fl = (f"{ma['floor']}->{mb['floor']}" if ma is not None else "?")
            same = ("同一" if ma is not None and ma["floor"] == mb["floor"]
                    else "またぎ")
            kind = junction_kind(a, b)
            head = (f"{a:5d}->{b:<5d} {kind:>5s} {fl:>8s} "
                    f"{b - a - 1:4d}f {same}")
            if c is None or not es:
                print(f"{head}  評価不能")
                continue
            print(f"{head} {c:7.2f}m {min(es):7.2f}m {max(es):7.2f}m")
        return
    if args.join:
        print(f"{'切れ目':>12s} {'mode':>6s} {'結合':>7s} {'島A単独':>8s} "
              f"{'島B単独':>8s} {'橋残差A':>8s} {'橋残差B':>8s}  判定")
        for a, b in js:
            for mode in args.modes:
                try:
                    r = join_eval(a, b, mode, gt)
                except Exception as e:
                    print(f"{a:5d}->{b:<5d} {mode:>6s}  失敗 "
                          f"({type(e).__name__}: {e})")
                    continue
                if r is None:
                    print(f"{a:5d}->{b:<5d} {mode:>6s}  島/GT不足")
                    continue
                rj, ra, rb, qa, qb, ta, tb, n = r
                # 結合しても単独より悪くならない = 橋が壊していない
                worst = max(ra, rb)
                mark = ("合格" if rj < worst * 1.5 + 0.2
                        else ("可" if rj < worst * 3 else "不可"))
                print(f"{a:5d}->{b:<5d} {mode:>6s} {rj:6.2f}m {ra:7.2f}m "
                      f"{rb:7.2f}m {qa:7.3f} {qb:7.3f}  {mark} "
                      f"({ta}+{tb}, {n}f)")
        return
    print(f"{'切れ目':>12s} {'mode':>5s} {'運ぶ':>7s} {'全体':>7s} "
          f"{'A側':>7s} {'B側':>7s} {'尺度比':>6s}  判定")
    for a, b in js:
        for mode in args.modes:
            try:
                r = evaluate(a, b, mode, gt)
            except Exception as e:
                print(f"{a:5d}->{b:<5d} {mode:>5s}  失敗 ({type(e).__name__}: {e})")
                continue
            if r is None:
                print(f"{a:5d}->{b:<5d} {mode:>5s}  GT不足")
                continue
            r_all, r_a, r_b, carry, sr, n, na, nb = r
            if carry is None:
                print(f"{a:5d}->{b:<5d} {mode:>5s}  B側不足 ({n}枚 A{na}/B{nb})")
                continue
            # 判定は「運ぶ誤差」で行う。島アンカーの成功判定が1mなので、
            # 橋がそれを食い潰さないこと = 0.5m未満を合格とする
            mark = "合格" if carry < 0.5 else ("可" if carry < 1.0 else "不可")
            print(f"{a:5d}->{b:<5d} {mode:>5s} {carry:6.2f}m {r_all:6.2f}m "
                  f"{r_a:6.2f}m {r_b:6.2f}m {sr:6.3f}  {mark} "
                  f"({n}枚 A{na}/B{nb})")


if __name__ == "__main__":
    main()
