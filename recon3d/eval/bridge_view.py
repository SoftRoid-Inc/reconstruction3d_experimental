"""橋渡しチャンクの接続結果を図面に重ねて目で確かめる

置き方が要点: **島Aだけ**をGTに相似フィットし、**同じ変換で**橋が運んだ島Bを
描く。こうすると「島Aが正しく図面に乗っているとき島Bがどこに来るか」が
そのまま絵になり、GTの島Bとのずれが**橋の誤差そのもの**として見える。
結合全体をGTにフィットすると誤差が両側に分配され、どちらがずれたのか
分からなくなる。

掌性は島A単独で決める (PanoVGGTのworld系は図面と鏡像。結合全体で選ぶと
橋が壊れたときに鏡映の選択まで引きずられる)。

使い方:
  IMG_SESSION=0925 python -m recon3d.eval.bridge_view
  IMG_SESSION=0925 python -m recon3d.eval.bridge_view --modes n8s1 --junction 916:935
出力: <物件>/results/<sess>/backend/bridge_<a>_<b>_<mode>.png
"""
import argparse

import numpy as np
from PIL import Image

from recon3d.config import OUT_DIR, PROP, SESSION
from recon3d.core.bridge import (bridge_transform, consensus, islands_at,
                                   junctions, manifest_islands, transported)
from recon3d.core.island_group import CONSENSUS_OK_M, build_groups
from recon3d.eval.bridge_eval import _apply2, _sim2_fit
from recon3d.eval.gt_traj import mm_per_px, load_gt

PAD_PX = 140
COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#17becf",
          "#8c564b", "#e377c2")


def _chain_xy(tag, horiz=(0, 2)):
    from recon3d.core.bridge import _chain_poses
    c = _chain_poses(tag)
    fr = sorted(c)
    return fr, np.stack([[c[f][horiz[0]], c[f][horiz[1]]] for f in fr])


def _gt_xy(gt, frames):
    ok = [f for f in frames if f in gt]
    return ok, np.array([[gt[f]["x_px"], gt[f]["y_px"]] for f in ok])


def draw(a_end, b_start, mode, gt, horiz=(0, 2)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]

    ma, mb = islands_at(a_end, b_start)
    if ma is None:
        return None
    tb_pos, _ = transported(a_end, b_start, mode)
    if tb_pos is None:
        return None
    fa, xa = _chain_xy(ma["tag"], horiz)
    fb, _ = _chain_xy(mb["tag"], horiz)
    xb = np.stack([tb_pos[:, horiz[0]], tb_pos[:, horiz[1]]], axis=1)

    # 島Aだけを使って図面への変換を決める (掌性もここで確定)
    ga, ya = _gt_xy(gt, fa)
    if len(ga) < 10:
        return None
    ia = [i for i, f in enumerate(fa) if f in gt]
    r_a, _, T = _sim2_fit(xa[ia], ya)
    pa, pb = _apply2(T, xa), _apply2(T, xb)

    gb, yb = _gt_xy(gt, fb)
    ib = [i for i, f in enumerate(fb) if f in gt]
    err = (float(np.median(np.linalg.norm(pb[ib] - yb, axis=1)))
           * mm_per_px() / 1000 if len(gb) >= 4 else None)
    m = mm_per_px() / 1000

    floor = ma.get("floor") or mb.get("floor")
    bg = np.asarray(Image.open(PROP.zumen_dir / f"{floor}.jpg").convert("L"))
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.imshow(bg, cmap="gray", vmin=0, vmax=255, alpha=0.35, zorder=0)
    ax.plot(ya[:, 0], ya[:, 1], "-", color="k", lw=5, alpha=0.20,
            label="GT 島A", zorder=1)
    ax.plot(yb[:, 0], yb[:, 1], "-", color="#8c564b", lw=5, alpha=0.30,
            label="GT 島B", zorder=1)
    ax.plot(pa[:, 0], pa[:, 1], "-o", ms=2.4, lw=1.5, color="#1f77b4",
            label=f"島A {ma['tag']} ({r_a * m:.2f}m)", zorder=4)
    ax.plot(pb[:, 0], pb[:, 1], "-o", ms=2.4, lw=1.5, color="#ff7f0e",
            label=f"島B {mb['tag']} 橋で運搬"
                  + (f" ({err:.2f}m)" if err else ""), zorder=5)
    # 繋ぎ目を線で示す (島Aの終端 → 島Bの始端)
    ax.plot([pa[-1, 0], pb[0, 0]], [pa[-1, 1], pb[0, 1]], "--",
            color="#d62728", lw=1.8, zorder=6, label="繋ぎ目")
    allxy = np.vstack([ya, yb, pa, pb])
    ax.set_xlim(max(0, allxy[:, 0].min() - PAD_PX),
                min(bg.shape[1], allxy[:, 0].max() + PAD_PX))
    ax.set_ylim(min(bg.shape[0], allxy[:, 1].max() + PAD_PX),
                max(0, allxy[:, 1].min() - PAD_PX))
    ax.set_axis_off()
    ax.legend(fontsize=9, loc="upper right", framealpha=0.85)
    ax.set_title(f"{SESSION} {floor} 切れ目 {a_end}->{b_start} ({mode}) — "
                 f"島Aを図面に合わせ、同じ変換で島Bを描く"
                 + (f"   島Bのずれ {err:.2f}m" if err else ""), fontsize=11)
    out = OUT_DIR / f"bridge_{a_end}_{b_start}_{mode}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out, err



def draw_floors(gt, modes, thresh=CONSENSUS_OK_M, horiz=(0, 2)):
    """フロアごとに全島を描く。橋で繋がった島は**同じ変換**で置く"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from recon3d.core.bridge import _chain_poses
    from recon3d.core.geometry import apply as apply3
    matplotlib.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]

    groups = build_groups(modes, thresh)
    # (floor, 群れindex) ごとに、その階の島のフレームと根ゲージ位置を集める
    per = {}
    for gi, (isl, T) in enumerate(groups):
        for m in isl:
            c = _chain_poses(m["tag"])
            fr = sorted(c)
            pos = apply3(T[m["tag"]], np.stack([c[f] for f in fr]))
            per.setdefault(m["floor"], []).append((gi, m["tag"], fr, pos))

    outs = []
    for floor, items in sorted(per.items()):
        bg = np.asarray(Image.open(PROP.zumen_dir / f"{floor}.jpg")
                        .convert("L"))
        fig, ax = plt.subplots(figsize=(13, 9))
        ax.imshow(bg, cmap="gray", vmin=0, vmax=255, alpha=0.35, zorder=0)
        gall = [v for _, v in sorted(gt.items()) if v["floor"] == floor]
        ya = np.array([[v["x_px"], v["y_px"]] for v in gall])
        if len(ya):
            ax.plot(ya[:, 0], ya[:, 1], "-", color="k", lw=5, alpha=0.18,
                    label="GT", zorder=1)
        allxy = [ya] if len(ya) else []
        # 群れ単位でGTに合わせる (同じ群れの島は1つの変換で置く = 接続の検証)
        for gi in sorted({i for i, *_ in items}):
            sub = [it for it in items if it[0] == gi]
            fr = np.concatenate([it[2] for it in sub])
            xy = np.vstack([np.stack([it[3][:, horiz[0]],
                                      it[3][:, horiz[1]]], axis=1)
                            for it in sub])
            k = [i for i, f in enumerate(fr) if f in gt]
            if len(k) < 10:
                continue
            dst = np.array([[gt[fr[i]]["x_px"], gt[fr[i]]["y_px"]]
                            for i in k])
            r, _, T2 = _sim2_fit(xy[k], dst)
            p = _apply2(T2, xy)
            allxy.append(p)
            n_isl = len(sub)
            ax.plot(p[:, 0], p[:, 1], "-o", ms=2.2, lw=1.4,
                    color=COLORS[gi % len(COLORS)], zorder=4,
                    label=f"群れ{gi} ({n_isl}島: "
                          f"{'+'.join(it[1].split('_')[-1] for it in sub)}) "
                          f"{r * mm_per_px() / 1000:.2f}m")
        A = np.vstack(allxy)
        ax.set_xlim(max(0, A[:, 0].min() - PAD_PX),
                    min(bg.shape[1], A[:, 0].max() + PAD_PX))
        ax.set_ylim(min(bg.shape[0], A[:, 1].max() + PAD_PX),
                    max(0, A[:, 1].min() - PAD_PX))
        ax.set_axis_off()
        ax.legend(fontsize=9, loc="upper right", framealpha=0.85)
        ax.set_title(f"{SESSION} {floor} — 橋で繋がった島は同じ変換で配置 "
                     f"(一致度 {thresh}m 未満を接続)", fontsize=11)
        out = OUT_DIR / f"bridge_floor_{floor}.png"
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        outs.append(out)
        print(f"-> {out}")
    return outs


COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#17becf",
          "#e377c2", "#8c564b")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="*", default=["n8s1"])
    ap.add_argument("--junction", help="a_end:b_start (省略時は全切れ目)")
    ap.add_argument("--floors", action="store_true",
                    help="フロアごとに全島を描く (橋で繋がった島は同じ変換)")
    ap.add_argument("--thresh", type=float, default=CONSENSUS_OK_M)
    args = ap.parse_args()
    gt = load_gt()
    if args.floors:
        draw_floors(gt, args.modes, args.thresh)
        return
    js = ([tuple(int(v) for v in args.junction.split(":"))] if args.junction
          else junctions())
    for a, b in js:
        for mode in args.modes:
            try:
                r = draw(a, b, mode, gt)
            except Exception as e:
                print(f"{a}->{b} {mode}: 失敗 ({type(e).__name__}: {e})")
                continue
            if r is None:
                print(f"{a}->{b} {mode}: 島/GT不足")
                continue
            out, err = r
            print(f"-> {out}" + (f"  島Bのずれ {err:.2f}m" if err else ""))


if __name__ == "__main__":
    main()
