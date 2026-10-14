"""階ごとに図面へ軌跡を重ねてモデルを比較する (PanoVGGT / Pi3 / Pi3-FT)

窓単位の小さなグラフは読みづらいので、**1階 = 1枚**にまとめる。背景は図面JPEG、
その上にGT軌跡と3モデルの軌跡を重ねる。各モデルの軌跡は島ごとにGTへSim(2)
フィットしてから図面ピクセル座標へ落とす (島は幾何的に独立なので島単位で合わせる
のが正しい。島をまたいで1つの変換を当てるとスケールが混ざる)。

GT点はフレーム間速度で色分けし、屋内歩行の上限1.5 m/sを超える箇所を赤で示す
— そこはGT自身が飛んでいる区間で、モデル3本が揃って外れる原因になる。

前提: 3モデル分の島を**同じGT区間**で作ってあること (島が揃っていないと比較に
ならない)。本番用の `<model>_manifest.json` (IMU区間) を潰さないよう、比較用は
`<model>_gtseg_manifest.json` に分ける:
  pipeline run --session S --model {pv,pi3,holo} --segments gt \
      --manifest-out <model>_gtseg_manifest.json

使い方:
  IMG_SESSION=0912 python -m img_align.eval.floor_overlay
  IMG_SESSION=0912 python -m img_align.eval.floor_overlay --models pv holo
出力: <物件>/results/<sess>/frontend_bench/floor_overlay_<階>.png
"""
import argparse
import json

import numpy as np
from PIL import Image

from img_align.config import BENCH_DIR, PROP, SESSION
from img_align.core.geometry import rotation_to_y
from img_align.eval.chunk_cmp import gt_interp
from img_align.eval.frontend import fit2d
from img_align.eval.gt_traj import MM_PER_PX, load_gt

DT = 0.5              # フレーム間隔[s]
WALK_MAX = 1.5        # 屋内歩行の速度上限の目安 [m/s]
PAD_PX = 160          # 図面の切り出し余白
COLORS = {"pv": "#1f77b4", "pi3": "#2ca02c", "holo": "#d62728"}
LABEL = {"pv": "PanoVGGT", "pi3": "Pi3", "holo": "Pi3-FT(Holo360D)"}


def island_xy(tag):
    """島の global_cameras.npz → (frames, 重力で水平化した平面座標)"""
    d = np.load(BENCH_DIR / tag / "global_cameras.npz")
    fr = np.array([int(str(n).replace("frame_", "")) for n in d["names"]])
    poses = d["poses"]
    u = poses[:, :3, 1]
    u = np.median(u / np.linalg.norm(u, axis=1, keepdims=True), 0)
    p = d["positions"] @ rotation_to_y(u / np.linalg.norm(u)).T
    return fr, p[:, [0, 2]]


def fit_island_px(tag, gt):
    """島をGTへ相似フィット → (図面px軌跡, GT残差[m])。GT不足ならNone

    フィットはメートルで行い、最後にpxへ戻す。鏡像は系の取り方で反転しうる
    ので両方試して残差の小さい方を採る。
    """
    fr, p = island_xy(tag)
    g, idx = gt_interp(gt, fr, MM_PER_PX)
    if len(idx) < 4:
        return None
    best = None
    for m in (1, -1):
        x = p.copy()
        x[:, 1] *= m
        T = fit2d(x[idx], g)
        e = np.linalg.norm(T(x[idx]) - g, axis=1)
        if best is None or np.median(e) < np.median(best[1]):
            best = (T(x), e)
    return best[0] / (MM_PER_PX / 1000), best[1]


def floor_of(info, gt):
    """島の所属階。マニフェストにfloorが無ければ (--segments imu) GT多数決"""
    if info.get("floor"):
        return info["floor"]
    fl = [gt[f]["floor"] for f in range(info["a"], info["b"] + 1) if f in gt]
    return max(set(fl), key=fl.count) if fl else None


def gt_track(gt, floor):
    """その階のGT軌跡 (px) と各点の直前からの速度 [m/s]"""
    g = [(f, v) for f, v in sorted(gt.items()) if v["floor"] == floor]
    xy = np.array([[v["x_px"], v["y_px"]] for _, v in g])
    fr = np.array([f for f, _ in g])
    d = np.linalg.norm(np.diff(xy, axis=0), axis=1) * MM_PER_PX / 1000
    return xy, np.r_[np.nan, d / (np.diff(fr) * DT)]


def draw_plan(ax, floor, gt):
    """図面背景 + GT軌跡 + 表示範囲"""
    bg = np.asarray(Image.open(PROP.zumen_dir / f"{floor}.jpg").convert("L"))
    # 図面は寸法線・注記が密なので薄く敷く (軌跡を主役にする)
    ax.imshow(bg, cmap="gray", vmin=0, vmax=255, alpha=0.38, zorder=0)
    xy, vel = gt_track(gt, floor)
    ax.plot(xy[:, 0], xy[:, 1], "-", color="k", lw=4.0, alpha=0.3,
            label="GT", zorder=1)
    bad = vel > WALK_MAX
    if bad.any():
        ax.scatter(xy[bad, 0], xy[bad, 1], s=28, c="#e31a1c", zorder=6,
                   label=f"GT速度>{WALK_MAX}m/s ({int(bad.sum())}点)")
    ax.set_xlim(max(0, xy[:, 0].min() - PAD_PX),
                min(bg.shape[1], xy[:, 0].max() + PAD_PX))
    ax.set_ylim(min(bg.shape[0], xy[:, 1].max() + PAD_PX),
                max(0, xy[:, 1].min() - PAD_PX))
    ax.set_axis_off()


def draw_model(ax, model, islands, gt):
    """1モデル分の島をまとめて描く → GT残差のリスト"""
    errs, first = [], True
    for info in islands:
        r = fit_island_px(info["tag"], gt)
        if r is None:
            continue
        xy, err = r
        errs.extend(err)
        ax.plot(xy[:, 0], xy[:, 1], "-", lw=1.5, alpha=0.9,
                color=COLORS[model], zorder=4,
                label=LABEL[model] if first else None)
        first = False
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["pv", "pi3", "holo"])
    ap.add_argument("--floors", nargs="*", default=None)
    ap.add_argument("--manifest", default="{model}_gtseg_manifest.json",
                    help="島台帳のファイル名パターン")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]

    gt = load_gt()
    mans = {}
    for m in args.models:
        p = BENCH_DIR / args.manifest.format(model=m)
        if p.exists():
            mans[m] = json.loads(p.read_text())
        else:
            print(f"[skip] {m}: {p.name} なし (pipeline --model {m} 未実行)")
    if not mans:
        raise SystemExit("比較できるマニフェストがありません")

    floors = args.floors or sorted(
        {floor_of(i, gt) for i in next(iter(mans.values()))} - {None})
    summary = {}
    for floor in floors:
        fig, ax = plt.subplots(figsize=(13, 9))
        draw_plan(ax, floor, gt)
        stats = {}
        for m in args.models:
            isl = [i for i in mans.get(m, []) if floor_of(i, gt) == floor]
            if isl:
                stats[m] = draw_model(ax, m, isl, gt)
        ax.legend(fontsize=9, loc="upper right", framealpha=0.85)
        s = "   ".join(f"{LABEL[m]} {np.median(v):.3f}"
                       for m, v in stats.items() if len(v))
        ax.set_title(f"{SESSION} {floor} — GT残差 median [m]:   {s}",
                     fontsize=11)
        out = BENCH_DIR / f"floor_overlay_{floor}.png"
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        summary[floor] = {m: round(float(np.median(v)), 3)
                          for m, v in stats.items() if len(v)}
        print(f"-> {out}  {summary[floor]}")
    (BENCH_DIR / "floor_overlay.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
