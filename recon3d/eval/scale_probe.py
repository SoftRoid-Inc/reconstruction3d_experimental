"""チャンク間スケール誤差を予測できるフレーム特徴があるかを調べる (スパイク)

誤差の支配項は**チャンク間スケールの不整合** (|log s| median 0.115)。
「品質を落とすフレームを除けば精度が上がる」という仮説を検証するには、
まず**どんなフレーム特徴がスケール誤差を予測するか**を見る必要がある。
何も相関しなければ、フレーム除外に伸びしろは無い。

既に効果が実証されている除外は**静止 (基線退化)** だけで、これは見た目でなく
幾何的退化だった (0828で−42%)。一方「テクスチャレス・鏡・狭所」に帰した破綻は
正式GT導入後に大半がGT誤差と判明している。そこで仮説を2系統に分けて測る:

  幾何系 (効く見込み): 並進量 / 回転÷並進 / 深度レンジ / 深度の平面性
  外観系 (効かない見込み): フレーム間画像差分 / エッジ分散 / マスク率

目的変数は chain_log.json の |log(scale)| (そのチャンクを前のチャンクへ
繋いだときのスケール比の対数絶対値)。

使い方:
  IMG_SESSION=0912 python -m recon3d.eval.scale_probe
出力: frontend_bench/scale_probe.json (チャンク毎の特徴と相関)
"""
import argparse
import json

import numpy as np

from recon3d.core.geometry import rotation_to_y
from recon3d.core.io_ply import read_ply_xyzrgb

SMALL = 256      # 画像特徴を測る縮小幅


def chunk_geom(cdir):
    """チャンクの幾何特徴 (カメラ高で正規化してスケール不定性を消す)"""
    d = np.load(cdir / "cameras.npz")
    P = d["poses"]
    u = P[:, :3, 1]
    u = np.median(u / np.linalg.norm(u, axis=1, keepdims=True), 0)
    R = rotation_to_y(u / np.linalg.norm(u))
    pos = P[:, :3, 3] @ R.T
    step = np.linalg.norm(np.diff(pos[:, [0, 2]], axis=0), axis=1)
    # フレーム間回転角 [deg]
    rot = []
    for i in range(1, len(P)):
        m = P[i - 1, :3, :3].T @ P[i, :3, :3]
        rot.append(np.degrees(np.arccos(np.clip((np.trace(m) - 1) / 2, -1, 1))))
    rot = np.array(rot)

    xyz, _ = read_ply_xyzrgb(cdir / "points.ply")
    q = xyz[::37] @ R.T
    cam_y = float(np.median(pos[:, 1]))
    h = np.abs(q[:, 1] - cam_y)
    scale_u = float(np.median(h[h > 0])) or 1.0      # カメラ高相当の長さ単位
    dist = np.linalg.norm(q[:, [0, 2]] - pos[:, [0, 2]].mean(0), axis=1)
    # 点群の平面性: 水平断面の主成分比 (1に近いほど等方=情報が多い)
    c = np.cov((q[:, [0, 2]] - q[:, [0, 2]].mean(0)).T)
    ev = np.sort(np.linalg.eigvalsh(c))
    return {
        "translation": float(np.median(step)) / scale_u,
        "rot_per_trans": float(np.median(rot)) / max(float(np.median(step)), 1e-6)
        * scale_u,
        "depth_range": float(np.percentile(dist, 90) - np.percentile(dist, 10))
        / scale_u,
        "depth_iso": float(ev[0] / max(ev[1], 1e-9)),
        "path_len": float(step.sum()) / scale_u,
    }


def chunk_appearance(paths, cdir):
    """チャンクの外観特徴 (入力フレームから)"""
    from PIL import Image
    names = [str(n) for n in np.load(cdir / "cameras.npz")["names"]]
    diffs, edges = [], []
    prev = None
    for n in names:
        p = paths.images / paths.frame_name(int(n.replace("frame_", "")))
        if not p.exists():
            continue
        a = np.asarray(Image.open(p).convert("L")
                       .resize((SMALL, SMALL // 2))).astype(np.float32)
        edges.append(float(np.var(np.abs(np.diff(a, axis=1)))))
        if prev is not None:
            diffs.append(float(np.abs(a - prev).mean()))
        prev = a
    return {"img_diff": float(np.median(diffs)) if diffs else np.nan,
            "edge_var": float(np.median(edges)) if edges else np.nan}


def collect(paths, chunks_dir="chunks", manifest="pv_manifest.json", appearance=True):
    """島の chain_log から |log s| を取り、該当チャンクの特徴を集める"""
    pcd = paths.results / "pcd" / chunks_dir
    rows = []
    for m in json.loads((paths.bench / manifest).read_text()):
        p = paths.bench / m["tag"] / "chain_log.json"
        if not p.exists():
            continue
        for c in json.loads(p.read_text()):
            k = m["chunks"][c["chunk"]] if c["chunk"] < len(m["chunks"]) \
                else None
            cdir = pcd / f"chunk_{k:02d}" if k is not None else None
            if cdir is None or not (cdir / "cameras.npz").exists():
                continue
            r = {"tag": m["tag"], "chunk": int(k),
                 "abs_log_s": abs(float(np.log(c["scale"]))),
                 "icp_resid": c.get("icp_resid"), **chunk_geom(cdir)}
            if appearance:
                r.update(chunk_appearance(paths, cdir))
            rows.append(r)
            print(f"  {m['tag']:16} chunk{k:3d} |log s|={r['abs_log_s']:.3f} "
                  f"trans={r['translation']:.3f}", flush=True)
    return rows


def main(paths, argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks-dir", default="chunks")
    ap.add_argument("--manifest", default="pv_manifest.json")
    ap.add_argument("--no-appearance", action="store_true")
    args = ap.parse_args(argv)
    rows = collect(paths, args.chunks_dir, args.manifest, not args.no_appearance)
    if len(rows) < 8:
        raise SystemExit(f"サンプルが{len(rows)}本しかありません")
    y = np.array([r["abs_log_s"] for r in rows])
    feats = [k for k in rows[0] if k not in
             ("tag", "chunk", "abs_log_s") and
             isinstance(rows[0][k], float)]
    print(f"\nチャンク {len(rows)}本  |log s| median {np.median(y):.3f}")
    print(f"{'特徴量':16} {'相関':>7} {'|相関|順':>8}")
    cors = {}
    for f in feats:
        x = np.array([r[f] for r in rows], float)
        ok = np.isfinite(x) & np.isfinite(y)
        cors[f] = float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() > 4 \
            else float("nan")
    for f in sorted(cors, key=lambda f: -abs(cors[f] or 0)):
        print(f"{f:16} {cors[f]:+7.2f}")
    (paths.bench / "scale_probe.json").write_text(json.dumps(
        {"n": len(rows), "median_abs_log_s": float(np.median(y)),
         "correlations": cors, "rows": rows}, indent=1, ensure_ascii=False))
    print(f"-> {paths.bench / 'scale_probe.json'}")

