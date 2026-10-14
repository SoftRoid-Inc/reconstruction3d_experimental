"""鉛直プロファイルによる階の自動決定 (図面・IMU・GT 不使用)

島は階段で切るため、島同士は幾何的に独立で階も不明だった。一方で
**階段チャンクも含めて全チャンクを逐次連結すれば鉛直座標は連続**する。
水平はドリフトしても、1フロア=約2.9mの段差は桁違いに大きく頑健に読める
(core/stair_check が「カメラ高の1.5〜19倍の変位」として検出済み)。

手順:
  1. 全チャンクを逐次連結 (階段を含む。水平の品質は問わない)
  2. 共通重力で整列し、カメラの鉛直座標を取る
  3. チャンクごとのカメラ高 (床スラブまでの距離) でメートル換算
  4. 鉛直プロファイルを平滑化 → 水平な踊り場 (レベル) をクラスタリング
  5. レベルを低い順に 1F, 2F, … と割り当て、各島の代表フレームの所属レベルで階を決定

注意: PanoVGGT座標はY+が物理的に下 (equirect由来)。本モジュールは
符号を反転して「上が正」に揃える。

使い方:
  IMG_SESSION=0912 python -m img_align.core.floor_profile
  IMG_SESSION=0912 python -m img_align.core.floor_profile --plot
出力: frontend_bench/floor_profile.json (島→階) + floor_profile.png
"""
import argparse
import json

import numpy as np

from img_align.config import BENCH_DIR, CAM_HEIGHT_M, FRAMES_DIR, RESULTS_DIR
from img_align.core.chaining import chain, load_chunk
from img_align.core.chunking import chunk_windows
from img_align.core.geometry import apply, rotation_to_y
from img_align.postproc.density_map import slab_peaks

SMOOTH_F = 9          # 鉛直プロファイルの平滑窓 (フレーム)
LEVEL_TOL_M = 0.9     # 同一レベルとみなす高さ差
MIN_LEVEL_F = 15      # レベルとして採用する最小滞在フレーム数


def chunk_scale_m(cdir, poses, rg):
    """カメラ高 (床スラブまでの距離) から m/unit を出す"""
    from img_align.core.io_ply import read_ply_xyzrgb
    xyz, _ = read_ply_xyzrgb(cdir / "points.ply")
    y = (xyz[::7] @ rg.T)[:, 1]
    peaks, _, _ = slab_peaks(y)
    cam = float(np.median((poses[:, :3, 3] @ rg.T)[:, 1]))
    if not len(peaks):
        return None
    h = float(np.min(np.abs(peaks - cam)))
    return CAM_HEIGHT_M / h if h > 1e-6 else None


def ensure_all_chunks(chunks_dir="chunks"):
    """鉛直の連続性には階段チャンクも要る。欠けている窓だけ推論して埋める"""
    from img_align.core.panovggt import PCD_DIR, run_panovggt, select_entries
    n = len(sorted(FRAMES_DIR.glob("frame_number_*.jpg")))
    windows = chunk_windows(n)
    entries = select_entries(f"0:{n}:1")
    miss = [k for k in range(len(windows))
            if not (PCD_DIR / chunks_dir / f"chunk_{k:02d}"
                    / "cameras.npz").exists()]
    if miss:
        print(f"[floor_profile] 連続化のため欠落チャンク {len(miss)} 本を推論: "
              f"{miss}", flush=True)
    for k in miss:
        a, b = windows[k]
        cdir = PCD_DIR / chunks_dir / f"chunk_{k:02d}"
        cdir.mkdir(parents=True, exist_ok=True)
        run_panovggt(entries[a:b], cdir, use_mask=True)
    return len(miss)


def vertical_profile(chunks_dir="chunks"):
    """全チャンク逐次連結 → (frames, 高さ[m]、上が正)"""
    ensure_all_chunks(chunks_dir)
    pcd = RESULTS_DIR / "pcd" / chunks_dir
    cdirs = sorted(pcd.glob("chunk_*"), key=lambda p: int(p.name.split("_")[1]))
    chunks = [load_chunk(c) for c in cdirs]
    _, T, _ = chain(cdirs)
    ups = []
    for _, _, poses, _, _ in chunks:
        u = poses[:, :3, 1]
        ups.append(np.median(u / np.linalg.norm(u, axis=1, keepdims=True), 0))
    g = np.median(np.stack(ups), axis=0)
    rg = rotation_to_y(g / np.linalg.norm(g))
    scales = [chunk_scale_m(c, ch[2], rg) for c, ch in zip(cdirs, chunks)]
    mpu = float(np.median([s for s in scales if s])) if any(scales) else 1.0
    fy = {}
    for k, (names, pos, _, _, unt) in enumerate(chunks):
        g_pos = apply(T[k], pos) @ rg.T
        for i, nm in enumerate(names):
            if nm not in unt:
                fy.setdefault(int(nm.replace("frame_", "")), []).append(
                    g_pos[i, 1])
    fr = np.array(sorted(fy))
    # Y+ が物理下向き → 反転して「上が正」、メートル換算
    y = -np.array([np.mean(fy[f]) for f in fr]) * mpu
    k = SMOOTH_F
    ys = np.convolve(np.pad(y, (k // 2, k - k // 2 - 1), mode="edge"),
                     np.ones(k) / k, mode="valid")
    return fr, ys - ys.min(), mpu


def levels_from_profile(fr, h, tol=LEVEL_TOL_M, min_f=MIN_LEVEL_F):
    """高さプロファイル → レベル区間 [(a, b, 代表高さ)] (時系列順)"""
    segs, start = [], 0
    for i in range(1, len(h)):
        if abs(h[i] - np.median(h[start:i])) > tol:
            if i - start >= min_f:
                segs.append((int(fr[start]), int(fr[i - 1]),
                             float(np.median(h[start:i]))))
            start = i
    if len(h) - start >= min_f:
        segs.append((int(fr[start]), int(fr[-1]),
                     float(np.median(h[start:]))))
    # 近い高さのレベルを統合し、低い順に階番号を振る
    hs = sorted({round(s[2] / tol) * tol for s in segs})
    merged = []
    for v in hs:
        if merged and abs(v - merged[-1]) < tol:
            continue
        merged.append(v)
    out = []
    for a, b, v in segs:
        idx = int(np.argmin([abs(v - m) for m in merged]))
        out.append({"a": a, "b": b, "h_m": round(v, 2), "level": idx})
    return out, merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks-dir", default="chunks")
    ap.add_argument("--manifest", default="pv_manifest.json")
    ap.add_argument("--tol", type=float, default=LEVEL_TOL_M)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    fr, h, mpu = vertical_profile(args.chunks_dir)
    segs, levels = levels_from_profile(fr, h, args.tol)
    print(f"m/unit={mpu:.4f}  高さ範囲 {h.min():.2f}〜{h.max():.2f}m")
    print(f"検出レベル (低い順): {[round(v, 2) for v in levels]}")
    for s in segs:
        print(f"  frames {s['a']:5d}-{s['b']:5d}  h={s['h_m']:6.2f}m "
              f"-> level {s['level']} (= {s['level'] + 1}F?)")

    man = json.loads((BENCH_DIR / args.manifest).read_text())
    res = []
    for m in man:
        mid = (m["a"] + m["b"]) / 2
        cand = [s for s in segs if s["a"] <= mid <= s["b"]] or \
            [min(segs, key=lambda s: min(abs(s["a"] - mid), abs(s["b"] - mid)))]
        lv = cand[0]["level"]
        res.append({"tag": m["tag"], "a": m["a"], "b": m["b"],
                    "level": lv, "floor": f"{lv + 1}F",
                    "h_m": cand[0]["h_m"]})
        print(f"  {m['tag']:20s} -> {lv + 1}F (h={cand[0]['h_m']:.2f}m)")
    (BENCH_DIR / "floor_profile.json").write_text(json.dumps(
        {"mpu": mpu, "levels": levels, "segments": segs, "islands": res},
        indent=1, ensure_ascii=False))
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(fr, h, "-", color="navy")
        for v in levels:
            ax.axhline(v, color="0.7", lw=0.8)
        for m in man:
            ax.axvspan(m["a"], m["b"], alpha=0.08, color="green")
        ax.set_xlabel("frame")
        ax.set_ylabel("高さ [m] (上が正)")
        fig.tight_layout()
        fig.savefig(BENCH_DIR / "floor_profile.png", dpi=110)
        print(f"-> {BENCH_DIR / 'floor_profile.png'}")


if __name__ == "__main__":
    main()
