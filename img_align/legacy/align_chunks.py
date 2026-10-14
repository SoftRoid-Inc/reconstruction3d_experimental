"""各チャンクを独立に図面へ照合し、図面座標系で軌跡を組み立てる

チャンク間の縫合(Umeyama)は階段区間で破綻するため行わない。
各チャンクは単体で歪みの少ないローカル地図になっており(chunk_testで確認)、
図面への相似変換照合が成立すれば、図面自体がグローバル座標系のアンカーになる。

手順: チャンクごとに density_map → align_floorplan を実行 → 選択フロアと
カメラ図面座標を収集 → 重複フレームはスコアの高いチャンクを採用 →
1F/2Fの軌跡図と summary を出力。
"""
import json
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from img_align.core.plan_grid import ZUMEN_DIR
from img_align.core.panovggt import PCD_DIR

CHUNKS_DIR = PCD_DIR / "chunks"
OUT_DIR = PCD_DIR / "by_chunk"


def run_stage(module, tag):
    subprocess.run([sys.executable, "-m", f"img_align.{module}", "--tag", tag],
                   check=True)


def main():
    chunk_dirs = sorted(CHUNKS_DIR.glob("chunk_*"), key=lambda p: int(p.name.split("_")[1]))
    print(f"{len(chunk_dirs)} chunks")
    for cdir in chunk_dirs:
        tag = f"chunks/{cdir.name}"
        if not (cdir / "density.npz").exists():
            run_stage("density_map", tag)
        if not (cdir / "align_summary.json").exists():
            run_stage("align_floorplan", tag)

    # フレームごとに (floor, px, score) を収集。重複は高スコアのチャンクを採用
    frames = {}
    chunk_results = {}
    for cdir in chunk_dirs:
        summ = json.loads((cdir / "align_summary.json").read_text())
        floor = summ["selected_floor"]
        res = summ["results"][floor]
        chunk_results[cdir.name] = {
            "floor": floor, "score": res["score"],
            "theta": res["theta"], "s": res["s"],
            "score_other": summ["results"][
                "2F" if floor == "1F" else "1F"]["score"],
        }
        for name, px in zip(summ["names"], res["cam_px"]):
            if name not in frames or res["score"] > frames[name]["score"]:
                frames[name] = {"floor": floor, "px": px,
                                "score": res["score"], "chunk": cdir.name}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(json.dumps(
        {"chunks": chunk_results, "frames": frames}, indent=2))
    for c, r in chunk_results.items():
        print(c, r)

    # 軌跡図 (階ごと、フレーム番号順に接続、番号を注記)
    for floor in ("1F", "2F"):
        pts = sorted(((n, f["px"]) for n, f in frames.items()
                      if f["floor"] == floor), key=lambda x: x[0])
        if not pts:
            continue
        bg = np.asarray(Image.open(ZUMEN_DIR / f"{floor}.jpg").convert("L"))
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(bg, cmap="gray")
        xs = [p[1][0] for p in pts]
        ys = [p[1][1] for p in pts]
        order = np.linspace(0, 1, len(pts))
        ax.plot(xs, ys, "-", color="red", linewidth=0.8, alpha=0.6)
        ax.scatter(xs, ys, c=order, cmap="viridis", s=25, zorder=3)
        for (n, (x, y)), o in zip(pts[::4], order[::4]):
            ax.annotate(n.replace("frame_", ""), (x, y), fontsize=6,
                        color="blue")
        ax.set_title(f"camera trajectory on {floor} "
                     f"({len(pts)} frames, color = time)")
        ax.set_axis_off()
        fig.savefig(OUT_DIR / f"trajectory_{floor}.jpg", dpi=120,
                    bbox_inches="tight")
        plt.close(fig)
    print(f"done -> {OUT_DIR}")


if __name__ == "__main__":
    main()
