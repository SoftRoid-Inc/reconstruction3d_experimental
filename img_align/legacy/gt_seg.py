"""gt_areas (手動ポリゴンアノテーション) を seg_ade20k と同形式のラベルマップへ変換

入力: 28-AAAAA/gt_areas/frame_number_XXXXXX.json (labelme polygon、元解像度5376x2688、
ラベル = floor / ceiling / door / window / stairs。壁ラベルは無い)
出力: results/seg_gt/frame_XXXXXX_labels.npz (+確認用オーバーレイ)

壁は明示ラベルが無いため未ラベル(0)のままとする。scan.py の ray_tag は
開口/壁系が共に無い帯を壁とみなすので、そのまま仮想スキャンに使える。
"""
import json

import numpy as np
from PIL import Image, ImageDraw

from img_align.config import EQUI_W, FRAMES_DIR, GROUP_IDS, PROJECT_ROOT, RESULTS_DIR
from img_align.seg_ade20k import make_overlay

GT_DIR = PROJECT_ROOT / "28-AAAAA" / "gt_areas"
OUT_DIR = RESULTS_DIR / "seg_gt"
DRAW_ORDER = ["floor", "ceiling", "stairs", "window", "door"]  # 後に描く方が勝ち


def rasterize(d):
    w0, h0 = d["imageWidth"], d["imageHeight"]
    sx, sy = EQUI_W / w0, (EQUI_W // 2) / h0
    canvas = Image.new("L", (EQUI_W, EQUI_W // 2), 0)
    draw = ImageDraw.Draw(canvas)
    shapes = [s for s in d["shapes"]
              if s.get("shape_type") == "polygon" and s["label"] in GROUP_IDS]
    shapes.sort(key=lambda s: DRAW_ORDER.index(s["label"])
                if s["label"] in DRAW_ORDER else 99)
    for s in shapes:
        pts = [(x * sx, y * sy) for x, y in s["points"]]
        if len(pts) >= 3:
            draw.polygon(pts, fill=GROUP_IDS[s["label"]])
    return np.array(canvas, dtype=np.uint8)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for jf in sorted(GT_DIR.glob("frame_number_*.json")):
        d = json.loads(jf.read_text())
        labels = rasterize(d)
        name = jf.stem.replace("frame_number_", "frame_")
        np.savez_compressed(OUT_DIR / f"{name}_labels.npz", labels=labels)
        img = np.asarray(Image.open(FRAMES_DIR / f"{jf.stem}.jpg").convert("RGB"))
        make_overlay(img, labels).save(OUT_DIR / f"{name}_overlay.jpg", quality=88)
        counts = {g: round(float((labels == gid).mean()), 4)
                  for g, gid in GROUP_IDS.items() if (labels == gid).any()}
        print(name, counts)
    print(f"done -> {OUT_DIR}")


if __name__ == "__main__":
    main()
