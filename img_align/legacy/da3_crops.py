"""DA3前段用のキューブマップ切り出し (等距円筒 → yaw4方向 × FOV90° 透視画像)

3刻み系列(チャンクと同じ152フレーム)を対象に生成する。
出力: results/da3/crops/frame_XXXXXX_y{000,090,180,270}.jpg + crops_meta.json
"""
import json

import numpy as np
import py360convert
from PIL import Image
from tqdm import tqdm

from img_align.config import RESULTS_DIR
from img_align.core.panovggt import select_entries

CROPS_DIR = RESULTS_DIR / "da3" / "crops"
FOV = 90
SIZE = 720
YAWS = [0, 90, 180, 270]


def main():
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    entries = select_entries("0:454:3")
    for e in tqdm(entries):
        name = f"frame_{e['index']:06d}"
        if (CROPS_DIR / f"{name}_y270.jpg").exists():
            continue
        img = np.asarray(Image.open(e["path"]).convert("RGB"))
        for yaw in YAWS:
            # u_deg: 0=正面(画像中央)。等距円筒→透視 (pitch 0)
            p = py360convert.e2p(img, fov_deg=(FOV, FOV), u_deg=yaw - 180,
                                 v_deg=0, out_hw=(SIZE, SIZE))
            Image.fromarray(p).save(CROPS_DIR / f"{name}_y{yaw:03d}.jpg",
                                    quality=92)
    meta = {"fov_deg": FOV, "size": SIZE, "yaws": YAWS,
            "note": "u_deg = yaw-180 (等距円筒の画像中央=正面をyaw0とする)",
            "focal_px": SIZE / 2 / np.tan(np.radians(FOV / 2))}
    (CROPS_DIR / "crops_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"done -> {CROPS_DIR} ({len(entries)*4} crops, "
          f"focal={meta['focal_px']:.1f}px)")


if __name__ == "__main__":
    main()
