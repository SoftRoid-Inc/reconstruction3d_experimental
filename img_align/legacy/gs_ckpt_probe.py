"""島ckpt(SH3・全ガウシアン)で任意ワープ視点をレンダリングする検証プローブ

合併ply(DC色)との比較で「DC化・間引き」の影響と「学習品質」を切り分ける。
docker compose exec -T gsplat env IMG_SESSION=0828 WARP_LABEL="2F 000267" TAG=panovggt_Bcore_yawfix \
    PYTHONPATH=/code2/reconstruction_3d python -m img_align.legacy.gs_ckpt_probe
"""
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rasterization

from img_align.config import RESULTS_DIR

SESSION = os.environ.get("IMG_SESSION", "0828")
TAG = os.environ.get("TAG", "panovggt_Bcore_yawfix")
WARP_LABEL = os.environ.get("WARP_LABEL", "2F 000267")
CKPT = os.environ.get("CKPT", "ckpt.npz")
OUT_DIR = RESULTS_DIR / "backend"
GS_DIR = RESULTS_DIR / "gs"


def main():
    dev = "cuda"
    d = np.load(GS_DIR / TAG / CKPT)
    means = torch.tensor(d["means"], device=dev)
    quats = torch.tensor(d["quats"], device=dev)
    scales = torch.exp(torch.tensor(d["log_scales"], device=dev))
    opac = torch.sigmoid(torch.tensor(d["logit_opacities"], device=dev))
    colors = torch.cat([torch.tensor(d["sh0"], device=dev),
                        torch.tensor(d["shN"], device=dev)], 1)

    # ワープはレンダラ系(図面→y下)+図面座標。ckptは「メートル島座標」なので
    # 学習ビューのポーズで代用: views.npz から WARP_LABEL のフレーム名を含む
    # ビューを1つ選ぶ (= 学習が実際に見た視点での実力を見る)
    v = np.load(GS_DIR / TAG / "views.npz")
    names = [str(n) for n in v["names"]]
    frame = "frame_" + WARP_LABEL.split()[-1]
    cand = [i for i, n in enumerate(names) if n.startswith(frame)]
    if not cand:
        raise SystemExit(f"{frame} のビューが {TAG} に無い")
    for i in cand[:4]:
        c2w = v["cam2world"][i]
        K = torch.tensor(v["K"], dtype=torch.float32, device=dev)[None]
        w2c = torch.tensor(np.linalg.inv(c2w), dtype=torch.float32,
                           device=dev)[None]
        img, _, _ = rasterization(means, F.normalize(quats, dim=-1), scales,
                                  opac, colors, w2c, K, 800, 800, sh_degree=3)
        arr = (img[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        cv2.imwrite(str(OUT_DIR / f"gs_ckpt_{CKPT.split(chr(46))[0]}_{names[i]}.png"), arr[:, :, ::-1])
        # 学習画像そのもの (GT) も並べて出す
        gt = cv2.imread(str(GS_DIR / TAG / "images" / f"{names[i]}.jpg"))
        if gt is not None:
            cv2.imwrite(str(OUT_DIR / f"gs_gt_{names[i]}.png"), gt)
        print(f"-> gs_ckpt_{CKPT}_{names[i]}.png (+gt)")


if __name__ == "__main__":
    main()
