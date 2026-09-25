"""島1つのGS学習 (コンテナgsplat内で実行)

docker compose exec -T gsplat env IMG_SESSION=<sess> PYTHONPATH=/code2/reconstruction_3d \
    python -m img_align.gs_train --tag <tag> [--iters 20000]

入力: results/gs/<tag>/ (gs_prepの出力)
出力: 同dir に ckpt.npz (means/quats/log_scales/logit_opacities/sh0/shN) + train_log.json
"""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rasterization
from gsplat.strategy import DefaultStrategy
from pytorch_msssim import SSIM

GS_DIR = Path(__file__).resolve().parent / "results" / "gs"
HOLDOUT_EVERY = 20      # 20ビューに1つを評価用に除外


class Dataset:
    def __init__(self, tag):
        self.dir = GS_DIR / tag
        v = np.load(self.dir / "views.npz")
        self.names = [str(n) for n in v["names"]]
        self.K = v["K"].astype(np.float32)
        self.c2w = v["cam2world"].astype(np.float32)
        self.size = int(self.K[0, 2] * 2)
        idx = np.arange(len(self.names))
        self.holdout = idx[::HOLDOUT_EVERY]
        self.train = np.setdiff1d(idx, self.holdout)
        cams = self.c2w[:, :3, 3]
        self.scene_scale = float(
            np.linalg.norm(cams - cams.mean(0), axis=1).max())
        self.rng = np.random.default_rng(0)

    def load(self, i, dev):
        img = cv2.imread(str(self.dir / "images" / f"{self.names[i]}.jpg"))
        img = torch.tensor(img[:, :, ::-1].copy(), dtype=torch.float32,
                           device=dev) / 255.0
        m = cv2.imread(str(self.dir / "masks" / f"{self.names[i]}.png"), 0)
        mask = torch.tensor(m > 127, device=dev)[..., None].float()
        w2c = torch.tensor(np.linalg.inv(self.c2w[i]), device=dev)
        K = torch.tensor(self.K, device=dev)
        return img, mask, w2c, K


def init_params(tag, dev):
    ip = np.load(GS_DIR / tag / "init_points.npz")
    xyz, rgb = ip["xyz"], ip["rgb"]
    n = len(xyz)
    means = torch.tensor(xyz, dtype=torch.float32, device=dev)
    log_scales = torch.full((n, 3), math.log(0.03), device=dev)
    quats = torch.zeros(n, 4, device=dev)
    quats[:, 0] = 1.0
    opac = torch.logit(torch.full((n,), 0.1, device=dev))
    sh0 = ((torch.tensor(rgb, dtype=torch.float32, device=dev) / 255.0 - 0.5)
           / 0.28209479).unsqueeze(1)
    shN = torch.zeros(n, 15, 3, device=dev)
    ps = {"means": means, "quats": quats, "scales": log_scales,
          "opacities": opac, "sh0": sh0, "shN": shN}
    return {k: torch.nn.Parameter(v) for k, v in ps.items()}


def render_view(params, w2c, K, size, sh_deg):
    colors = torch.cat([params["sh0"], params["shN"]], 1)
    return rasterization(
        params["means"], F.normalize(params["quats"], dim=-1),
        torch.exp(params["scales"]), torch.sigmoid(params["opacities"]),
        colors, w2c[None], K[None], size, size,
        sh_degree=sh_deg, packed=False)


@torch.no_grad()
def eval_holdout(params, data, dev):
    psnrs = []
    for i in data.holdout:
        img, mask, w2c, K = data.load(int(i), dev)
        render, _, _ = render_view(params, w2c, K, data.size, 3)
        mse = (((render[0] - img) ** 2) * mask).sum() / (mask.sum() * 3 + 1)
        psnrs.append(-10 * math.log10(max(mse.item(), 1e-10)))
    return float(np.mean(psnrs))


def train(tag, iters):
    dev = "cuda"
    data = Dataset(tag)
    params = init_params(tag, dev)
    print(f"{tag}: {len(data.train)} train / {len(data.holdout)} holdout, "
          f"init {len(params['means']):,} gs, scene_scale {data.scene_scale:.2f}")
    lrs = {"means": 1.6e-4 * data.scene_scale, "quats": 1e-3, "scales": 5e-3,
           "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    opts = {k: torch.optim.Adam([p], lr=lrs[k]) for k, p in params.items()}
    strategy = DefaultStrategy(verbose=False)
    strategy.check_sanity(params, opts)
    state = strategy.initialize_state(scene_scale=data.scene_scale)
    ssim = SSIM(data_range=1.0, channel=3)
    for it in range(iters):
        i = int(data.rng.choice(data.train))
        img, mask, w2c, K = data.load(i, dev)
        sh_deg = min(it // 1000, 3)
        render, alpha, info = render_view(params, w2c, K, data.size, sh_deg)
        strategy.step_pre_backward(params, opts, state, it, info)
        l1 = (torch.abs(render[0] - img) * mask).sum() / (mask.sum() * 3 + 1)
        ls = 1 - ssim((render[0] * mask).permute(2, 0, 1)[None],
                      (img * mask).permute(2, 0, 1)[None])
        loss = 0.8 * l1 + 0.2 * ls
        loss.backward()
        strategy.step_post_backward(params, opts, state, it, info,
                                    packed=False)
        for o in opts.values():
            o.step()
            o.zero_grad(set_to_none=True)
        if it % 2000 == 0:
            print(f"{it}: loss {loss.item():.4f} "
                  f"n_gs {len(params['means']):,}", flush=True)
    psnr = eval_holdout(params, data, dev)
    with torch.no_grad():
        np.savez_compressed(
            GS_DIR / tag / "ckpt.npz",
            means=params["means"].cpu().numpy(),
            quats=F.normalize(params["quats"], dim=-1).cpu().numpy(),
            log_scales=params["scales"].cpu().numpy(),
            logit_opacities=params["opacities"].cpu().numpy(),
            sh0=params["sh0"].cpu().numpy(),
            shN=params["shN"].cpu().numpy())
    (GS_DIR / tag / "train_log.json").write_text(json.dumps(
        {"psnr_holdout": round(psnr, 2), "iters": iters,
         "n_gs": len(params["means"])}))
    print(f"{tag}: holdout PSNR {psnr:.2f} dB, "
          f"{len(params['means']):,} gaussians -> ckpt.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--iters", type=int, default=20000)
    args = ap.parse_args()
    train(args.tag, args.iters)


if __name__ == "__main__":
    main()
