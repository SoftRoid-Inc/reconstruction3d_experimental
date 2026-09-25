"""合併GS(図面座標)をgsplatで直接レンダリングする検証プローブ

Webビューアを介さずに向き・品質を確認する。カメラは図面座標で明示構築
(カメラ系 x右/y下/z前、図面は z上)。

docker compose exec -T gsplat env IMG_SESSION=0828 PYTHONPATH=/code2/reconstruction_3d \
    python -m img_align.gs_render_probe
出力: results/backend/gs_probe_<sess>_*.png
"""
import numpy as np
import torch
from gsplat import rasterization
from pathlib import Path
import cv2
import os

SESSION = os.environ.get("IMG_SESSION", "0828")
WARP_EXTRA = int(os.environ.get("WARP_EXTRA", "30"))
SFX = "" if SESSION == "0828" else f"_{SESSION}"
OUT_DIR = Path(__file__).resolve().parent / "results" / "backend"


def read_3dgs_full(path):
    with open(path, "rb") as fp:
        header = b""
        while not header.endswith(b"end_header\n"):
            header += fp.readline()
        n = int([ln for ln in header.decode().splitlines()
                 if ln.startswith("element vertex")][0].split()[-1])
        d = np.frombuffer(fp.read(n * 14 * 4), dtype=np.float32).reshape(n, 14)
    return d


def cam_pose(pos, yaw_deg, pitch_deg=0.0):
    """図面座標(z上)でのcam2world。yaw=0で+x方向、pitch>0で見下ろし"""
    cy, sy = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    cp, sp = np.cos(np.radians(pitch_deg)), np.sin(np.radians(pitch_deg))
    fwd = np.array([cy * cp, sy * cp, -sp])          # z_cam
    down = np.array([0.0, 0.0, -1.0])                # 仮のy_cam(下)
    x = np.cross(down, fwd)
    x /= np.linalg.norm(x)
    y = np.cross(fwd, x)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = x, y, fwd, pos
    return c2w


def main():
    d = read_3dgs_full(OUT_DIR / f"gaussian_house{SFX}.ply")
    dev = "cuda"
    means = torch.tensor(d[:, :3], device=dev)
    rgb = torch.tensor(np.clip(0.5 + 0.28209479 * d[:, 3:6], 0, 1), device=dev)
    opac = torch.sigmoid(torch.tensor(d[:, 6], device=dev))
    scales = torch.exp(torch.tensor(d[:, 7:10], device=dev))
    quats = torch.tensor(d[:, 10:14], device=dev)
    cx = float(np.median(d[:, 0]))
    cyy = float(np.median(d[:, 1]))
    S, f = 900, 550.0
    K = torch.tensor([[f, 0, S / 2], [0, f, S / 2], [0, 0, 1.0]],
                     device=dev)[None]
    views = [
        ("1f_yaw0", cam_pose(np.array([cx - 3, cyy, 1.5]), 0)),
        ("1f_yaw90", cam_pose(np.array([cx, cyy - 3, 1.5]), 90)),
        ("iso", cam_pose(np.array([cx - 8, cyy - 8, 7.0]), 45, 35)),
        ("elev_x", cam_pose(np.array([cx - 14, cyy, 2.5]), 0)),
        ("elev_y", cam_pose(np.array([cx, cyy - 14, 2.5]), 90)),
    ]
    # ワープ点と同じポーズ (ユーザーが見るもの)
    import json
    w = json.loads((OUT_DIR / f"warps_{SESSION}.json").read_text())
    for wi in (2, 10, 20, WARP_EXTRA):
        vm = np.array(w[wi]["v"]).reshape(4, 4).T   # 列優先→行列
        c2w_r = np.linalg.inv(vm)
        # レンダラ座標(y下)→図面座標: 逆回転 Rx(-90)
        Rb = np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]])
        c2w = np.eye(4)
        c2w[:3, :3] = Rb @ c2w_r[:3, :3]
        c2w[:3, 3] = Rb @ c2w_r[:3, 3]
        views.append((f"warp{wi}_{w[wi]['l'].replace(' ', '_')}", c2w))
    for name, c2w in views:
        w2c = torch.tensor(np.linalg.inv(c2w), dtype=torch.float32,
                           device=dev)[None]
        img, _, _ = rasterization(means, torch.nn.functional.normalize(
            quats, dim=-1), scales, opac, rgb, w2c, K, S, S)
        arr = (img[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        cv2.imwrite(str(OUT_DIR / f"gs_probe_{SESSION}_{name}.png"),
                    arr[:, :, ::-1])
        print(f"-> gs_probe_{SESSION}_{name}.png")


if __name__ == "__main__":
    main()
