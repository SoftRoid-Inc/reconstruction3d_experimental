"""Pi3 / Pi3-Finetuned-Holo360D による360°チャンク推論 (PanoVGGT互換の出力契約)

PanoVGGTの比較対象。Pi3はequirectを直接食べないので、Holo360D同梱の
`PanoramaToPerspective` で1枚のパノラマを**8枚の透視投影**(方位45°刻み・FOV90°)に
分割し、全フレーム×8ビューを**1回のforward**でPi3に通す。

Pi3が返すポーズは「ビュー毎」(N×8本) なので、そのままでは軌跡にならない。
8ビューはパノラマ中心を共有し、パノラマ座標系での各ビューの外部パラメータ E_v は
既知の固定値なので、
    x_world = P_(f,v) @ E_v @ x_pano   →   P_f = P_(f,v) @ E_v
が各vについて成り立つ。8本の P_f を平均 (回転はSVDで直交化) して
**フレーム毎のパノラマ姿勢**を得る。8本のばらつき `pose_spread` は
そのフレームの推定一貫性を表すので診断値として保存する。

座標系: Holo360Dのパノラマ系はZ上。PanoVGGTはY下なので、下流 (chaining /
gravity / floor) をそのまま使えるよう固定回転 `M` (X軸-90°) で揃えてから書き出す。

重み (いずれもホストの / パーティション、コンテナからも同じ絶対パス):
  pi3  = /home/maren/models/pi3/model.safetensors      (元祖Pi3, BSD-2-Clause)
  holo = /home/maren/models/holo360d/8views.bin        (Holo360D fine-tuned, MIT)

実行はpi3コンテナ内:
  docker compose exec -e IMG_SESSION=0925 pi3 \\
      python -m img_align.core.pi3 --tag pi3_c29 --frames 464:512:1 --ckpt holo
出力: results/pcd/<tag>/{points.ply, cameras.npz, pcd_stats.json}
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from img_align.config import RESULTS_DIR
from img_align.core.io_ply import write_ply
from img_align.core.panovggt import build_mask, select_entries

PCD_DIR = RESULTS_DIR / "pcd"
HOLO_DIR = "/opt/holo360d/Pi3_Finetuned_Holo360d"
CKPTS = {
    "pi3": "/home/maren/models/pi3/model.safetensors",
    "holo": "/home/maren/models/holo360d/8views.bin",
}
PANO_WH = (2880, 1440)   # Holo360Dの既定入力解像度 (透視分割はこの解像度前提)
CONF_KEEP = 0.5          # 信頼度上位この割合の画素のみ点群に採用
PF_POINTS = 36_000       # 連結ICP用のフレーム毎点数 (PanoVGGT側の[::15]と同程度)
# points.ply の点数上限。Holo360Dのckptは信頼度ヘッドを含まずconf間引きが効かない
# ため、素のままだと1チャンク2900万点=435MBになり55チャンクで24GB食う
# (2026-10-09に /mnt/data を満杯にした)。連結は [::8] しか使わないので支障はない
MAX_PLY_POINTS = 4_000_000

# パノラマ系(Z上) -> PanoVGGT系(Y下) の固定回転 (X軸まわり-90°)
M = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)


def _import_holo():
    """Holo360Dのリポジトリをimport可能にする (コンテナ内 /opt/holo360d)"""
    if not Path(HOLO_DIR).exists():
        raise RuntimeError(
            f"{HOLO_DIR} がありません。pi3コンテナ内で実行してください:\n"
            "  docker compose exec pi3 python -m img_align.core.pi3 ...")
    if HOLO_DIR not in sys.path:
        sys.path.insert(0, HOLO_DIR)
    os.chdir(HOLO_DIR)       # 相対パスのアセットを参照するコードがあるため
    from datasets.base.PanoramaToPerspective import PanoramaToPerspective
    from pi3.models.pi3 import Pi3
    from pi3.utils.geometry import depth_edge
    return PanoramaToPerspective, Pi3, depth_edge


def load_model(ckpt, device):
    import torch
    _, Pi3, _ = _import_holo()
    path = CKPTS.get(ckpt, ckpt)
    model = Pi3().to(device).eval()
    if str(path).endswith(".safetensors"):
        from safetensors.torch import load_file
        weight = load_file(path)
    else:
        weight = torch.load(path, map_location="cpu", weights_only=False)
    weight = {k.replace("module.", ""): v for k, v in weight.items()}
    miss = model.load_state_dict(weight, strict=False)
    # strict=Falseで黙って落ちると別モデルを評価してしまうので必ず出す
    print(f"[pi3] {ckpt} <- {path}\n"
          f"      missing={len(miss.missing_keys)} unexpected={len(miss.unexpected_keys)}")
    # Holo360Dのfine-tuned ckptは conf_decoder/conf_head (66本) を含まない。
    # 信頼度がランダム初期化のまま出るので、上位%の間引きは無意味になる
    has_conf = not any(k.startswith(("conf_decoder", "conf_head"))
                       for k in miss.missing_keys)
    if not has_conf:
        print("      信頼度ヘッド未収録 -> conf間引きを無効化 (mask+depth_edgeのみ)")
    elif miss.missing_keys:
        print(f"      missing例: {miss.missing_keys[:5]}")
    return model, has_conf


def load_panos(entries, use_mask=True):
    """(frames, H, W, 3) uint8 と マスク (frames, H, W) bool|None を返す"""
    import cv2
    from PIL import Image
    w, h = PANO_WH
    imgs, masks, names = [], [], []
    for e in entries:
        name = f"frame_{e['index']:06d}"
        names.append(name)
        rgb = cv2.resize(np.array(Image.open(e["path"]).convert("RGB")), (w, h))
        m = build_mask(name) if use_mask else None
        if m is not None:
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) > 0
            rgb = rgb * m[:, :, None]
        imgs.append(rgb)
        masks.append(m)
    n_mask = sum(m is not None for m in masks)
    print(f"[pi3] {len(names)} frames / masks {n_mask}")
    return imgs, masks, names


def pano_poses(view_c2w, extrinsics):
    """ビュー毎c2w (N*V,4,4) → フレーム毎パノラマc2w (N,4,4) + ばらつき

    P_f = P_(f,v) @ E_v を v で平均。回転はSVDで最近接回転に射影する。
    """
    v = len(extrinsics)
    n = len(view_c2w) // v
    cand = view_c2w.reshape(n, v, 4, 4) @ extrinsics[None].astype(np.float64)
    out = np.zeros((n, 4, 4))
    out[:, 3, 3] = 1.0
    spread = np.zeros((n, 2))
    for i in range(n):
        t = cand[i, :, :3, 3]
        out[i, :3, 3] = t.mean(0)
        spread[i, 0] = float(np.linalg.norm(t - t.mean(0), axis=1).mean())
        u, _, vt = np.linalg.svd(cand[i, :, :3, :3].mean(0))
        r = u @ vt
        if np.linalg.det(r) < 0:
            u[:, -1] *= -1
            r = u @ vt
        out[i, :3, :3] = r
        ang = [np.degrees(np.arccos(np.clip(
            (np.trace(r.T @ cand[i, k, :3, :3]) - 1) / 2, -1, 1)))
            for k in range(v)]
        spread[i, 1] = float(np.mean(ang))
    return out, spread


def run_pi3(entries, out_dir, ckpt="holo", view_type="8views",
            use_mask=True, conf_keep=CONF_KEEP, device_name="cuda",
            model=None):
    """model を渡すと読み込み済みのものを使う (チャンク一括推論用)"""
    import torch
    P2P, _, depth_edge = _import_holo()
    device = torch.device(device_name)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    p2p = P2P(output_resolution=518, pano_w=PANO_WH[0], pano_h=PANO_WH[1],
              view_type=view_type)
    imgs, masks, names = load_panos(entries, use_mask)

    views, view_masks = [], []
    eye = np.eye(4, dtype=np.float32)
    for rgb, m in zip(imgs, masks):
        splits, _, _ = p2p.process_single_image(rgb, None, eye)
        views.extend(torch.from_numpy(s).permute(2, 0, 1).float() / 255.0
                     for s in splits)
        if m is None:
            view_masks.extend([None] * len(splits))
        else:
            m3 = np.repeat((m * 255).astype(np.uint8)[:, :, None], 3, axis=2)
            ms, _, _ = p2p.process_single_image(m3, None, eye)
            view_masks.extend(s[:, :, 0] > 127 if s.ndim == 3 else s > 127
                              for s in ms)
    v_per_pano = len(views) // len(names)
    images = torch.stack(views).to(device)
    print(f"[pi3] perspectives {images.shape[0]} "
          f"({v_per_pano}/pano) -> forward", flush=True)

    model, has_conf = model if model is not None else load_model(ckpt, device)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        res = model(images[None])
    torch.cuda.empty_cache()

    keep = (~depth_edge(res["local_points"][0][..., 2], rtol=0.1)).cpu().numpy()
    if has_conf and conf_keep > 0:
        conf = torch.sigmoid(res["conf"][0][..., 0]).float().cpu().numpy()
        keep &= conf > np.percentile(conf.reshape(-1), (1.0 - conf_keep) * 100.0)
    else:
        conf_keep = 0.0
    if any(m is not None for m in view_masks):
        keep &= np.stack([np.ones(keep.shape[1:], bool) if m is None else m
                          for m in view_masks])

    wpts = res["points"][0].float().cpu().numpy()
    pts = wpts[keep].reshape(-1, 3) @ M.T
    cols = (images.permute(0, 2, 3, 1).cpu().numpy()[keep].reshape(-1, 3)
            * 255).astype(np.uint8)
    if len(pts) > MAX_PLY_POINTS:
        st = -(-len(pts) // MAX_PLY_POINTS)
        pts, cols = pts[::st], cols[::st]
    poses, spread = pano_poses(res["camera_poses"][0].float().cpu().numpy(),
                               p2p.np_extrinsics)
    m4 = np.eye(4)
    m4[:3, :3] = M
    poses = m4 @ poses @ np.linalg.inv(m4)

    # チャンク連結(core/chaining)は**フレーム毎のworld点群**で重なりICPをかける。
    # Pi3は1パノラマ=8ビューなので、その8枚分をまとめて1フレーム分とする。
    per_frame = {}
    for i, nm in enumerate(names):
        sl = slice(i * v_per_pano, (i + 1) * v_per_pano)
        q = wpts[sl][keep[sl]].reshape(-1, 3)
        if len(q) > PF_POINTS:
            q = q[:: -(-len(q) // PF_POINTS)]
        per_frame[nm] = (q @ M.T).astype(np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_ply(out_dir / "points.ply", pts, cols)
    np.savez_compressed(out_dir / "per_frame.npz", **per_frame)
    np.savez_compressed(out_dir / "cameras.npz", poses=poses,
                        positions=poses[:, :3, 3], names=np.array(names),
                        pose_spread=spread)
    stats = {
        "model": ckpt, "view_type": view_type, "conf_keep": conf_keep,
        "n_points": int(len(pts)), "n_cameras": int(len(poses)),
        "bbox_min": pts.min(0).tolist(), "bbox_max": pts.max(0).tolist(),
        "camera_span": (poses[:, :3, 3].max(0) - poses[:, :3, 3].min(0)).tolist(),
        "pose_spread_t_median": float(np.median(spread[:, 0])),
        "pose_spread_r_deg_median": float(np.median(spread[:, 1])),
    }
    (out_dir / "pcd_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    return stats


def run_chunks(windows, chunks_dir, ckpt="holo", **kw):
    """チャンク列をまとめて推論する (モデルの読み込みは1回だけ)

    windows: [(k, a, b)] — k=チャンク番号、[a:b) がフレーム範囲。
    既に cameras.npz があるチャンクはスキップするので中断・再開できる。
    """
    import torch
    from img_align.config import FRAMES_DIR
    n = len(sorted(FRAMES_DIR.glob("frame_number_*.jpg")))
    entries = select_entries(f"0:{n}:1")
    model = load_model(ckpt, torch.device("cuda"))
    done = 0
    for k, a, b in windows:
        cdir = PCD_DIR / chunks_dir / f"chunk_{k:02d}"
        if (cdir / "cameras.npz").exists():
            continue
        print(f"=== chunk {k}: frames[{a}:{b}]", flush=True)
        cdir.mkdir(parents=True, exist_ok=True)
        run_pi3(entries[a:b], cdir, ckpt=ckpt, model=model, **kw)
        torch.cuda.empty_cache()
        done += 1
    print(f"[pi3] {done} チャンクを推論 -> {PCD_DIR / chunks_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None, help="results/pcd/<tag>/ に出力")
    ap.add_argument("--frames", default=None, help="'start:end:step'")
    ap.add_argument("--chunk-windows", default=None,
                    help="チャンク一括: 'k:a:b,k:a:b,...' (--chunks-dirと併用)")
    ap.add_argument("--chunks-dir", default=None)
    ap.add_argument("--ckpt", default="holo", help=f"{list(CKPTS)} またはパス")
    ap.add_argument("--view-type", default="8views",
                    choices=("4views", "6views", "8views", "10views"))
    ap.add_argument("--conf-keep", type=float, default=CONF_KEEP)
    ap.add_argument("--no-mask", action="store_true")
    args = ap.parse_args()
    kw = dict(view_type=args.view_type, conf_keep=args.conf_keep,
              use_mask=not args.no_mask)
    if args.chunk_windows:
        wins = [tuple(int(v) for v in w.split(":"))
                for w in args.chunk_windows.split(",")]
        run_chunks(wins, args.chunks_dir or "chunks", ckpt=args.ckpt, **kw)
    else:
        run_pi3(select_entries(args.frames), PCD_DIR / args.tag,
                ckpt=args.ckpt, **kw)


if __name__ == "__main__":
    main()
