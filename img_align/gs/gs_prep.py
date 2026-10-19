"""GS学習用データセット生成: equirect原画 → 透視ビュー+ポーズ+初期点群

ビュー構成: ヨー8方向(45°刻み) × ピッチ{0°, 35°下} = 16ビュー/フレーム。
カメラ系は meshing.py と同一 (x右/y下/z前)。Rx(+35°)が下向き35°。
personはマスク画像として出力し、学習の損失から除外する(黒塗りしない)。

使い方 (env=img_align):
  IMG_SESSION=0312 conda run -n img_align python -m img_align.gs.gs_prep [--gate]
出力: results/gs/<tag>/{images,masks}/*.jpg|png, views.npz, init_points.npz
"""
import argparse
import math

import cv2
import numpy as np

from img_align.config import BENCH_DIR
from img_align.core.chaining import RESULTS_DIR
from img_align.config import FRAMES_DIR, SESSION, frame_name
from img_align.core.plan_grid import PlanGrid
from img_align.mesh.meshing import ISLANDS, _rot, island_scale, load_anchor, resolve_frames
from img_align.core.io_ply import read_ply_xyzrgb
from img_align.core.crops import load_person_mask

GS_DIR = RESULTS_DIR / "gs"
VIEW_SIZE = 800
FOV_DEG = 70.0
YAWS = list(range(0, 360, 45))
PITCH_DOWN = [0.0, 35.0]
MIN_VALID = 0.40          # 有効画素率がこれ未満のビューは捨てる
INIT_VOXEL = 0.05         # 3cmだと~290万点でVRAM過大。densificationが密度は回復する
MAX_INIT = 800_000


def view_rotations():
    return [(f"y{y:03d}p{int(p):02d}", _rot("y", y) @ _rot("x", p))
            for y in YAWS for p in PITCH_DOWN]


def view_dirs_and_K():
    """ビューローカルの方向ベクトル(S,S,3)と内部パラメータK"""
    f = 0.5 * VIEW_SIZE / math.tan(math.radians(FOV_DEG / 2))
    c = VIEW_SIZE / 2.0
    i, j = np.meshgrid(np.arange(VIEW_SIZE), np.arange(VIEW_SIZE),
                       indexing="xy")
    d = np.stack([(i - c) / f, (j - c) / f, np.ones_like(i, float)], -1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    K = np.array([[f, 0, c], [0, f, c], [0, 0, 1.0]])
    return d, K


def equirect_lut(d_world, W, H):
    """カメラ系方向 → equirect(W,H)の (row, col)"""
    lon = np.arctan2(d_world[..., 0], d_world[..., 2])
    lat = np.arcsin(np.clip(-d_world[..., 1], -1, 1))
    col = np.rint((lon / (2 * math.pi) + 0.5) * W - 0.5).astype(np.int32) % W
    row = np.clip(np.rint((0.5 - lat / math.pi) * H - 0.5), 0,
                  H - 1).astype(np.int32)
    return row, col


def build_luts(img_wh, seg_wh):
    """全16ビューぶんのLUTを一度だけ構築 (原画用とセグ用)"""
    d, K = view_dirs_and_K()
    luts = []
    for vname, Rv in view_rotations():
        dw = d @ Rv.T
        luts.append((vname, Rv, equirect_lut(dw, *img_wh),
                     equirect_lut(dw, *seg_wh)))
    return luts, K


def make_init_points(tag, k):
    xyz, rgb = read_ply_xyzrgb(BENCH_DIR / tag / "global_points.ply")
    xyz = xyz * k
    cell = np.floor(xyz / INIT_VOXEL).astype(np.int64)
    _, idx = np.unique(cell, axis=0, return_index=True)
    if len(idx) > MAX_INIT:
        idx = np.random.default_rng(0).choice(idx, MAX_INIT, replace=False)
    return xyz[idx].astype(np.float32), rgb[idx]


def prep_island(isl):
    frames, anchor = resolve_frames(isl)
    k = island_scale(anchor, PlanGrid(isl["floor"]))
    out = GS_DIR / isl["tag"]
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(exist_ok=True)

    luts, K = build_luts((5376, 2688), (2048, 1024))
    names, c2ws = [], []
    n_skip_img = n_skip_masked = 0
    for fi, fr in enumerate(frames):
        num = int(fr["name"].replace("frame_", ""))
        src = FRAMES_DIR / frame_name(num)
        if not src.exists():
            print(f"  warn: {src.name} なし -> skip")
            n_skip_img += 1
            continue
        img = cv2.imread(str(src))          # BGRのまま扱いjpg出力もBGRで一貫
        person = load_person_mask(fr["name"])
        for vname, Rv, (ri, ci), (rs, cs) in luts:
            valid = np.ones((VIEW_SIZE, VIEW_SIZE), bool)
            if person is not None:
                valid = ~person[rs, cs]
            if valid.mean() < MIN_VALID:
                n_skip_masked += 1
                continue
            name = f"{fr['name']}_{vname}"
            cv2.imwrite(str(out / "images" / f"{name}.jpg"), img[ri, ci],
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            cv2.imwrite(str(out / "masks" / f"{name}.png"),
                        valid.astype(np.uint8) * 255)
            c2w = np.eye(4)
            c2w[:3, :3] = fr["R"] @ Rv
            c2w[:3, 3] = fr["t_m"]
            names.append(name)
            c2ws.append(c2w)
        if fi % 20 == 0:
            print(f"  {fi}/{len(frames)} frames")
    np.savez_compressed(out / "views.npz", names=np.array(names), K=K,
                        cam2world=np.stack(c2ws))
    xyz, rgb = make_init_points(isl["tag"], k)
    np.savez_compressed(out / "init_points.npz", xyz=xyz, rgb=rgb)
    print(f"{isl['tag']}: {len(names)} views "
          f"(欠画像 {n_skip_img}f, マスク超過 {n_skip_masked}v), "
          f"init {len(xyz):,} pts -> {out}")


def gate():
    """幾何ゲート: 初期点群を先頭ビューへ投影し、画像色と一致するか

    実装ミス(放射→方向・ポーズ・LUT)は全島で大きくずれる。島内部ドリフト
    (0828島A/Bの既知問題)は統合点群由来で島ごとに差が出るため、
    判定は「最良の島 < 40」とし、全島の値を報告する。
    """
    meds = []
    for isl in ISLANDS[SESSION]:
        med, n = _gate_one(isl)
        print(f"GATE {isl['tag']}: 投影点 {n:,}, 色差 median {med:.1f}")
        meds.append(med)
    print(f"best {min(meds):.1f} (閾値40)")
    print("PASS" if min(meds) < 40 else "FAIL")


def _gate_one(isl):
    out = GS_DIR / isl["tag"]
    v = np.load(out / "views.npz")
    ip = np.load(out / "init_points.npz")
    name = str(v["names"][0])
    c2w, K = v["cam2world"][0], v["K"]
    img = cv2.imread(str(out / "images" / f"{name}.jpg"))[:, :, ::-1]
    p_cam = (ip["xyz"] - c2w[:3, 3]) @ c2w[:3, :3]
    front = p_cam[:, 2] > 0.3
    p = p_cam[front]
    rgb = ip["rgb"][front]
    uv = (p[:, :2] / p[:, 2:3]) * K[0, 0] + K[0, 2]
    inb = ((uv[:, 0] >= 0) & (uv[:, 0] < VIEW_SIZE - 1)
           & (uv[:, 1] >= 0) & (uv[:, 1] < VIEW_SIZE - 1))
    p, rgb, uv = p[inb], rgb[inb], np.rint(uv[inb]).astype(int)
    # zバッファ: 画素ごとに最小zの点だけ残す
    pix = uv[:, 1] * VIEW_SIZE + uv[:, 0]
    order = np.lexsort((p[:, 2], pix))
    keep = np.ones(len(order), bool)
    keep[1:] = pix[order][1:] != pix[order][:-1]
    sel = order[keep]
    diff = np.abs(img[uv[sel, 1], uv[sel, 0]].astype(float)
                  - rgb[sel].astype(float)).mean(axis=1)
    return float(np.median(diff)), len(sel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--init-only", action="store_true",
                    help="init_points.npz だけ再生成 (ビューは触らない)")
    args = ap.parse_args()
    if args.gate:
        gate()
        return
    for isl in ISLANDS[SESSION]:
        print(f"=== {isl['tag']}")
        if args.init_only:
            _, anchor = resolve_frames(isl, verbose=False)
            k = island_scale(anchor, PlanGrid(isl["floor"]))
            xyz, rgb = make_init_points(isl["tag"], k)
            np.savez_compressed(GS_DIR / isl["tag"] / "init_points.npz",
                                xyz=xyz, rgb=rgb)
            print(f"  init {len(xyz):,} pts")
        else:
            prep_island(isl)


if __name__ == "__main__":
    main()
