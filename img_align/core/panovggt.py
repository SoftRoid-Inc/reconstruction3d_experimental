"""PanoVGGTで360°フレームから点群+カメラ姿勢を推定する (SceneAligner-lite 前段)

PanoVGGT の inference.py (無改変) を**同一プロセスで**呼ぶ。撮影者は
人物マスクで除外する (PanoVGGTのマスク仕様: 黒=無効)。

かつては専用 conda env の subprocess だったが、単一 Docker イメージ化で
不要になった (2026-10-19)。推論段は `img_align` イメージの中でだけ動く。

出力: results/pcd/points.ply (統合点群) / cameras.npz (姿勢+フレームID)
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from img_align.core.io_ply import read_ply_xyzrgb
from img_align.config import (FRAMES_DIR, GROUP_IDS, PROP, RESULTS_DIR,
                              frame_name)
from img_align.legacy.sample_frames import load_selected

PCD_DIR = RESULTS_DIR / "pcd"
INPUT_DIR = PCD_DIR / "input"
MASK_DIR = PCD_DIR / "input_mask"
PANOVGGT_OUT = PCD_DIR / "panovggt_out"

PANOVGGT_CONFIG = os.environ.get("PANOVGGT_CONFIG",
                                 "training/config/default.yaml")


def _panovggt_paths():
    """(PanoVGGTディレクトリ, 重み) を返す。使えなければ SystemExit

    **`PANOVGGT_DIR` を持つのはイメージだけ** (Dockerfile が設定する)。
    未設定なら推論段で止める — 開発ホストの python には torch 2.9.1 が
    入っていて `import inference` が**通ってしまう**が、PanoVGGT が想定
    するのは 2.5 系。「動くが結果が違う」が一番まずいので入口で止める。
    """
    d = os.environ.get("PANOVGGT_DIR")
    if not d:
        raise SystemExit(
            "PANOVGGT_DIR が未設定: 推論段は img_align イメージ内で"
            "実行してください\n"
            "  docker compose exec img_align python main.py run input=...\n"
            "(既存チャンクを使う段 — posegraph / anchor / export / 評価 — は"
            "ホストのままで動きます)")
    d = Path(d)
    ckpt = Path(os.environ.get("PANOVGGT_CKPT",
                               d / "checkpoints" / "model.pt"))
    if not ckpt.exists():
        raise SystemExit(
            f"PanoVGGT の重みが無い: {ckpt}\n"
            "  PANOVGGT_CKPT で場所を指すか、"
            f"{d / 'checkpoints'} へボリュームをマウントしてください")
    return d, str(ckpt)


PERSON_DILATE_ITER = 12   # personマスクの膨張 (境界の取りこぼし対策)
NADIR_INVALID_FRAC = 0.07  # 画像下端の無効帯 (自撮り棒・手元)


MASK_WH = (2048, 1024)  # PanoVGGTに渡すマスク解像度 (seg結果と同じ)


def build_mask(name):
    """有効画素マスク(白=有効)を作る。
    物件に既存人物マスク(白=有効/黒=人物)があればそれを縮小して使い、
    無ければseg_ade20kのperson結果から作る。どちらも無ければNone"""
    if PROP.person_mask_dir is not None:
        num = int("".join(c for c in name if c.isdigit()))
        p = PROP.person_mask_dir / frame_name(num)
        if not p.exists():
            return None
        img = Image.open(p).convert("L").resize(MASK_WH, Image.BILINEAR)
        person = np.asarray(img) < 128
    else:
        from img_align.config import SEG_DIR
        npz = SEG_DIR / f"{name}_labels.npz"
        if not npz.exists():
            return None
        labels = np.load(npz)["labels"]
        person = labels == GROUP_IDS["person"]
    person = ndimage.binary_dilation(person, iterations=PERSON_DILATE_ITER)
    valid = ~person
    h = valid.shape[0]
    valid[int(h * (1 - NADIR_INVALID_FRAC)):, :] = False
    return (valid * 255).astype(np.uint8)


def select_entries(frames_spec):
    """frames_spec 'start:end:step' 指定があれば連続レンジ、無ければ代表フレーム"""
    if frames_spec is None:
        return load_selected()
    start, end, step = (int(v) for v in frames_spec.split(":"))
    entries = []
    for i in range(start, end, step):
        p = FRAMES_DIR / frame_name(i)
        if p.exists():
            entries.append({"index": i, "path": str(p)})
    return entries


def prepare_inputs(entries, input_dir, mask_dir, use_mask=True):
    input_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    names = []
    n_masks = 0
    for entry in entries:
        name = f"frame_{entry['index']:06d}"
        names.append(name)
        dst = input_dir / f"{name}.jpg"
        if not dst.exists():
            dst.symlink_to(entry["path"])
        mask = build_mask(name) if use_mask else None
        if mask is not None:
            Image.fromarray(mask).save(mask_dir / f"{name}.png")
            n_masks += 1
    print(f"masks: {n_masks}/{len(names)}")
    return names, n_masks


def local_floor_height(ply_path):
    """per-frameローカル点群から直下床までの高さ(units)を推定。失敗時None"""
    if not ply_path.exists():
        return None
    xyz, _ = read_ply_xyzrgb(ply_path)
    y = xyz[:, 1]
    y = y[(y < -0.2) & (y > -4)]
    if len(y) < 5000:
        return None
    hist, edges = np.histogram(y, bins=120)
    i = int(hist.argmax())
    return -(edges[i] + edges[i + 1]) / 2




def _infer(chunks):
    """PanoVGGT 推論の唯一の入口 (同一プロセス。モデル読み込みは1回)"""
    from img_align.core.panovggt_batch import run_spec
    pdir, ckpt = _panovggt_paths()
    run_spec(chunks, pdir, PANOVGGT_CONFIG, ckpt)


def run_panovggt(entries, out_dir, use_mask=True):
    """フレーム集合に対してPanoVGGT推論を実行し、out_dir に標準形式で保存する"""
    input_dir = out_dir / "input"
    mask_dir = out_dir / "input_mask"
    vggt_out = out_dir / "panovggt_out"
    names, n_masks = prepare_inputs(entries, input_dir, mask_dir, use_mask)
    print(f"{len(names)} frames prepared -> {out_dir}")

    _infer([{"image_dir": str(input_dir),
             "mask_dir": str(mask_dir) if n_masks else None,
             "output_dir": str(vggt_out)}])

    xyz, _ = read_ply_xyzrgb(vggt_out / "pointclouds" / "merged.ply")
    poses = np.load(vggt_out / "poses" / "all_poses.npy")  # (S,4,4)
    (out_dir / "points.ply").write_bytes(
        (vggt_out / "pointclouds" / "merged.ply").read_bytes())
    np.savez_compressed(
        out_dir / "cameras.npz",
        poses=poses,
        positions=poses[:, :3, 3],
        names=np.array(names),
    )
    stats = {
        "n_points": int(len(xyz)),
        "bbox_min": xyz.min(axis=0).tolist(),
        "bbox_max": xyz.max(axis=0).tolist(),
        "n_cameras": int(len(poses)),
        "camera_span": (poses[:, :3, 3].max(axis=0)
                        - poses[:, :3, 3].min(axis=0)).tolist(),
    }
    (out_dir / "pcd_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


def run_chunks(windows, entries, chunks_dir, use_mask=True):
    """複数チャンクをまとめて推論する (モデル読み込みは1回)

    windows: [(k, a, b)] — k=チャンク番号、[a:b) がフレーム範囲。
    既に cameras.npz があるチャンクはスキップするので中断・再開できる。
    1チャンクずつモデルを読み直すと固定コスト12.2秒が毎回かかるため
    (実推論は16フレームで9秒)、panovggt_batch にまとめて渡す。
    """
    todo, names_of = [], {}
    for k, a, b in windows:
        cdir = PCD_DIR / chunks_dir / f"chunk_{k:02d}"
        if (cdir / "cameras.npz").exists():
            continue
        cdir.mkdir(parents=True, exist_ok=True)
        names, n_masks = prepare_inputs(entries[a:b], cdir / "input",
                                        cdir / "input_mask", use_mask)
        names_of[k] = names
        todo.append({"image_dir": str(cdir / "input"),
                     "mask_dir": str(cdir / "input_mask") if n_masks else None,
                     "output_dir": str(cdir / "panovggt_out")})
    if not todo:
        print("[batch] 推論すべきチャンクなし")
        return
    print(f"[batch] {len(todo)} チャンクを推論", flush=True)
    _infer(todo)
    for k, a, b in windows:
        if k in names_of:
            _collect(PCD_DIR / chunks_dir / f"chunk_{k:02d}", names_of[k])


def _collect(out_dir, names):
    """inference.py の出力をパイプラインの契約へ読み替える"""
    vggt_out = out_dir / "panovggt_out"
    xyz, _ = read_ply_xyzrgb(vggt_out / "pointclouds" / "merged.ply")
    poses = np.load(vggt_out / "poses" / "all_poses.npy")
    (out_dir / "points.ply").write_bytes(
        (vggt_out / "pointclouds" / "merged.ply").read_bytes())
    np.savez_compressed(out_dir / "cameras.npz", poses=poses,
                        positions=poses[:, :3, 3], names=np.array(names))
    (out_dir / "pcd_stats.json").write_text(json.dumps({
        "n_points": int(len(xyz)), "n_cameras": int(len(poses)),
        "bbox_min": xyz.min(axis=0).tolist(),
        "bbox_max": xyz.max(axis=0).tolist()}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default=None,
                    help="'start:end:step' の連続レンジ。省略時は代表16フレーム")
    ap.add_argument("--tag", default=None,
                    help="出力サブディレクトリ名 (results/pcd/<tag>/)。省略時はpcd直下")
    ap.add_argument("--no-mask", action="store_true")
    args = ap.parse_args()

    out_dir = PCD_DIR / args.tag if args.tag else PCD_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = select_entries(args.frames)
    run_panovggt(entries, out_dir, use_mask=not args.no_mask)


if __name__ == "__main__":
    main()
