"""PanoVGGTで360°フレームから点群+カメラ姿勢を推定する (SceneAligner-lite 前段)

third_party/PanoVGGT の inference.py (無改変) を専用conda env `panovggt` で
subprocess実行する。撮影者は手法A(seg_ade20k)のpersonマスクで除外する
(PanoVGGTのマスク仕様: 黒=無効)。

出力: results/pcd/points.ply (統合点群) / cameras.npz (姿勢+フレームID)
"""
import argparse
import json
import os
import subprocess

import numpy as np
from PIL import Image
from scipy import ndimage

from img_align.core.io_ply import read_ply_xyzrgb
from img_align.config import (FRAMES_DIR, GROUP_IDS, PROJECT_ROOT, PROP,
                              RESULTS_DIR, THIRD_PARTY)
from img_align.legacy.sample_frames import load_selected

PCD_DIR = RESULTS_DIR / "pcd"
INPUT_DIR = PCD_DIR / "input"
MASK_DIR = PCD_DIR / "input_mask"
PANOVGGT_DIR = THIRD_PARTY / "PanoVGGT"
PANOVGGT_OUT = PCD_DIR / "panovggt_out"
# env内pythonを絶対パスで直接起動する。`conda run -n panovggt python` は
# 呼び出し側のPATH (pyenv等) に負けて別のpythonが起動することがあり、
# env側のsite-packagesが一切載らずImportErrorになる (2026-10-07に実際に発生)
PANOVGGT_PY = "/home/maren/miniconda3/envs/panovggt/bin/python"

PERSON_DILATE_ITER = 12   # personマスクの膨張 (境界の取りこぼし対策)
NADIR_INVALID_FRAC = 0.07  # 画像下端の無効帯 (自撮り棒・手元)


MASK_WH = (2048, 1024)  # PanoVGGTに渡すマスク解像度 (seg結果と同じ)


def build_mask(name):
    """有効画素マスク(白=有効)を作る。
    物件に既存人物マスク(白=有効/黒=人物)があればそれを縮小して使い、
    無ければseg_ade20kのperson結果から作る。どちらも無ければNone"""
    if PROP.person_mask_dir is not None:
        num = int("".join(c for c in name if c.isdigit()))
        p = PROP.person_mask_dir / f"frame_number_{num:06d}.jpg"
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
        p = FRAMES_DIR / f"frame_number_{i:06d}.jpg"
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




def run_panovggt(entries, out_dir, use_mask=True):
    """フレーム集合に対してPanoVGGT推論を実行し、out_dir に標準形式で保存する"""
    input_dir = out_dir / "input"
    mask_dir = out_dir / "input_mask"
    vggt_out = out_dir / "panovggt_out"
    names, n_masks = prepare_inputs(entries, input_dir, mask_dir, use_mask)
    print(f"{len(names)} frames prepared -> {out_dir}")

    cmd = [
        PANOVGGT_PY, "inference.py",
        "--config", "training/config/default.yaml",
        "--checkpoint", "checkpoints/model.pt",
        "--image_dir", str(input_dir),
        "--output_dir", str(vggt_out),
    ]
    if n_masks:
        cmd += ["--mask_dir", str(mask_dir)]
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=PANOVGGT_DIR, check=True)

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
    1チャンクずつ subprocess を起動すると固定コスト12.2秒が毎回かかるため
    (実推論は16フレームで9秒)、panovggt_batch にまとめて渡す。
    """
    import tempfile
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
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(todo, f)
        spec = f.name
    print(f"[batch] {len(todo)} チャンクを1プロセスで推論", flush=True)
    subprocess.run([PANOVGGT_PY, "-m", "img_align.core.panovggt_batch",
                    "--spec", spec, "--panovggt-dir", str(PANOVGGT_DIR)],
                   check=True, env={**os.environ,
                                    "PYTHONPATH": str(PROJECT_ROOT)})
    os.unlink(spec)
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
