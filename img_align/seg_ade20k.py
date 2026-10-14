"""手法A: ADE20Kセマンティックセグメンテーション (Mask2Former)

等距円筒フレームを6面キューブマップに変換して面ごとに推論し、
結果を等距円筒に逆投影して統合ラベルマップを作る。
対象7要素 + person(撮影者マスク)をグループIDに集約して保存する。
"""
import json

import numpy as np
import py360convert
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

from img_align.config import CUBE_FACE_SIZE, EQUI_W, GROUP_COLORS, GROUP_IDS, RESULTS_DIR, SEG_MODEL, TARGET_LABELS
from img_align.legacy.sample_frames import load_selected

from img_align.config import SEG_DIR
FACE_KEYS = ["F", "R", "B", "L", "U", "D"]


def build_id_lookup(id2label):
    """ADE20KクラスID → グループID の変換表 (該当なしは0)"""
    lookup = np.zeros(len(id2label) + 1, dtype=np.uint8)
    for cid, label in id2label.items():
        names = {s.strip().lower() for s in label.split(",")}
        for group, aliases in TARGET_LABELS.items():
            if names & {a.lower() for a in aliases}:
                lookup[int(cid)] = GROUP_IDS[group]
                break
    return lookup


def segment_frame(img, model, processor, lookup, device):
    """1フレーム(等距円筒RGB ndarray)→ グループラベルマップ (EQUI_W/2, EQUI_W)"""
    faces = py360convert.e2c(img, face_w=CUBE_FACE_SIZE, cube_format="dict")
    face_imgs = [faces[k] for k in FACE_KEYS]

    inputs = processor(images=face_imgs, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
    seg_maps = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[(CUBE_FACE_SIZE, CUBE_FACE_SIZE)] * len(face_imgs)
    )

    group_faces = {}
    for k, seg in zip(FACE_KEYS, seg_maps):
        g = lookup[seg.cpu().numpy()]
        group_faces[k] = np.repeat(g[..., None], 3, axis=2).astype(np.float32)

    equi = py360convert.c2e(
        group_faces, h=EQUI_W // 2, w=EQUI_W, mode="nearest", cube_format="dict"
    )
    return equi[..., 0].astype(np.uint8)


def area_weighted_fraction(group_map, gid):
    """等距円筒の面積補正(cos緯度)つきの被覆率"""
    h = group_map.shape[0]
    lat = (np.arange(h) + 0.5) / h * np.pi - np.pi / 2
    w = np.cos(lat)[:, None] * np.ones_like(group_map, dtype=np.float64)
    return float((w * (group_map == gid)).sum() / w.sum())


def make_overlay(img, group_map, alpha=0.45):
    base = np.asarray(
        Image.fromarray(img).resize((EQUI_W, EQUI_W // 2), Image.BILINEAR)
    ).astype(np.float32)
    color = np.zeros_like(base)
    mask = group_map > 0
    for name, gid in GROUP_IDS.items():
        color[group_map == gid] = GROUP_COLORS[name]
    out = base.copy()
    out[mask] = base[mask] * (1 - alpha) + color[mask] * alpha
    overlay = Image.fromarray(out.astype(np.uint8))

    draw = ImageDraw.Draw(overlay)
    for i, (name, gid) in enumerate(GROUP_IDS.items()):
        y = 8 + i * 22
        draw.rectangle([8, y, 26, y + 16], fill=GROUP_COLORS[name])
        draw.text((32, y + 2), name, fill=(255, 255, 255))
    return overlay


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default=None,
                    help="'start:end:step' の連続レンジ。省略時は代表16フレーム")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()
    if args.frames:
        from img_align.core.panovggt import select_entries
        entries = select_entries(args.frames)
    else:
        entries = load_selected()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(SEG_MODEL)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(SEG_MODEL)
    model.to(device).eval()
    lookup = build_id_lookup(model.config.id2label)

    SEG_DIR.mkdir(parents=True, exist_ok=True)
    stats_path = SEG_DIR / "stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    for entry in tqdm(entries, desc="seg"):
        name = f"frame_{entry['index']:06d}"
        if args.skip_existing and (SEG_DIR / f"{name}_labels.npz").exists():
            continue
        img = np.asarray(Image.open(entry["path"]).convert("RGB"))
        group_map = segment_frame(img, model, processor, lookup, device)

        np.savez_compressed(SEG_DIR / f"{name}_labels.npz", labels=group_map)
        make_overlay(img, group_map).save(SEG_DIR / f"{name}_overlay.jpg", quality=90)
        stats[name] = {
            g: round(area_weighted_fraction(group_map, gid), 4)
            for g, gid in GROUP_IDS.items()
        }

    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"done -> {SEG_DIR}")


if __name__ == "__main__":
    main()
