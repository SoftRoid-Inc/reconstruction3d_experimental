#!/usr/bin/env python3
"""CubiCasa5k 推論結果 (prediction.json) と labelme 正解データの比較評価。

クラスごと (wall / window / door / column) に以下を算出する:
- ピクセルレベル: IoU, precision, recall, F1 (マスクのラスタライズ比較)
- インスタンスレベル (カバレッジ方式):
    GT矩形は面積の coverage_th 以上が同クラス予測マスクに覆われれば「検出」
    予測ポリゴンは面積の coverage_th 以上がGTマスクに覆われれば「有効」
  ※壁はGTと予測で分割粒度が違うため 1対1 IoU ではなくこの方式を使う

実行例 (コンテナ内):
    python compare_gt.py <gt labelme.json> <prediction.json> \
        --image <元画像> --out-dir <出力先>

出力: metrics.json と comparison_<class>.png (緑=一致, 赤=過検出, 青=見逃し)
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

CLASSES = ["wall", "window", "door", "column"]
COVERAGE_TH = 0.5


def load_gt_polygons(gt_path):
    """labelme JSON → {class: [ndarray(N,2)]} と画像サイズ。"""
    data = json.loads(Path(gt_path).read_text(encoding="utf-8"))
    polys = {c: [] for c in CLASSES}
    for s in data["shapes"]:
        if s["label"] not in polys:
            continue
        if s["shape_type"] == "rectangle":
            (x1, y1), (x2, y2) = s["points"]
            pts = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        else:
            pts = s["points"]
        polys[s["label"]].append(np.asarray(pts, dtype=np.float64))
    return polys, (data["imageWidth"], data["imageHeight"])


def load_pred_polygons(pred_path):
    """prediction.json → {class: [ndarray(N,2)]}。

    wall はクラス Wall のみ (Railing 除外)。column は is_column の壁。
    """
    data = json.loads(Path(pred_path).read_text(encoding="utf-8"))
    polys = {c: [] for c in CLASSES}
    for w in data["walls"]:
        if w["class"] != "Wall":
            continue
        pts = np.asarray(w["polygon"], dtype=np.float64)
        polys["wall"].append(pts)
        if w.get("is_column"):
            polys["column"].append(pts)
    for key in ("windows", "doors"):
        for it in data[key]:
            polys[key[:-1]].append(np.asarray(it["polygon"], dtype=np.float64))
    return polys


def rasterize(polys, size):
    w, h = size
    mask = np.zeros((h, w), dtype=np.uint8)
    for p in polys:
        cv2.fillPoly(mask, [np.round(p).astype(np.int32)], 1)
    return mask


def pixel_metrics(gt_mask, pred_mask):
    inter = int(np.count_nonzero(gt_mask & pred_mask))
    gt_a = int(np.count_nonzero(gt_mask))
    pr_a = int(np.count_nonzero(pred_mask))
    union = gt_a + pr_a - inter
    prec = inter / pr_a if pr_a else 0.0
    rec = inter / gt_a if gt_a else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"iou": round(inter / union, 4) if union else 0.0,
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4),
            "gt_px": gt_a, "pred_px": pr_a}


def coverage_counts(polys, other_mask, size, th):
    """各ポリゴンの面積が other_mask に th 以上覆われている個数を数える。"""
    n_ok = 0
    for p in polys:
        m = rasterize([p], size)
        area = int(np.count_nonzero(m))
        if area == 0:
            continue
        covered = int(np.count_nonzero(m & other_mask))
        if covered / area >= th:
            n_ok += 1
    return n_ok


def render_comparison(img, gt_mask, pred_mask, out_png):
    """緑=一致, 赤=過検出(予測のみ), 青=見逃し(GTのみ) を重ね描き。"""
    vis = cv2.addWeighted(img, 0.45, np.full_like(img, 255), 0.55, 0)
    both = (gt_mask & pred_mask).astype(bool)
    fp = (pred_mask & ~gt_mask).astype(bool)
    fn = (gt_mask & ~pred_mask).astype(bool)
    vis[both] = (0, 180, 0)
    vis[fp] = (0, 0, 255)
    vis[fn] = (255, 80, 0)
    cv2.imwrite(str(out_png), vis)


def evaluate(gt_path, pred_path, image_path, out_dir, coverage_th):
    gt_polys, size = load_gt_polygons(gt_path)
    pred_polys = load_pred_polygons(pred_path)
    img = cv2.imread(str(image_path))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {"gt": str(gt_path), "pred": str(pred_path),
               "coverage_th": coverage_th, "classes": {}}

    for cls in CLASSES:
        gt_mask = rasterize(gt_polys[cls], size)
        pred_mask = rasterize(pred_polys[cls], size)
        n_gt, n_pred = len(gt_polys[cls]), len(pred_polys[cls])
        gt_det = coverage_counts(gt_polys[cls], pred_mask, size, coverage_th)
        pred_ok = coverage_counts(pred_polys[cls], gt_mask, size, coverage_th)
        metrics["classes"][cls] = {
            "pixel": pixel_metrics(gt_mask, pred_mask),
            "instance": {
                "gt_total": n_gt, "gt_detected": gt_det,
                "gt_recall": round(gt_det / n_gt, 4) if n_gt else None,
                "pred_total": n_pred, "pred_valid": pred_ok,
                "pred_precision": round(pred_ok / n_pred, 4) if n_pred else None,
            },
        }
        if n_gt or n_pred:
            render_comparison(img, gt_mask, pred_mask,
                              out_dir / f"comparison_{cls}.png")

    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def format_row(name, m):
    px, inst = m["pixel"], m["instance"]
    return (f"{name:<8} pxIoU={px['iou']:.3f} P={px['precision']:.3f} "
            f"R={px['recall']:.3f} | GT検出 {inst['gt_detected']}/{inst['gt_total']} "
            f"予測有効 {inst['pred_valid']}/{inst['pred_total']}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("gt_json")
    p.add_argument("prediction_json")
    p.add_argument("--image", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--coverage-th", type=float, default=COVERAGE_TH)
    args = p.parse_args()

    metrics = evaluate(args.gt_json, args.prediction_json, args.image,
                       args.out_dir, args.coverage_th)
    for cls in CLASSES:
        print("   ", format_row(cls, metrics["classes"][cls]))


if __name__ == "__main__":
    main()
