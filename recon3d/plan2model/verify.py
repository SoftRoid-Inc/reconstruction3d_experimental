"""3Dモデルの2D再投影がGTと一致するかの検証。壁+柱のフットプリントIoUと
オーバーレイPNGを出力する。wall_infill(開口充填)とpaneは、GTでは壁ラベルに
含まれない領域なのでIoU計算から除外する。"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from recon3d.plan2model.calibrate import scale_mm_per_px
from recon3d.plan2model.config import ReconConfig
from recon3d.plan2model.geometry import plan_to_boxes
from recon3d.plan2model.loader import Rect, load_floor

SOLID = ("wall", "column")


def rasterize_rects(rects_px, image_size, grid=1):
    w, h = image_size
    mask = np.zeros((h * grid, w * grid), dtype=bool)
    for r in rects_px:
        x1, y1 = int(round(r.x1 * grid)), int(round(r.y1 * grid))
        x2, y2 = int(round(r.x2 * grid)), int(round(r.y2 * grid))
        mask[max(y1, 0):min(y2, h * grid), max(x1, 0):min(x2, w * grid)] = True
    return mask


def _boxes_to_rects_px(boxes, mm_per_px, image_height, categories=SOLID):
    rects = []
    for b in boxes:
        if b.category not in categories:
            continue
        rects.append(Rect.from_points(
            b.x1 / mm_per_px, image_height - b.y1 / mm_per_px,
            b.x2 / mm_per_px, image_height - b.y2 / mm_per_px))
    return rects


def reprojection_iou(plan, boxes, mm_per_px):
    gt = rasterize_rects(plan.walls + plan.columns, plan.image_size)
    pred = rasterize_rects(
        _boxes_to_rects_px(boxes, mm_per_px, plan.image_size[1]),
        plan.image_size)
    inter = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()
    return float(inter) / float(union) if union else 0.0


def save_overlay(plan, boxes, mm_per_px, out_png):
    fig, ax = plt.subplots(figsize=(12, 9))
    for r in plan.walls + plan.columns:
        ax.add_patch(plt.Rectangle((r.x1, r.y1), r.w, r.h,
                                   facecolor="green", alpha=0.4, edgecolor="none"))
    for r in _boxes_to_rects_px(boxes, mm_per_px, plan.image_size[1]):
        ax.add_patch(plt.Rectangle((r.x1, r.y1), r.w, r.h,
                                   fill=False, edgecolor="red", linewidth=0.6))
    for cat, color in (("pane_window", "blue"), ("pane_door", "orange"),
                       ("wall_infill", "purple")):
        for r in _boxes_to_rects_px(boxes, mm_per_px, plan.image_size[1], (cat,)):
            ax.add_patch(plt.Rectangle((r.x1, r.y1), r.w, r.h,
                                       fill=False, edgecolor=color, linewidth=0.6))
    ax.set_xlim(0, plan.image_size[0])
    ax.set_ylim(plan.image_size[1], 0)
    ax.set_aspect("equal")
    ax.set_title(f"{plan.name}: GT(green) vs walls(red) / infill(purple) / "
                 "window(blue) / door(orange)")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", required=True, help="NAME=PATH")
    ap.add_argument("--calib-floor", default=None,
                    help="校正に使う階 NAME=PATH (省略時は--floorと同じ)")
    ap.add_argument("--width-mm", type=float, default=9100.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = ReconConfig(known_width_mm=args.width_mm)
    name, path = args.floor.split("=", 1)
    plan = load_floor(path, name)
    cname, cpath = (args.calib_floor or args.floor).split("=", 1)
    s = scale_mm_per_px(load_floor(cpath, cname), cfg.known_width_mm)
    boxes = plan_to_boxes(plan, s, cfg, base_z=0.0)
    iou = reprojection_iou(plan, boxes, s)
    os.makedirs(args.out, exist_ok=True)
    save_overlay(plan, boxes, s, os.path.join(args.out, f"overlay_{name}.png"))
    print(json.dumps({"floor": name, "iou": iou, "mm_per_px": s}))


if __name__ == "__main__":
    main()
