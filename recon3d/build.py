"""物件の複数階GT/predictionから3Dモデル一式(glb/obj/report.json)を生成する。

使い方(コンテナ内):
  python -m recon3d.build --floors 1F=/code2/.../1F.json 2F=/code2/.../2F.json \
      --width-mm 9100 --depth-mm 9555 --out /code2/reconstruction_3d/recon3d/output/28-AAAAA
"""
import argparse
import json

from recon3d.calibrate import cross_check_scale, scale_mm_per_px
from recon3d.config import ReconConfig
from recon3d.geometry import plan_to_boxes, rect_px_to_mm
from recon3d.loader import load_floor
from recon3d.mesh import boxes_to_scene, export_scene, footprint_polygon


def build_model(floor_paths, cfg, out_dir):
    plans = [load_floor(p, name) for name, p in floor_paths]
    mm_per_px = scale_mm_per_px(plans[0], cfg.known_width_mm)
    err = cross_check_scale(plans[0], mm_per_px, cfg.known_depth_mm)

    boxes, slabs, counts = [], [], {}
    for i, plan in enumerate(plans):
        base_z = i * cfg.floor_height_mm
        boxes.extend(plan_to_boxes(plan, mm_per_px, cfg, base_z))
        walls_mm = [rect_px_to_mm(w, mm_per_px, plan.image_size[1])
                    for w in plan.walls]
        fp_poly = footprint_polygon(walls_mm)
        slabs.append((fp_poly, base_z - cfg.slab_thickness_mm, base_z))
        if i == len(plans) - 1:  # 最上階の天井スラブ
            top = base_z + cfg.floor_height_mm
            slabs.append((fp_poly, top, top + cfg.slab_thickness_mm))
        counts[plan.name] = {
            "walls": len(plan.walls), "columns": len(plan.columns),
            "windows": len(plan.windows), "doors": len(plan.doors),
        }

    scene = boxes_to_scene(boxes, slabs)
    outputs = export_scene(scene, out_dir)
    report = {
        "mm_per_px": mm_per_px,
        "scale_cross_check_err": err,
        "counts": counts,
        "num_boxes": len(boxes),
        "assumptions": {
            "floor_height_mm": cfg.floor_height_mm,
            "window_sill_mm": cfg.window_sill_mm,
            "window_height_mm": cfg.window_height_mm,
            "door_height_mm": cfg.door_height_mm,
        },
        "outputs": outputs,
    }
    with open(f"{out_dir}/report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return report


def main():
    ap = argparse.ArgumentParser(description="平面図JSON→3Dモデル再構成")
    ap.add_argument("--floors", nargs="+", required=True,
                    help="下階から順に NAME=PATH (例: 1F=/code2/.../1F.json)")
    ap.add_argument("--width-mm", type=float, default=9100.0,
                    help="通り芯 横全幅mm(スケール校正用)")
    ap.add_argument("--depth-mm", type=float, default=9555.0,
                    help="通り芯 縦全幅mm(交差検証用)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = ReconConfig(known_width_mm=args.width_mm, known_depth_mm=args.depth_mm)
    floor_paths = [tuple(f.split("=", 1)) for f in args.floors]
    report = build_model(floor_paths, cfg, args.out)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
