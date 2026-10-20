"""物件の複数階GT/predictionから3Dモデル一式(glb/obj/report.json)を生成する。

撮影データは要らず `floors/` だけで走るので、パイプラインのステップでは
なく独立したサブコマンドにしてある。

使い方:
  python main.py plan2model build  input=/data/x
  python main.py plan2model verify input=/data/x     # 2D再投影IoU

階の一覧・図面の実寸・出力先はすべて入力ディレクトリから決まる
(`floors/*.json` と `floors/meta.json` の `known_width_mm`、
 出力は `<input>/results/plan2model/`)。
"""
import argparse
import json

from recon3d.plan2model.calibrate import cross_check_scale, scale_mm_per_px
from recon3d.plan2model.config import ReconConfig
from recon3d.plan2model.geometry import plan_to_boxes, rect_px_to_mm
from recon3d.plan2model.loader import load_floor
from recon3d.plan2model.mesh import boxes_to_scene, export_scene, footprint_polygon


def build_model(floor_paths, cfg, out_dir):
    plans = [load_floor(p, name) for name, p in floor_paths]
    mm_per_px = scale_mm_per_px(plans[0], cfg.known_width_mm)
    err = cross_check_scale(plans[0], mm_per_px, cfg.known_depth_mm)

    boxes, slabs, counts = [], [], {}
    for i, plan in enumerate(plans):
        base_z = i * cfg.floor_height_mm
        boxes.extend(plan_to_boxes(plan, mm_per_px, cfg, base_z))
        # 壁は開口で途切れるため、柱・開口も含めてフットプリントを閉じる
        solid_rects = plan.walls + plan.columns + plan.windows + plan.doors
        rects_mm = [rect_px_to_mm(r, mm_per_px, plan.image_size[1])
                    for r in solid_rects]
        fp_poly = footprint_polygon(rects_mm)
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


def floors_from_paths(paths):
    """入力契約から (階名, JSONパス文字列) を下階から順に返す"""
    cands = paths.floor_jsons()
    if not cands:
        raise SystemExit(f"間取り図のJSONが無い: {paths.floors}")
    return [(name, str(p)) for name, p in cands]


def build_from_paths(paths, out_dir=None):
    """入力契約から3Dモデルを作る (main.py plan2model build の実体)

    `--floors` / `--width-mm` を手で並べる代わりに、`floors/*.json` と
    `floors/meta.json` の `known_width_mm` から埋める。
    """
    floor_paths = floors_from_paths(paths)
    kw = paths.known_width_mm
    if kw is None:
        raise SystemExit(
            f"図面の実寸が未設定: {paths.floors / 'meta.json'} に "
            '{"known_width_mm": 13877.5} を書いてください '
            "(壁注釈だけでは px→m が決まらない)")
    # 階でスパンが違う物件は辞書。校正は最下階の値を使う
    if isinstance(kw, dict):
        kw = kw.get(floor_paths[0][0]) or next(iter(kw.values()))
    cfg = ReconConfig(known_width_mm=kw)
    out = out_dir or (paths.results / "plan2model")
    report = build_model(floor_paths, cfg, str(out))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return out


def main(paths, argv=None):
    ap = argparse.ArgumentParser(description="平面図JSON→3Dモデル再構成")
    ap.add_argument("--floors", nargs="+", required=True,
                    help="下階から順に NAME=PATH (例: 1F=/code2/.../1F.json)")
    ap.add_argument("--width-mm", type=float, default=9100.0,
                    help="通り芯 横全幅mm(スケール校正用)")
    ap.add_argument("--depth-mm", type=float, default=9555.0,
                    help="通り芯 縦全幅mm(交差検証用)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    cfg = ReconConfig(known_width_mm=args.width_mm,
                      known_depth_mm=args.depth_mm)
    floor_paths = [tuple(f.split("=", 1)) for f in args.floors]
    report = build_model(floor_paths, cfg, args.out)
    print(json.dumps(report, indent=2, ensure_ascii=False))

