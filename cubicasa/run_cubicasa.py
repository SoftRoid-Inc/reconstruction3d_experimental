#!/usr/bin/env python3
"""CubiCasa5k の学習済みモデルで任意の図面画像から壁・柱(推定)・窓・ドアを抽出する。

コンテナ (compose サービス: cubicasa) 内での実行例:
    python /code2/reconstruction3d_experimental/cubicasa/run_cubicasa.py \
        /code2/reconstruction3d_experimental/28-DDDDD/zumen/1F.jpg \
        /code2/reconstruction3d_experimental/28-DDDDD/zumen/2F.jpg \
        --scales 0.5 0.75 1.0 --fp16

出力 (--out 以下、画像名/スケールごと):
    prediction.json  壁/柱/窓/ドア/部屋ポリゴン (元画像ピクセル座標)
    overlay.png      元図面への重ね描き
    rooms_seg.png    部屋クラスのセグメンテーション
    icons_seg.png    アイコンクラスのセグメンテーション

注意: CubiCasa5k に「柱」クラスは無い (データセット上、柱は壁に統合されている)。
柱はリポジトリの wall_is_pillar 規則 (幅 > 平均壁厚 かつ 長さ < 3*幅) を
予測済み壁ポリゴンに適用したヒューリスティックとして出力する。
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOM_CLASSES = ["Background", "Outdoor", "Wall", "Kitchen", "Living Room",
                "Bed Room", "Bath", "Entry", "Railing", "Storage", "Garage",
                "Undefined"]
ICON_CLASSES = ["No Icon", "Window", "Door", "Closet", "Electrical Appliance",
                "Toilet", "Sink", "Sauna Bench", "Fire Place", "Bathtub",
                "Chimney"]

WALL_CLASS = 2
RAILING_CLASS = 8
WINDOW_ICON = 1
DOOR_ICON = 2

N_CLASSES = 44
SPLIT = [21, 12, 11]
ROTATIONS = [(0, 0), (1, -1), (2, 2), (-1, 1)]

# BGR (OpenCV 描画用)
COLOR_WALL = (0, 0, 255)      # 赤
COLOR_COLUMN = (0, 128, 255)  # 橙
COLOR_RAILING = (255, 0, 255) # マゼンタ
COLOR_WINDOW = (255, 0, 0)    # 青
COLOR_DOOR = (0, 180, 0)      # 緑

ROOM_LUT = np.array([
    [255, 255, 255],  # Background
    [200, 230, 200],  # Outdoor
    [60, 60, 60],     # Wall
    [255, 220, 150],  # Kitchen
    [255, 255, 150],  # Living Room
    [200, 200, 255],  # Bed Room
    [150, 220, 255],  # Bath
    [255, 200, 200],  # Entry
    [255, 150, 255],  # Railing
    [200, 255, 255],  # Storage
    [220, 220, 220],  # Garage
    [240, 240, 200],  # Undefined
], dtype=np.uint8)

ICON_LUT = np.array([
    [255, 255, 255],  # No Icon
    [0, 0, 255],      # Window
    [0, 200, 0],      # Door
    [180, 120, 60],   # Closet
    [255, 0, 255],    # Electrical Appliance
    [0, 128, 255],    # Toilet
    [255, 128, 0],    # Sink
    [128, 128, 0],    # Sauna Bench
    [255, 0, 0],      # Fire Place
    [0, 255, 255],    # Bathtub
    [128, 0, 128],    # Chimney
], dtype=np.uint8)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("images", nargs="+", help="入力図面画像 (jpg/png)")
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "results"),
                   help="出力ルートフォルダ")
    p.add_argument("--repo", default="/code2/third_party/CubiCasa5k",
                   help="CubiCasa5k リポジトリのパス")
    p.add_argument("--weights", default=None,
                   help="学習済み重み (省略時: <repo>/model_best_val_loss_var.pkl)")
    p.add_argument("--scales", type=float, nargs="+", default=[0.5],
                   help="入力スケール (複数指定で比較)")
    p.add_argument("--threshold", type=float, default=0.2,
                   help="接合点ヒートマップ閾値 (samples.ipynb=0.2, eval.py=0.4)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--fp16", action="store_true", help="CUDA autocast fp16 で推論")
    return p.parse_args()


def load_model(repo, weights_path, device):
    """モデル構築と重みロード。init_weights が相対パスを使うため repo に chdir する。"""
    sys.path.insert(0, repo)
    os.chdir(repo)
    from floortrans.models import get_model

    model = get_model("hg_furukawa_original", 51)
    model.conv4_ = torch.nn.Conv2d(256, N_CLASSES, bias=True, kernel_size=1)
    model.upsample = torch.nn.ConvTranspose2d(N_CLASSES, N_CLASSES,
                                              kernel_size=4, stride=4)
    checkpoint = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    model.to(device)
    return model


def apply_compat_patches():
    """ドメイン外入力向けの安全化パッチ (リポジトリ本体は改変しない)。

    post_prosessing.adjust_door_points は、開口部に対応する壁ラインが
    見つからない場合 (find_line_map_single が -1 を返す / 壁ライン0本) に
    IndexError で落ちるため、対応不能な開口部をスキップする版に差し替える。
    """
    import floortrans.post_prosessing as pp

    def safe_adjust_door_points(door_points, door_lines, wall_points,
                                wall_lines, door_wall_map):
        for door_line_index, door_line in enumerate(door_lines):
            wall_idx = door_wall_map[door_line_index]
            if wall_idx < 0 or wall_idx >= len(wall_lines):
                continue  # 対応する壁が無い開口部は位置調整せずそのまま
            line_dim = pp.calc_line_dim(door_points, door_line)
            wall_line = wall_lines[wall_idx]
            fixed_value = (wall_points[wall_line[0]][1 - line_dim]
                           + wall_points[wall_line[1]][1 - line_dim]) / 2
            for end_point_index in range(2):
                door_points[door_line[end_point_index]][1 - line_dim] = fixed_value

    pp.adjust_door_points = safe_adjust_door_points


def preprocess(img_bgr, scale):
    """スケール → 白パディング (縦横4の倍数) → 正規化テンソル化。

    Returns: (tensor(1,3,Hp,Wp), (hs, ws) パディング前サイズ)
    """
    if scale != 1.0:
        img_bgr = cv2.resize(img_bgr, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_AREA)
    hs, ws = img_bgr.shape[:2]
    pad_h = (4 - hs % 4) % 4
    pad_w = (4 - ws % 4) % 4
    img_bgr = cv2.copyMakeBorder(img_bgr, 0, pad_h, 0, pad_w,
                                 cv2.BORDER_CONSTANT, value=(255, 255, 255))
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = 2.0 * (img / 255.0) - 1.0
    tensor = torch.from_numpy(np.moveaxis(img, -1, 0)).unsqueeze(0)
    return tensor, (hs, ws)


def predict_tta(model, tensor, unpadded_size, device, fp16):
    """4回転 TTA 推論。パディングを切り落とした平均予測 (1,44,hs,ws) を CPU で返す。"""
    from floortrans.loaders.augmentations import RotateNTurns
    rot = RotateNTurns()

    hs, ws = unpadded_size
    _, _, hp, wp = tensor.shape
    preds = torch.zeros([len(ROTATIONS), N_CLASSES, hs, ws])
    with torch.no_grad():
        for i, (forward, back) in enumerate(ROTATIONS):
            rot_image = rot(tensor, "tensor", forward).to(device)
            if fp16 and device.startswith("cuda"):
                # float16 はこのモデルでオーバーフローして NaN になるため bfloat16 を使う
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred = model(rot_image)
                pred = pred.float()
            else:
                pred = model(rot_image)
            pred = pred.cpu()
            pred = rot(pred, "tensor", back)
            pred = rot(pred, "points", back)
            pred = F.interpolate(pred, size=(hp, wp), mode="bilinear",
                                 align_corners=True)
            preds[i] = pred[0, :, :hs, :ws]
            del rot_image, pred
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
    return preds.mean(0, keepdim=True)


def run_postprocessing(prediction, size, threshold):
    """接合点ヒートマップ + セグメンテーション → ポリゴン群。"""
    from floortrans.post_prosessing import split_prediction, get_polygons

    heatmaps, rooms, icons = split_prediction(prediction, size, SPLIT)
    polygons, types, room_polygons, room_types = get_polygons(
        (heatmaps, rooms, icons), threshold, [WINDOW_ICON, DOOR_ICON])
    return polygons, types, room_polygons, room_types, rooms, icons


def quad_width_length(quad):
    """4点矩形の (短辺, 長辺) = (幅, 長さ)。"""
    edges = [np.linalg.norm(quad[i] - quad[(i + 1) % 4]) for i in range(4)]
    edges = sorted(edges)
    return float(edges[0]), float(edges[-1])


def extract_elements(polygons, types, room_polygons, room_types, inv_scale):
    """予測ポリゴンから壁/柱(推定)/手すり/窓/ドア/部屋を抽出し、元解像度座標で返す。"""
    walls, windows, doors = [], [], []
    for poly, t in zip(polygons, types):
        pts = (np.asarray(poly, dtype=np.float64) * inv_scale)
        if t["type"] == "wall":
            width, length = quad_width_length(pts)
            walls.append({
                "id": len(walls),
                "class": ROOM_CLASSES[t["class"]],  # Wall or Railing
                "polygon": pts.round(1).tolist(),
                "width_px": round(width, 1),
                "length_px": round(length, 1),
                "is_column": False,  # 後段で判定
            })
        elif t["type"] == "icon":
            entry = {
                "polygon": pts.round(1).tolist(),
                "prob": round(float(t.get("prob", 0.0)), 4),
            }
            if t["class"] == WINDOW_ICON:
                entry["id"] = len(windows)
                windows.append(entry)
            elif t["class"] == DOOR_ICON:
                entry["id"] = len(doors)
                doors.append(entry)

    # 柱ヒューリスティック (svg_utils.wall_is_pillar と同じ規則)
    real_walls = [w for w in walls if w["class"] == "Wall"]
    avg_wall_width = (float(np.mean([w["width_px"] for w in real_walls]))
                      if real_walls else 0.0)
    columns = []
    for w in real_walls:
        if w["width_px"] > avg_wall_width and w["length_px"] < 3 * w["width_px"]:
            w["is_column"] = True
            columns.append(w["id"])

    rooms = []
    for rp, rt in zip(room_polygons, room_types):
        geoms = rp.geoms if hasattr(rp, "geoms") else [rp]
        for g in geoms:
            if g.is_empty:
                continue
            coords = (np.asarray(g.exterior.coords, dtype=np.float64) * inv_scale)
            rooms.append({
                "class": ROOM_CLASSES[rt["class"]],
                "polygon": coords.round(1).tolist(),
            })

    return walls, columns, windows, doors, rooms, avg_wall_width


def draw_overlay(img_bgr, walls, windows, doors):
    """元図面に抽出結果を重ね描きする。"""
    overlay = img_bgr.copy()

    def draw_polys(items, color, fill=False, thickness=3):
        for it in items:
            pts = np.asarray(it["polygon"], dtype=np.int32).reshape(-1, 1, 2)
            if fill:
                cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(overlay, [pts], True, color, thickness)

    draw_polys([w for w in walls if w["class"] == "Wall" and not w["is_column"]],
               COLOR_WALL)
    draw_polys([w for w in walls if w["class"] == "Railing"], COLOR_RAILING)
    draw_polys([w for w in walls if w["is_column"]], COLOR_COLUMN, fill=True)
    draw_polys(windows, COLOR_WINDOW, thickness=4)
    draw_polys(doors, COLOR_DOOR, thickness=4)

    blended = cv2.addWeighted(overlay, 0.6, img_bgr, 0.4, 0)
    legend = [("Wall", COLOR_WALL), ("Column(est.)", COLOR_COLUMN),
              ("Railing", COLOR_RAILING), ("Window", COLOR_WINDOW),
              ("Door", COLOR_DOOR)]
    for i, (name, color) in enumerate(legend):
        y = 40 + i * 36
        cv2.rectangle(blended, (20, y - 22), (48, y), color, -1)
        cv2.putText(blended, name, (56, y - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 0, 0), 2, cv2.LINE_AA)
    return blended


def save_segmentations(rooms_prob, icons_prob, out_dir, original_size):
    """rooms/icons の argmax セグメンテーションを元解像度の色画像で保存。"""
    w, h = original_size
    for prob, lut, name in [(rooms_prob, ROOM_LUT, "rooms_seg.png"),
                            (icons_prob, ICON_LUT, "icons_seg.png")]:
        seg = np.argmax(prob, axis=0).astype(np.uint8)
        color = lut[seg][:, :, ::-1]  # RGB→BGR
        color = cv2.resize(color, (w, h), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(out_dir / name), color)


def process_image(model, image_path, scale, threshold, device, fp16, out_root):
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"画像を読めません: {image_path}")
    h0, w0 = img_bgr.shape[:2]

    tensor, unpadded = preprocess(img_bgr, scale)
    try:
        prediction = predict_tta(model, tensor, unpadded, device, fp16)
    except torch.cuda.OutOfMemoryError:
        print(f"    [WARN] CUDA OOM (scale={scale}) → CPU で再試行")
        torch.cuda.empty_cache()
        model.cpu()
        prediction = predict_tta(model, tensor, unpadded, "cpu", False)
        model.to(device)

    hs, ws = unpadded
    polygons, types, room_polygons, room_types, rooms_prob, icons_prob = \
        run_postprocessing(prediction, (hs, ws), threshold)

    walls, columns, windows, doors, rooms, avg_w = extract_elements(
        polygons, types, room_polygons, room_types, inv_scale=1.0 / scale)

    stem = Path(image_path).stem
    out_dir = Path(out_root) / stem / f"s{scale:.2f}"
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "image": str(image_path),
        "image_size": [w0, h0],
        "scale": scale,
        "threshold": threshold,
        "avg_wall_width_px": round(avg_w / scale, 1),
        "counts": {
            "walls": sum(1 for w in walls if w["class"] == "Wall"),
            "columns_estimated": len(columns),
            "railings": sum(1 for w in walls if w["class"] == "Railing"),
            "windows": len(windows),
            "doors": len(doors),
            "rooms": len(rooms),
        },
        "note": "columns は wall_is_pillar 規則による推定 (CubiCasa5k に柱クラスは無い)",
        "walls": walls,
        "column_wall_ids": columns,
        "windows": windows,
        "doors": doors,
        "rooms": rooms,
    }
    with open(out_dir / "prediction.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    cv2.imwrite(str(out_dir / "overlay.png"), draw_overlay(img_bgr, walls, windows, doors))
    save_segmentations(rooms_prob, icons_prob, out_dir, (w0, h0))

    print(f"    -> {out_dir}  counts={result['counts']}")
    return result["counts"]


def main():
    args = parse_args()
    repo = str(Path(args.repo).resolve())
    weights = args.weights or str(Path(repo) / "model_best_val_loss_var.pkl")
    images = [str(Path(p).resolve()) for p in args.images]
    out_root = str(Path(args.out).resolve())

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA が使えないため CPU で実行します")
        device = "cpu"

    print(f"モデルをロード中: {weights}")
    model = load_model(repo, weights, device)
    apply_compat_patches()

    for image_path in images:
        for scale in args.scales:
            print(f"[{Path(image_path).name}] scale={scale} threshold={args.threshold}")
            process_image(model, image_path, scale, args.threshold,
                          device, args.fp16, out_root)

    print("完了")


if __name__ == "__main__":
    main()
