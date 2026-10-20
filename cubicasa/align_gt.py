#!/usr/bin/env python3
"""labelme 正解データ (rectangle) の整列後処理。

- 共線エッジのスナップ (垂直・水平の完全アライン)
- 同一壁ライン上の壁/窓/ドアの厚み帯の統一と隙間閉鎖
- 壁端の直交壁への延長 (コーナー/T字の隙間閉鎖)
- 柱断面を壁帯に一致させ壁からのはみ出しを解消

実行例 (コンテナ内):
    python /code2/reconstruction3d_experimental/cubicasa/align_gt.py \
        /code2/reconstruction3d_experimental/<物件>/zumen/1F.json \
        /code2/reconstruction3d_experimental/<物件>/zumen/2F.json \
        --out-dir /code2/reconstruction3d_experimental/<物件>/gt_aligned

出力: <out-dir>/<stem>.json (labelme形式, 元は無改変) と <stem>_after.png (確認用)
"""

import argparse
import json
import statistics
from pathlib import Path

BAND_LABELS = {"wall", "window", "door"}  # 壁ラインを構成する要素
SNAP_TOL = 4.0      # エッジ座標クラスタリング許容 (壁厚の半分未満であること)
GAP_TOL = 30.0      # 同一ライン上の隙間/壁端延長の最大距離
CROSS_OVERLAP = 0.5  # 同一ラインとみなす厚み方向の重なり率 (薄い方基準)


class Rect:
    def __init__(self, shape):
        (x1, y1), (x2, y2) = shape["points"]
        self.x1, self.x2 = sorted((float(x1), float(x2)))
        self.y1, self.y2 = sorted((float(y1), float(y2)))
        self.label = shape["label"]
        self.shape = shape

    @property
    def horizontal(self):
        return (self.x2 - self.x1) >= (self.y2 - self.y1)

    def band(self, horizontal):
        """厚み方向の区間 (横要素なら y帯, 縦要素なら x帯)。"""
        return (self.y1, self.y2) if horizontal else (self.x1, self.x2)

    def set_band(self, horizontal, lo, hi):
        if horizontal:
            self.y1, self.y2 = lo, hi
        else:
            self.x1, self.x2 = lo, hi

    def span(self, horizontal):
        """長手方向の区間。"""
        return (self.x1, self.x2) if horizontal else (self.y1, self.y2)

    def set_span(self, horizontal, lo, hi):
        if horizontal:
            self.x1, self.x2 = lo, hi
        else:
            self.y1, self.y2 = lo, hi

    def points(self):
        return [[round(self.x1, 2), round(self.y1, 2)],
                [round(self.x2, 2), round(self.y2, 2)]]


def overlap(a1, a2, b1, b2):
    return min(a2, b2) - max(a1, b1)


# ---------------------------------------------------------------- step 1
def snap_edge_coordinates(rects, tol):
    """全矩形のエッジ座標を軸ごとにクラスタ化し中央値へスナップする。"""
    for axis in ("x", "y"):
        refs = []
        for r in rects:
            refs.append((r, axis + "1"))
            refs.append((r, axis + "2"))
        refs.sort(key=lambda rv: getattr(rv[0], rv[1]))

        cluster = []
        clusters = [cluster]
        prev = None
        for ref in refs:
            v = getattr(ref[0], ref[1])
            if prev is not None and v - prev > tol:
                cluster = []
                clusters.append(cluster)
            cluster.append(ref)
            prev = v
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            snapped = statistics.median(getattr(r, a) for r, a in cluster)
            for r, a in cluster:
                setattr(r, a, snapped)


# ---------------------------------------------------------------- step 2
def group_lines(rects):
    """壁/窓/ドアを向きと厚み帯の重なりで「壁ライン」にグループ化する。"""
    lines = []
    for horizontal in (True, False):
        members = [r for r in rects
                   if r.label in BAND_LABELS and r.horizontal == horizontal]
        groups = []  # 各要素: list[Rect]
        for r in sorted(members, key=lambda r: r.band(horizontal)[0]):
            lo, hi = r.band(horizontal)
            placed = None
            for g in groups:
                glo, ghi = g["lo"], g["hi"]
                ov = overlap(lo, hi, glo, ghi)
                if ov > CROSS_OVERLAP * min(hi - lo, ghi - glo):
                    placed = g
                    break
            if placed is None:
                placed = {"lo": lo, "hi": hi, "members": []}
                groups.append(placed)
            placed["members"].append(r)
            placed["lo"] = min(placed["lo"], lo)
            placed["hi"] = max(placed["hi"], hi)
        for g in groups:
            lines.append((horizontal, g["members"]))
    return lines


def unify_line_bands(lines):
    """ライン内の厚み帯を壁メンバーの中央値に統一する。"""
    n_changed = 0
    for horizontal, members in lines:
        basis = [r for r in members if r.label == "wall"] or members
        lo = statistics.median(r.band(horizontal)[0] for r in basis)
        hi = statistics.median(r.band(horizontal)[1] for r in basis)
        for r in members:
            if r.band(horizontal) != (lo, hi):
                r.set_band(horizontal, lo, hi)
                n_changed += 1
    return n_changed


# ---------------------------------------------------------------- step 3
def close_line_gaps(lines, columns, gap_tol):
    """同一ライン上の隣接要素間の小さな隙間/重なりを解消する。

    隙間に柱 (方立) が入っている場合は柱の両エッジへスナップし、
    柱を突き抜けて要素同士を繋げない。それ以外は境界を中点で共有する。
    """
    closures = []
    for horizontal, members in lines:
        members = sorted(members, key=lambda r: r.span(horizontal)[0])
        for a, b in zip(members, members[1:]):
            a_lo, a_hi = a.span(horizontal)
            b_lo, b_hi = b.span(horizontal)
            gap = b_lo - a_hi
            if gap == 0 or abs(gap) > gap_tol:
                continue
            blo, bhi = a.band(horizontal)
            gap_col = None
            for c in columns:
                clo, chi = c.span(horizontal)
                if (overlap(*c.band(horizontal), blo, bhi)
                        > CROSS_OVERLAP * (bhi - blo)
                        and a_hi - SNAP_TOL <= clo and chi <= b_lo + SNAP_TOL):
                    gap_col = c
                    break
            if gap_col is not None:
                clo, chi = gap_col.span(horizontal)
                a.set_span(horizontal, a_lo, clo)
                b.set_span(horizontal, chi, b_hi)
                kind = f"{a.label}-柱-{b.label}"
            else:
                mid = (a_hi + b_lo) / 2
                a.set_span(horizontal, a_lo, mid)
                b.set_span(horizontal, mid, b_hi)
                kind = f"{a.label}-{b.label}"
            closures.append((("H" if horizontal else "V"), kind,
                             round((a_hi + b_lo) / 2), round(gap, 1)))
    return closures


def extend_ends_to_perpendicular(rects, gap_tol):
    """壁ライン要素の端点を近傍の直交壁まで延長する。

    壁は直交壁の外縁まで (コーナーを完全に重ねる)、窓/ドアは内縁まで。
    """
    walls = [r for r in rects if r.label == "wall"]
    n_ext = 0
    for r in rects:
        if r.label not in BAND_LABELS:
            continue
        horizontal = r.horizontal
        blo, bhi = r.band(horizontal)
        for end in (0, 1):  # 0: 始端(小さい側), 1: 終端(大きい側)
            slo, shi = r.span(horizontal)
            e = shi if end else slo
            for w in walls:
                if w is r or w.horizontal == horizontal:
                    continue
                wlo, whi = w.band(not horizontal)      # 直交壁の長手 = rの厚み方向
                nlo, nhi = w.span(not horizontal)      # 直交壁の帯 = rの長手方向
                if overlap(blo, bhi, nlo, nhi) < CROSS_OVERLAP * (bhi - blo):
                    continue
                if end:  # 終端を +方向へ延長
                    near, far = wlo, whi
                    if near - gap_tol <= e < (far if r.label == "wall" else near):
                        r.set_span(horizontal, slo,
                                   far if r.label == "wall" else near)
                        n_ext += 1
                else:    # 始端を -方向へ延長
                    near, far = whi, wlo
                    if (far if r.label == "wall" else near) < e <= near + gap_tol:
                        r.set_span(horizontal,
                                   far if r.label == "wall" else near, shi)
                        n_ext += 1
    return n_ext


# ---------------------------------------------------------------- step 4
def fit_columns_to_walls(rects, tol):
    """柱の断面を重なる壁の帯に一致させ、壁の範囲内へクランプする。

    壁に重ならない柱 (窓間の方立など) は窓/ドアの帯へフォールバックする。
    """
    walls = [r for r in rects if r.label == "wall"]
    openings = [r for r in rects if r.label in ("window", "door")]
    n_fit, unmatched = 0, []

    def best_of(cands, col, horizontal):
        """col と重なる要素のうち向きが horizontal のもので重なり最大を返す。"""
        best, best_area = None, 0.0
        for w in cands:
            if w.horizontal != horizontal:
                continue
            ox = overlap(col.x1, col.x2, w.x1 - tol, w.x2 + tol)
            oy = overlap(col.y1, col.y2, w.y1 - tol, w.y2 + tol)
            if ox > 0 and oy > 0 and ox * oy > best_area:
                best, best_area = w, ox * oy
        return best

    def best_wall(col, horizontal):
        return best_of(walls, col, horizontal)

    def clamp_span(col, wall, horizontal):
        """柱の長手方向位置を壁の長手範囲内へ平行移動でクランプする。"""
        clo, chi = col.span(horizontal)
        wlo, whi = wall.span(horizontal)
        shift = max(wlo - clo, 0) or min(whi - chi, 0)
        if shift:
            col.set_span(horizontal, clo + shift, chi + shift)

    for col in [r for r in rects if r.label == "column"]:
        h_wall = best_wall(col, True)
        v_wall = best_wall(col, False)
        if h_wall is None and v_wall is None:
            # 方立: 窓/ドアの帯に断面だけ合わせる (長手位置は動かさない)
            elem = best_of(openings, col, True) or best_of(openings, col, False)
            if elem is None:
                unmatched.append((round(col.x1), round(col.y1)))
                continue
            if elem.horizontal:
                col.y1, col.y2 = elem.y1, elem.y2
            else:
                col.x1, col.x2 = elem.x1, elem.x2
            n_fit += 1
            continue
        if h_wall is not None:
            col.y1, col.y2 = h_wall.y1, h_wall.y2   # 断面を壁厚に一致
            if v_wall is None:
                clamp_span(col, h_wall, True)       # 壁の長手内へ
        if v_wall is not None:
            col.x1, col.x2 = v_wall.x1, v_wall.x2
            if h_wall is None:
                clamp_span(col, v_wall, False)
        n_fit += 1
    return n_fit, unmatched


# ---------------------------------------------------------------- I/O
def process_file(json_path, out_dir, snap_tol, gap_tol):
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    rects = [Rect(s) for s in data["shapes"]
             if s["shape_type"] == "rectangle"]
    original = {id(r): (r.x1, r.y1, r.x2, r.y2) for r in rects}

    snap_edge_coordinates(rects, snap_tol)
    lines = group_lines(rects)
    n_band = unify_line_bands(lines)
    columns = [r for r in rects if r.label == "column"]
    closures = close_line_gaps(lines, columns, gap_tol)
    n_ext = extend_ends_to_perpendicular(rects, gap_tol)
    n_fit, unmatched = fit_columns_to_walls(rects, snap_tol)

    max_shift = 0.0
    for r in rects:
        ox1, oy1, ox2, oy2 = original[id(r)]
        max_shift = max(max_shift, abs(r.x1 - ox1), abs(r.y1 - oy1),
                        abs(r.x2 - ox2), abs(r.y2 - oy2))
        r.shape["points"] = r.points()

    stem = Path(json_path).stem
    rel = Path("..") / Path(json_path).parent.name / data["imagePath"]
    data["imagePath"] = str(rel)
    data["imageData"] = None
    out_path = Path(out_dir) / f"{stem}.json"
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    print(f"[{stem}] 帯統一 {n_band}件, 隙間閉鎖 {len(closures)}件, "
          f"端部延長 {n_ext}件, 柱フィット {n_fit}件, 最大移動 {max_shift:.1f}px")
    for c in closures:
        print(f"    隙間閉鎖 {c[0]} {c[1]} @{c[2]} (gap {c[3]}px)")
    if unmatched:
        print(f"    [WARN] 壁に対応しない柱 {len(unmatched)}個: {unmatched}")
    return out_path


def render_overlay(json_path, image_path, out_png):
    import cv2
    colors = {"wall": (0, 0, 255), "column": (0, 165, 255),
              "window": (255, 0, 0), "door": (0, 180, 0)}
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    img = cv2.imread(str(image_path))
    for s in data["shapes"]:
        (x1, y1), (x2, y2) = s["points"]
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)),
                      colors.get(s["label"], (128, 128, 128)), 2)
    cv2.imwrite(str(out_png), img)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("jsons", nargs="+", help="labelme JSON (元ファイルは無改変)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--snap-tol", type=float, default=SNAP_TOL)
    p.add_argument("--gap-tol", type=float, default=GAP_TOL)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for jp in args.jsons:
        out_json = process_file(jp, out_dir, args.snap_tol, args.gap_tol)
        img = Path(jp).parent / json.loads(Path(jp).read_text())["imagePath"]
        render_overlay(out_json, img, out_dir / f"{Path(jp).stem}_after.png")


if __name__ == "__main__":
    main()
