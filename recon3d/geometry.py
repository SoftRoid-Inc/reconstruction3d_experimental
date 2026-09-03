"""軸平行ボックス分割による形状生成。全入力が軸平行矩形なので、開口の引き算は
長手方向の1次元区間分割+z方向の腰壁/垂れ壁生成で完結する(ブーリアン不要)。"""
import warnings
from dataclasses import dataclass

from recon3d.loader import Rect


@dataclass(frozen=True)
class Box3D:
    x1: float
    y1: float
    z1: float
    x2: float
    y2: float
    z2: float
    category: str


def rect_px_to_mm(r, mm_per_px, image_height):
    return Rect.from_points(
        r.x1 * mm_per_px, (image_height - r.y1) * mm_per_px,
        r.x2 * mm_per_px, (image_height - r.y2) * mm_per_px,
    )


def _overlap_area(a, b):
    w = min(a.x2, b.x2) - max(a.x1, b.x1)
    h = min(a.y2, b.y2) - max(a.y1, b.y1)
    return w * h if (w > 0 and h > 0) else 0.0


def assign_openings(walls, openings):
    assigned = {}
    for o in openings:
        areas = [_overlap_area(w, o) for w in walls]
        best = max(range(len(walls)), key=lambda i: areas[i]) if walls else None
        if best is None or areas[best] <= 0:
            warnings.warn(f"開口 {o} はどの壁とも交差しないため無視する")
            continue
        assigned.setdefault(best, []).append(o)
    return assigned


def subtract_openings_1d(start, end, holes):
    segs = [(start, end)]
    for h1, h2 in sorted(holes):
        out = []
        for s1, s2 in segs:
            a, b = max(s1, h1), min(s2, h2)
            if a >= b:
                out.append((s1, s2))
                continue
            if s1 < a:
                out.append((s1, a))
            if b < s2:
                out.append((b, s2))
        segs = out
    return segs


def _make_box(along, thick, z1, z2, horizontal, category):
    (a1, a2), (t1, t2) = along, thick
    if horizontal:
        return Box3D(a1, t1, z1, a2, t2, z2, category)
    return Box3D(t1, a1, z1, t2, a2, z2, category)


def wall_to_boxes(wall_mm, openings_mm, cfg, base_z):
    horizontal = not wall_mm.is_vertical()
    if horizontal:
        along, thick = (wall_mm.x1, wall_mm.x2), (wall_mm.y1, wall_mm.y2)
    else:
        along, thick = (wall_mm.y1, wall_mm.y2), (wall_mm.x1, wall_mm.x2)
    top = base_z + cfg.floor_height_mm
    boxes = []
    holes = []
    for rect, kind in openings_mm:
        h1, h2 = (rect.x1, rect.x2) if horizontal else (rect.y1, rect.y2)
        h1, h2 = max(h1, along[0]), min(h2, along[1])
        if h1 >= h2:
            continue
        if kind == "window":
            z1 = base_z + cfg.window_sill_mm
            z2 = z1 + cfg.window_height_mm
        else:
            z1, z2 = base_z, base_z + cfg.door_height_mm
        holes.append((h1, h2, z1, z2, kind))
    # 長手方向の残り(全高)
    for s1, s2 in subtract_openings_1d(along[0], along[1], [(h[0], h[1]) for h in holes]):
        boxes.append(_make_box((s1, s2), thick, base_z, top, horizontal, "wall"))
    # 開口区間の腰壁・垂れ壁とpane
    tc = (thick[0] + thick[1]) / 2
    pane_thick = (tc - cfg.pane_thickness_mm / 2, tc + cfg.pane_thickness_mm / 2)
    for h1, h2, z1, z2, kind in holes:
        if z1 > base_z:
            boxes.append(_make_box((h1, h2), thick, base_z, z1, horizontal, "wall"))
        if z2 < top:
            boxes.append(_make_box((h1, h2), thick, z2, top, horizontal, "wall"))
        boxes.append(_make_box((h1, h2), pane_thick, z1, z2, horizontal, f"pane_{kind}"))
    return boxes


def plan_to_boxes(plan, mm_per_px, cfg, base_z):
    ih = plan.image_size[1]
    walls_mm = [rect_px_to_mm(w, mm_per_px, ih) for w in plan.walls]
    openings = ([(rect_px_to_mm(o, mm_per_px, ih), "window") for o in plan.windows]
                + [(rect_px_to_mm(o, mm_per_px, ih), "door") for o in plan.doors])
    assigned = assign_openings(walls_mm, [r for r, _ in openings])
    kind_of = {id(r): k for r, k in openings}
    boxes = []
    for i, w in enumerate(walls_mm):
        ops = [(r, kind_of[id(r)]) for r in assigned.get(i, [])]
        boxes.extend(wall_to_boxes(w, ops, cfg, base_z))
    for c in plan.columns:
        r = rect_px_to_mm(c, mm_per_px, ih)
        boxes.append(Box3D(r.x1, r.y1, base_z, r.x2, r.y2,
                           base_z + cfg.floor_height_mm, "column"))
    return boxes
