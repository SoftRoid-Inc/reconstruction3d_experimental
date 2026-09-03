"""px→mmスケール校正。図面の通り芯寸法(既知mm)と、GT外周壁の中心線スパン(px)の
比から係数を求める。理論値(1/50・300dpi→4.233mm/px)は本データでは成立しない
(実測≈11.968mm/px)ため、必ず寸法線ベースで校正する。"""


def centerline_span_px(plan, axis):
    if axis == "x":
        coords = [w.cx for w in plan.walls if w.is_vertical()]
    elif axis == "y":
        coords = [w.cy for w in plan.walls if not w.is_vertical()]
    else:
        raise ValueError(f"axis must be 'x' or 'y': {axis}")
    if len(coords) < 2:
        raise ValueError(f"axis={axis} の壁が2本未満でスパンを測れない")
    span = max(coords) - min(coords)
    if span <= 0:
        raise ValueError(f"axis={axis} のスパンが0")
    return span


def scale_mm_per_px(plan, known_width_mm):
    return known_width_mm / centerline_span_px(plan, "x")


def cross_check_scale(plan, mm_per_px, known_depth_mm):
    depth_mm = centerline_span_px(plan, "y") * mm_per_px
    return abs(depth_mm - known_depth_mm) / known_depth_mm
