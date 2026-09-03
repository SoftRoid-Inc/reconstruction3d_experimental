"""GT(labelme)とCubiCasa prediction.jsonを共通スキーマFloorPlanへ正規化する。"""
import json
from dataclasses import dataclass, field

LABELS = ("wall", "column", "window", "door")


@dataclass(frozen=True)
class Rect:
    x1: float
    y1: float
    x2: float
    y2: float

    @classmethod
    def from_points(cls, xa, ya, xb, yb):
        return cls(min(xa, xb), min(ya, yb), max(xa, xb), max(ya, yb))

    @classmethod
    def from_polygon(cls, poly):
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        return cls(min(xs), min(ys), max(xs), max(ys))

    @property
    def w(self):
        return self.x2 - self.x1

    @property
    def h(self):
        return self.y2 - self.y1

    @property
    def cx(self):
        return (self.x1 + self.x2) / 2

    @property
    def cy(self):
        return (self.y1 + self.y2) / 2

    def is_vertical(self):
        return self.h > self.w


@dataclass
class FloorPlan:
    name: str
    image_size: tuple
    walls: list = field(default_factory=list)
    columns: list = field(default_factory=list)
    windows: list = field(default_factory=list)
    doors: list = field(default_factory=list)


def load_labelme(path, name):
    with open(path) as f:
        d = json.load(f)
    fp = FloorPlan(name=name, image_size=(d["imageWidth"], d["imageHeight"]))
    buckets = {"wall": fp.walls, "column": fp.columns,
               "window": fp.windows, "door": fp.doors}
    for s in d["shapes"]:
        if s.get("shape_type") != "rectangle" or s["label"] not in buckets:
            continue
        (xa, ya), (xb, yb) = s["points"]
        buckets[s["label"]].append(Rect.from_points(xa, ya, xb, yb))
    return fp


def load_prediction(path, name):
    with open(path) as f:
        d = json.load(f)
    fp = FloorPlan(name=name, image_size=tuple(d["image_size"]))
    for w in d.get("walls", []):
        r = Rect.from_polygon(w["polygon"])
        (fp.columns if w.get("is_column") else fp.walls).append(r)
    for key, bucket in (("windows", fp.windows), ("doors", fp.doors)):
        for o in d.get(key, []):
            bucket.append(Rect.from_polygon(o["polygon"]))
    return fp


def load_floor(path, name):
    with open(path) as f:
        d = json.load(f)
    if "shapes" in d:
        return load_labelme(path, name)
    return load_prediction(path, name)
