"""入力データから「何ができるか」を判定する

パイプラインは入力の欠けに対して**黙って縮退してはいけない**。既定セッション
(28-AAAAA) のGTや図面を読んで、もっともらしい数字を出したまま完走する事故が
実際にあった (2026-10-09)。ここで能力を宣言的に出し、steps.py が
「できないステップは理由つきでskip」「必須が欠けていれば明示エラー」に振る。

  IMG_SESSION=/data/reconstruction python -m img_align.core.caps
"""
import json
from dataclasses import dataclass, field

from img_align.config import (CFG, FRAMES_DIR, IMU_DIR, PROP, SESSION,
                              frame_indices)

# 能力名 -> 人間向けの説明
DESC = {
    "frames": "360°等距円筒フレーム (images/frame_number_*.jpg)",
    "masks": "人物マスク (masks/frame_number_*.jpg、白=有効)",
    "imu_segments": "IMU区間 (imu/transitions.json)",
    "imu_idle": "静止区間 (imu/idle_spans.json)",
    "floors": "間取り図の壁注釈 (floors/*.json、labelme形式)",
    "scale": "図面の実寸 (floors/meta.json の known_width_mm)",
    "anchor": "図面アンカー = floors + scale の両方",
    "gt": "GT軌跡 (trajectory_points.csv)",
    "floor_assign": "島→階の手動割当 (backend/floor_assign.json)",
}


@dataclass
class Caps:
    have: set = field(default_factory=set)
    why: dict = field(default_factory=dict)   # 能力名 -> 欠けている理由

    def __contains__(self, name):
        return name in self.have

    def reason(self, name):
        return self.why.get(name, DESC.get(name, name) + " が無い")

    def as_dict(self):
        return {k: (k in self.have) for k in DESC}


def _labelme_floors(zumen_dir):
    """labelme として読める図面JSONの階名 (壁が1本も無いものは数えない)"""
    out = []
    if not zumen_dir.is_dir():
        return out
    for p in sorted(zumen_dir.glob("*.json")):
        if len(p.stem) > 4:
            continue
        try:
            d = json.loads(p.read_text())
            shapes = d["shapes"]
        except Exception:
            continue
        if any(s.get("label") == "wall" for s in shapes):
            out.append(p.stem)
    return out


def detect():
    """いまのセッションの能力を判定する"""
    c = Caps()

    n = len(frame_indices())
    if n:
        c.have.add("frames")
    else:
        c.why["frames"] = f"フレーム画像が0枚: {FRAMES_DIR}"

    md = PROP.person_mask_dir
    if md and md.is_dir() and next(md.glob("frame_number_*"), None):
        c.have.add("masks")
    else:
        c.why["masks"] = (f"人物マスクが無い ({md}) — "
                          "マスク無しで続行するか seg_ade20k を実行")

    # prep が入力の imu/ を IMU_DIR へ取り込む前でも「できる」と判定する
    # (そうしないと inspect が常に「IMUなし」と言う)
    from img_align.core.ingest import source_dir
    src = source_dir()
    for cap, fn in (("imu_segments", "transitions.json"),
                    ("imu_idle", "idle_spans.json")):
        if (IMU_DIR / fn).exists() or (src and (src / fn).exists()):
            c.have.add(cap)
        else:
            c.why[cap] = (f"{fn} が無い。入力の imu/ に置けば prep が "
                          f"{IMU_DIR} へ取り込む")

    floors = _labelme_floors(PROP.zumen_dir)
    if floors:
        c.have.add("floors")
    else:
        c.why["floors"] = (f"壁注釈つきの図面JSONが無い: {PROP.zumen_dir}"
                           " (labelme形式、label=wall)")

    # **壁注釈だけではアンカーできない**。px→m を決める図面の実寸が要る
    # (core/plan_grid.PlanGrid -> recon3d.calibrate.scale_mm_per_px)
    if PROP.known_width_mm is not None or PROP.mm_per_px is not None:
        c.have.add("scale")
    else:
        c.why["scale"] = ("図面の実寸が未設定。floors/meta.json に "
                          '{"known_width_mm": 13877.5} '
                          '(階で違うなら {"1F": ..., "2F": ...}) を書く')
    if {"floors", "scale"} <= c.have:
        c.have.add("anchor")
    else:
        miss = [k for k in ("floors", "scale") if k not in c.have]
        c.why["anchor"] = " / ".join(c.reason(k) for k in miss)

    if _has_gt():
        c.have.add("gt")
    else:
        c.why["gt"] = (f"GT軌跡が無い ({PROP.root}/trajectory_points.csv)。"
                       "評価はできないが軌跡生成とアンカーは動く")

    from img_align.config import OUT_DIR
    if (OUT_DIR / "floor_assign.json").exists():
        c.have.add("floor_assign")
    else:
        c.why["floor_assign"] = (
            "島→階の割当が無い。`python -m img_align.core.floor_assign "
            "--template` で雛形を作って backend/floor_assign.json を編集")
    return c


def _has_gt():
    from img_align.eval.gt_traj import gt_dir
    if (gt_dir() / "trajectory_points.csv").exists():
        return True
    from img_align.config import OUT_DIR
    return (OUT_DIR / "gt_trajectory.csv").exists()


def main():
    c = detect()
    print(f"session {SESSION}  root {PROP.root}")
    print(f"frames  {len(frame_indices())}枚  range {CFG['frames']}")
    for k in DESC:
        ok = k in c.have
        print(f"  [{'x' if ok else ' '}] {k:13} {DESC[k]}")
        if not ok:
            print(f"        -> {c.reason(k)}")


if __name__ == "__main__":
    main()
