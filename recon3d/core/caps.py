"""入力データから「何ができるか」を判定する

パイプラインは入力の欠けに対して**黙って縮退してはいけない**。既定セッション
(28-AAAAA) のGTや図面を読んで、もっともらしい数字を出したまま完走する事故が
実際にあった (2026-10-09)。ここで能力を宣言的に出し、steps.py が
「できないステップは理由つきでskip」「必須が欠けていれば明示エラー」に振る。

  python main.py inspect input=/data/reconstruction
"""
import json
import os
from dataclasses import dataclass, field
from pathlib import Path


CGROUP_V2 = Path("/sys/fs/cgroup/cpu.max")
CGROUP_V1 = Path("/sys/fs/cgroup/cpu")


def cpus(root="/"):
    """このプロセスが実際に使える CPU 数

    **`os.cpu_count()` を直接使ってはいけない。** ノードのコア数を返すので、
    CPU limit 付きの Kubernetes ポッドでは桁違いに過剰な並列数になる
    (ノード64コア → 16並列、ポッドの割当は4コア)。島ごとに点群を読む
    アンカーでは、過剰並列はそのまま OOMKilled になる。

    cgroup の quota → cpuset → ノードのコア数 の順で解決する。
    `root` はテスト用 (cgroup ファイルを模したディレクトリを指す)。
    """
    root = Path(root)
    v2 = root / CGROUP_V2.relative_to("/")
    if v2.is_file():                              # cgroup v2: "<quota> <period>"
        f = v2.read_text().split()
        n = _quota(f[0], f[1]) if len(f) >= 2 else None
        if n:
            return n
    v1 = root / CGROUP_V1.relative_to("/")
    if v1.is_dir():                               # cgroup v1
        try:
            n = _quota((v1 / "cpu.cfs_quota_us").read_text(),
                       (v1 / "cpu.cfs_period_us").read_text())
        except OSError:
            n = None
        if n:
            return n
    try:
        return len(os.sched_getaffinity(0))       # cpuset を反映する
    except AttributeError:                        # Linux 以外
        pass
    return os.cpu_count() or 4


def _quota(quota, period):
    """quota/period -> CPU数 (切り上げ)。無制限・不正なら None"""
    try:
        q, p = float(quota), float(period or 100000)
    except (TypeError, ValueError):
        return None
    if q <= 0 or p <= 0:                          # "max" / -1 = 無制限
        return None
    return max(1, int(-(-q // p)))

# 能力名 -> 人間向けの説明
DESC = {
    "frames": "360°等距円筒フレーム (images/frame_number_*.jpg)",
    "masks": "人物マスク (masks/frame_number_*.jpg、白=有効)",
    "imu_segments": "IMU区間 (imu/transitions.json)",
    "imu_idle": "静止区間 (imu/idle_spans.json)",
    "floors": "間取り図の壁注釈 (floors/*.json、labelme形式)",
    "scale": "図面の実寸 (floors/meta.json の known_width_mm)",
    "anchor": "図面アンカー = floors + scale の両方",
    "scale_prior": "島のメートル尺度の供給源 (backend/scale_bias.json "
                   "または macond タグ)",
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
        if name in self.why:
            return self.why[name]
        return f"{DESC.get(name) or name} が無い"

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


def detect(paths):
    """いまのセッションの能力を判定する"""
    c = Caps()

    n = len(paths.frame_indices)
    if n:
        c.have.add("frames")
    else:
        c.why["frames"] = f"フレーム画像が0枚: {paths.images}"

    md = paths.masks
    if md and md.is_dir() and next(md.glob("frame_number_*"), None):
        c.have.add("masks")
    else:
        c.why["masks"] = (f"人物マスクが無い ({md}) — "
                          "マスク無しで続行するか seg_ade20k を実行")

    # prep が入力の imu/ を paths.imu へ取り込む前でも「できる」と判定する
    # (そうしないと inspect が常に「IMUなし」と言う)
    from recon3d.core.ingest import source_dir
    src = source_dir(paths)
    for cap, fn in (("imu_segments", "transitions.json"),
                    ("imu_idle", "idle_spans.json")):
        if (paths.imu / fn).exists() or (src and (src / fn).exists()):
            c.have.add(cap)
        else:
            c.why[cap] = (f"{fn} が無い。入力の imu/ に置けば prep が "
                          f"{paths.imu} へ取り込む")

    floors = _labelme_floors(paths.floors)
    if floors:
        c.have.add("floors")
    else:
        c.why["floors"] = (f"壁注釈つきの図面JSONが無い: {paths.floors}"
                           " (labelme形式、label=wall)")

    # **壁注釈だけではアンカーできない**。px→m を決める図面の実寸が要る
    # (core/plan_grid.PlanGrid -> recon3d.calibrate.scale_mm_per_px)
    if paths.known_width_mm is not None or paths.mm_per_px is not None:
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

    # アンカーの候補生成は事前スケールの ±40% しか探索しないので、
    # m_per_unit がずれていると**正解の仮説がそもそも作られない**。
    # 既定の slab (カメラ高/床スラブピーク) は0912実測で誤差97〜383%。
    # 動きはするが黙って悪くなるので、どの供給源が使えるかを出す
    src = scale_prior_source(paths)
    if src == "slab":
        c.why["scale_prior"] = (
            "mapany も較正も無いので slab (カメラ高/床スラブ) に落ちる — "
            "0912実測で誤差97〜383%。アンカーは動くが当たりにくい。"
            "steps に mapany を入れて macond を作るか、1度アンカーを通して "
            "`main.py tool scale_prior --calibrate` で scale_bias.json を書く")
    else:
        c.have.add("scale_prior")

    if _has_gt(paths):
        c.have.add("gt")
    else:
        c.why["gt"] = (f"GT軌跡が無い ({paths.root}/trajectory_points.csv)。"
                       "評価はできないが軌跡生成とアンカーは動く")

    if (paths.backend / "floor_assign.json").exists():
        c.have.add("floor_assign")
    else:
        c.why["floor_assign"] = (
            "島→階の割当が無い。`python -m recon3d.core.floor_assign "
            "--template` で雛形を作って backend/floor_assign.json を編集")
    return c


def scale_prior_source(paths):
    """アンカーが実際に使う m_per_unit の供給源 ("calib"/"mapany"/"slab")

    優先順は postproc/scale_prior.m_per_unit と同じ:
    ①島自身のアンカー由来の較正 ②MapAnything条件付け出力 ③床スラブ推定
    """
    if (paths.backend / "scale_bias.json").exists():
        return "calib"
    if paths.bench.is_dir() and next(paths.bench.glob("macond*"), None):
        return "mapany"
    return "slab"


def _has_gt(paths):
    if (paths.root / "trajectory_points.csv").exists():
        return True
    return (paths.backend / "gt_trajectory.csv").exists()


def main(paths, argv=None):
    """入力の能力 (何が揃っていて何が足りないか) を表示する"""
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    c = detect(paths)
    print(f"dataset {paths.name}  root {paths.root}")
    print(f"frames  {len(paths.frame_indices)}枚  (0..{paths.n_frames - 1})")
    for k in DESC:
        ok = k in c.have
        note = (f" -> {scale_prior_source(paths)}"
                if k == "scale_prior" else "")
        print(f"  [{'x' if ok else ' '}] {k:13} {DESC[k]}{note}")
        if not ok:
            print(f"        -> {c.reason(k)}")

