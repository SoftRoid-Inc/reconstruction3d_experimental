"""軌跡GT (<物件>/trajectory_points.csv) による位置合わせ推定の定量評価

GT形式: frame_number + 図面画像幅で正規化した (x, y, z)。階はフレーム範囲で決まる
(trajectory_data_by_scan_data_id.csv の floor_start/end_frame_number)。
GTは物件ごとに置く。28-BBBBBは2026-10-13に正式GTが入り、それ以前に使っていた
手動登録 (results/<s>/backend/gt_trajectory.csv) は不要になった
(未配備の物件ではフォールバックとして今も使う)。
評価対象: results/fuse/fused.json (PF統合軌跡) / results/loc*/summary.json (単フレーム照合)
"""
import csv
import functools
import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from img_align.config import PROP, PROJECT_ROOT, RESULTS_DIR, SESSION
from img_align.core.plan_grid import ZUMEN_DIR

def gt_dir(session=None):
    """GT CSV群の置き場。物件ごとに持つ

    28-AAAAA へのフォールバックは**プロジェクト内の物件だけ**に効かせる。
    外部データセット (入力ディレクトリ渡し) でこれを許すと、GTを持たない
    新しい建物を 28-AAAAA のGTで評価して例外ゼロで完走する
    (2026-10-09に実際に踏んだ事故の経路)。
    """
    from img_align.config import PROPERTY
    root = (PROPERTY[session] if session else PROP).root
    if (root / "trajectory_points.csv").exists():
        return root
    if PROJECT_ROOT in root.parents or root == PROJECT_ROOT:
        return PROJECT_ROOT / "28-AAAAA"
    return root     # 外部データセット: 存在しないので後段が FileNotFoundError


GT_DIR = gt_dir()
OUT_DIR = RESULTS_DIR / "eval_gt"
DRAWING_W, DRAWING_H = PROP.drawing_wh  # 図面画像サイズ (yは下原点→上原点に反転が必要)


def _plan_json():
    """スケール導出に使う図面JSON (階名, Path)。無ければ None

    「1F」決め打ちをやめた。B1F や 2F から始まる部分スキャンでは 1F.json が
    存在せず、黙って mm/px=None に落ちていた。
    """
    if not ZUMEN_DIR.is_dir():
        return None
    cands = sorted(p for p in ZUMEN_DIR.glob("*.json") if len(p.stem) <= 4)
    if not cands:
        return None
    pick = next((p for p in cands if p.stem == "1F"), cands[0])
    return pick.stem, pick


@functools.lru_cache(maxsize=None)
def mm_per_px():
    """図面実測スケール [mm/px]。導出できなければ None

    **import 時に評価してはいけない** (旧 `MM_PER_PX` 定数)。図面JSONが
    labelme でない・`known_width_mm` が未設定といった入力で例外が飛ぶと、
    このモジュールを top-level import する20モジュールと、`plan_icp` 経由の
    二次伝播16モジュール (= アンカー経路まるごと) が import 不能になる。
    """
    if PROP.mm_per_px is not None:
        return PROP.mm_per_px
    if PROP.known_width_mm is None:
        return None             # 図面の実寸が無ければ px→m は決められない
    found = _plan_json()
    if found is None:
        return None
    floor, path = found
    kw = PROP.known_width_mm
    kw = kw.get(floor) if isinstance(kw, dict) else kw
    if kw is None:
        return None
    try:
        from recon3d.calibrate import scale_mm_per_px
        from recon3d.loader import load_labelme
        return scale_mm_per_px(load_labelme(path, floor), kw)
    except (FileNotFoundError, ValueError, KeyError, TypeError,
            ZeroDivisionError) as e:
        print(f"[gt] mm/px を {path.name} から導出できない ({type(e).__name__}"
              f": {e}) -> None")
        return None

# 階のフレーム範囲 (0828セッション。後方互換用。一般にはload_gtがCSVから導出)
FLOOR_RANGES = [(0, 233, "1F"), (234, 411, "2F"), (412, 453, "1F")]


def _floor_names(gdir, rows):
    """floor_id -> 階名。floor_info.csv が無ければ floor_id 昇順で 1F,2F,… とする

    28-BBBBB では floor_info.csv が未配備。昇順=下階からであることは、旧GT
    (手動登録) の階ラベルとの突き合わせで実証済み
    (0912: 37258→1F×236, 37259→2F×208, 37260→3F×184)。
    """
    p = gdir / "floor_info.csv"
    if p.exists():
        return {r["floor_id"]: r["floor_name"] for r in csv.DictReader(open(p))}
    ids = sorted({r["floor_id"] for r in rows}, key=int)
    names = {fid: f"{i + 1}F" for i, fid in enumerate(ids)}
    print(f"[gt] floor_info.csv なし -> floor_id昇順で階名を推定: {names}")
    return names


def session_meta(session=None):
    """セッション(MMDD) -> {'traj': trajectory_id集合, 'floors': [(a,b,name)]}
    trajectory_data_by_scan_data_id.csv + floor_info.csv から導出"""
    from img_align.config import PROP, PROPERTY, SESSION
    session = session or SESSION
    # 外部データセットは撮影日とセッション名が一致しないので gt_session を見る
    prop = PROPERTY.get(session, PROP)
    key = getattr(prop, "gt_session", None) or session
    gdir = gt_dir(session)
    rows = list(csv.DictReader(
        open(gdir / "trajectory_data_by_scan_data_id.csv")))
    floor_names = _floor_names(gdir, rows)
    meta = {}
    for r in rows:
        # metadata_s3_key: v3/28/<物件>/<YYYY>/<MM>/<DD>/... → MMDD
        p = r["metadata_s3_key"].split("/")
        sess = p[4] + p[5]
        m = meta.setdefault(sess, {"traj": set(), "floors": []})
        m["traj"].add(r["trajectory_id"])
        rng = (int(float(r["floor_start_frame_number"])),
               int(float(r["floor_end_frame_number"])),
               floor_names[r["floor_id"]])
        if rng not in m["floors"]:
            m["floors"].append(rng)
    if key not in meta:
        raise KeyError(f"session {session} (key {key}) not in metadata "
                       f"({list(meta)})")
    return meta[key]


def load_gt(session=None):
    """frame -> {floor, x_px, y_px} (GTはほぼ1-2フレーム刻み)。
    sessionはconfig.SESSION(環境変数IMG_SESSION)既定。
    正式GTは <物件>/trajectory_points.csv。そこに無いセッションだけ
    手動登録由来の backend/gt_trajectory.csv (align_toolが書く正規化座標、
    yは下原点) へフォールバックする。"""
    # フォールバック経路は results/<session>/ を読むので、引数で渡された
    # セッションの物件・図面サイズに揃える (凍結済みの定数を使うと別セッションの
    # GTを黙って返してしまう。2026-10-09に実際に踏んだ)
    from img_align.config import PROPERTY
    prop = PROPERTY[session] if session else PROP
    dw, dh = prop.drawing_wh
    try:
        m = session_meta(session)
    except KeyError:
        csv_p = (prop.root / "results" / (session or SESSION)
                 / "backend" / "gt_trajectory.csv")
        if not csv_p.exists():
            raise
        gt = {}
        with open(csv_p) as f:
            for r in csv.DictReader(f):
                gt[int(r["frame_number"])] = {
                    "floor": r["floor"],
                    "x_px": float(r["width_normalized_drawing_x"]) * dw,
                    "y_px": dh
                    - float(r["width_normalized_drawing_y"]) * dw,
                }
        return gt
    gt = {}
    with open(gt_dir(session) / "trajectory_points.csv") as f:
        for r in csv.DictReader(f):
            if r.get("trajectory_id") and r["trajectory_id"] not in m["traj"]:
                continue
            fr = int(r["frame_number"])
            floor = next(name for a, b, name in m["floors"] if a <= fr <= b)
            gt[fr] = {
                "floor": floor,
                "x_px": float(r["width_normalized_drawing_x"]) * dw,
                # GTのyは下原点 (ユーザー確認 2026-09-08) → 画像座標へ反転
                "y_px": dh - float(r["width_normalized_drawing_y"]) * dw,
            }
    return gt


def gt_at(gt, frame, max_gap=4):
    """最近傍のGT点 (gapが大きければNone)"""
    best = min(gt, key=lambda f: abs(f - frame))
    if abs(best - frame) > max_gap:
        return None
    return gt[best]


def evaluate_track(name, samples, gt):
    """samples: [{frame, floor, x_px, y_px}] を評価"""
    floor_ok, errs, errs_all = 0, [], []
    n = 0
    mpx = mm_per_px()
    if mpx is None:
        raise ValueError(
            "mm/px が導出できないので誤差をメートルで出せない。"
            "図面の壁注釈 (labelme) と known_width_mm の両方が要る")
    for s in samples:
        g = gt_at(gt, s["frame"])
        if g is None:
            continue
        n += 1
        e = math.dist((s["x_px"], s["y_px"]), (g["x_px"], g["y_px"])) \
            * mpx / 1000
        errs_all.append(e)
        if s["floor"] == g["floor"]:
            floor_ok += 1
            errs.append(e)
    res = {
        "method": name,
        "n": n,
        "floor_acc": round(floor_ok / n, 3) if n else None,
        "err_median_m": round(float(np.median(errs)), 2) if errs else None,
        "err_p90_m": round(float(np.percentile(errs, 90)), 2) if errs else None,
        "err_median_all_m": round(float(np.median(errs_all)), 2) if errs_all else None,
    }
    print(res)
    return res


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gt = load_gt()
    print(f"GT points: {len(gt)}")
    results = []

    # 1) PF統合軌跡
    fused = json.loads((RESULTS_DIR / "fuse" / "fused.json").read_text())
    results.append(evaluate_track("temporal_fuse (PF)", fused["track"], gt))

    # 2) 単フレーム照合 (Mask2Former / GTセグ)
    for tag, label in (("loc", "localize (Mask2Former)"),
                       ("loc_gt", "localize (GTセグ)")):
        summ = json.loads((RESULTS_DIR / tag / "summary.json").read_text())
        samples = [{"frame": int(k.replace("frame_", "")), **v}
                   for k, v in summ.items() if "x_px" in v]
        results.append(evaluate_track(label, samples, gt))

    (OUT_DIR / "summary.json").write_text(json.dumps(results, indent=1))

    # GT vs PF軌跡の重畳図 (階ごと)
    for floor in ("1F", "2F"):
        bg = np.asarray(Image.open(ZUMEN_DIR / f"{floor}.jpg").convert("L"))
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(bg, cmap="gray")
        g = [(f, v) for f, v in sorted(gt.items()) if v["floor"] == floor]
        ax.plot([v["x_px"] for _, v in g], [v["y_px"] for _, v in g],
                "-", color="lime", linewidth=2, label="GT")
        t = [s for s in fused["track"] if s["floor"] == floor]
        ax.plot([s["x_px"] for s in t], [s["y_px"] for s in t],
                "-o", color="red", markersize=3, linewidth=1, alpha=0.7,
                label="PF estimate")
        ax.legend()
        ax.set_title(f"{floor}: GT vs PF trajectory")
        ax.set_axis_off()
        fig.savefig(OUT_DIR / f"gt_vs_pf_{floor}.jpg", dpi=120,
                    bbox_inches="tight")
        plt.close(fig)
    print(f"done -> {OUT_DIR}")


if __name__ == "__main__":
    main()
