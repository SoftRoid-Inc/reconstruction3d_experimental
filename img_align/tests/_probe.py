"""config の凍結値を1行JSONで印字する (必ず子プロセスで実行すること)

`img_align.config` は import 時に SESSION/PROP/... を凍結し、約90モジュールが
`from img_align.config import X` で値を**束ねる**。`importlib.reload` しても
束ねられた側は戻らないので、セッションを変えるにはプロセスを分けるしかない。

  IMG_SESSION=0912 python -m img_align.tests._probe
"""
import json
import sys


def snapshot():
    import img_align.config as c
    kw = c.PROP.known_width_mm
    return {
        "session": c.SESSION,
        "root": str(c.PROP.root),
        "frames_dir": str(c.FRAMES_DIR),
        "results_dir": str(c.RESULTS_DIR),
        "seg_dir": str(c.SEG_DIR),
        "imu_dir": str(c.IMU_DIR),
        "bench_dir": str(c.BENCH_DIR),
        "out_dir": str(c.OUT_DIR),
        "zumen_dir": str(c.PROP.zumen_dir),
        "person_mask_dir": str(c.PROP.person_mask_dir),
        "colmap_model": str(c.PROP.colmap_model),
        "known_width_mm": kw,
        "mm_per_px": c.PROP.mm_per_px,
        "gt_session": c.PROP.gt_session,
        "drawing_wh": list(c.PROP.drawing_wh),
        "frames": [c.CFG["frames"].start, c.CFG["frames"].stop,
                   c.CFG["frames"].step],
        "ascend": list(c.STAIR_ASCEND),
        "descend": list(c.STAIR_DESCEND),
        "registry": sorted(c.PROPERTY),
    }


def main():
    try:
        out = snapshot()
    except Exception as e:                       # noqa: BLE001
        out = {"error": f"{type(e).__name__}: {e}"}
    json.dump(out, sys.stdout, ensure_ascii=False, sort_keys=True)
    print()


if __name__ == "__main__":
    main()
