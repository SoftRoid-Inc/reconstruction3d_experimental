"""代表フレームの選定: 全フレームから等間隔に抽出し frames.json に保存する"""
import json

import numpy as np

from img_align.config import FRAMES_DIR, NUM_SAMPLE_FRAMES, RESULTS_DIR


def sample_frames():
    frames = sorted(FRAMES_DIR.glob("frame_number_*.jpg"))
    if not frames:
        raise FileNotFoundError(f"no frames in {FRAMES_DIR}")
    idx = np.unique(
        np.linspace(0, len(frames) - 1, NUM_SAMPLE_FRAMES).round().astype(int)
    )
    selected = [{"index": int(i), "path": str(frames[i])} for i in idx]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "frames.json"
    out.write_text(json.dumps(
        {"total_frames": len(frames), "selected": selected},
        indent=2,
    ))
    return selected


def load_selected():
    data = json.loads((RESULTS_DIR / "frames.json").read_text())
    return data["selected"]


if __name__ == "__main__":
    sel = sample_frames()
    print(f"selected {len(sel)} frames -> {RESULTS_DIR / 'frames.json'}")
    for s in sel:
        print(f"  [{s['index']:4d}] {s['path']}")
