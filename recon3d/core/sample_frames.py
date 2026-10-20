"""代表フレームの選定: 全フレームから等間隔に抽出し frames.json に保存する"""
import json

import numpy as np

from recon3d.constants import NUM_SAMPLE_FRAMES


def sample_frames(paths):
    frames = sorted(paths.images.glob("frame_number_*.jpg"))
    if not frames:
        raise FileNotFoundError(f"no frames in {paths.images}")
    idx = np.unique(
        np.linspace(0, len(frames) - 1, NUM_SAMPLE_FRAMES).round().astype(int)
    )
    selected = [{"index": int(i), "path": str(frames[i])} for i in idx]

    paths.results.mkdir(parents=True, exist_ok=True)
    out = paths.results / "frames.json"
    out.write_text(json.dumps(
        {"total_frames": len(frames), "selected": selected},
        indent=2,
    ))
    return selected


def load_selected(paths):
    data = json.loads((paths.results / "frames.json").read_text())
    return data["selected"]
