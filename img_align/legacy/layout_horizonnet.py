"""手法C: HorizonNet によるルームレイアウト推定

HorizonNet(third_party/ にclone、無改変)を subprocess で呼び出す。
前処理(消失点による水平化)→ 推論(壁-床-天井境界・コーナー)。
入力は 1024x512 にリサイズした等距円筒フレーム。
"""
import subprocess
import sys

from PIL import Image

from img_align.config import HORIZONNET_DIR, HORIZONNET_REPO, HORIZONNET_WEIGHTS, RESULTS_DIR
from img_align.legacy.sample_frames import load_selected

LAYOUT_DIR = RESULTS_DIR / "layout"
INPUT_DIR = LAYOUT_DIR / "input"
ALIGNED_DIR = LAYOUT_DIR / "aligned"
OUT_DIR = LAYOUT_DIR / "out"


def run(cmd, cwd=None):
    print("+", " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], cwd=cwd, check=True)


def ensure_repo():
    if not HORIZONNET_DIR.exists():
        run(["git", "clone", HORIZONNET_REPO, HORIZONNET_DIR])


def ensure_weights():
    """HuggingFace Hub (sunset1995/HorizonNet) から st3d 学習済み重みを取得"""
    if HORIZONNET_WEIGHTS.exists():
        return
    from huggingface_hub import hf_hub_download

    HORIZONNET_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    hf_hub_download(
        repo_id="sunset1995/HorizonNet",
        filename=HORIZONNET_WEIGHTS.name,
        local_dir=HORIZONNET_WEIGHTS.parent,
    )


def prepare_inputs():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    for entry in load_selected():
        name = f"frame_{entry['index']:06d}"
        dst = INPUT_DIR / f"{name}.png"
        if not dst.exists():
            Image.open(entry["path"]).convert("RGB").resize(
                (1024, 512), Image.LANCZOS
            ).save(dst)


def main():
    ensure_repo()
    ensure_weights()
    prepare_inputs()

    ALIGNED_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    run(
        [sys.executable, "preprocess.py",
         "--img_glob", str(INPUT_DIR / "*.png"),
         "--output_dir", ALIGNED_DIR],
        cwd=HORIZONNET_DIR,
    )
    run(
        [sys.executable, "inference.py",
         "--pth", HORIZONNET_WEIGHTS,
         "--img_glob", str(ALIGNED_DIR / "*aligned_rgb.png"),
         "--output_dir", OUT_DIR,
         "--visualize"],
        cwd=HORIZONNET_DIR,
    )
    print(f"done -> {OUT_DIR}")


if __name__ == "__main__":
    main()
