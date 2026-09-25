"""VidMap入力準備: equirectフレーム → 前方ピンホール列の合成

VidMapのフロントエンド(GeoCalib/DA3/RoMav2)は透視カメラ前提でequirectは不可。
COLMAP公式のpanorama_sfm推奨と同じく、equirectから透視ビューを描画して入力する。
フレーム番号との対応表 (name_map.json) を保存し、COLMAPモデル読み取り時に
frame番号へ引き戻せるようにする。

使い方: IMG_SESSION=0312 conda run -n img_align python -m img_align.vidmap_prep \
          [--fov 120] [--yaw 0] [--size 1100]
出力: results/vidmap/<sess>/frames/ (連番jpg) + name_map.json
"""
import argparse
import json

import numpy as np
import py360convert
from PIL import Image
from scipy import ndimage

from .config import FRAMES_DIR, GROUP_IDS, RESULTS_DIR, SEG_DIR, SESSION


def load_person_mask(frame_name, dilate=6):
    """personのequirect boolマスク (セグ解像度 1024x2048 のまま)。無ければNone

    frame_name は 'frame_000123' 形式。gs_prep とも共用。
    """
    fr = int(frame_name.replace("frame_", ""))
    seg_p = SEG_DIR / f"frame_{fr:06d}_labels.npz"
    if not seg_p.exists():
        return None
    lab = np.load(seg_p)["labels"]
    person = lab == GROUP_IDS["person"]
    if dilate:
        person = ndimage.binary_dilation(person, iterations=dilate)
    return person


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fov", type=float, default=100.0)
    ap.add_argument("--yaw", type=float, default=0.0,
                    help="equirect内の視線方位 (deg)")
    ap.add_argument("--pitch", type=float, default=15.0,
                    help="仰角 (deg)。撮影者は仰角-40°以下に集中するため"
                         "上向きで排除する (0312実測)")
    ap.add_argument("--size", type=int, default=1100, help="出力解像度(正方)")
    ap.add_argument("--tag", default="frames")
    args = ap.parse_args()

    out = RESULTS_DIR / "vidmap" / SESSION / args.tag
    out.mkdir(parents=True, exist_ok=True)
    frames = sorted(FRAMES_DIR.glob("frame_number_*.jpg"))
    name_map = {}
    for i, p in enumerate(frames):
        fr = int(p.stem.replace("frame_number_", ""))
        dst = out / f"{i:06d}.jpg"
        name_map[dst.name] = fr
        if dst.exists():
            continue
        e = np.asarray(Image.open(p))
        persp = py360convert.e2p(e, fov_deg=(args.fov, args.fov),
                                 u_deg=args.yaw, v_deg=args.pitch,
                                 out_hw=(args.size, args.size))
        # 残留する撮影者を黒塗り (vidmapはマスク入力非対応のため画素で消す。
        # 境界の静止特徴はPANORAMIC縮退分類で幾何検証から落ちる)
        person = load_person_mask(f"frame_{fr:06d}")
        if person is not None:
            pm = py360convert.e2p(person[..., None].astype(np.float32),
                                  fov_deg=(args.fov, args.fov),
                                  u_deg=args.yaw, v_deg=args.pitch,
                                  out_hw=(args.size, args.size))
            persp = persp.copy()
            persp[pm[..., 0] > 0.3] = 0
        Image.fromarray(persp).save(dst, quality=92)
        if i % 50 == 0:
            print(f"{i}/{len(frames)}")
    (out.parent / f"name_map_{args.tag}.json").write_text(
        json.dumps(name_map, indent=1))
    print(f"-> {out} ({len(frames)} frames, fov={args.fov}, yaw={args.yaw})")


if __name__ == "__main__":
    main()
