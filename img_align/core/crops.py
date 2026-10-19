"""VidMap入力準備: equirectフレーム → 前方ピンホール列の合成

VidMapのフロントエンド(GeoCalib/DA3/RoMav2)は透視カメラ前提でequirectは不可。
COLMAP公式のpanorama_sfm推奨と同じく、equirectから透視ビューを描画して入力する。
フレーム番号との対応表 (name_map.json) を保存し、COLMAPモデル読み取り時に
frame番号へ引き戻せるようにする。

使い方: IMG_SESSION=0312 conda run -n img_align python -m img_align.core.crops \
          [--fov 120] [--yaw 0] [--size 1100]
出力: results/vidmap/<sess>/frames/ (連番jpg) + name_map.json
"""
import argparse
import json

import numpy as np
from PIL import Image
from scipy import ndimage

from img_align.config import FRAMES_DIR, GROUP_IDS, PROP, RESULTS_DIR, SEG_DIR


def load_person_mask(frame_name, dilate=6):
    """personのequirect boolマスク (セグ解像度 1024x2048 のまま)。無ければNone

    frame_name は 'frame_000123' 形式。gs_prep とも共用。
    seg npzが無い物件 (28-BBBBB等) は既存の人物マスクjpg (白=有効/黒=人物) を使う。
    """
    fr = int(frame_name.replace("frame_", ""))
    seg_p = SEG_DIR / f"frame_{fr:06d}_labels.npz"
    if seg_p.exists():
        person = np.load(seg_p)["labels"] == GROUP_IDS["person"]
    else:
        from img_align.config import frame_name as _fname
        jpg = (PROP.person_mask_dir / _fname(fr)
               if PROP.person_mask_dir else None)
        if jpg is None or not jpg.exists():
            return None
        m = Image.open(jpg).convert("L").resize((2048, 1024), Image.BILINEAR)
        person = np.asarray(m) < 128
    if dilate:
        person = ndimage.binary_dilation(person, iterations=dilate)
    return person


def generate(fov=130.0, yaw=0.0, pitch=10.0, size=518, tag="mapany_f130"):
    """equirect全フレーム→ピンホールクロップ列 (既存はスキップ=再開可能)

    透視変換はこの関数専用 (load_person_maskだけ使う環境に依存を強いない)。
    FOV130/pitch10はVidMap/MapAnything検証で実証済みの設定。"""
    import py360convert

    out = RESULTS_DIR / "vidmap" / tag
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
        persp = py360convert.e2p(e, fov_deg=(fov, fov),
                                 u_deg=yaw, v_deg=pitch,
                                 out_hw=(size, size))
        # 残留する撮影者を黒塗り (vidmapはマスク入力非対応のため画素で消す。
        # 境界の静止特徴はPANORAMIC縮退分類で幾何検証から落ちる)
        person = load_person_mask(f"frame_{fr:06d}")
        if person is not None:
            pm = py360convert.e2p(person[..., None].astype(np.float32),
                                  fov_deg=(fov, fov),
                                  u_deg=yaw, v_deg=pitch,
                                  out_hw=(size, size))
            persp = persp.copy()
            persp[pm[..., 0] > 0.3] = 0
        Image.fromarray(persp).save(dst, quality=92)
        if i % 50 == 0:
            print(f"{i}/{len(frames)}")
    (out.parent / f"name_map_{tag}.json").write_text(
        json.dumps(name_map, indent=1))
    print(f"-> {out} ({len(frames)} frames, fov={fov}, yaw={yaw})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fov", type=float, default=130.0)
    ap.add_argument("--yaw", type=float, default=0.0,
                    help="equirect内の視線方位 (deg)")
    ap.add_argument("--pitch", type=float, default=10.0, help="仰角 (deg)")
    ap.add_argument("--size", type=int, default=518, help="出力解像度(正方)")
    ap.add_argument("--tag", default="mapany_f130")
    args = ap.parse_args()
    generate(args.fov, args.yaw, args.pitch, args.size, args.tag)


if __name__ == "__main__":
    main()
