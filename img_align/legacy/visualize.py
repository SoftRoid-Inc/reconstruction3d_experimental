"""手法A(セグメンテーション)と手法C(レイアウト)の結果を1枚に合成して確認用画像を作る"""
import json

from PIL import Image, ImageDraw

from img_align.config import EQUI_W, RESULTS_DIR
from img_align.legacy.sample_frames import load_selected

SEG_DIR = RESULTS_DIR / "seg"
OUT_DIR = RESULTS_DIR / "layout" / "out"
COMBINED_DIR = RESULTS_DIR / "combined"


def layout_panel(name):
    """HorizonNetの可視化画像(境界線+コーナー描画)を読み込む。無ければNone"""
    raw = OUT_DIR / f"{name}_aligned_rgb.raw.png"
    if not raw.exists():
        return None
    panel = Image.open(raw).convert("RGB")

    js = OUT_DIR / f"{name}_aligned_rgb.json"
    if js.exists():
        data = json.loads(js.read_text())
        pts = data.get("uv") or data.get("cor_id") or []
        draw = ImageDraw.Draw(panel)
        for u, v in pts:
            if u <= 1.0 and v <= 1.0:  # 正規化座標ならピクセルへ
                u, v = u * panel.width, v * panel.height
            r = 5
            draw.ellipse([u - r, v - r, u + r, v + r], outline=(255, 0, 0), width=2)
    return panel


def main():
    COMBINED_DIR.mkdir(parents=True, exist_ok=True)
    combined_paths = []
    for entry in load_selected():
        name = f"frame_{entry['index']:06d}"
        seg = SEG_DIR / f"{name}_overlay.jpg"
        if not seg.exists():
            continue
        seg_img = Image.open(seg)
        panels = [seg_img]
        layout = layout_panel(name)
        if layout is not None:
            panels.append(layout.resize((EQUI_W, EQUI_W // 2), Image.BILINEAR))

        h = sum(p.height for p in panels)
        canvas = Image.new("RGB", (EQUI_W, h))
        y = 0
        for p in panels:
            canvas.paste(p, (0, y))
            y += p.height
        out = COMBINED_DIR / f"{name}.jpg"
        canvas.save(out, quality=88)
        combined_paths.append(out)

    # コンタクトシート (4列)
    if combined_paths:
        thumbs = []
        for p in combined_paths:
            im = Image.open(p)
            thumbs.append(im.resize((512, 512 * im.height // im.width)))
        cols = 4
        rows = (len(thumbs) + cols - 1) // cols
        th = max(t.height for t in thumbs)
        sheet = Image.new("RGB", (cols * 512, rows * th), (30, 30, 30))
        for i, t in enumerate(thumbs):
            sheet.paste(t, ((i % cols) * 512, (i // cols) * th))
        sheet.save(COMBINED_DIR / "contact_sheet.jpg", quality=85)
    print(f"done -> {COMBINED_DIR}")


if __name__ == "__main__":
    main()
