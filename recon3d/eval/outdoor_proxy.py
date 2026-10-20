"""島が屋外かを簡易判定する (評価から除くための暫定版)

屋外・半屋外の島は**図面に写っていない場所**を撮っているので、図面アンカーが
当たらないのは当然であり、失敗として数えると評価が歪む。
2026-10-14に0925の失敗島を視察したところ、13.87mで最悪だった s002 は
道路と駐車場、4.66mの s03 は屋外テラスだった。

本来の屋外・敷地内外判定は別棟 (tools/boundary_mask / seg_ade20k) で検討済み。
ここはあくまで**評価時に屋外島を外すための代用**で、本番の判定には使わない。

判定: equirect の上端帯 (天頂側) が「明るく青い」フレームの割合。
屋外では上半球が空で埋まり、室内では天井 (白〜木目=青くない) になる。

使い方:
  IMG_SESSION=0925 python -m recon3d.eval.outdoor_proxy
出力: backend/outdoor_proxy.json (島ごとの屋外率)
"""
import argparse
import json

import numpy as np
from PIL import Image


TOP_FRAC = 0.22      # 上端から見る帯 (天頂側)
BRIGHT = 120         # 空とみなす明るさ
BLUE = 12            # B - R がこれ以上なら青い
SKY_FRAC = 0.35      # 帯のうちこの割合が空ならそのフレームは屋外
OUT_FRAC = 0.5       # 島のフレームのうちこの割合が屋外なら屋外島
SAMPLE = 6           # 何フレームに1枚見るか


def sky_ratio(paths, frame):
    """1フレームの上端帯における空らしさ [0,1]"""
    p = paths.images / paths.frame_name(frame)
    if not p.exists():
        return None
    im = Image.open(p).convert("RGB").resize((256, 128))
    a = np.asarray(im).astype(np.int16)
    band = a[:int(128 * TOP_FRAC)]
    sky = (band.mean(axis=2) > BRIGHT) & (band[:, :, 2] - band[:, :, 0] > BLUE)
    return float(sky.mean())


def island_outdoor(paths, a, b, sample=SAMPLE):
    """島の屋外率と判定"""
    r = [v for f in range(a, b + 1, sample) if (v := sky_ratio(paths, f)) is not None]
    if not r:
        return None
    frac = float(np.mean([v > SKY_FRAC for v in r]))
    return {"outdoor_frac": round(frac, 3),
            "sky_median": round(float(np.median(r)), 3),
            "outdoor": frac >= OUT_FRAC, "n": len(r)}


def main(paths, argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="pv_manifest.json")
    args = ap.parse_args(argv)
    man = json.loads((paths.bench / args.manifest).read_text())
    rows = []
    print(f"{'island':20} {'屋外率':>7} {'空(中央)':>9}  判定")
    for m in man:
        r = island_outdoor(paths, m["a"], m["b"])
        if r is None:
            continue
        rows.append({"tag": m["tag"], **r})
        print(f"{m['tag']:20} {r['outdoor_frac']:7.2f} {r['sky_median']:9.3f}  "
              f"{'屋外' if r['outdoor'] else '室内'}")
    (paths.backend / "outdoor_proxy.json").write_text(
        json.dumps(rows, indent=1, ensure_ascii=False))
    print(f"-> {paths.backend / 'outdoor_proxy.json'}")

