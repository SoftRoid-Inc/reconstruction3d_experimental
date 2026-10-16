"""外部テストデータセット (polish-pipeline sfm_testdataset) の評価レポート

図面の壁アノテーションが無いので**図面アンカーは評価できない**。GT軌跡だけで
測れるもの — 島の形状品質・橋渡しの接続精度・カバレッジ — をまとめる。

誤差は**図面幅で正規化した単位** (このデータセット本来の単位)。壁注釈が無く
mm/px が出ないため。既存物件換算で 0.01 ≒ 0.30m。

使い方:
  python -m img_align.out.ext_report --data <集計JSON> --out <出力HTML>
"""
import argparse
import json

TPL_HEAD = """<style>
:root{color-scheme:light dark}
body{font-family:-apple-system,"Hiragino Sans","Noto Sans JP",sans-serif;
     max-width:1100px;margin:0 auto;padding:24px 18px;line-height:1.7}
h1{font-size:1.6rem;border-bottom:2px solid currentColor;padding-bottom:6px}
h2{font-size:1.25rem;margin-top:2.2em}
h3{font-size:1.05rem;margin-top:1.6em;opacity:.9}
table{border-collapse:collapse;width:100%;margin:1em 0;font-size:.92rem}
th,td{border:1px solid color-mix(in srgb,currentColor 25%,transparent);
      padding:5px 9px;text-align:right}
th{text-align:center;background:color-mix(in srgb,currentColor 8%,transparent)}
td:first-child,th:first-child{text-align:left}
.ok{color:#1a7f37;font-weight:600}.ng{color:#cf222e;font-weight:600}
.warn{color:#9a6700;font-weight:600}
.note{background:color-mix(in srgb,currentColor 7%,transparent);
      border-left:4px solid currentColor;padding:10px 14px;margin:1.2em 0;
      border-radius:0 6px 6px 0}
.figs{display:flex;flex-wrap:wrap;gap:14px;margin:1em 0}
figure{margin:0;flex:1 1 320px}
figcaption{font-size:.82rem;opacity:.75;text-align:center;margin-top:4px}
svg{width:100%;height:auto;background:color-mix(in srgb,currentColor 5%,transparent);
    border-radius:6px}
.kpi{display:flex;flex-wrap:wrap;gap:12px;margin:1.2em 0}
.kpi div{flex:1 1 150px;padding:10px 14px;border-radius:8px;
         background:color-mix(in srgb,currentColor 8%,transparent)}
.kpi b{display:block;font-size:1.5rem}
code{background:color-mix(in srgb,currentColor 10%,transparent);
     padding:1px 5px;border-radius:4px;font-size:.9em}
</style>"""

COL = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#17becf"]


def _poly(pts, color, w=1.6, op=1.0):
    d = " ".join(f"{x * 1000:.1f},{y * 700:.1f}" for x, y in pts)
    return (f'<polyline points="{d}" fill="none" stroke="{color}" '
            f'stroke-width="{w}" stroke-opacity="{op}" '
            f'stroke-linejoin="round"/>')


def floor_svg(islands, floor):
    """その階の GT (灰) と 島の軌跡 (色) を重ねた SVG"""
    sel = [i for i in islands if i["floor"] == floor and i["traj"]]
    if not sel:
        return ""
    xs = [p[0] for i in sel for p in i["traj"] + i["gt"]]
    ys = [p[1] for i in sel for p in i["traj"] + i["gt"]]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    pad = 0.06 * max(x1 - x0, y1 - y0, 1e-6)
    sx = 1.0 / (x1 - x0 + 2 * pad)
    sy = 1.0 / (y1 - y0 + 2 * pad)
    k = min(sx, sy)

    def tf(p):
        return ((p[0] - x0 + pad) * k, (p[1] - y0 + pad) * k)

    body = []
    for i in sel:
        body.append(_poly([tf(p) for p in i["gt"]], "currentColor", 5, .22))
    for n, i in enumerate(sel):
        body.append(_poly([tf(p) for p in i["traj"]], COL[n % len(COL)], 1.8))
    leg = "".join(
        f'<text x="12" y="{20 + 16 * n}" font-size="13" '
        f'fill="{COL[n % len(COL)]}">{i["tag"].split("_")[-1]} '
        f'({i["resid"]:.4f})</text>'
        for n, i in enumerate(sel) if i["resid"])
    return (f'<svg viewBox="0 0 1000 700">{"".join(body)}'
            f'<text x="12" y="690" font-size="13" fill="currentColor" '
            f'opacity=".6">灰=GT / 色=島 (括弧内は残差)</text>{leg}</svg>')


def junction_svg(j):
    """橋渡し: 島Aの変換で運んだ島B (橙) と GT の島B (灰)"""
    if not j.get("b_traj"):
        return ""
    pts = j["b_traj"] + j["b_gt"]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    pad = 0.08 * max(x1 - x0, y1 - y0, 1e-6)
    k = min(1.0 / (x1 - x0 + 2 * pad), 1.0 / (y1 - y0 + 2 * pad))

    def tf(p):
        return ((p[0] - x0 + pad) * k, (p[1] - y0 + pad) * k)
    return (f'<svg viewBox="0 0 1000 700">'
            f'{_poly([tf(p) for p in j["b_gt"]], "currentColor", 6, .25)}'
            f'{_poly([tf(p) for p in j["b_traj"]], "#ff7f0e", 2.0)}'
            f'<text x="12" y="690" font-size="13" fill="currentColor" '
            f'opacity=".6">灰=GTの島B / 橙=橋で運んだ島B</text></svg>')


def render(data):
    h = [TPL_HEAD, "<h1>外部テストデータセットでの検証 (2026-10-16)</h1>"]
    isl = [i for v in data.values() for i in v["islands"]]
    jun = [j for v in data.values() for j in v["junctions"]]
    good = [i["resid"] for i in isl if i["resid"] is not None]
    carry = [j["carry"] for j in jun if j.get("carry") is not None]
    import statistics as st
    h.append(f"""<div class="kpi">
<div><b>{len(data)}</b>データセット (初見の物件)</div>
<div><b>{len(isl)}</b>島</div>
<div><b>{st.median(good):.4f}</b>島の形状残差 中央値</div>
<div><b>{len(jun)}</b>切れ目 (橋渡しの対象)</div>
<div><b>{st.median(carry):.4f}</b>橋の運搬誤差 中央値</div>
</div>""")
    h.append("""<div class="note"><b>単位について</b> — 誤差はすべて
<b>図面幅で正規化した値</b>。このデータセットには図面の壁アノテーション
(labelme) が無く mm/px が導出できないため、データセット本来の単位で示す。
既存物件 (28-AAAAA) 換算で <code>0.01 ≒ 0.30m</code>。
<b>図面アンカーはこのデータでは評価できない</b> (壁注釈が要るため)。
評価したのは GT軌跡だけで測れる項目。</div>""")

    h.append("<h2>1. 島の形状品質 — 初見の物件でも既存と同等</h2>")
    h.append("<table><tr><th>データセット</th><th>島</th><th>階</th>"
             "<th>フレーム</th><th>GT点</th><th>形状残差</th><th>≈m</th></tr>")
    for s, v in data.items():
        for i in v["islands"]:
            r = i["resid"]
            cls = "ok" if r and r < 0.01 else ("warn" if r and r < 0.02
                                               else "ng")
            h.append(f"<tr><td>{s}</td><td>{i['tag'].split('_')[-1]}</td>"
                     f"<td>{i['floor']}</td><td>{i['n_frames']}</td>"
                     f"<td>{i['n_gt']}</td>"
                     f"<td class='{cls}'>{r:.4f}</td>"
                     f"<td>{r * 29.68:.2f}</td></tr>" if r else
                     f"<tr><td>{s}</td><td>{i['tag']}</td><td>{i['floor']}</td>"
                     f"<td>{i['n_frames']}</td><td>{i['n_gt']}</td>"
                     f"<td colspan=2>GT不足</td></tr>")
    h.append("</table>")
    h.append(f"""<div class="note">中央値 <b>{st.median(good):.4f}</b>
(≈{st.median(good) * 29.68:.2f}m) は既存5セッションの 0.12〜0.50m と同水準。
<b>PanoVGGT のチャンク化・連結は初見の物件でも調整なしで動く</b>。
外れは {sum(1 for r in good if r > 0.015)}/{len(good)} 島で、既存の破綻率
(0428 の 1.33m など) と同程度。</div>""")

    h.append("<h2>2. 軌跡の比較 (階ごと)</h2>")
    for s, v in data.items():
        h.append(f"<h3>{s}</h3><div class='figs'>")
        for fl in sorted({i["floor"] for i in v["islands"] if i["floor"]}):
            svg = floor_svg(v["islands"], fl)
            if svg:
                h.append(f"<figure>{svg}<figcaption>{s} {fl}</figcaption>"
                         f"</figure>")
        h.append("</div>")

    h.append("<h2>3. カバレッジ</h2>")
    h.append("<table><tr><th>データセット</th><th>総フレーム</th>"
             "<th>GT点</th><th>島がGTを覆う率</th><th>島がフレームを覆う率</th>"
             "<th>階区間</th></tr>")
    for s, v in data.items():
        h.append(f"<tr><td>{s}</td><td>{v['n_frames']}</td>"
                 f"<td>{v['n_gt']}</td>"
                 f"<td>{v['coverage']:.1%}</td>"
                 f"<td>{v['frame_coverage']:.1%}</td>"
                 f"<td>{len(v['floors'])}</td></tr>")
    h.append("</table>")
    h.append("""<div class="note">島に入らないフレームは、階の変わり目
(階段) と <code>MIN_SEG_FRAMES=24</code> 未満の短い区間。e11196 は2F区間が
2つ (20f / 19f) 落ちている。<b>フレームの17〜18%が島に入らない</b>のは
既存セッションと同じ傾向。</div>""")

    h.append("<h2>4. 橋渡しによる島間接続 — 採否基準がホールドアウトで破綻</h2>")
    h.append("<table><tr><th>データセット</th><th>切れ目</th><th>階</th>"
             "<th>隙間</th><th>運搬誤差</th><th>≈m</th><th>方式</th></tr>")
    for s, v in data.items():
        for j in v["junctions"]:
            c = j.get("carry")
            cls = "ok" if c and c < 0.02 else ("warn" if c and c < 0.06
                                               else "ng")
            h.append(f"<tr><td>{s}</td>"
                     f"<td>{j['a_end']}→{j['b_start']}</td>"
                     f"<td>{j['floors']}{' ⚠' if j['cross'] else ''}</td>"
                     f"<td>{j['gap']}f</td>"
                     + (f"<td class='{cls}'>{c:.4f}</td><td>{c * 29.68:.2f}</td>"
                        f"<td>{j.get('mode', '')}</td>" if c
                        else "<td colspan=3>評価不能</td>")
                     + "</tr>")
    h.append("</table>")
    same = [j["carry"] for j in jun if not j["cross"] and j.get("carry")]
    cross = [j["carry"] for j in jun if j["cross"] and j.get("carry")]
    h.append(f"""<div class="note"><b>ここが今回いちばん重要な結果。</b>
運搬誤差 (島Aを正しく置いたとき、橋で運んだ島Bがどれだけずれるか) の中央値は
<b>{st.median(carry):.4f}</b> (≈{st.median(carry) * 29.68:.2f}m) で、
島単独の形状残差 {st.median(good):.4f} の<b>{st.median(carry) / st.median(good):.0f}倍</b>。
既存2物件では 0.5m以内が4/11あったが、ここでは<b>1つも無い</b>。<br><br>
さらに<b>採否基準 (方式間の一致度) が機能しない</b>。一致度は島ゲージ単位で
計算しており、尺度が物件で大きく違う (28-BBBBB が 0.45〜0.49 m/unit に対し
このデータは <b>1.23〜1.60</b>)。<code>CONSENSUS_OK_M=2.5</code> は名前に反して
メートルではなく、実質 1.2m〜<b>4.0m</b> に化けていた。メートルに正規化して
測り直しても 3.7〜45.0m で<b>全切れ目が棄却</b>される。<br><br>
同一階 {len(same)}件 中央 {st.median(same):.4f} / 階またぎ {len(cross)}件 中央
{st.median(cross):.4f} — <b>階またぎが難しいという傾向はここでも出ている</b>。</div>""")

    for s, v in data.items():
        js = [j for j in v["junctions"] if j.get("b_traj")]
        if not js:
            continue
        h.append(f"<h3>{s} の切れ目</h3><div class='figs'>")
        for j in js:
            h.append(f"<figure>{junction_svg(j)}<figcaption>"
                     f"{j['a_end']}→{j['b_start']} {j['floors']} "
                     f"誤差 {j['carry']:.4f}</figcaption></figure>")
        h.append("</div>")

    h.append("""<h2>5. 結論</h2>
<ul>
<li><b>チェーン生成は初見の物件でも成立する</b> — 島の形状残差の中央値は
既存と同水準で、前処理からチャンク化・連結まで調整なしで通った</li>
<li><b>橋渡しの採否基準はホールドアウトで破綻した</b> — 一致度が島ゲージ単位
だったため物件をまたぐと意味が変わる。メートル正規化は実装したが、それでも
全切れ目が棄却される</li>
<li><b>しきい値は 21〜22島で決めたもの</b>で、やはり過学習だった。
壁の高さ帯 (0.7m)・外れ値判定 (1.5倍)・最小島長 (36f) も同じ 21島由来なので、
同様の検証が要る</li>
<li><b>図面アンカーの検証には壁アノテーションが要る</b> — 368フロアの図面画像は
あるが labelme 注釈が無い。ここが最大のボトルネック</li>
</ul>""")
    return "\n".join(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    html = render(json.loads(open(a.data).read()))
    open(a.out, "w").write(html)
    print(f"-> {a.out} ({len(html):,} bytes)")


if __name__ == "__main__":
    main()
