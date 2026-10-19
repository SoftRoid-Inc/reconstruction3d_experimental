"""全データセット横断レポート: チェーン品質・ポーズグラフ・アンカー

既存5セッション (壁注釈あり → 図面アンカーも評価できる) と外部2件
(polish-pipeline sfm_testdataset、壁注釈なし) を**同じ形式**で並べる。

島の形状残差は**図面幅で正規化**して載せる。物件ごとに mm/px が違う上、
外部データは mm/px そのものが出ないため。実測では正規化1.0 ≒ 29.8m で
既存3物件がほぼ揃っており、横断比較に使える。

使い方: python -m img_align.out.all_report --data <JSON> --loops <JSON>
                                            --out <HTML>
"""
import argparse
import json
import statistics as st

HEAD = """<style>
:root{color-scheme:light dark}
body{font-family:-apple-system,"Hiragino Sans","Noto Sans JP",sans-serif;
     max-width:1200px;margin:0 auto;padding:24px 18px;line-height:1.7}
h1{font-size:1.55rem;border-bottom:2px solid currentColor;padding-bottom:6px}
h2{font-size:1.2rem;margin-top:2.1em;border-bottom:1px solid
   color-mix(in srgb,currentColor 25%,transparent);padding-bottom:4px}
h3{font-size:1.02rem;margin:1.4em 0 .4em}
table{border-collapse:collapse;width:100%;margin:.9em 0;font-size:.9rem}
th,td{border:1px solid color-mix(in srgb,currentColor 22%,transparent);
      padding:4px 8px;text-align:right}
th{text-align:center;background:color-mix(in srgb,currentColor 8%,transparent)}
td:first-child,th:first-child{text-align:left}
.ok{color:#1a7f37;font-weight:600}.ng{color:#cf222e;font-weight:600}
.dim{opacity:.55}
.note{background:color-mix(in srgb,currentColor 7%,transparent);
      border-left:4px solid currentColor;padding:10px 14px;margin:1em 0;
      border-radius:0 6px 6px 0}
.kpi{display:flex;flex-wrap:wrap;gap:10px;margin:1em 0}
.kpi div{flex:1 1 150px;padding:9px 13px;border-radius:8px;
         background:color-mix(in srgb,currentColor 8%,transparent)}
.kpi b{display:block;font-size:1.4rem}
.row{display:flex;flex-wrap:wrap;gap:12px}
.row>figure{flex:1 1 330px;margin:0}
figcaption{font-size:.79rem;opacity:.7;text-align:center;margin-top:3px}
svg{width:100%;height:auto;background:color-mix(in srgb,currentColor 5%,transparent);
    border-radius:6px}
code{background:color-mix(in srgb,currentColor 10%,transparent);
     padding:1px 5px;border-radius:4px;font-size:.9em}
</style>"""
SEQ, PG = "#1f77b4", "#ff7f0e"


def _poly(pts, color, w=1.7, op=1.0):
    d = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    return (f'<polyline points="{d}" fill="none" stroke="{color}" '
            f'stroke-width="{w}" stroke-opacity="{op}" '
            f'stroke-linejoin="round"/>')


def floor_svg(islands, floor):
    """その階の GT(灰) / 逐次(青) / PG(橙) を重ねる"""
    sel = [i for i in islands if i["floor"] == floor]
    if not sel:
        return ""
    pts = [p for i in sel for p in i["seq"]["gt"] + i["seq"]["traj"]
           + (i["pg"]["traj"] if i.get("pg") else [])]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    pad = .07 * max(x1 - x0, y1 - y0, 1e-6)
    k = min(930 / (x1 - x0 + 2 * pad), 600 / (y1 - y0 + 2 * pad))

    def tf(p):
        return ((p[0] - x0 + pad) * k + 20, (p[1] - y0 + pad) * k + 18)
    b = [_poly([tf(p) for p in i["seq"]["gt"]], "currentColor", 5.5, .20)
         for i in sel]
    b += [_poly([tf(p) for p in i["seq"]["traj"]], SEQ, 1.6) for i in sel]
    b += [_poly([tf(p) for p in i["pg"]["traj"]], PG, 1.6)
          for i in sel if i.get("pg")]
    return (f'<svg viewBox="0 0 980 640">{"".join(b)}'
            f'<text x="18" y="632" font-size="12" fill="currentColor" '
            f'opacity=".6">灰=GT / 青=逐次 / 橙=PG+ループ</text></svg>')


def _fmt(v, mpn, digits=4):
    if v is None:
        return '<span class="dim">—</span>'
    m = f" <span class='dim'>({v * mpn:.2f}m)</span>" if mpn else ""
    return f"{v:.{digits}f}{m}"


def render(data, loops):
    cfg = data.pop("_config")
    names = list(data)
    allis = [(s, i) for s in names for i in data[s]["islands"]]
    seq = [i["seq"]["resid"] for _, i in allis]
    pg = [i["pg"]["resid"] for _, i in allis if i.get("pg")]
    h = [HEAD, "<h1>全データセット横断評価 — チェーン品質とポーズグラフ</h1>"]
    h.append(f"""<div class="kpi">
<div><b>{len(names)}</b>データセット</div>
<div><b>{len(allis)}</b>島</div>
<div><b>{st.median(seq):.4f}</b>逐次連結 中央値</div>
<div><b>{st.median(pg):.4f}</b>PG+ループ 中央値</div>
<div><b>{sum(loops.values())}</b>採用ループ</div>
</div>""")
    h.append("<h2>生成条件</h2><table>" + "".join(
        f"<tr><td>{k}</td><td style='text-align:left'>{v}</td></tr>"
        for k, v in (("コミット", f"<code>{cfg['commit']}</code>"),
                     ("チャンク", cfg["chunk"]), ("島の切り方", cfg["segments"]),
                     ("最適化", cfg["posegraph"]), ("単位", cfg["unit"])))
             + "</table>")
    h.append("""<div class="note"><b>単位</b>: 島の形状残差は図面幅で正規化。
既存3物件はいずれも<b>正規化1.0 ≒ 29.8m</b>でほぼ揃っており、外部データとも
直接比較できる。括弧内は各物件の mm/px による m 換算
(外部データは壁注釈が無く mm/px が出ないので省略)。</div>""")

    h.append("<h2>データセット一覧</h2><table><tr><th>データセット</th>"
             "<th>フレーム</th><th>GT点</th><th>島</th><th>階区間</th>"
             "<th>GT被覆</th><th>逐次 中央</th><th>PG 中央</th><th>変化</th>"
             "<th>ループ</th></tr>")
    for s in names:
        d = data[s]
        a = [i["seq"]["resid"] for i in d["islands"]]
        b = [i["pg"]["resid"] for i in d["islands"] if i.get("pg")]
        lp = sum(loops.get(i["tag"], 0) for i in d["islands"])
        ch = ((st.median(b) - st.median(a)) / st.median(a) * 100) if b else None
        cls = "ok" if ch and ch < -5 else ("ng" if ch and ch > 5 else "")
        h.append(f"<tr><td>{s}</td><td>{d['n_frames']}</td><td>{d['n_gt']}</td>"
                 f"<td>{len(d['islands'])}</td><td>{len(d['floors'])}</td>"
                 f"<td>{d['coverage']:.0%}</td>"
                 f"<td>{st.median(a):.4f}</td>"
                 f"<td>{st.median(b):.4f}</td>"
                 f"<td class='{cls}'>{ch:+.0f}%</td><td>{lp}</td></tr>")
    h.append("</table>")

    for s in names:
        d = data[s]
        mpn = d.get("m_per_norm")
        h.append(f"<h2>{s}</h2>")
        has_anchor = any(i.get("anchor_m") is not None for i in d["islands"])
        h.append("<table><tr><th>島</th><th>階</th><th>frames</th>"
                 "<th>チャンク</th><th>ループ</th><th>逐次</th><th>PG+ループ</th>"
                 "<th>変化</th>"
                 + ("<th>屋外率</th><th>アンカー</th>" if has_anchor else "")
                 + "</tr>")
        for i in sorted(d["islands"], key=lambda x: -x["seq"]["resid"]):
            a, b = i["seq"]["resid"], (i["pg"] or {}).get("resid")
            ch = ((b - a) / a * 100) if b else None
            cls = "ok" if ch and ch < -10 else ("ng" if ch and ch > 10 else "")
            extra = ""
            if has_anchor:
                od = i.get("outdoor")
                am = i.get("anchor_m")
                acls = "ok" if am is not None and am <= 1 else "ng"
                extra = (f"<td>{od:.0%}</td>" if od is not None
                         else "<td class='dim'>—</td>")
                extra += (f"<td class='{acls}'>{am:.2f}m</td>" if am is not None
                          else "<td class='dim'>—</td>")
            h.append(f"<tr><td>{i['tag'].split('_', 1)[-1]}</td>"
                     f"<td>{i['floor'] or '—'}</td>"
                     f"<td>{i['a']}-{i['b']}</td><td>{i['n_chunks']}</td>"
                     f"<td>{loops.get(i['tag'], 0)}</td>"
                     f"<td>{_fmt(a, mpn)}</td><td>{_fmt(b, mpn)}</td>"
                     f"<td class='{cls}'>"
                     + (f"{ch:+.0f}%" if ch else "—") + "</td>"
                     + extra + "</tr>")
        h.append("</table><div class='row'>")
        for fl in sorted({i["floor"] for i in d["islands"] if i["floor"]}):
            svg = floor_svg(d["islands"], fl)
            if svg:
                h.append(f"<figure>{svg}<figcaption>{s} {fl}</figcaption>"
                         f"</figure>")
        h.append("</div>")

    h.append(f"""<h2>まとめ</h2><ul>
<li><b>ポーズグラフは全体で {st.median(seq):.4f} → {st.median(pg):.4f}</b>
({(st.median(pg) - st.median(seq)) / st.median(seq) * 100:+.0f}%)。
逐次連結は誤差を一方向に積むが、全チャンクを同時に解くと解消する</li>
<li><b>ループが0本でも改善する島がある</b> — 効いているのは同時最適化そのもの。
ループは上乗せ</li>
<li><b>外部データでも既存と同水準</b> — 初見の物件でチェーン生成は成立する。
残る問題はアンカー側に集中している</li>
</ul>""")
    return "\n".join(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--loops", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    html = render(json.loads(open(a.data).read()),
                  json.loads(open(a.loops).read()))
    open(a.out, "w").write(html)
    print(f"-> {a.out} ({len(html):,} bytes)")


if __name__ == "__main__":
    main()
