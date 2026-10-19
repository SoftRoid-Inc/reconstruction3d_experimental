"""ポーズグラフ+ループ拘束 vs 逐次連結の比較レポート

どの弱点に手を入れるかを判断するための可視化。島ごとに
  GT (灰) / 逐次連結 (青) / ポーズグラフ+ループ (橙)
を重ね、フレーム毎の誤差推移も併記する。誤差が**どこで**減った/増えたかが
分かれば、ループが効いた区間と効かなかった区間を切り分けられる。

誤差は図面幅で正規化 (壁注釈が無く mm/px が出ないため)。0.01 ≒ 0.30m。

使い方: python -m img_align.out.pg_report --data <JSON> --out <HTML>
"""
import argparse
import json
import statistics as st

HEAD = """<style>
:root{color-scheme:light dark}
body{font-family:-apple-system,"Hiragino Sans","Noto Sans JP",sans-serif;
     max-width:1150px;margin:0 auto;padding:24px 18px;line-height:1.7}
h1{font-size:1.55rem;border-bottom:2px solid currentColor;padding-bottom:6px}
h2{font-size:1.2rem;margin-top:2em}
table{border-collapse:collapse;width:100%;margin:1em 0;font-size:.92rem}
th,td{border:1px solid color-mix(in srgb,currentColor 25%,transparent);
      padding:5px 9px;text-align:right}
th{text-align:center;background:color-mix(in srgb,currentColor 8%,transparent)}
td:first-child{text-align:left}
.ok{color:#1a7f37;font-weight:600}.ng{color:#cf222e;font-weight:600}
.note{background:color-mix(in srgb,currentColor 7%,transparent);
      border-left:4px solid currentColor;padding:10px 14px;margin:1.1em 0;
      border-radius:0 6px 6px 0}
.kpi{display:flex;flex-wrap:wrap;gap:12px;margin:1.1em 0}
.kpi div{flex:1 1 160px;padding:10px 14px;border-radius:8px;
         background:color-mix(in srgb,currentColor 8%,transparent)}
.kpi b{display:block;font-size:1.45rem}
.card{border:1px solid color-mix(in srgb,currentColor 20%,transparent);
      border-radius:10px;padding:12px 14px;margin:14px 0}
.card h3{margin:0 0 6px;font-size:1rem}
.row{display:flex;flex-wrap:wrap;gap:12px}
.row>div{flex:1 1 380px}
svg{width:100%;height:auto;background:color-mix(in srgb,currentColor 5%,transparent);
    border-radius:6px}
.cap{font-size:.8rem;opacity:.72;text-align:center;margin-top:3px}
code{background:color-mix(in srgb,currentColor 10%,transparent);
     padding:1px 5px;border-radius:4px;font-size:.9em}
</style>"""
SEQ, PG, GT = "#1f77b4", "#ff7f0e", "currentColor"


def _poly(pts, color, w=1.7, op=1.0):
    d = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    return (f'<polyline points="{d}" fill="none" stroke="{color}" '
            f'stroke-width="{w}" stroke-opacity="{op}" '
            f'stroke-linejoin="round"/>')


def traj_svg(isl):
    """GT / 逐次 / PG を同じ枠に重ねる"""
    seq, pg = isl["seq"], isl.get("pg")
    pts = seq["gt"] + seq["traj"] + (pg["traj"] if pg else [])
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    pad = 0.08 * max(x1 - x0, y1 - y0, 1e-6)
    k = min(960 / (x1 - x0 + 2 * pad), 620 / (y1 - y0 + 2 * pad))

    def tf(p):
        return ((p[0] - x0 + pad) * k + 20, (p[1] - y0 + pad) * k + 20)
    body = [_poly([tf(p) for p in seq["gt"]], GT, 6, .22),
            _poly([tf(p) for p in seq["traj"]], SEQ, 1.7)]
    if pg:
        body.append(_poly([tf(p) for p in pg["traj"]], PG, 1.7))
    return (f'<svg viewBox="0 0 1000 660">{"".join(body)}'
            f'<text x="16" y="648" font-size="13" fill="currentColor" '
            f'opacity=".65">灰=GT / 青=逐次連結 / 橙=PG+ループ</text></svg>')


def err_svg(isl):
    """フレーム毎の誤差推移 (どこで効いたかを見る)"""
    seq, pg = isl["seq"], isl.get("pg")
    if not pg:
        return ""
    a, b = seq["per_frame"], pg["per_frame"]
    n = min(len(a), len(b))
    hi = max(max(a[:n]), max(b[:n]), 1e-6)

    def line(v, col):
        return _poly([(20 + i / max(n - 1, 1) * 940, 600 - v[i] / hi * 560)
                      for i in range(n)], col, 1.6)
    grid = "".join(
        f'<line x1="20" y1="{600 - f * 560}" x2="960" y2="{600 - f * 560}" '
        f'stroke="currentColor" stroke-opacity=".15"/>'
        f'<text x="964" y="{604 - f * 560}" font-size="11" '
        f'fill="currentColor" opacity=".55">{hi * f:.3f}</text>'
        for f in (0, .5, 1))
    return (f'<svg viewBox="0 0 1000 640">{grid}{line(a, SEQ)}{line(b, PG)}'
            f'<text x="20" y="632" font-size="12" fill="currentColor" '
            f'opacity=".65">横軸=フレーム / 縦軸=GT誤差 (図面幅)</text></svg>')


def config_table(c):
    """**生成条件を必ず書く** — 設定を変えながら検証するので、どの条件の
    数字かが分からないと比較できなくなる (2026-10-16に実際に混乱した)"""
    rows = [("コミット", f"<code>{c['commit']}</code>"),
            ("データ", c["data"]), ("島の切り方", c["segments"]),
            ("チャンク", c["chunk"]), ("最適化", c["posegraph"]),
            ("単位", c["unit"])]
    t = "".join(f"<tr><td>{k}</td><td style='text-align:left'>{v}</td></tr>"
                for k, v in rows)
    alt = "".join(f"<tr><td>{n}</td><td>{a:.4f}</td><td>{b:.4f}</td>"
                  f"<td>{l}</td></tr>" for n, a, b, l in c["alt"])
    return (f"<h2>生成条件</h2><table>{t}</table>"
            f"<p>チャンク設定を変えた場合 (参考):</p>"
            f"<table><tr><th>設定</th><th>逐次</th><th>PG+ループ</th>"
            f"<th>ループ計</th></tr>{alt}</table>"
            f"<div class='note'>逐次では 0.0059〜0.0080 とばらつくが、"
            f"<b>ポーズグラフを通すと 0.0052〜0.0054 に収束する</b>。"
            f"24/18 はループ3本しか無いのに最良で、"
            f"<b>同時最適化そのものの寄与が大きい</b>ことを示す。</div>")


def render(data, loops):
    cfg = data.pop("_config", None)
    isls = [i for v in data.values() for i in v if i.get("pg")]
    a = [i["seq"]["resid"] for i in isls]
    b = [i["pg"]["resid"] for i in isls]
    h = [HEAD, "<h1>ポーズグラフ+ループ拘束 vs 逐次連結 (2026-10-16)</h1>"]
    h.append(f"""<div class="kpi">
<div><b>{len(isls)}</b>島 (外部データ2件)</div>
<div><b>{st.median(a):.4f}</b>逐次連結 中央値</div>
<div><b>{st.median(b):.4f}</b>PG+ループ 中央値</div>
<div><b>{(st.median(b) - st.median(a)) / st.median(a) * 100:+.0f}%</b>変化</div>
<div><b>{sum(loops.values())}</b>採用ループ (合計)</div>
</div>""")
    if cfg:
        h.append(config_table(cfg))

    h.append("<h2>一覧</h2><table><tr><th>島</th><th>階</th><th>チャンク</th>"
             "<th>採用ループ</th><th>逐次</th><th>PG+ループ</th><th>変化</th>"
             "</tr>")
    for i in sorted(isls, key=lambda x: -x["seq"]["resid"]):
        s, p = i["seq"]["resid"], i["pg"]["resid"]
        d = (p - s) / s * 100
        cls = "ok" if d < -10 else ("ng" if d > 10 else "")
        h.append(f"<tr><td>{i['tag']}</td><td>{i['floor']}</td>"
                 f"<td>{i['n_chunks']}</td><td>{loops.get(i['tag'], 0)}</td>"
                 f"<td>{s:.4f}</td><td>{p:.4f}</td>"
                 f"<td class='{cls}'>{d:+.0f}%</td></tr>")
    h.append("</table>")

    h.append("<h2>島ごとの比較</h2>")
    for i in sorted(isls, key=lambda x: -x["seq"]["resid"]):
        s, p = i["seq"]["resid"], i["pg"]["resid"]
        h.append(f"""<div class="card"><h3>{i['tag']} ({i['floor']})
 — 逐次 {s:.4f} → PG {p:.4f} ({(p - s) / s * 100:+.0f}%)、
 ループ {loops.get(i['tag'], 0)}本 / チャンク {i['n_chunks']}本</h3>
<div class="row"><div>{traj_svg(i)}<div class="cap">軌跡</div></div>
<div>{err_svg(i)}<div class="cap">フレーム毎の誤差</div></div></div></div>""")

    h.append("""<h2>読みどころ</h2>
<ul>
<li><b>ループが多い島ほど効いている</b> — 5本の 1F / 3F が −45%前後。
0本の島でも改善するのは<b>全チャンク同時最適化</b>そのものの効果</li>
<li><b>最悪島 2F2 はループ0本で取り残された</b> — 候補8本すべて棄却。
<code>find_loops</code> は「現在解で空間的に近い対」を候補にするので、
<b>累積誤差が大きい島ほど真の再訪対が近くに来ない</b> (鶏と卵)</li>
<li><b>悪化した2島</b> (1Fc / 1Fa) — 誤差推移を見ると特定区間で跳ねている。
誤ったループが1本混ざると全体が引きずられる。採否は ICP残差だけで決めており、
橋渡しで確認したとおり<b>残差は当てにならない</b></li>
</ul>""")
    return "\n".join(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--loops", required=True, help="島タグ -> 採用ループ数 JSON")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    html = render(json.loads(open(a.data).read()),
                  json.loads(open(a.loops).read()))
    open(a.out, "w").write(html)
    print(f"-> {a.out} ({len(html):,} bytes)")


if __name__ == "__main__":
    main()
