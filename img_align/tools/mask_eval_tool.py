"""boundary mask の目視評価ビューア (ローカルFS版)

敷地境界マスク (masks_boundary/*.png, 白=表示/黒=隠す) を元フレームに赤く
オーバーレイし、フレーム単位で 正解 / マスク漏れ(FN) / 過剰マスク(FP) /
誤マスク(両方) を評価する。結果はセッションごとに
<sess>/masks_boundary_eval.json へ自動保存される。

polish-pipeline の scripts/mask_eval_viewer.py (S3人物マスク版) の移植。
依存は標準ライブラリ + Pillow のみ (img_align.config には依存しない)。

起動:
    python3 -m img_align.tools.mask_eval_tool --root 28-BBBBB [--port 8766]
ホーム画面で --root 直下の撮影 (images/ + masks_boundary/ を持つディレクトリ)
を選択する。root 自身がその構成を持つ場合も1セッションとして扱う。
"""
import argparse
import html
import io
import json
import re
import threading
import traceback
import urllib.parse
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image

FRAME_NAME_PATTERN = re.compile(r"frame_number_\d{6}")
MASK_SHOW_THRESHOLD = 128        # mask >= 128 = 表示 (白) / それ未満 = 隠す (黒)
OVERLAY_COLOR = (255, 40, 40)
OVERLAY_ALPHA = 115              # 0-255 (~45%)。masked_preview 動画と同じ見た目
DISPLAY_WIDTH = 1600
JPEG_QUALITY = 85
EVAL_JSON_NAME = "masks_boundary_eval.json"

# 評価カテゴリ: 保存キー → (ラベル, 色)。dict 順 = キー/ボタン 1〜4 の順
CATEGORIES = {
    "ok": ("正解", "#2e7d32"),
    "fn": ("マスク漏れ FN", "#c62828"),
    "fp": ("過剰マスク FP", "#ef6c00"),
    "both": ("誤マスク 両方", "#6a1b9a"),
}
CATEGORY_DESCRIPTIONS = {
    "ok": "マスクが適切 (隠すべき敷地外が隠れ、敷地内は見えている)",
    "fn": "隠すべき敷地外が見えている (マスク漏れ)",
    "fp": "見せるべき敷地内が隠れている (過剰マスク)",
    "both": "マスク漏れと過剰マスクの両方がある",
}
UNRATED_COLOR = "#dddddd"


# ---------------------------------------------------------------- データ層

class Session:
    """1撮影分: フレームjpg一覧 / マスクpng / 評価JSON の置き場"""

    def __init__(self, name, sdir):
        self.name = name
        self.dir = sdir
        self.frames_dir = sdir / "images"
        self.masks_dir = sdir / "masks_boundary"
        self.eval_path = sdir / EVAL_JSON_NAME
        self.frames = sorted(p.stem for p in
                             self.frames_dir.glob("frame_number_*.jpg"))
        self.masks = {p.stem for p in
                      self.masks_dir.glob("frame_number_*.png")}


def discover_sessions(root):
    """root 直下 (および root 自身) から images+masks_boundary を持つ撮影を探す"""
    sessions = []
    cands = [root] + sorted(p for p in root.iterdir() if p.is_dir())
    for d in cands:
        if not (d / "images").is_dir() or not (d / "masks_boundary").is_dir():
            continue
        s = Session(root.name if d == root else d.name, d)
        if s.frames and s.masks:
            sessions.append(s)
    return sessions


# ---------------------------------------------------------------- 評価ストア

class EvalStore:
    """セッションごとの評価JSON (atomic 書き込み)"""

    def __init__(self, sessions):
        self.lock = threading.Lock()
        self.data = {}           # name -> {"ratings": {...}, ...}
        for s in sessions:
            if s.eval_path.is_file():
                try:
                    self.data[s.name] = json.loads(s.eval_path.read_text())
                except json.JSONDecodeError as e:
                    raise SystemExit(
                        f"評価JSONが壊れています。修復するか退避してください: "
                        f"{s.eval_path} ({e})")
            else:
                self.data[s.name] = {"session": s.name, "ratings": {}}
            self.data[s.name].setdefault("ratings", {})
            self.data[s.name]["frame_count"] = len(s.frames)

    def ratings(self, name):
        with self.lock:
            return dict(self.data[name]["ratings"])

    def set_rating(self, sess, frame, category):
        """評価を保存 (category=None でクリア)。カテゴリ別件数を返す"""
        with self.lock:
            entry = self.data[sess.name]
            if category is None:
                entry["ratings"].pop(frame, None)
            else:
                entry["ratings"][frame] = category
            entry["updated_at"] = datetime.now().isoformat(timespec="seconds")
            tmp = sess.eval_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(entry, ensure_ascii=False, indent=1))
            tmp.replace(sess.eval_path)
            return self.counts_locked(entry)

    @staticmethod
    def counts_locked(entry):
        counts = {c: 0 for c in CATEGORIES}
        for v in entry["ratings"].values():
            if v in counts:
                counts[v] += 1
        return counts

    def counts(self, name):
        with self.lock:
            return self.counts_locked(self.data[name])


# ---------------------------------------------------------------- 画像合成

def _decode_draft(path, mode, target_width):
    """JPEG を draft モードで縮小デコード (5376幅のequirectでも高速)"""
    img = Image.open(path)
    img.draft(mode, (target_width, 1))
    return img.convert(mode)


def render_display_jpeg(sess, frame, overlay):
    """フレームにマスク黒領域を赤くオーバーレイした表示用JPEGを作る"""
    fp = sess.frames_dir / f"{frame}.jpg"
    img = _decode_draft(fp, "RGB", DISPLAY_WIDTH)
    w, h = img.size
    size = (DISPLAY_WIDTH, max(1, round(h * DISPLAY_WIDTH / w)))
    img = img.resize(size, Image.BILINEAR)
    if overlay and frame in sess.masks:
        mask = Image.open(sess.masks_dir / f"{frame}.png").convert("L") \
            .resize(size, Image.NEAREST)
        alpha_lut = [OVERLAY_ALPHA if v < MASK_SHOW_THRESHOLD else 0
                     for v in range(256)]
        color = Image.new("RGB", size, OVERLAY_COLOR)
        img = Image.composite(color, img, mask.point(alpha_lut))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


# ---------------------------------------------------------------- HTML

PAGE_CSS = """
body { font-family: sans-serif; margin: 1.2rem; background: #16171c; color: #ddd; }
h1 { font-size: 1.3rem; } a { color: #7cc4ff; text-decoration: none; }
a:hover { text-decoration: underline; }
table { border-collapse: collapse; font-size: 0.9rem; }
th, td { border: 1px solid #444; padding: 0.3rem 0.7rem; text-align: right; }
th { background: #23242b; } td:first-child, th:first-child { text-align: left; }
.chip { display: inline-block; color: #fff; border-radius: 3px;
  padding: 0.1rem 0.5rem; margin-right: 0.4rem; font-size: 0.85rem; }
.chip.unrated { color: #333; }
.bar { display: flex; width: 280px; height: 14px; border: 1px solid #555;
  border-radius: 3px; overflow: hidden; background: #2a2b31; }
.bar-seg { height: 100%; }
.toolbar { display: flex; gap: 0.4rem; align-items: center; flex-wrap: wrap;
  margin: 0.5rem 0; }
.toolbar button { font-size: 0.95rem; padding: 0.35rem 0.8rem;
  border: 1px solid #555; border-radius: 5px; background: #2d3748;
  color: #ddd; cursor: pointer; }
.toolbar button:hover { border-color: #7cc4ff; }
button.rate-btn { border-bottom-width: 3px; }
button.rate-btn.current { outline: 2px solid #7cc4ff; background: #1f3350;
  font-weight: bold; }
.badge { display: inline-block; color: #fff; border-radius: 3px;
  padding: 0.15rem 0.5rem; margin-left: 0.5rem; font-size: 0.85rem; }
#mask-badge { background: #c62828; display: none; }
#img-error { display: none; background: #5f2b2b; color: #f3b7b7;
  padding: 0.4rem 0.8rem; border-radius: 5px; }
#view { max-width: 100%; border: 1px solid #444; margin-top: 0.4rem; }
p.legend { color: #889; font-size: 0.85rem; margin: 0.3rem 0; }
nav.pager { margin: 0.4rem 0; display: flex; gap: 1rem; align-items: baseline; }
nav.pager span.disabled { color: #555; }
"""

SESSION_JS = r"""
const view = document.getElementById("view");
let overlay = true;
let cur = 0;
const hm = location.hash.match(/f=(\d+)/);
if (hm) cur = Math.min(S.frames.length - 1, Math.max(0, parseInt(hm[1])));

function imgUrl(i, ov) {
  return "/image/" + encodeURIComponent(S.session) + "/" + S.frames[i]
       + "?mode=" + (ov ? "overlay" : "raw");
}
function counts() {
  const c = {};
  for (const k of S.categories) c[k] = 0;
  for (const v of Object.values(S.ratings)) if (v in c) c[v]++;
  return c;
}
function updateStatus() {
  const frame = S.frames[cur];
  const rating = S.ratings[frame];
  document.getElementById("frame-pos").textContent =
    "フレーム " + (cur + 1) + " / " + S.frames.length + "  " + frame;
  const badge = document.getElementById("rating-badge");
  badge.textContent = rating ? S.labels[rating] : "未評価";
  badge.style.background = rating ? S.colors[rating] : "#666";
  document.getElementById("mask-badge").style.display =
    S.masks[frame] ? "none" : "inline-block";
  document.getElementById("overlay-state").textContent =
    "オーバーレイ: " + (overlay ? "ON" : "OFF");
  const c = counts();
  document.getElementById("progress").textContent =
    "評価済み " + Object.keys(S.ratings).length + " / " + S.frames.length +
    "  (" + S.categories.map(k => S.labels[k] + " " + c[k]).join(" / ") + ")";
  document.querySelectorAll("button.rate-btn").forEach(
    b => b.classList.toggle("current", b.dataset.cat === rating));
}
function show(i) {
  if (!S.frames.length) return;
  cur = Math.min(S.frames.length - 1, Math.max(0, i));
  location.hash = "f=" + cur;
  document.getElementById("img-error").style.display = "none";
  view.src = imgUrl(cur, overlay);
  updateStatus();
  if (cur + 1 < S.frames.length) new Image().src = imgUrl(cur + 1, overlay);
}
function nextUnrated() {
  for (let d = 1; d <= S.frames.length; d++) {
    const i = (cur + d) % S.frames.length;
    if (!(S.frames[i] in S.ratings)) { show(i); return; }
  }
  toast("全フレーム評価済みです");
}
async function rate(cat) {
  if (!S.frames.length) return;
  const frame = S.frames[cur];
  let ok = false;
  try {
    const res = await fetch("/api/rate", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({session: S.session, frame: frame, category: cat})});
    ok = res.ok;
  } catch (e) { ok = false; }
  if (!ok) { toast("評価の保存に失敗しました"); return; }
  if (cat) { S.ratings[frame] = cat; show(cur + 1); }
  else { delete S.ratings[frame]; updateStatus(); }
}
function toast(msg) {
  const err = document.getElementById("img-error");
  err.textContent = msg;
  err.style.display = "block";
  setTimeout(() => err.style.display = "none", 2000);
}
function toggleOverlay() { overlay = !overlay; show(cur); }
view.addEventListener("error", () => toast("画像の読み込みに失敗: " + S.frames[cur]));
document.addEventListener("keydown", (e) => {
  if (e.target.closest("input, textarea") || e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.key === "ArrowLeft" && e.shiftKey) { if (S.prevUrl) location.href = S.prevUrl; return; }
  if (e.key === "ArrowRight" && e.shiftKey) { if (S.nextUrl) location.href = S.nextUrl; return; }
  if (e.key === "ArrowLeft") { e.preventDefault(); show(cur - 1); return; }
  if (e.key === "ArrowRight") { e.preventDefault(); show(cur + 1); return; }
  if (e.key >= "1" && e.key <= "4") { rate(S.categories[parseInt(e.key) - 1]); return; }
  if (e.key === "0" || e.key === "Backspace") { rate(null); return; }
  if (e.key === "m") { toggleOverlay(); return; }
  if (e.key === "u") nextUnrated();
});
if (S.frames.length) show(cur);
"""


def _page(title, body):
    return f"""<!DOCTYPE html>
<html lang="ja"><head><meta charset="utf-8">
<title>{html.escape(title)}</title><style>{PAGE_CSS}</style></head>
<body><nav><a href="/">&larr; ホーム</a></nav>
{body}
</body></html>"""


def _embed_json(value):
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


def _stacked_bar(counts, total):
    if not total:
        return '<div class="bar"></div>'
    segs, rated = [], 0
    for cat, (label, color) in CATEGORIES.items():
        n = counts.get(cat, 0)
        rated += n
        if n:
            segs.append(f'<div class="bar-seg" style="width:{n/total*100:.2f}%;'
                        f'background:{color}" title="{label} {n}"></div>')
    if total - rated > 0:
        segs.append(f'<div class="bar-seg" style="width:{(total-rated)/total*100:.2f}%;'
                    f'background:{UNRATED_COLOR}" title="未評価 {total-rated}"></div>')
    return f'<div class="bar">{"".join(segs)}</div>'


def frame_metrics(counts):
    """フレーム単位の品質指標。各フレームを1サンプルとして
    FNあり = fn+both (隠すべき敷地外が見えた) / FPあり = fp+both (敷地内を隠した)。
    Accuracy = okの割合, Precision = FPが無い割合 (マスクした領域の正しさの代理),
    Recall = FNが無い割合 (隠すべきものを隠せた割合の代理), F1 = 調和平均"""
    rated = sum(counts.values())
    if not rated:
        return None
    has_fn = counts["fn"] + counts["both"]
    has_fp = counts["fp"] + counts["both"]
    acc = counts["ok"] / rated
    prec = (rated - has_fp) / rated
    rec = (rated - has_fn) / rated
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return {"rated": rated, "accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1}


def _metric_cells(m):
    if m is None:
        return '<td colspan="4" style="text-align:center;color:#889">—</td>'
    return "".join(f"<td>{v * 100:.1f}%</td>"
                   for v in (m["accuracy"], m["precision"], m["recall"],
                             m["f1"]))


def render_home(root, sessions, store):
    total = {c: 0 for c in CATEGORIES}
    total_frames = sum(len(s.frames) for s in sessions)
    rows = []
    for s in sessions:
        counts = store.counts(s.name)
        for c in CATEGORIES:
            total[c] += counts[c]
        rated = sum(counts.values())
        rows.append(
            f'<tr><td><a href="/session/{urllib.parse.quote(s.name)}">'
            f'{html.escape(s.name)}</a></td>'
            f'<td>{len(s.frames)}</td><td>{rated}</td>'
            f'{_metric_cells(frame_metrics(counts))}'
            f'<td>{_stacked_bar(counts, len(s.frames))}</td></tr>')
    total_rated = sum(total.values())
    tm = frame_metrics(total)
    rows.append(
        f'<tr style="border-top:2px solid #7cc4ff"><td><b>全体</b></td>'
        f'<td>{total_frames}</td><td>{total_rated}</td>'
        f'{_metric_cells(tm)}'
        f'<td>{_stacked_bar(total, total_frames)}</td></tr>')
    chips = "".join(
        f'<span class="chip" style="background:{color}" '
        f'title="{html.escape(CATEGORY_DESCRIPTIONS[cat])}">{label} {total[cat]}</span>'
        for cat, (label, color) in CATEGORIES.items())
    chips += (f'<span class="chip unrated" style="background:{UNRATED_COLOR}">'
              f'未評価 {max(total_frames - total_rated, 0)}</span>')
    body = f"""
<h1>boundary mask 評価 — {html.escape(str(root))}</h1>
<p>撮影 {len(sessions)} 件 / 総フレーム {total_frames} / 評価済み {total_rated}</p>
<p>{chips}</p>
<table><thead><tr><th>撮影</th><th>フレーム数</th><th>評価済み</th>
<th title="正解フレームの割合 = ok/評価済み">Accuracy</th>
<th title="過剰マスク(FP)の無いフレームの割合 = 1 - (fp+both)/評価済み">Precision</th>
<th title="マスク漏れ(FN)の無いフレームの割合 = 1 - (fn+both)/評価済み">Recall</th>
<th title="PrecisionとRecallの調和平均">F1</th>
<th>内訳</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table>
<p class="legend">指標はフレーム単位: Accuracy=正解率 /
Precision=過剰マスク(FP)が無い割合 / Recall=マスク漏れ(FN)が無い割合 / F1=調和平均。
分母は評価済みフレーム数。</p>
<p class="legend">撮影をクリックして評価を開始。評価は {EVAL_JSON_NAME} に自動保存されます。</p>
"""
    return _page("boundary mask 評価", body)


def render_session(sessions, idx, store):
    s = sessions[idx]
    prev_url = f"/session/{urllib.parse.quote(sessions[idx-1].name)}" if idx > 0 else ""
    next_url = (f"/session/{urllib.parse.quote(sessions[idx+1].name)}"
                if idx < len(sessions) - 1 else "")

    def pager_link(url, label):
        return (f'<a href="{url}">{label}</a>' if url
                else f'<span class="disabled">{label}</span>')

    rating_buttons = "".join(
        f'<button class="rate-btn" data-cat="{cat}" '
        f'style="border-bottom-color:{color}" '
        f'title="{html.escape(CATEGORY_DESCRIPTIONS[cat])}" '
        f"onclick=\"rate('{cat}')\">{i} {label}</button>"
        for i, (cat, (label, color)) in enumerate(CATEGORIES.items(), start=1))
    state = {
        "session": s.name, "frames": s.frames,
        "masks": {n: 1 for n in s.masks},
        "ratings": store.ratings(s.name),
        "categories": list(CATEGORIES),
        "labels": {c: lb for c, (lb, _) in CATEGORIES.items()},
        "colors": {c: col for c, (_, col) in CATEGORIES.items()},
        "prevUrl": prev_url, "nextUrl": next_url,
    }
    body = f"""
<h1>{html.escape(s.name)} <span style="color:#889;font-size:0.8em">
{html.escape(str(s.dir))}</span></h1>
<nav class="pager">{pager_link(prev_url, "← 前の撮影")}
<span style="color:#889">{idx + 1} / {len(sessions)}</span>
{pager_link(next_url, "次の撮影 →")}<span id="progress"></span></nav>
<div class="toolbar">
  <button onclick="show(cur - 1)">&#9664; 前</button>
  <button onclick="show(cur + 1)">次 &#9654;</button>
  {rating_buttons}
  <button onclick="rate(null)" title="このフレームの評価を取り消す">0 クリア</button>
  <button onclick="toggleOverlay()"><span id="overlay-state">オーバーレイ: ON</span></button>
  <button onclick="nextUnrated()" title="次の未評価フレームへ">u 次の未評価</button>
</div>
<div><span id="frame-pos"></span>
  <span id="rating-badge" class="badge"></span>
  <span id="mask-badge" class="badge">マスクなし</span></div>
<p id="img-error"></p>
<img id="view" alt="frame">
<p class="legend">キー: ←/→ 移動 / 1〜4 評価して次へ / 0・Backspace クリア /
m オーバーレイ切替 / u 次の未評価 / Shift+←/→ 撮影移動</p>
<p class="legend">{" / ".join(f"{i} {lb}: {html.escape(CATEGORY_DESCRIPTIONS[c])}"
                              for i, (c, (lb, _)) in enumerate(CATEGORIES.items(), 1))}</p>
<script>const S = {_embed_json(state)};{SESSION_JS}</script>
"""
    return _page(f"{s.name} — boundary mask 評価", body)


# ---------------------------------------------------------------- HTTP 層

class ViewerHandler(BaseHTTPRequestHandler):
    root: Path
    sessions: list
    store: EvalStore
    _img_cache = {}
    _img_cache_lock = threading.Lock()

    def log_message(self, *a):
        pass

    def _session_by_name(self, name):
        for i, s in enumerate(self.sessions):
            if s.name == name:
                return i, s
        return None, None

    def do_GET(self):  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        parts = [urllib.parse.unquote(p) for p in parsed.path.split("/") if p]
        query = {k: v[0] for k, v in urllib.parse.parse_qs(parsed.query).items()}
        try:
            if not parts:
                self._send_html(render_home(self.root, self.sessions, self.store))
            elif parts[0] == "session" and len(parts) == 2:
                idx, s = self._session_by_name(parts[1])
                if s is None:
                    self._send_html("<h1>404</h1>", 404)
                else:
                    self._send_html(render_session(self.sessions, idx, self.store))
            elif parts[0] == "image" and len(parts) == 3:
                self._send_image(parts[1], parts[2], query.get("mode", "overlay"))
            else:
                self._send_html("<h1>404</h1>", 404)
        except BrokenPipeError:
            pass
        except Exception:
            self._send_html(f"<h1>500</h1><pre>{html.escape(traceback.format_exc())}</pre>", 500)

    def _send_image(self, name, frame, mode):
        _, s = self._session_by_name(name)
        if s is None or not FRAME_NAME_PATTERN.fullmatch(frame) \
                or frame not in set(s.frames):
            self._send_html("<h1>404</h1>", 404)
            return
        key = (name, frame, mode)
        with self._img_cache_lock:
            data = self._img_cache.get(key)
        if data is None:
            data = render_display_jpeg(s, frame, mode != "raw")
            with self._img_cache_lock:
                if len(self._img_cache) > 30:
                    self._img_cache.pop(next(iter(self._img_cache)))
                self._img_cache[key] = data
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):  # noqa: N802
        parts = [p for p in urllib.parse.urlparse(self.path).path.split("/") if p]
        try:
            if parts == ["api", "rate"]:
                self._handle_rate()
            else:
                self._send_json({"ok": False, "error": "not found"}, 404)
        except BrokenPipeError:
            pass
        except Exception:
            self._send_json({"ok": False, "error": traceback.format_exc()[-500:]}, 500)

    def _handle_rate(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(n))
        except json.JSONDecodeError:
            self._send_json({"ok": False, "error": "不正なJSON"}, 400)
            return
        _, s = self._session_by_name(payload.get("session", ""))
        frame = payload.get("frame", "")
        category = payload.get("category")
        if s is None:
            self._send_json({"ok": False, "error": "不正な撮影名"}, 400)
            return
        if not isinstance(frame, str) or frame not in set(s.frames):
            self._send_json({"ok": False, "error": "不正なフレーム名"}, 400)
            return
        if category is not None and category not in CATEGORIES:
            self._send_json({"ok": False, "error": "不正なカテゴリ"}, 400)
            return
        counts = self.store.set_rating(s, frame, category)
        self._send_json({"ok": True, "counts": counts})

    def _send_json(self, payload, status=200):
        data = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_html(self, body, status=200):
        data = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True,
                    help="ホームフォルダー (この直下の撮影を一覧する。例: 28-BBBBB)")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--host", default="0.0.0.0",
                    help="バインド先 (Tailscale経由なら 0.0.0.0 のまま)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        raise SystemExit(f"--root が見つかりません: {root}")
    sessions = discover_sessions(root)
    if not sessions:
        raise SystemExit(
            f"{root} 直下に images/ + masks_boundary/ を持つ撮影がありません")
    ViewerHandler.root = root
    ViewerHandler.sessions = sessions
    ViewerHandler.store = EvalStore(sessions)

    print(f"撮影 {len(sessions)} 件: {', '.join(s.name for s in sessions)}")
    print(f"評価JSON: <撮影>/{EVAL_JSON_NAME}")
    print(f"http://{args.host}:{args.port}/ (Ctrl-Cで終了)")
    srv = ThreadingHTTPServer((args.host, args.port), ViewerHandler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.server_close()


if __name__ == "__main__":
    main()
