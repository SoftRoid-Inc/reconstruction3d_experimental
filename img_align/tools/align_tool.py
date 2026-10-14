"""手動軌跡アライメントツール: ブラウザで島の軌跡を図面に合わせてアンカー登録する

自動照合(密度グリッド/コーナーRANSAC)が偽解を出すセッション向けの確実な手段。
ホーム画面で全セッションを一覧し、島(階)ごとに軌跡+密度マップを図面に重ねて
並進・回転・スケールを手で合わせ、「登録」で保存する:
  - anchor_<tag>.json       (パイプライン標準スキーマ。boundary_mask等が無変更で使う)
  - manual_traj_<tag>.json  (フレーム毎の図面px。corner_anchor と同形式)
  - gt_trajectory.csv       (フレーム毎の正規化座標 = GT。28-AAAAA CSV規約:
                             x/yとも図面幅Wで正規化、yは下原点)

起動:   python3 -m img_align.tools.align_tool [--port 8765]   (IMG_SESSION 不要)
内部:   config.SESSION が import 時固定のため、セッションデータの抽出は
        IMG_SESSION を変えたサブプロセス (--dump-session) で行いキャッシュする。
"""
import argparse
import base64
import csv
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

TEMPLATE = Path(__file__).resolve().parent / "align_tool_template.html"
FRAME_VIEW_W = 1344  # フレーム表示の縮小幅

# 密度オーバーレイの色 (RGBA、アルファは密度値に比例)
DENSITY_RGB = (60, 220, 255)


# ================================================================ ワーカー側
# IMG_SESSION 固定のサブプロセスで実行され、セッションペイロードJSONを stdout へ出す

def _img_dataurl(img, fmt="jpeg", **kw):
    buf = io.BytesIO()
    img.save(buf, format=fmt.upper(), **kw)
    return f"data:image/{fmt};base64," + base64.b64encode(buf.getvalue()).decode()


def dump_session():
    # prepare_island 等の print で stdout の JSON が汚れないよう退避
    import contextlib
    with contextlib.redirect_stdout(sys.stderr):
        payload = _build_session_payload()
    json.dump(payload, sys.stdout)


def _build_session_payload():
    from PIL import Image

    from img_align.config import OUT_DIR
    from img_align.postproc.anchor import MIN_Y_BINS, Y_BIN_M, prepare_island
    from img_align.config import PROP, RESULTS_DIR, SESSION
    from img_align.eval.gt_traj import MM_PER_PX
    from img_align.core.plan_grid import PlanGrid, ZUMEN_DIR
    from img_align.mesh.meshing import islands_for, load_chunk_light

    payload = {"sess": SESSION, "backend_dir": str(OUT_DIR),
               "frames_dir": str(PROP.frames_dir),
               "drawing_wh": list(PROP.drawing_wh),
               "mm_per_px": MM_PER_PX, "floors": {}, "islands": []}
    grids = {}
    for isl in islands_for(SESSION):
        tag, floor = isl["tag"], isl["floor"]
        try:
            d = prepare_island(tag)
        except FileNotFoundError as e:
            print(f"skip {tag}: {e}", file=sys.stderr)
            continue
        # フレーム→チャンク番号 (resolve_frames と同じ isl["chunks"] 順の先勝ち)。
        # 区間微調整の「どの区切りに属すか」の判定に使う。
        # dir名の0詰め桁はセッション依存 (chunk_0/chunk_00) → 数値キーで引く
        croot = RESULTS_DIR / isl["root"]
        cdir_by_num = {int(q.name.split("_")[1]): q
                       for q in croot.glob("chunk_*")}
        name_chunk = {}
        for c in isl["chunks"]:
            try:
                names_c, _, _, _ = load_chunk_light(cdir_by_num[c])
            except (KeyError, FileNotFoundError):
                continue
            for n in names_c:
                name_chunk.setdefault(n, c)
        if floor not in payload["floors"]:
            g = grids[floor] = PlanGrid(floor)
            img = Image.open(ZUMEN_DIR / f"{floor}.jpg").convert("RGB")
            payload["floors"][floor] = {
                "drawing": _img_dataurl(img, "jpeg", quality=85),
                "x0_px": g.x0_px, "y0_px": g.y0_px, "cell_px": g.cell_px,
                "nx": g.nx, "ny": g.ny}
        cam = d["cam_pos_aligned"]
        cam_rc = np.stack([(cam[:, 2] - d["origin"][1]) / d["cell_units"],
                           (cam[:, 0] - d["origin"][0]) / d["cell_units"]], 1)
        frames = [int(re.sub(r"\D", "", str(n))) for n in d["names"]]
        order = np.argsort(frames)

        # 密度 → RGBA PNG (白壁をシアンで、値→アルファ)
        den = np.clip(d["density"], 0, 1)
        rgba = np.zeros((*den.shape, 4), np.uint8)
        rgba[..., 0], rgba[..., 1], rgba[..., 2] = DENSITY_RGB
        rgba[..., 3] = (den * 255).astype(np.uint8)
        den_url = _img_dataurl(Image.fromarray(rgba), "png")

        # 既存アンカー (ISLANDSの参照ファイル優先、無ければ素のanchor_<tag>.json)
        # UIの変換状態は 2x2 アフィン A + t (非等方スケール対応)。
        # 既存アンカーに "A" が無ければ s*R(th) から構成する
        def _anchor_At(a):
            if a.get("A") is not None:
                return np.asarray(a["A"], float).tolist()
            co, si = math.cos(a["th"]), math.sin(a["th"])
            return (a["s"] * np.array([[co, -si], [si, co]])).tolist()

        init, saved, mirror_col, segments = None, False, None, None
        for name in (isl["anchor"], f"anchor_{tag}.json"):
            p = OUT_DIR / name
            if p.exists():
                a = json.loads(p.read_text())
                init = {"A": _anchor_At(a), "t": list(a["t"])}
                mirror_col = a.get("mirror_col")
                saved = True
                # 区間微調整の復元: セグメントアンカーから (chunks, A, t)
                if a.get("segments"):
                    segments = []
                    for sg in a["segments"]:
                        sp = OUT_DIR / sg["anchor"]
                        if not sp.exists():
                            segments = None
                            break
                        sa = json.loads(sp.read_text())
                        segments.append({"chunks": sg["chunks"],
                                         "A": _anchor_At(sa),
                                         "t": list(sa["t"])})
                break
        if mirror_col is not None:
            # 表示用に鏡映をセル座標へ焼き込む。保存時は mirror_col を
            # そのまま引き継ぐ (パイプラインは生セルに mirror→(s,th,t) を
            # 適用するため、UIの(s,th,t)と意味が一致する)
            cam_rc[:, 1] = mirror_col - cam_rc[:, 1]
            rgba = rgba[:, ::-1]
            den_url = _img_dataurl(Image.fromarray(rgba), "png")
        if init is None:
            g = grids[floor]
            lo, hi = cam_rc.min(0), cam_rc.max(0)
            span = float(max(hi[0] - lo[0], hi[1] - lo[1], 1e-6))
            s0 = 0.6 * min(g.ny, g.nx) / span
            c_cell = np.array([g.ny / 2, g.nx / 2])
            init = {"A": (s0 * np.eye(2)).tolist(),
                    "t": list(c_cell - s0 * (lo + hi) / 2)}

        # チャンク別密度 (区間微調整の表示用): 点を最寄りカメラのフレーム→
        # チャンクへ帰属させ、親と同じグリッドで span 密度を作る。
        # PNG は白+アルファで送り、クライアントが区間色に着色して
        # 区間ごとの変換で描画する (区切り直してもサーバ再計算は不要)
        den_chunks = {}
        w = d.get("density_pts")
        if w is not None and len(w) and len(isl["chunks"]) > 1:
            from scipy.spatial import cKDTree
            w2 = w[::max(1, len(w) // 4_000_000)]
            _, ci = cKDTree(d["cam_pos_aligned"]).query(w2, k=1)
            cam_chunk = np.array([name_chunk.get(str(n), -1)
                                  for n in d["names"]])
            pt_chunk = cam_chunk[ci]
            nz, nx = d["density"].shape
            ox, oz = d["origin"]
            cu = float(d["cell_units"])
            ix = ((w2[:, 0] - ox) / cu).astype(np.int64)
            iz = ((w2[:, 2] - oz) / cu).astype(np.int64)
            iy = ((w2[:, 1] - w2[:, 1].min())
                  / (Y_BIN_M / d["m_per_unit"])).astype(np.int64)
            ny = int(iy.max()) + 1
            inb = (ix >= 0) & (ix < nx) & (iz >= 0) & (iz < nz)
            for c in isl["chunks"]:
                m = inb & (pt_chunk == c)
                if not m.any():
                    continue
                flat = (iz[m] * nx + ix[m]) * ny + iy[m]
                span2 = np.bincount(np.unique(flat) // ny,
                                    minlength=nz * nx).reshape(nz, nx)
                # 閾値は親 (MIN_Y_BINS) より緩め: チャンク分割+間引きで
                # セルあたりのyビン数が減り、壁が消えすぎるため
                dc = np.where(span2 >= max(2, MIN_Y_BINS - 2),
                              span2, 0).astype(np.float32)
                if dc.max() <= 0:
                    continue
                rgbac = np.zeros((nz, nx, 4), np.uint8)
                rgbac[..., :3] = 255
                rgbac[..., 3] = (dc / dc.max() * 255).astype(np.uint8)
                if mirror_col is not None:
                    rgbac = rgbac[:, ::-1]
                den_chunks[str(c)] = _img_dataurl(
                    Image.fromarray(rgbac), "png")

        payload["islands"].append({
            "tag": tag, "floor": floor,
            "anchor_file": isl["anchor"],
            "frames": [frames[i] for i in order],
            "cam_rc": cam_rc[order].round(3).tolist(),
            "frame_chunk": [name_chunk.get(str(d["names"][i]), -1)
                            for i in order],
            "chunks": list(isl["chunks"]),
            "density_png": den_url,
            "density_chunks": den_chunks,
            "density_shape": list(d["density"].shape),
            "mirror_col": mirror_col,
            "init": init, "saved": saved, "segments": segments,
            "anchor_base": {
                "R_align": d["R_align"].tolist(),
                "origin": np.asarray(d["origin"]).tolist(),
                "cell_units": float(d["cell_units"]),
                "floor_y_units": float(d["floor_y_units"]),
                "floor_phys_y_units": float(d["floor_phys_y_units"]),
                "m_per_unit": float(d["m_per_unit"])}})
    return payload


# ================================================================ サーバ側

def list_sessions():
    """ホーム画面用のセッション概況 (セッションimport不要の静的情報のみ)"""
    from img_align.config import PROPERTY
    from img_align.mesh.meshing import ISLANDS, islands_for
    out = []
    for sess in ISLANDS:
        islands = islands_for(sess)
        prop = PROPERTY.get(sess)
        if prop is None:
            continue
        backend = prop.root / "results" / sess / "backend"
        n_saved = sum((backend / i["anchor"]).exists()
                      or (backend / f"anchor_{i['tag']}.json").exists()
                      for i in islands)
        out.append({"sess": sess, "root": prop.root.name,
                    "n_islands": len(islands), "n_saved": n_saved,
                    "floors": sorted({i["floor"] for i in islands})})
    return sorted(out, key=lambda r: (r["root"], r["sess"]))


class AlignServer:
    def __init__(self):
        self.payloads = {}       # sess -> payload dict
        self.lock = threading.Lock()
        self.frame_cache = {}    # (sess,num) -> jpeg bytes

    def payload(self, sess):
        with self.lock:
            if sess in self.payloads:
                return self.payloads[sess]
        env = dict(os.environ, IMG_SESSION=sess)
        r = subprocess.run([sys.executable, "-m", "img_align.tools.align_tool",
                            "--dump-session"], env=env, capture_output=True,
                           cwd=Path(__file__).resolve().parent.parent)
        if r.returncode != 0:
            raise RuntimeError(r.stderr.decode()[-2000:])
        data = json.loads(r.stdout.decode())
        with self.lock:
            self.payloads[sess] = data
        return data

    def frame_jpeg(self, sess, num, width=FRAME_VIEW_W):
        width = max(512, min(int(width), 5376))
        key = (sess, num, width)
        if key in self.frame_cache:
            return self.frame_cache[key]
        from PIL import Image
        p = self.payload(sess)
        fp = Path(p["frames_dir"]) / f"frame_number_{num:06d}.jpg"
        img = Image.open(fp)
        if width < img.width:
            img = img.resize((width, width // 2), Image.BILINEAR)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=82)
        data = buf.getvalue()
        if len(self.frame_cache) > 30:
            self.frame_cache.pop(next(iter(self.frame_cache)))
        self.frame_cache[key] = data
        return data

    # ------------------------------------------------------------ 保存

    @staticmethod
    def _anchor_dict(tag, isl, A, t, anchored_by):
        """標準スキーマのアンカー辞書 (キー順固定)"""
        A = np.asarray(A, float)
        # 後方互換の s/th は極分解の等価値 (A が純相似ならそのまま一致)
        anchor = {"tag": tag, "floor": isl["floor"], **isl["anchor_base"],
                  "mirror_col": isl.get("mirror_col"),
                  "s": float(math.sqrt(abs(np.linalg.det(A)))),
                  "th": float(math.atan2(A[1, 0] - A[0, 1],
                                         A[0, 0] + A[1, 1])),
                  "t": list(map(float, t)), "A": A.tolist(),
                  "wall_resid_m": None, "anchored_by": anchored_by}
        return {k: anchor[k] for k in (
            "tag", "floor", "R_align", "floor_y_units", "floor_phys_y_units",
            "m_per_unit", "origin", "cell_units", "mirror_col",
            "s", "th", "t", "A", "wall_resid_m", "anchored_by")}

    def save(self, req):
        sess, tag = req["sess"], req["tag"]
        p = self.payload(sess)
        isl = next(i for i in p["islands"] if i["tag"] == tag)
        backend = Path(p["backend_dir"])
        backend.mkdir(parents=True, exist_ok=True)
        segs = req.get("segments") or []
        if len(segs) < 2:
            segs = []

        # 親の代表 (A,t): セグメントありならフレーム数最大の区間、無ければ要求値
        # (segments 未対応の消費者 = export_aligned2 等の近似フォールバック用)
        if segs:
            fc = isl["frame_chunk"]
            counts = [sum(1 for c in fc if c in set(sg["chunks"]))
                      for sg in segs]
            rep = segs[int(np.argmax(counts))]
            A, t = rep["A"], list(map(float, rep["t"]))
        else:
            A, t = req["A"], list(map(float, req["t"]))

        anchor_path = backend / isl["anchor_file"]
        if anchor_path.exists():
            shutil.copy2(anchor_path, anchor_path.with_suffix(".json.bak"))
        ordered = self._anchor_dict(tag, isl, A, t, "manual")

        # セグメントアンカー: base は親と同一、A/t だけ区間固有。
        # パイプラインは meshing.islands_expanded が仮想島として展開する
        seg_meta = []
        for k, sg in enumerate(segs):
            sname = f"anchor_{tag}_seg{k}.json"
            (backend / sname).write_text(json.dumps(
                self._anchor_dict(f"{tag}__seg{k}", isl, sg["A"], sg["t"],
                                  "manual-seg"), indent=1))
            seg_meta.append({"chunks": list(map(int, sg["chunks"])),
                             "anchor": sname})
        if seg_meta:
            ordered["segments"] = seg_meta
        # 区間数が減った/解除された場合の残骸セグメントファイルを削除
        for q in backend.glob(f"anchor_{tag}_seg*.json"):
            kq = q.stem.rsplit("seg", 1)[-1]
            if kq.isdigit() and int(kq) >= len(seg_meta):
                q.unlink()
        anchor_path.write_text(json.dumps(ordered, indent=1))

        # フレーム毎の図面px (corner_anchor と同形式)。フレームの属する
        # チャンクを含むセグメントの変換で計算する (無ければ親の変換)
        fl = p["floors"][isl["floor"]]
        px = self._traj_px(isl, fl, A, t)
        if segs:
            chunk_seg = {c: k for k, sg in enumerate(segs)
                         for c in sg["chunks"]}
            px_seg = [self._traj_px(isl, fl, sg["A"], sg["t"])
                      for sg in segs]
            for j, c in enumerate(isl["frame_chunk"]):
                k = chunk_seg.get(c)
                if k is not None:
                    px[j] = px_seg[k][j]
        traj = {f"frame_{n:06d}": [round(float(x), 2), round(float(y), 2)]
                for n, (x, y) in zip(isl["frames"], px)}
        (backend / f"manual_traj_{tag}.json").write_text(
            json.dumps(traj, indent=1))
        isl["init"] = {"A": np.asarray(A, float).tolist(), "t": t}
        isl["segments"] = [{"chunks": sg["chunks"], "A": sg["A"],
                            "t": list(map(float, sg["t"]))}
                           for sg in segs] or None
        isl["saved"] = True

        self._write_gt_csv(p, backend)
        return {"anchor": str(anchor_path), "n_segments": len(seg_meta),
                "traj": str(backend / f"manual_traj_{tag}.json"),
                "csv": str(backend / "gt_trajectory.csv")}

    @staticmethod
    def _traj_px(isl, fl, A, t):
        rc = np.asarray(isl["cam_rc"], float)
        pc = rc @ np.asarray(A).T + np.asarray(t)   # meshing.anchor_transform_cells
        x = fl["x0_px"] + pc[:, 1] * fl["cell_px"]  # refine_align.cells_to_px
        y = fl["y0_px"] + pc[:, 0] * fl["cell_px"]
        return np.stack([x, y], 1)

    def _write_gt_csv(self, p, backend):
        """登録済み全島の manual_traj から正規化GT CSVを再生成。
        正規化は 28-AAAAA trajectory_points.csv 規約 (eval_gt_traj.load_gt の逆):
        x_norm = x_px/W, y_norm = (H - y_px)/W (x/yとも幅Wで正規化、yは下原点)"""
        W, H = p["drawing_wh"]
        rows = {}
        for isl in p["islands"]:
            tp = backend / f"manual_traj_{isl['tag']}.json"
            if not tp.exists():
                continue
            for name, (x, y) in json.loads(tp.read_text()).items():
                fr = int(re.sub(r"\D", "", name))
                if fr in rows:      # 島間の重複フレームは先勝ち
                    continue
                rows[fr] = (isl["floor"], x / W, (H - y) / W)
        with open(backend / "gt_trajectory.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame_number", "floor",
                        "width_normalized_drawing_x",
                        "width_normalized_drawing_y"])
            for fr in sorted(rows):
                fl, xn, yn = rows[fr]
                w.writerow([fr, fl, round(xn, 6), round(yn, 6)])


SERVER = AlignServer()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *a):
        pass

    def _send(self, body, ctype="text/html; charset=utf-8", code=200):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _page(self, data):
        tpl = TEMPLATE.read_text()
        return tpl.replace("/*DATA*/", f"const BOOT = {json.dumps(data)};")

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path == "/":
                self._send(self._page({"mode": "home",
                                       "sessions": list_sessions()}))
            elif u.path == "/align":
                self._send(self._page({"mode": "align", "sess": q["sess"]}))
            elif u.path == "/session-data":
                self._send(json.dumps(SERVER.payload(q["sess"])),
                           "application/json")
            elif u.path == "/frame":
                self._send(SERVER.frame_jpeg(q["sess"], int(q["num"]),
                                             int(q.get("w", FRAME_VIEW_W))),
                           "image/jpeg")
            else:
                self._send("not found", code=404)
        except Exception as e:  # noqa: BLE001 — エラーはブラウザに返す
            self._send(f"error: {e}", "text/plain; charset=utf-8", 500)

    def do_POST(self):
        u = urlparse(self.path)
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n))
            if u.path == "/save":
                self._send(json.dumps(SERVER.save(req)), "application/json")
            else:
                self._send("not found", code=404)
        except Exception as e:  # noqa: BLE001
            self._send(f"error: {e}", "text/plain; charset=utf-8", 500)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1",
                    help="バインド先 (Tailscale経由なら tailscale ip を指定、"
                         "全インタフェースは 0.0.0.0)")
    ap.add_argument("--dump-session", action="store_true",
                    help="(内部用) IMG_SESSIONのペイロードJSONをstdoutへ")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    if args.dump_session:
        dump_session()
        return
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"align_tool: {url} (Ctrl-Cで終了)")
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, [url]).start()
    srv.serve_forever()


if __name__ == "__main__":
    main()
