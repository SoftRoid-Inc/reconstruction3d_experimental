"""合併GSの単体HTMLビューア生成 (antimatter15/splat ベース、MIT)

- gaussian_house{SFX}.ply を .splat 形式(32B/個)に量子化してbase64同梱
- 描画系はvendorのmain.jsを最小パッチ: fetch→同梱データ、carousel無効、
  初期カメラ=1F中央・高さ1.5m
- UI: 階フィルタ(全/1F/2F)、視点高さプリセット(1.3m/1.5m)
- 座標系: 図面座標(z上)をレンダラ規約(y下)へ Rx(-90°) で回して格納

使い方 (env=panovggt):
  IMG_SESSION=0312 conda run -n panovggt python -m img_align.gs_viewer
出力: results/backend/gs_viewer{SFX}.html
"""
import base64
import json
import math
from pathlib import Path

import numpy as np

from .config import SESSION
from .localize import PlanGrid
from .meshing import (ISLANDS, OUT_DIR, SFX, apply_matrix, drawing_matrix,
                      load_anchor, resolve_frames)

VENDOR = Path(__file__).resolve().parent / "vendor" / "splat_main.js"
MAX_SPLATS = 1_200_000
FLOOR_SPLIT = 2.6          # 図面座標での1F/2F境界 [m]
WARP_STEP_M = 1.0          # 軌跡ワープ点の間隔 (経路に沿って)


MIN_CLEARANCE_M = 1.2      # 前方にこの距離未満で面があるワープ点は捨てる


def forward_clearance(p, fwd, pts2d):
    """カメラ高さ帯の点群に対する前方(±30°)の見通し距離 [m]"""
    rel = pts2d - p[None, :2]
    dist = np.linalg.norm(rel, axis=1)
    near = dist < 6.0
    if not near.any():
        return np.inf
    rel, dist = rel[near], dist[near]
    cosang = (rel @ fwd[:2]) / np.maximum(dist * np.hypot(*fwd[:2]), 1e-9)
    in_cone = cosang > np.cos(np.radians(30))
    if in_cone.sum() < 30:
        return np.inf
    return float(np.percentile(dist[in_cone], 10))


def build_warps(means_draw=None):
    """軌跡に沿った約1m間隔のワープ点 [{l: ラベル, v: viewMatrix(16)}]

    位置=実カメラ(図面座標→レンダラy下系)、視線=進行方向・水平。
    means_draw(図面座標のガウシアン中心)を渡すと、前方1.2m未満に面がある
    地点(クローゼット内・壁直前)を除外する。
    viewMatrix は antimatter15 の列優先レイアウト。
    """
    R = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])   # 図面→レンダラ
    warps = []
    for isl in ISLANDS[SESSION]:
        frames, anchor = resolve_frames(isl, verbose=False)
        frames = sorted(frames, key=lambda f: f["name"])
        M, _ = drawing_matrix(load_anchor(isl), PlanGrid(isl["floor"]),
                              isl["z_off"], metric_input=True)
        pos = apply_matrix(M, np.stack([f["t_m"] for f in frames]))
        keep = [0]
        acc = 0.0
        for i in range(1, len(pos)):
            acc += float(np.linalg.norm(pos[i, :2] - pos[i - 1, :2]))
            if acc >= WARP_STEP_M:
                keep.append(i)
                acc = 0.0
        # クリアランス判定用: この階のカメラ高さ帯にある中心点のxy
        pts2d = None
        if means_draw is not None:
            zc = float(np.median(pos[:, 2]))
            band = means_draw[(means_draw[:, 2] > zc - 0.5)
                              & (means_draw[:, 2] < zc + 0.5)]
            pts2d = band[:, :2]
        n_drop = 0
        for j, i in enumerate(keep):
            nxt = pos[keep[j + 1]] if j + 1 < len(keep) else pos[keep[j - 1]]
            d = (nxt - pos[i]) if j + 1 < len(keep) else (pos[i] - nxt)
            a, b = float(d[0]), float(d[1])          # 図面xy → レンダラxz
            n = math.hypot(a, b) or 1.0
            a, b = a / n, b / n
            if pts2d is not None and forward_clearance(
                    pos[i], np.array([a, b]), pts2d) < MIN_CLEARANCE_M:
                n_drop += 1
                continue
            p = pos[i] @ R.T
            # c2w列: x=(b,0,-a), y=(0,1,0), z=(a,0,b) → viewMatrix=逆行列
            x, y, z = np.array([b, 0, -a]), np.array([0, 1, 0]), \
                np.array([a, 0, b])
            vm = [x[0], y[0], z[0], 0, x[1], y[1], z[1], 0,
                  x[2], y[2], z[2], 0,
                  -float(x @ p), -float(y @ p), -float(z @ p), 1]
            warps.append({"l": f"{isl['floor']} {frames[i]['name'][-6:]}",
                          "v": [round(float(v), 4) for v in vm]})
        if n_drop:
            print(f"  {isl['tag']}: warp {n_drop} 点をクリアランス不足で除外")
    return warps


def read_3dgs_full(path):
    with open(path, "rb") as fp:
        header = b""
        while not header.endswith(b"end_header\n"):
            header += fp.readline()
        n = int([ln for ln in header.decode().splitlines()
                 if ln.startswith("element vertex")][0].split()[-1])
        d = np.frombuffer(fp.read(n * 14 * 4), dtype=np.float32).reshape(n, 14)
    means, f_dc, logit = d[:, :3], d[:, 3:6], d[:, 6]
    log_scales, quats = d[:, 7:10], d[:, 10:14]
    rgb = np.clip(0.5 + 0.28209479 * f_dc, 0, 1)
    opac = 1 / (1 + np.exp(-logit))
    return means, quats, log_scales, rgb, opac


def quat_mul(r, q):
    """(4,) ⊗ (N,4) wxyz"""
    w1, x1, y1, z1 = r
    w2, x2, y2, z2 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], axis=1)


def to_splat_bytes(means, quats, log_scales, rgb, opac, max_n=MAX_SPLATS):
    """antimatter15 .splat 形式 (32B/個)。寄与順(不透明度×体積)で間引き"""
    idx = np.argsort(-opac)[:max_n]
    buf = np.zeros(len(idx), dtype=[("pos", "f4", 3), ("scale", "f4", 3),
                                    ("rgba", "u1", 4), ("quat", "u1", 4)])
    buf["pos"] = means[idx]
    buf["scale"] = np.exp(log_scales[idx])
    buf["rgba"][:, :3] = np.clip(rgb[idx] * 255, 0, 255)
    buf["rgba"][:, 3] = np.clip(opac[idx] * 255, 0, 255)
    q = quats[idx] / np.linalg.norm(quats[idx], axis=1, keepdims=True)
    buf["quat"] = np.clip(q * 128 + 128, 0, 255)
    return buf.tobytes(), len(idx)


HTML = """<!doctype html><html><head><meta charset="utf-8">
<title>gaussian house __SESS__</title>
<style>
 body{margin:0;background:#000;color:#eee;font-family:sans-serif;overflow:hidden}
 #canvas{width:100vw;height:100vh;display:block}
 #ui{position:fixed;top:8px;left:8px;z-index:5;background:#000a;padding:8px 10px;
     border-radius:8px;font-size:13px}
 #ui button{margin:2px;padding:3px 10px;background:#333;color:#eee;border:1px solid #666;
     border-radius:5px;cursor:pointer}
 #ui button.on{background:#2b6cb0}
 #fps{position:fixed;top:8px;right:12px;z-index:5;color:#9f9}
 #spinner{position:fixed;top:50%;left:50%;color:#fff;z-index:6}
 #message{position:fixed;top:40%;left:10%;color:#f66;z-index:6}
 #camid,#caminfo,#progress{display:none}
</style></head><body>
<div id="ui">
 <b>gaussian house __SESS__</b><br>
 階: <button id="fAll" class="on">全体</button><button id="f1F">1F</button><button id="f2F">2F</button><br>
 高さ: <button id="h13">1.3m</button><button id="h15">1.5m</button><br>
 ワープ: <button id="wprev">◀</button><select id="wsel" style="max-width:110px"></select><button id="wnext">▶</button> (N/P)<br>
 <span style="color:#aaa">移動: ↑↓ / 旋回: A D / 見回し: ドラッグ</span>
</div>
<div id="fps"></div><div id="camid"></div><div id="caminfo"></div>
<div id="spinner">loading...</div><div id="message"></div>
<progress id="progress"></progress>
<canvas id="canvas"></canvas>
<script>
const B64 = "__B64__";
function b64ToU8(s){const bin=atob(s);const u=new Uint8Array(bin.length);
 for(let i=0;i<bin.length;i++)u[i]=bin.charCodeAt(i);return u;}
const EMBED_FULL = b64ToU8(B64);
const WARPS = __WARPS__;
</script>
<script>
__MAINJS__
</script></body></html>
"""

UI_JS = """
    // ---- 階フィルタ + 高さプリセット (パッチ追加) ----
    const ROW = 32;
    function postFloor(mode) {
        // y'(=下向き) = -高さ。1F: 高さ<FLOOR_SPLIT
        if (mode === "all") {
            worker.postMessage({ buffer: splatData.buffer,
                vertexCount: Math.floor(splatData.length / ROW) });
            return;
        }
        const n = Math.floor(splatData.length / ROW);
        const f = new Float32Array(splatData.buffer);
        const out = new Uint8Array(splatData.length);
        let m = 0;
        for (let i = 0; i < n; i++) {
            const h = -f[i * 8 + 1];
            const is1F = h < __SPLIT__;
            if ((mode === "1F") === is1F) {
                out.set(splatData.subarray(i * ROW, (i + 1) * ROW), m * ROW);
                m++;
            }
        }
        worker.postMessage({ buffer: out.buffer.slice(0, m * ROW),
                             vertexCount: m });
    }
    // 高さプリセットはロック式トグル + 常時オートレベリング
    // (ドラッグ回転はカメラローカル軸なのでロールが蓄積する。ロールのみを
    //  連続除去し、ヨー・ピッチ・位置は保持する)
    let lockedHeight = null;
    setInterval(() => {
        let inv = invert4(viewMatrix);
        const z = [inv[8], inv[9], inv[10]];
        if (Math.hypot(z[0], z[2]) > 0.2) {       // ほぼ真上/真下は触らない
            let x = [z[2], 0, -z[0]];
            const n = Math.hypot(x[0], x[2]);
            x = [x[0] / n, 0, x[2] / n];
            const y = [z[1] * x[2] - z[2] * x[1],
                       z[2] * x[0] - z[0] * x[2],
                       z[0] * x[1] - z[1] * x[0]];
            inv[0] = x[0]; inv[1] = x[1]; inv[2] = x[2];
            inv[4] = y[0]; inv[5] = y[1]; inv[6] = y[2];
        }
        if (lockedHeight !== null) inv[13] = -lockedHeight;
        viewMatrix = invert4(inv);
    }, 50);
    function setHeight(h, btn) {
        const was = lockedHeight === h;
        lockedHeight = was ? null : h;
        for (const b of ["h13", "h15"])
            document.getElementById(b).classList.remove("on");
        if (!was) btn.classList.add("on");
    }
    // 軌跡ワープ (実撮影カメラの位置・進行方向へジャンプ)
    let warpIdx = -1;
    const wsel = document.getElementById("wsel");
    for (let i = 0; i < WARPS.length; i++) {
        const o = document.createElement("option");
        o.value = i;
        o.text = WARPS[i].l;
        wsel.add(o);
    }
    function warpTo(i) {
        if (!WARPS.length) return;
        warpIdx = ((i % WARPS.length) + WARPS.length) % WARPS.length;
        viewMatrix = WARPS[warpIdx].v.slice();
        wsel.value = warpIdx;
    }
    wsel.addEventListener("change", () => warpTo(+wsel.value));
    document.getElementById("wprev").addEventListener("click",
        () => warpTo(warpIdx - 1));
    document.getElementById("wnext").addEventListener("click",
        () => warpTo(warpIdx + 1));
    window.addEventListener("keydown", (e) => {
        if (e.code === "KeyN") warpTo(warpIdx + 1);
        if (e.code === "KeyP") warpTo(warpIdx - 1);
    });
    for (const [id, fn] of [["fAll", (e) => postFloor("all")],
                            ["f1F", (e) => postFloor("1F")],
                            ["f2F", (e) => postFloor("2F")],
                            ["h13", (e) => setHeight(1.3, e.target)],
                            ["h15", (e) => setHeight(1.5, e.target)]]) {
        document.getElementById(id).addEventListener("click", (e) => {
            if (id[0] === "f") {
                for (const b of ["fAll", "f1F", "f2F"])
                    document.getElementById(b).classList.remove("on");
                e.target.classList.add("on");
            }
            fn(e);
        });
    }
"""


def patch_mainjs(js, view0):
    # 1) fetch~splatData確保 を同梱データに差し替え
    a = js.index("    const url = new URL(")
    b = js.index('let splatData = new Uint8Array(req.headers.get("content-length"));')
    b = js.index("\n", b) + 1
    js = (js[:a]
          + "    const rowLength = 3 * 4 + 3 * 4 + 4 + 4;\n"
          + "    let splatData = EMBED_FULL;\n"
          + js[b:])
    # 元のrowLength宣言(重複)を除去
    js = js.replace("    const rowLength = 3 * 4 + 3 * 4 + 4 + 4;\n"
                    "    const reader = req.body.getReader();\n", "", 1)
    # 2) 読み込みループ → 一括postMessage + UI配線
    a = js.index("    let bytesRead = 0;")
    b = js.index("}\n\nmain().catch")
    js = (js[:a]
          + "    worker.postMessage({ buffer: splatData.buffer,\n"
          + "        vertexCount: Math.floor(splatData.length / rowLength) });\n"
          + "    // 初回ソートをRAFに依存させない (非表示タブではframeループが止まる)\n"
          + "    worker.postMessage({ view: multiply4(projectionMatrix, viewMatrix) });\n"
          + UI_JS
          + js[b:])
    # 3) carousel無効 + 初期カメラ
    js = js.replace("let carousel = true;", "let carousel = false;", 1)
    js = js.replace(
        "let defaultViewMatrix = [\n"
        "    0.47, 0.04, 0.88, 0, -0.11, 0.99, 0.02, 0, -0.88, -0.11, 0.47, 0, 0.07,\n"
        "    0.03, 6.55, 1,\n"
        "];",
        f"let defaultViewMatrix = {json.dumps(view0)};", 1)
    return js


def main():
    means, quats, log_scales, rgb, opac = read_3dgs_full(
        OUT_DIR / f"gaussian_house{SFX}.ply")
    warps = build_warps(means_draw=means[opac > 0.3])
    # 図面座標(z上) → レンダラ規約(y下): Rx(-90°)
    R = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])
    z_draw = means[:, 2].copy()
    means = means @ R.T
    # 位置の回転Rと同一の回転 (Rx(+90°)相当)。符号を誤ると斜め向きの
    # 異方性スプラットが鏡映方向に傾く (レビューC1)
    r_quat = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0])
    quats = quat_mul(r_quat, quats)
    blob, n = to_splat_bytes(means, quats, log_scales, rgb, opac)

    # 初期カメラ: 1F重心の上、高さ1.5m、+z'(図面+y)向き
    m1 = means[z_draw < FLOOR_SPLIT]
    cx, cz = float(np.median(m1[:, 0])), float(np.median(m1[:, 2]))
    # viewMatrixの並進は −(カメラ位置)。カメラ高さ1.5m = ワールドy'=-1.5 →
    # 第14要素は +1.5 (符号を誤ると床下からの初期視点になる。レビューI1)
    view0 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
             round(-cx, 3), 1.5, round(-cz, 3), 1]

    js = patch_mainjs(VENDOR.read_text(), view0)
    html = (HTML.replace("__SESS__", SESSION)
                .replace("__WARPS__", json.dumps(warps))
                .replace("__B64__", base64.b64encode(blob).decode())
                .replace("__MAINJS__", js.replace("__SPLIT__", str(FLOOR_SPLIT))))
    out = OUT_DIR / f"gs_viewer{SFX}.html"
    out.write_text(html)
    print(f"-> {out} ({n:,} splats, {out.stat().st_size/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
