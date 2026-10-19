"""外部テストデータセットを大量に回す実走スクリプト

`polish-pipeline/scripts/sfm_testdataset` の240件 (うちINSTA 156件) を
取り込み → PanoVGGT推論 → チェーン → ポーズグラフ → 評価 まで通し、
**小さな結果JSONだけ残してチャンクと画像を捨てる**。

## ディスクが律速なので、1件ずつ片付ける

1件あたり画像0.4GB + チャンク13GB。156件を残したら2TBになるので無理。
必要なのは評価結果 (数十KB) とチェーン (13KB/島) だけなので、
**1件終わるごとに images/ masks/ pcd/ を消す**。ピークは1件分で済む。

## 再開可能

`<dir>/result.json` があればその件はskipする。途中で止めても続きから走る。

## 計算時間の内訳 (実測、388枚5フロアの1件 = 471秒)

| 工程 | 秒 | 割合 |
|---|---|---|
| **推論+チェーン** | **357** | **76%** |
| **取込 (S3同期)** | **82** | **17%** |
| ポーズグラフ | 32 | 7% |
| 評価 | 0.4 | 0% |

ポーズグラフと評価は誤差。**GPU (推論) と ネットワーク (S3) は別資源**なので
**次の件のダウンロードを現在の件の推論と重ねる** (`--prefetch`、既定ON)。
これで取込の82秒が隠れ、1件あたり約17%短縮する。
さらに縮めるなら推論そのもの (1チャンク9.4秒 × チャンク数) を削るしかない。
subprocess の起動コストは**0.28秒で無視できる** — 実測せずに当たりを付けると
的を外す。

## 使い方
    # 候補を見る (INSTA + GTあり + 複数フロア)
    python -m img_align.tools.ext_batch list --limit 20
    # 実走 (5件ずつなど小分けに)
    python -m img_align.tools.ext_batch run --limit 5
    # 集計してレポート
    python -m img_align.tools.ext_batch report
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

SRC = Path("/home/maren/git/polish-pipeline/scripts/sfm_testdataset")
BUCKET = "s3://p2-test-data-bucket/person_mask_gt"
DRAWING_BUCKET = "s3://zenshot-private-bucket-production/"
ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT / "ext"
MIN_FRAMES, MAX_FRAMES = 150, 900   # 短すぎ=島が立たない / 長すぎ=1件が重い


def _sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def candidates():
    """INSTA かつ GTあり の候補 [(s3key, 枚数, フロア数)] を長さ順に返す"""
    rows = list(csv.DictReader(open(SRC / "trajectory_data_by_scan_data_id.csv")))
    info = json.loads((SRC / "s3_key_info.json").read_text())
    insta = {r[0]: r[2] for r in info if r[1] == "INSTA"}
    floors = {}
    for r in rows:
        k = r["metadata_s3_key"].rsplit("/data_number_0", 1)[0]
        if k in insta:
            floors.setdefault(k, set()).add(r["floor_id"])
    out = [(k, insta[k], len(f)) for k, f in floors.items()
           if MIN_FRAMES <= insta[k] <= MAX_FRAMES]
    # フロア数が多く・短いものを優先 (情報量が多くて軽い)
    return sorted(out, key=lambda x: (-x[2], x[1]))


def name_of(key):
    """s3キー → ディレクトリ名 / セッション名"""
    p = key.split("/")
    return f"sfm{p[2]}_{p[3]}{p[4]}{p[5]}", f"e{p[2]}x{p[4]}{p[5]}"


def setup(key):
    """取り込み: 画像・マスク・図面・GT を揃えて meta.json を書く"""
    d, sess = name_of(key)
    out = EXT / d
    if (out / "meta.json").exists():
        return out, sess
    out.mkdir(parents=True, exist_ok=True)
    r = _sh(["aws", "s3", "sync", f"{BUCKET}/{key}/", str(out),
             "--only-show-errors"])
    if r.returncode or not (out / "images").is_dir():
        raise RuntimeError(f"同期失敗: {r.stderr[:200]}")
    rows = [x for x in csv.DictReader(
        open(SRC / "trajectory_data_by_scan_data_id.csv"))
        if x["metadata_s3_key"].startswith(key)]
    tids = {x["trajectory_id"] for x in rows}
    fids = {x["floor_id"] for x in rows}
    pts = [p for p in csv.DictReader(open(SRC / "trajectory_points.csv"))
           if p["trajectory_id"] in tids]
    fl = [f for f in csv.DictReader(open(SRC / "floor_info.csv"))
          if f["floor_id"] in fids]
    for fn, data in (("trajectory_data_by_scan_data_id.csv", rows),
                     ("trajectory_points.csv", pts), ("floor_info.csv", fl)):
        with open(out / fn, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    (out / "zumen").mkdir(exist_ok=True)
    for f in fl:
        dst = out / "zumen" / f"{f['floor_name']}.jpg"
        if not dst.exists():
            _sh(["aws", "s3", "cp",
                 DRAWING_BUCKET + f["image_url"].split("amazonaws.com/", 1)[1],
                 str(dst), "--only-show-errors"])
    p = key.split("/")
    (out / "meta.json").write_text(json.dumps(
        {"session": sess, "gt_session": p[4] + p[5], "dir": d, "s3key": key,
         "n_frames": len(list((out / "images").glob("*.jpg"))),
         "n_gt": len(pts), "floors": len(fids)}, indent=1))
    return out, sess


def evaluate(sess):
    """島ごとに 逐次 / PG+ループ の GT残差 (図面幅で正規化) を測る"""
    code = r'''
import json
import numpy as np
from img_align.config import BENCH_DIR, PROP, CFG
from img_align.eval.bridge_eval import _sim2_fit
from img_align.eval.gt_traj import load_gt, session_meta
gt = load_gt(); dw, dh = PROP.drawing_wh
def ev(tag, refl=None):
    d = BENCH_DIR/tag
    if not (d/"global_cameras.npz").exists(): return None
    cam = np.load(d/"global_cameras.npz")
    fr = [int(str(n).replace("frame_","")) for n in cam["names"]]
    k = [i for i,f in enumerate(fr) if f in gt]
    if len(k) < 8: return None
    y = np.array([[gt[fr[i]]["x_px"], gt[fr[i]]["y_px"]] for i in k])
    r, fit, T = _sim2_fit(cam["positions"][k][:,[0,2]], y, refl=refl)
    return {"resid": r/dw, "traj": (fit/[dw,dh]).tolist(),
            "gt": (y/[dw,dh]).tolist(), "n": len(k), "refl": bool(T[3])}
isl, covered = [], set()
for m in json.loads((BENCH_DIR/"pv_manifest.json").read_text()):
    a = ev(m["tag"])
    if a is None: continue
    cam = np.load(BENCH_DIR/m["tag"]/"global_cameras.npz")
    covered |= {int(str(n).replace("frame_","")) for n in cam["names"]}
    isl.append({"tag": m["tag"], "floor": m.get("floor"), "a": m["a"],
                "b": m["b"], "n_chunks": len(m["chunks"]), "seq": a,
                "pg": ev("pg12_"+m["tag"], refl=a["refl"])})
print("@@" + json.dumps({"n_frames": len(CFG["frames"]), "n_gt": len(gt),
    "floors": sorted(session_meta()["floors"]),
    "coverage": len(covered & set(gt))/max(len(gt),1),
    "frame_cov": len(covered)/max(len(CFG["frames"]),1), "islands": isl}))
'''
    r = _sh([sys.executable, "-c", code],
            env={**os.environ, "IMG_SESSION": sess, "PYTHONPATH": str(ROOT)})
    for ln in r.stdout.splitlines():
        if ln.startswith("@@"):
            return json.loads(ln[2:])
    raise RuntimeError(f"評価失敗: {r.stderr[-400:]}")


def process(key, keep_chunks=False, verbose=True):
    """1件を通す。戻り値 result.json の中身

    **工程別の所要時間を必ず測る** — 最適化の的を外さないため。実測なしで
    「subprocessが重い」等と当たりを付けると外す (import は0.28秒だった)。
    """
    t0 = time.time()
    lap = {}

    def tick(k, fn):
        s = time.time()
        r = fn()
        lap[k] = round(time.time() - s, 1)
        if verbose:
            print(f"    [{k}] {lap[k]}秒", flush=True)
        return r

    out, sess = tick("取込", lambda: setup(key))
    res_p = out / "result.json"
    if res_p.exists():
        print(f"  [skip] {sess} は済み")
        return json.loads(res_p.read_text())
    env = {**os.environ, "IMG_SESSION": sess, "PYTHONPATH": str(ROOT)}
    r = tick("推論+チェーン", lambda: _sh(
        [sys.executable, "-m", "img_align.pipeline", "run", "--session", sess,
         "--segments", "gt", "--max-seg", "160", "--no-eval"],
        env=env, cwd=str(ROOT)))
    if r.returncode:
        raise RuntimeError(f"pipeline失敗: {r.stderr[-400:]}")
    man = json.loads(
        (out / "results" / sess / "frontend_bench" / "pv_manifest.json")
        .read_text())
    def _pg():
        d = {}
        for m in man:
            g = _sh([sys.executable, "-m", "img_align.core.posegraph",
                     "--from-tag", m["tag"], "--tag", "pg12_" + m["tag"]],
                    env=env, cwd=str(ROOT))
            n = 0
            for ln in g.stdout.splitlines():
                if "loop:" in ln and "採用" in ln:
                    n = int(ln.split("採用")[1].split()[0])
            d[m["tag"]] = n
        return d
    loops = tick("ポーズグラフ", _pg)
    res = tick("評価", lambda: evaluate(sess))
    res.update({"session": sess, "s3key": key, "loops": loops,
                "sec": round(time.time() - t0, 1), "lap": lap})
    res_p.write_text(json.dumps(res))
    if not keep_chunks:      # **ディスクが律速**: 評価が済んだら素材を捨てる
        for sub in ("images", "masks", "results/%s/pcd" % sess):
            p = out / sub
            if p.exists():
                shutil.rmtree(p, ignore_errors=True)
    return res


class Prefetcher:
    """次の件のS3取込を裏で走らせる (GPUが空かないので重ねられる)

    取込82秒 / 推論357秒なので、**取込は完全に隠れる**。
    """

    def __init__(self, keys):
        self.keys = list(keys)
        self.i = 0
        self.fut = None
        self.pool = ThreadPoolExecutor(max_workers=1)

    def _start(self, k):
        return self.pool.submit(setup, k)

    def next(self):
        """次の件の (dir, session) を返す。裏で更に次を取りに行く"""
        if self.i >= len(self.keys):
            return None, None, None
        k = self.keys[self.i]
        fut, self.fut = (self.fut, None), None
        got = fut[0] if fut[0] is not None else self._start(k)
        self.i += 1
        if self.i < len(self.keys):      # 次を先読み開始
            self.fut = self._start(self.keys[self.i])
        out, sess = got.result()
        return k, out, sess

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("list", "run"):
        s = sub.add_parser(c)
        s.add_argument("--limit", type=int, default=5)
        s.add_argument("--skip", type=int, default=0)
        if c == "run":
            s.add_argument("--keep-chunks", action="store_true")
            s.add_argument("--no-prefetch", action="store_true",
                           help="次の件のS3取込を先読みしない (既定は先読み)")
    sub.add_parser("report").add_argument("--out", default=str(
        ROOT / "img_align" / "results" / "ext_batch_report.html"))
    a = ap.parse_args()

    if a.cmd == "report":
        rows = [json.loads(p.read_text())
                for p in sorted(EXT.glob("*/result.json"))]
        print(f"{len(rows)}件の結果を集計")
        import statistics as st
        seq = [i["seq"]["resid"] for r in rows for i in r["islands"]]
        pg = [i["pg"]["resid"] for r in rows for i in r["islands"] if i["pg"]]
        if seq:
            print(f"  島{len(seq)}本  逐次 {st.median(seq):.4f} -> "
                  f"PG {st.median(pg):.4f}")
        out = {r["session"]: r for r in rows}
        p = ROOT / "img_align" / "results" / "ext_batch_data.json"
        p.write_text(json.dumps(out))
        print(f"-> {p}")
        return

    cand = candidates()[a.skip:a.skip + a.limit]
    if a.cmd == "list":
        print(f"{'s3key':52s} {'枚数':>5s} {'階':>3s}  {'名前'}")
        for k, n, f in cand:
            print(f"{k:52s} {n:5d} {f:3d}  {name_of(k)[1]}")
        return
    ok = ng = 0
    pf = None if a.no_prefetch else Prefetcher([k for k, _, _ in cand])
    for i, (k, n, f) in enumerate(cand, 1):
        print(f"\n===== [{i}/{len(cand)}] {k} ({n}枚 {f}フロア)", flush=True)
        if pf is not None:
            pf.next()        # ここで現在の件の取込完了を待ち、次を先読み開始
        try:
            r = process(k, a.keep_chunks)
            isl = r["islands"]
            med = (np.median([x["seq"]["resid"] for x in isl])
                   if isl else float("nan"))
            print(f"  -> {len(isl)}島 逐次中央 {med:.4f} "
                  f"({r.get('sec', 0):.0f}秒)")
            ok += 1
        except Exception as e:
            print(f"  !! 失敗 ({type(e).__name__}: {e})")
            ng += 1
        print(f"  ディスク: {shutil.disk_usage(EXT).free / 2**30:.0f}GB 空き")
    if pf is not None:
        pf.close()
    print(f"\n完了 成功{ok} / 失敗{ng}")


if __name__ == "__main__":
    main()
