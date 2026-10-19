"""外部テストデータセットを大量に回す実走 (CLI=typer / 設定=hydra)

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
| **取込 (S3)** | **82** | **17%** |
| ポーズグラフ | 32 | 7% |
| 評価 | 0.4 | 0% |

**GPU (推論) と ネットワーク (S3) は別資源**なので**次の件のダウンロードを
現在の件の推論と重ねる** (`run.prefetch`、既定ON)。これで取込が隠れ1件あたり
約17%短縮する。subprocess の起動コストは**0.28秒で無視できる** — 実測せずに
当たりを付けると的を外す。

## 使い方

    python -m img_align.tools.ext_batch list
    python -m img_align.tools.ext_batch run  select.limit=10
    python -m img_align.tools.ext_batch report

設定は `img_align/conf/ext_batch.yaml`。`key=value` で上書きできる:

    ... run select.limit=20 select.skip=10 run.keep_chunks=true
    ... run pipeline.stride=10 pipeline.chunk_size=24
"""
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

import typer
from omegaconf import DictConfig, OmegaConf

from img_align.tools._hydra import CONF_DIR, ROOT  # noqa: F401
from img_align.tools._hydra import load_cfg as _load_cfg

app = typer.Typer(add_completion=False,
                  help="外部テストデータセットの大量実走")


def load_cfg(overrides: Optional[List[str]] = None,
             name: str = "ext_batch") -> DictConfig:
    """hydra設定を読む。overrides は `key=value` の列"""
    return _load_cfg(overrides, name)


def _p(cfg, key) -> Path:
    v = Path(cfg.paths[key])
    return v if v.is_absolute() else ROOT / v


def _sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


# ----------------------------------------------------------------- 候補選び

def candidates(cfg):
    """条件に合う候補 [(s3key, 枚数, フロア数)]"""
    src = _p(cfg, "src")
    rows = list(csv.DictReader(
        open(src / "trajectory_data_by_scan_data_id.csv")))
    info = json.loads((src / "s3_key_info.json").read_text())
    cam = {r[0]: r[2] for r in info if r[1] == cfg.select.camera}
    floors = {}
    for r in rows:
        k = r["metadata_s3_key"].rsplit("/data_number_0", 1)[0]
        if k in cam:
            floors.setdefault(k, set()).add(r["floor_id"])
    out = [(k, cam[k], len(f)) for k, f in floors.items()
           if cfg.select.min_frames <= cam[k] <= cfg.select.max_frames]
    if cfg.select.order == "floors_desc":
        out.sort(key=lambda x: (-x[2], x[1]))   # 情報量が多くて軽い順
    else:
        out.sort(key=lambda x: x[1])
    return out


def name_of(key):
    """s3キー → (ディレクトリ名, セッション名)"""
    p = key.split("/")
    return f"sfm{p[2]}_{p[3]}{p[4]}{p[5]}", f"e{p[2]}x{p[4]}{p[5]}"


# ------------------------------------------------------------------- 取り込み

def setup(cfg, key):
    """画像・マスク・図面・GT を揃えて meta.json を書く"""
    d, sess = name_of(key)
    out = _p(cfg, "ext") / d
    if (out / "meta.json").exists():
        return out, sess
    out.mkdir(parents=True, exist_ok=True)
    r = _sh(["aws", "s3", "sync", f"{cfg.paths.bucket}/{key}/", str(out),
             "--only-show-errors"])
    if r.returncode or not (out / "images").is_dir():
        raise RuntimeError(f"同期失敗: {r.stderr[:200]}")
    src = _p(cfg, "src")
    rows = [x for x in csv.DictReader(
        open(src / "trajectory_data_by_scan_data_id.csv"))
        if x["metadata_s3_key"].startswith(key)]
    tids = {x["trajectory_id"] for x in rows}
    fids = {x["floor_id"] for x in rows}
    pts = [p for p in csv.DictReader(open(src / "trajectory_points.csv"))
           if p["trajectory_id"] in tids]
    fl = [f for f in csv.DictReader(open(src / "floor_info.csv"))
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
                 cfg.paths.drawing_bucket
                 + f["image_url"].split("amazonaws.com/", 1)[1],
                 str(dst), "--only-show-errors"])
    p = key.split("/")
    (out / "meta.json").write_text(json.dumps(
        {"session": sess, "gt_session": p[4] + p[5], "dir": d, "s3key": key,
         "n_frames": len(list((out / "images").glob("*.jpg"))),
         "n_gt": len(pts), "floors": len(fids)}, indent=1))
    return out, sess


class Prefetcher:
    """次の件のS3取込を裏で走らせる (GPUが空かないので重ねられる)

    取込82秒 / 推論357秒なので**取込は完全に隠れる**。
    """

    def __init__(self, cfg, keys):
        self.cfg, self.keys, self.i, self.fut = cfg, list(keys), 0, None
        self.pool = ThreadPoolExecutor(max_workers=1)

    def next(self):
        """現在の件の取込完了を待ち、次の件の取込を開始する"""
        if self.i >= len(self.keys):
            return
        got = self.fut or self.pool.submit(setup, self.cfg, self.keys[self.i])
        self.i += 1
        self.fut = (self.pool.submit(setup, self.cfg, self.keys[self.i])
                    if self.i < len(self.keys) else None)
        got.result()

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)


# --------------------------------------------------------------------- 評価

_EVAL_CODE = r'''
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


def evaluate(sess):
    """島ごとに 逐次 / PG+ループ の GT残差 (図面幅で正規化) を測る"""
    r = _sh([sys.executable, "-c", _EVAL_CODE],
            env={**os.environ, "IMG_SESSION": sess, "PYTHONPATH": str(ROOT)})
    for ln in r.stdout.splitlines():
        if ln.startswith("@@"):
            return json.loads(ln[2:])
    raise RuntimeError(f"評価失敗: {r.stderr[-400:]}")


# ------------------------------------------------------------------- 1件処理

def process(cfg, key):
    """1件を通す。戻り値は result.json の中身

    **工程別の所要時間を必ず測る** — 最適化の的を外さないため。
    """
    t0, lap = time.time(), {}

    def tick(k, fn):
        s = time.time()
        r = fn()
        lap[k] = round(time.time() - s, 1)
        if cfg.run.verbose:
            typer.echo(f"    [{k}] {lap[k]}秒")
        return r

    out, sess = tick("取込", lambda: setup(cfg, key))
    res_p = out / "result.json"
    if res_p.exists():
        typer.echo(f"  [skip] {sess} は済み")
        return json.loads(res_p.read_text())
    env = {**os.environ, "IMG_SESSION": sess, "PYTHONPATH": str(ROOT)}
    argv = [sys.executable, "-m", "img_align.pipeline", "run",
            "--session", sess, "--segments", str(cfg.pipeline.segments),
            "--max-seg", str(cfg.pipeline.max_seg), "--no-eval"]
    for opt, val in (("--stride", cfg.pipeline.stride),
                     ("--chunk-size", cfg.pipeline.chunk_size)):
        if val is not None:
            argv += [opt, str(val)]
    r = tick("推論+チェーン", lambda: _sh(argv, env=env, cwd=str(ROOT)))
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

    loops = tick("ポーズグラフ", _pg) if cfg.pipeline.posegraph else {}
    res = tick("評価", lambda: evaluate(sess))
    # **どの設定で得た数字か**を結果自体に残す (レポートで同じ問題に当たった)
    res.update({"session": sess, "s3key": key, "loops": loops,
                "sec": round(time.time() - t0, 1), "lap": lap,
                "cfg": OmegaConf.to_container(cfg.pipeline, resolve=True)})
    res_p.write_text(json.dumps(res))
    if not cfg.run.keep_chunks:   # 評価が済んだら素材を捨てる (ディスク律速)
        for sub in ("images", "masks", f"results/{sess}/pcd"):
            shutil.rmtree(out / sub, ignore_errors=True)
    return res


# --------------------------------------------------------------------- CLI

@app.command("list")
def cmd_list(overrides: List[str] = typer.Argument(
        None, help="hydra上書き (例 select.limit=20)")):
    """候補を一覧する"""
    cfg = load_cfg(overrides)
    sel = candidates(cfg)[cfg.select.skip:cfg.select.skip + cfg.select.limit]
    typer.echo(f"{'s3key':52s} {'枚数':>5s} {'階':>3s}  名前")
    for k, n, f in sel:
        typer.echo(f"{k:52s} {n:5d} {f:3d}  {name_of(k)[1]}")


@app.command("run")
def cmd_run(overrides: List[str] = typer.Argument(
        None, help="hydra上書き (例 select.limit=10 run.keep_chunks=true)")):
    """実走する"""
    import numpy as np
    cfg = load_cfg(overrides)
    sel = candidates(cfg)[cfg.select.skip:cfg.select.skip + cfg.select.limit]
    typer.echo(OmegaConf.to_yaml({"select": cfg.select,
                                  "pipeline": cfg.pipeline, "run": cfg.run}))
    pf = Prefetcher(cfg, [k for k, _, _ in sel]) if cfg.run.prefetch else None
    ok = ng = 0
    for i, (k, n, f) in enumerate(sel, 1):
        typer.echo(f"\n===== [{i}/{len(sel)}] {k} ({n}枚 {f}フロア)")
        if pf is not None:
            pf.next()
        try:
            r = process(cfg, k)
            isl = r["islands"]
            med = (np.median([x["seq"]["resid"] for x in isl]) if isl
                   else float("nan"))
            typer.echo(f"  -> {len(isl)}島 逐次中央 {med:.4f} "
                       f"({r.get('sec', 0):.0f}秒)")
            ok += 1
        except Exception as e:
            typer.secho(f"  !! 失敗 ({type(e).__name__}: {e})",
                        fg=typer.colors.RED)
            ng += 1
        free = shutil.disk_usage(_p(cfg, "ext")).free / 2 ** 30
        typer.echo(f"  ディスク: {free:.0f}GB 空き")
    if pf is not None:
        pf.close()
    typer.secho(f"\n完了 成功{ok} / 失敗{ng}",
                fg=typer.colors.GREEN if not ng else typer.colors.YELLOW)


@app.command("report")
def cmd_report(overrides: List[str] = typer.Argument(None)):
    """result.json を集計する"""
    import statistics as st
    cfg = load_cfg(overrides)
    rows = [json.loads(p.read_text())
            for p in sorted(_p(cfg, "ext").glob("*/result.json"))]
    typer.echo(f"{len(rows)}件の結果を集計")
    seq = [i["seq"]["resid"] for r in rows for i in r["islands"]]
    pg = [i["pg"]["resid"] for r in rows for i in r["islands"] if i.get("pg")]
    if seq:
        typer.echo(f"  島{len(seq)}本  逐次 {st.median(seq):.4f}"
                   + (f" -> PG {st.median(pg):.4f}" if pg else ""))
    p = ROOT / cfg.report.data
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({r["session"]: r for r in rows}))
    typer.echo(f"-> {p}")


if __name__ == "__main__":
    app()
