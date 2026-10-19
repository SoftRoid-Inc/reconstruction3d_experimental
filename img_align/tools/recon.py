"""外部入力ディレクトリからパイプラインを回す (CLI=typer / 設定=hydra)

S3から落とした 360°フレーム・人物マスク・IMU派生物・間取り図ラベルが
1つのディレクトリに揃っている前提で、**設定で選んだ段だけ**を走らせる。

    reconstruction/
      images/   frame_number_000000.jpg ...     必須
      masks/    frame_number_000000.jpg ...     人物マスク (白=有効)
      imu/      transitions.json idle_spans.json
      floors/   1F.json 1F.jpg 2F.json ... meta.json
      results/<session>/                        ← 出力はここだけ

**図面の実寸は `floors/meta.json` に書く**。壁注釈 (labelme) だけでは
px→m が決まらず図面アンカーは動かない:

    {"known_width_mm": {"1F": 13877.5, "2F": 13877.5, "3F": 11375.0}}

使い方:

    python -m img_align.tools.recon inspect input=/data/reconstruction
    python -m img_align.tools.recon run     input=/data/reconstruction
    python -m img_align.tools.recon run     input=/data/reconstruction \\
        'steps=[chains]' chunking.chunk_size=24
    python -m img_align.tools.recon steps

設定は `img_align/conf/recon.yaml`。`key=value` で上書きできる。
セッション名は既定で入力ディレクトリ名。同じ入力を条件違いで回すときは
`session=myrun` を付けると出力が `results/myrun/` に分かれる。
"""
import json
import os
import subprocess
import sys
from typing import List

import typer

from img_align.tools._hydra import ROOT, load_cfg, to_plain

app = typer.Typer(add_completion=False, help=__doc__)

ARG = typer.Argument(None, help="hydra上書き (例 input=/data/x steps=[anchor])")


def _cfg(overrides):
    cfg = load_cfg(overrides, name="recon")
    if not cfg.input:
        raise typer.BadParameter(
            "input= が要る (例 input=/data/reconstruction)")
    d = to_plain(cfg)
    d.pop("hydra", None)
    return d


def _session_ref(d):
    from img_align.api import session_ref
    return session_ref(d["input"], d.get("session"))


def _child(code, ref, check=True):
    """IMG_SESSION を立てた子プロセスで実行する

    config はプロセス単位で凍結するので、セッションを跨ぐ処理は必ず分ける。
    """
    env = {**os.environ, "PYTHONPATH": str(ROOT), "IMG_SESSION": ref}
    p = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env)
    if check and p.returncode:
        raise typer.Exit(p.returncode)
    return p.returncode


@app.command()
def run(overrides: List[str] = ARG):
    """設定で選んだ段を走らせる (前提の段は自動で補う)"""
    d = _cfg(overrides)
    steps = list(d.pop("steps") or [])
    inp, sess = d.pop("input"), d.pop("session")
    from img_align.api import run as api_run
    res = api_run(inp, steps=steps, session=sess, **d)
    if res.returncode:
        raise typer.Exit(res.returncode)
    typer.echo(f"\nsession {res.session}\n出力 {res.results_dir}")
    if res.skipped:
        typer.echo("skip: " + ", ".join(res.skipped))


@app.command()
def inspect(overrides: List[str] = ARG):
    """入力を点検し、何が走って何がskipされるかを表示する (実行しない)"""
    d = _cfg(overrides)
    steps = json.dumps(list(d.get("steps") or []))
    # 点検なので能力不足でも**止めない** (strict=False)。止めると
    # 「何が足りないか一覧したい」という用途で使えない
    _child(
        "from img_align.core.caps import main as caps_main;"
        "from img_align.steps import plan;"
        "caps_main();"
        f"order, skip = plan({steps}, strict=False);"
        "print();"
        "print('実行できる段:', ' -> '.join(order) or '(なし)');"
        "[print(f'skip {k}: {v}') for k, v in skip.items()]",
        _session_ref(d))


@app.command()
def steps(overrides: List[str] = ARG):
    """段の一覧と依存関係"""
    from img_align.steps import STEPS
    w = max(len(s) for s in STEPS)
    for s in STEPS.values():
        dep = f"  <- {','.join(s.needs)}" if s.needs else ""
        cap = f"  [要 {','.join(s.caps)}]" if s.caps else ""
        typer.echo(f"{s.name:{w}}  {s.doc}{dep}{cap}")


def main():
    app()


if __name__ == "__main__":
    main()
