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

    python -m recon3d.tools.recon inspect input=/data/reconstruction
    python -m recon3d.tools.recon run     input=/data/reconstruction
    python -m recon3d.tools.recon run     input=/data/reconstruction \\
        'steps=[chains]' chunking.chunk_size=24
    python -m recon3d.tools.recon steps

設定は `recon3d/conf/recon.yaml`。`key=value` で上書きできる。
セッション名は既定で入力ディレクトリ名。同じ入力を条件違いで回すときは
`session=myrun` を付けると出力が `results/myrun/` に分かれる。
"""
from typing import List

import typer

from recon3d.tools._hydra import load_cfg, to_plain

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


@app.command()
def run(overrides: List[str] = ARG):
    """設定で選んだ段を走らせる (前提の段は自動で補う)

    **このプロセスで完結する**ので、ポッドの PID 1 としてそのまま `exec`
    できる (SIGTERM が中継なしで届く)。失敗は例外のまま落として非ゼロ終了する。
    """
    d = _cfg(overrides)
    steps = list(d.pop("steps") or [])
    inp, sess = d.pop("input"), d.pop("session")
    from recon3d.api import run as api_run
    res = api_run(inp, steps=steps, session=sess, **d)
    typer.echo(f"\nsession {res.session}\n出力 {res.results_dir}")
    if res.skipped:
        typer.echo("skip: " + ", ".join(res.skipped))


@app.command()
def inspect(overrides: List[str] = ARG):
    """入力を点検し、何が走って何がskipされるかを表示する (実行しない)"""
    d = _cfg(overrides)
    from recon3d.api import plan as api_plan
    # 点検なので能力不足でも**止めない** (strict=False)。止めると
    # 「何が足りないか一覧したい」という用途で使えない
    order, skip = api_plan(d["input"], list(d.get("steps") or []),
                           d.get("session"), strict=False)
    from recon3d.core.caps import main as caps_main
    caps_main()
    typer.echo("")
    typer.echo("実行できる段: " + (" -> ".join(order) or "(なし)"))
    for k, v in skip.items():
        typer.echo(f"skip {k}: {v}")


@app.command()
def steps():
    """段の一覧と依存関係 (入力に依らないので設定は要らない)"""
    from recon3d.steps import describe
    typer.echo("\n".join(describe()))


def main():
    # k8s の SIGTERM は PID 1 にしか来ない。GPU推論の subprocess と
    # アンカーの fork ワーカーを孤児にしないよう、グループ全体へ中継する
    from recon3d.core.signals import forward_to_process_group
    forward_to_process_group()
    app()


if __name__ == "__main__":
    main()
