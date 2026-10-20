"""外部入力ディレクトリからパイプラインを回す (CLI=typer / 設定=hydra)

S3から落とした 360°フレーム・人物マスク・IMU派生物・間取り図ラベルが
1つのディレクトリに揃っている前提で、**設定で選んだステップだけ**を走らせる。

    reconstruction/
      images/   frame_number_000000.jpg ...     必須
      masks/    frame_number_000000.jpg ...     人物マスク (白=有効)
      imu/      transitions.json idle_spans.json
      floors/   1F.json 1F.jpg 2F.json ... meta.json
      results/                                  ← 出力はここだけ

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
    """設定で選んだステップを走らせる (前提のステップは自動で補う)

    **このプロセスで完結する**ので、ポッドの PID 1 としてそのまま `exec`
    できる (SIGTERM が中継なしで届く)。失敗は例外のまま落として非ゼロ終了する。
    """
    d = _cfg(overrides)
    steps = list(d.pop("steps") or [])
    inp = d.pop("input")
    from recon3d.api import run as api_run
    res = api_run(inp, steps=steps, **d)
    typer.echo(f"\n入力 {res.root}\n出力 {res.results_dir}")
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
                           strict=False)
    from recon3d.core.caps import main as caps_main
    from recon3d.paths import from_input
    paths = from_input(d["input"])
    # argv=[] を渡すこと。既定の None だと caps_main 内の argparse が
    # sys.argv を読み、typer が食い終わった "inspect input=..." を
    # 自分への引数だと思って usage で落ちる
    caps_main(paths, [])                  # 能力の内訳をそのまま出す
    typer.echo("")
    typer.echo("実行できるステップ: " + (" -> ".join(order) or "(なし)"))
    for k, v in skip.items():
        typer.echo(f"skip {k}: {v}")


@app.command()
def steps():
    """ステップの一覧と依存関係 (入力に依らないので設定は要らない)"""
    from recon3d.steps import describe
    typer.echo("\n".join(describe()))


# 図面→3Dモデル (plan2model)。撮影データは要らず floors/ だけで走るので、
# パイプラインのステップではなく独立したサブコマンドにしてある
plan2model_app = typer.Typer(add_completion=False,
                             help="間取り図 → 壁・柱・開口・スラブの3Dモデル")
app.add_typer(plan2model_app, name="plan2model")


@plan2model_app.command("build")
def plan2model_build(overrides: List[str] = ARG):
    """floors/ から glb / obj / report.json を作る"""
    d = _cfg(overrides)
    from recon3d.paths import from_input
    from recon3d.plan2model.build import build_from_paths
    paths = from_input(d["input"])
    out = build_from_paths(paths)
    typer.echo(f"-> {out}")


@plan2model_app.command("verify")
def plan2model_verify(overrides: List[str] = ARG):
    """壁+柱の2D再投影を図面と比べて IoU とオーバーレイPNGを出す"""
    d = _cfg(overrides)
    from recon3d.paths import from_input
    from recon3d.plan2model.verify import verify_from_paths
    paths = from_input(d["input"])
    out = verify_from_paths(paths)
    typer.echo(f"-> {out}")


# 開発・調査用のモジュール CLI。argparse のまま各モジュールの
# main(paths, argv) へ委譲する (typer へ書き直さない)
TOOLS = {
    "ingest": "recon3d.core.ingest",
    "floor_assign": "recon3d.core.floor_assign",
    "stair_check": "recon3d.core.stair_check",
    "chunking": "recon3d.core.chunking",
    "chaining": "recon3d.core.chaining",
    "posegraph": "recon3d.core.posegraph",
    "island_group": "recon3d.core.island_group",
    "bridge": "recon3d.core.bridge",
    "caps": "recon3d.core.caps",
    "panovggt": "recon3d.core.panovggt",
    "crops": "recon3d.core.crops",
    "anchor": "recon3d.postproc.anchor",
    "corners": "recon3d.postproc.corners",
    "plan_icp": "recon3d.postproc.plan_icp",
    "plan_match": "recon3d.postproc.plan_match",
    "scale_prior": "recon3d.postproc.scale_prior",
    "hier_anchor": "recon3d.postproc.hier_anchor",
    "interp": "recon3d.postproc.interp",
    "yaw_fix": "recon3d.postproc.yaw_fix",
    "density_map": "recon3d.postproc.density_map",
    "colmap_prior": "recon3d.postproc.colmap_prior",
    "mapany": "recon3d.postproc.mapany",
    "export": "recon3d.out.export",
    "viewer": "recon3d.out.viewer",
    "meshing": "recon3d.mesh.meshing",
    "collision_mesh": "recon3d.mesh.collision_mesh",
    "idle": "recon3d.imu.idle",
    "transitions": "recon3d.imu.transitions",
    "timeline": "recon3d.imu.timeline",
    "flowstate_yaw": "recon3d.imu.flowstate_yaw",
}
TOOLS.update({n: f"recon3d.eval.{n}" for n in (
    "frontend", "chunk_cmp", "anchor_confidence", "anchor_oracle",
    "anchor_blame", "anchor_overlay", "bridge_eval", "bridge_view",
    "island_diag", "island_viewer", "overview", "error_view", "depth",
    "density_view", "floor_overlay", "outdoor_label", "outdoor_proxy",
    "scale_probe", "wall_band_probe")})


@app.command(context_settings={"allow_extra_args": True,
                               "ignore_unknown_options": True})
def tool(ctx: typer.Context,
         name: str = typer.Argument(None, help="ツール名 (省略で一覧)"),
         input: str = typer.Option(None, "--input", help="入力ディレクトリ")):
    """開発・調査用のモジュール CLI を呼ぶ

        python main.py tool --list
        python main.py tool posegraph --input /data/x --from-tag pv_s001
    """
    if not name:
        w = max(len(k) for k in TOOLS)
        for k in sorted(TOOLS):
            typer.echo(f"  {k:{w}}  {TOOLS[k]}")
        return
    if name not in TOOLS:
        raise typer.BadParameter(
            f"未知のツール {name!r} (一覧: python main.py tool)")
    if not input:
        raise typer.BadParameter("--input /data/reconstruction が要る")
    import importlib
    from recon3d.paths import from_input
    mod = importlib.import_module(TOOLS[name])
    if not hasattr(mod, "main"):
        raise typer.BadParameter(f"{TOOLS[name]} に main(paths, argv) が無い")
    mod.main(from_input(input), ctx.args)


def main():
    # k8s の SIGTERM は PID 1 にしか来ない。GPU推論の subprocess と
    # アンカーの fork ワーカーを孤児にしないよう、グループ全体へ中継する
    from recon3d.core.signals import forward_to_process_group
    forward_to_process_group()
    app()


if __name__ == "__main__":
    main()
