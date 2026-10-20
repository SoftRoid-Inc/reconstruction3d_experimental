"""外部から呼ぶためのエントリポイント

    from recon3d import api
    res = api.run("/data/reconstruction", steps=["anchor"],
                  chunking={"max_seg": 160})

入力は**固定構造の1ディレクトリ**で、それがデータセットそのもの。
`Paths` を組み立てて各ステップへ引数で配るだけなので、環境変数もモジュール凍結も
使わない — 渡し忘れは `TypeError` で即座に出る。

1ポッド=1データセット (Kubernetes / Dagster Pipes) を前提にしているので、
子プロセスも立てない。同一プロセスから**別の**データセットを回したい場合も
`Paths` を作り直すだけでよい。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from recon3d.paths import from_input
from recon3d.types import PathLike


@dataclass
class RunResult:
    root: str                    # 入力ディレクトリ
    results_dir: str             # <root>/results
    steps: list[str] = field(default_factory=list)      # 実際に走らせた指定
    # 能力不足で飛ばしたステップと理由 (ステップ名 -> 理由)
    skipped: dict[str, str] = field(default_factory=dict)
    islands: list[str] = field(default_factory=list)    # 最終の島/群れタグ


def run(input_dir: PathLike, steps: Sequence[str] | None = None,
        **opt: Any) -> RunResult:
    """ステップを実行する。失敗は例外のまま伝播させる

    能力不足で実行できないステップがあれば `steps.plan` が `SystemExit` を上げる
    (メッセージ付きで非ゼロ終了するので、pipe 側が失敗として扱える)。
    """
    from recon3d.steps import DEFAULT_STEPS, execute
    paths = from_input(input_dir)
    steps = list(steps or DEFAULT_STEPS)
    ctx = execute(paths, steps, opt)
    return RunResult(
        root=str(paths.root), results_dir=str(paths.results), steps=steps,
        skipped=ctx.skipped,
        islands=[m["tag"] for m in (ctx.manifest or [])])


def plan(input_dir: PathLike, steps: Sequence[str] | None = None,
         strict: bool = True) -> tuple[list[str], dict[str, str]]:
    """実行せずに「何が走って何がskipされるか」を返す

    `strict=False` は点検用 — 能力不足で止めずに skip 扱いで報告する。

    返り値は (実行順, {skipしたステップ: 理由})。
    """
    from recon3d.steps import DEFAULT_STEPS
    from recon3d.steps import plan as _plan
    paths = from_input(input_dir)
    return _plan(paths, list(steps or DEFAULT_STEPS), strict=strict)
