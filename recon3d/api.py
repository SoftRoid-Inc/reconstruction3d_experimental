"""外部から呼ぶためのエントリポイント

    from recon3d import api
    res = api.run("/data/reconstruction", steps=["anchor"],
                  chunking={"max_seg": 160})

## プロセスは分けない

`recon3d.config` は最初の import でセッションを凍結し、137箇所が
`from recon3d.config import X` で値を**束ねる** (reload では戻らない)。
つまり **1プロセス = 1データセット**。

この制約は「同一プロセス内で2つのデータセットが混ざる」事故を防ぐためのもので、
本番 (Kubernetes で1ポッド=1データセット、Dagster Pipes で起動) では
**ポッド境界がすでに分離を担保している**。だからここで子プロセスを立てる
必要はない。むしろ立てると:

  - k8s の SIGTERM は PID 1 にしか行かないので、子が孤児化して GPU を
    掴んだまま走り続ける (親は即死するのでキャンセルは成功して見える)
  - 失敗が終了コードになり、呼び出し元が見落とすと成功扱いになる

同一プロセスから**別の**データセットを回したくなったら、ここに分岐を足さず
`tools/ext_batch.py` と同じく `python -m recon3d.tools.recon run input=...`
を子プロセスで起こすこと。

## 入口の規則

`IMG_SESSION` が決まる前に `recon3d.*` を import しないこと。
Dagster Pipes の入口なら、top-level ではなく `open_dagster_pipes()` の中で
import して `run()` を呼ぶ。破った場合は `_freeze` が
`RuntimeError: config は既に '...' で凍結済み` で落とす (黙って進まない)。
"""
import os
import sys
from dataclasses import dataclass, field


@dataclass
class RunResult:
    session: str
    results_dir: str
    steps: list = field(default_factory=list)
    skipped: dict = field(default_factory=dict)   # 能力不足で飛ばした段と理由
    islands: list = field(default_factory=list)


def run(input_dir=None, steps=None, session=None, **opt) -> RunResult:
    """段を実行する。失敗は例外のまま伝播させる

    能力不足で実行できない段があれば `steps.plan` が `SystemExit` を上げる
    (メッセージ付きで非ゼロ終了するので、pipe 側が失敗として扱える)。
    """
    from recon3d.steps import DEFAULT_STEPS, execute
    _freeze(input_dir, session)
    import recon3d.config as cfg
    steps = list(steps or DEFAULT_STEPS)
    ctx = execute(steps, opt)
    return RunResult(
        session=cfg.SESSION, results_dir=str(cfg.RESULTS_DIR), steps=steps,
        skipped=ctx.skipped,
        islands=[m["tag"] for m in (ctx.manifest or [])])


def plan(input_dir=None, steps=None, session=None, strict=True):
    """実行せずに「何が走って何がskipされるか」を返す

    `strict=False` は点検用 — 能力不足で止めずに skip 扱いで報告する。
    """
    from recon3d.steps import DEFAULT_STEPS
    from recon3d.steps import plan as _plan
    _freeze(input_dir, session)
    return _plan(list(steps or DEFAULT_STEPS), strict=strict)


def session_ref(input_dir, session):
    """`IMG_SESSION` に入れる値

    config は "名前"・"パス"・"名前=パス" の3形を受ける。環境変数を2本に
    増やさないための意図的な設計 (tools/ext_batch.py の
    `{**os.environ, "IMG_SESSION": sess}` が自然に勝つ)。
    """
    if input_dir and session:
        return f"{session}={input_dir}"
    if input_dir:
        return str(input_dir)
    if session:
        return str(session)
    raise ValueError("input_dir か session のどちらかが要る")


def _freeze(input_dir, session):
    """config を import する**前**に `IMG_SESSION` を決める

    既に別のセッションで凍結済みなら落とす。1プロセス1データセットなので
    ここで黙って続けると、別物件の成果物を読み書きして**例外なしに**
    完走してしまう (2026-10-09 に実際に踏んだ形)。
    """
    ref = session_ref(input_dir, session) if (input_dir or session) \
        else os.environ.get("IMG_SESSION")
    if "recon3d.config" in sys.modules:
        import recon3d.config as cfg
        if ref and cfg.SESSION_REF != ref:
            raise RuntimeError(
                f"config は既に '{cfg.SESSION_REF}' で凍結済みなので "
                f"'{ref}' は回せない。IMG_SESSION を決める前に recon3d を "
                "import していないか確認してください "
                "(別データセットを回すなら別プロセスで)")
        return
    if not ref:
        raise ValueError("input_dir か session のどちらかが要る")
    os.environ["IMG_SESSION"] = ref
