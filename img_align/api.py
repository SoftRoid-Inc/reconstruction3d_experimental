"""外部から呼ぶためのエントリポイント

    from img_align import api
    api.run("/data/reconstruction", steps=["anchor"], chunking={"max_seg": 160})

`img_align.config` は**プロセス単位で凍結**する (約90モジュールが
`from img_align.config import X` で値を束ねるので reload では戻らない)。
よって 1プロセス = 1データセット。`run()` は必ず子プロセスを立てる。
同一プロセスで済ませたい場合 (CLI 等、config を未 import の状態) は
`run_inprocess()` を使う。
"""
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class RunResult:
    session: str
    results_dir: str
    steps: list = field(default_factory=list)
    skipped: dict = field(default_factory=dict)
    islands: list = field(default_factory=list)
    returncode: int = 0


def run_inprocess(input_dir=None, steps=None, session=None, **opt):
    """このプロセスで実行する。config 未 import か、同じ入力であること"""
    from img_align.steps import DEFAULT_STEPS, execute
    _freeze(input_dir, session)
    import img_align.config as cfg
    ctx = execute(list(steps or DEFAULT_STEPS), opt)
    return RunResult(
        session=cfg.SESSION, results_dir=str(cfg.RESULTS_DIR),
        steps=[s for s in (steps or DEFAULT_STEPS)], skipped=ctx.skipped,
        islands=[m["tag"] for m in (ctx.manifest or [])])


def plan(input_dir=None, steps=None, session=None):
    """実行せずに「何が走って何がskipされるか」を返す"""
    from img_align.steps import DEFAULT_STEPS
    from img_align.steps import plan as _plan
    _freeze(input_dir, session)
    order, skip = _plan(list(steps or DEFAULT_STEPS))
    return order, skip


def run(input_dir=None, steps=None, session=None, python=None, **opt):
    """子プロセスで実行する (呼び出し元の config を汚さない)"""
    import tempfile
    with tempfile.NamedTemporaryFile("r", suffix=".json") as fh:
        payload = {"input": str(input_dir) if input_dir else None,
                   "steps": list(steps or []), "session": session,
                   "opt": opt, "summary_out": fh.name}
        env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT),
               "IMG_SESSION": session_ref(input_dir, session)}
        p = subprocess.run([python or sys.executable, "-m", "img_align.api",
                            json.dumps(payload)], cwd=PROJECT_ROOT, env=env)
        if p.returncode:
            return RunResult(session=session or "", results_dir="",
                             steps=list(steps or []), returncode=p.returncode)
        return RunResult(**json.loads(fh.read()))


def session_ref(input_dir, session):
    """子に渡す IMG_SESSION の値

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
    """config を import する**前**に IMG_SESSION を決める"""
    ref = session_ref(input_dir, session) if (input_dir or session) \
        else os.environ.get("IMG_SESSION")
    if "img_align.config" in sys.modules:
        import img_align.config as cfg
        if ref and cfg.SESSION_REF != ref:
            raise RuntimeError(
                f"config は既に '{cfg.SESSION_REF}' で凍結済み。'{ref}' を回す"
                "には別プロセスが要る (api.run を使ってください)")
        return
    if not ref:
        raise ValueError("input_dir か session のどちらかが要る")
    os.environ["IMG_SESSION"] = ref


def _main():
    payload = json.loads(sys.argv[1])
    res = run_inprocess(payload["input"], payload["steps"] or None,
                        payload["session"], **payload["opt"])
    out = {"session": res.session, "results_dir": res.results_dir,
           "steps": res.steps, "skipped": res.skipped, "islands": res.islands}
    if payload.get("summary_out"):
        Path(payload["summary_out"]).write_text(
            json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    _main()
