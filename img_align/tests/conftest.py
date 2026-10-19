"""img_align のテスト共通ヘルパ (pytest 非依存)

`img_align.config` はプロセス単位で凍結する (約90モジュールが
`from img_align.config import X` で値を束ねるため `reload` では戻らない)。
よってセッションに依存する検査は**すべて子プロセス**で行う。

ホストに pytest が無くても困らないよう、検査は素のスクリプトとして書き
(`checks/`)、pytest があれば `test_checks.py` が同じ関数を呼ぶ。
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]      # reconstruction_3d/
SNAP_DIR = Path(__file__).resolve().parent / "snapshots"


def run_py(code, session=None, timeout=300, **env_extra):
    """`python -c code` を子プロセスで実行して (returncode, stdout, stderr)"""
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    env.pop("IMG_SESSION", None)
    if session is not None:
        env["IMG_SESSION"] = str(session)
    env.update({k: str(v) for k, v in env_extra.items()})
    p = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def run_json(code, session=None, **kw):
    """子プロセスの**最終行**をJSONとして読む (前段のprintに邪魔されない)"""
    rc, out, err = run_py(code, session=session, **kw)
    if rc != 0:
        raise RuntimeError(f"exit {rc}\n{err[-3000:]}")
    lines = [ln for ln in out.splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError(f"出力なし\n{err[-3000:]}")
    return json.loads(lines[-1])


def known_sessions():
    """いまレジストリに居るセッション名 (ext/ の有無で変わる)"""
    return run_json("import json, img_align.config as c; "
                    "print(json.dumps(sorted(c.PROPERTY)))")
