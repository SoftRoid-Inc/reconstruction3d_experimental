"""recon3d のテスト共通ヘルパ (pytest 非依存)

ホストに pytest が無くても困らないよう、検査は素のスクリプトとして書き
(`check_*.py`)、pytest があれば `test_checks.py` が同じ関数を呼ぶ。

検査を子プロセスで回すのは、かつて `recon3d.config` がプロセス単位で
凍結していた名残ではなく、**import 時の副作用を素の状態から観測する**
ため。親プロセスで一度 import してしまうと「import しただけで何が起きるか」
が測れない。
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]      # reconstruction3d_experimental/


def run_py(code, timeout=300, **env_extra):
    """`python -c code` を子プロセスで実行して (returncode, stdout, stderr)"""
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    env.update({k: str(v) for k, v in env_extra.items()})
    p = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def run_json(code, **kw):
    """子プロセスの**最終行**をJSONとして読む (前段のprintに邪魔されない)"""
    rc, out, err = run_py(code, **kw)
    if rc != 0:
        raise RuntimeError(f"exit {rc}\n{err[-3000:]}")
    lines = [ln for ln in out.splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError(f"出力なし\n{err[-3000:]}")
    return json.loads(lines[-1])


def fixture_dir():
    """回帰用の入力ディレクトリ (`RECON3D_FIXTURE` で指す)。無ければ None"""
    d = os.environ.get("RECON3D_FIXTURE")
    return Path(d) if d and Path(d).is_dir() else None
