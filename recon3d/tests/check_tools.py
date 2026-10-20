"""`main.py tool <name>` の全ツールが呼べることを確認する

  RECON3D_FIXTURE=/data/recon_0912 python -m recon3d.tests.check_tools

各ツールに `-h` を渡して argparse の構築まで通す。これで拾えるのは:

- `main(paths, argv)` でない署名 (ディスパッチャは `ctx.args` を第2引数に
  渡すので、`main(paths, sfx="")` だとリストが値として流れ込む)
- `parse_args()` に `argv` を渡し忘れ (親の `sys.argv` を食って誤動作する)
- `help` 文字列の `%` 取りこぼし (argparse が書式化するので `%%` が要る)

いずれも「実行して初めて落ちる」種類で、import スモークでは出ない。
"""
import contextlib
import importlib
import io
import sys

from recon3d.paths import from_input
from recon3d.tests.conftest import fixture_dir

# argparse を持たない (引数を取らない) ツール。`-h` でも本来の処理が走る
NO_ARGPARSE = set()


def failures(root):
    """[(ツール名, 理由)] を返す ([] なら健全)"""
    from recon3d.tools.recon import TOOLS
    paths = from_input(root)
    bad = []
    for name in sorted(TOOLS):
        try:
            mod = importlib.import_module(TOOLS[name])
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                try:
                    mod.main(paths, ["-h"])
                except SystemExit as e:
                    if e.code not in (0, None):
                        raise RuntimeError(f"exit {e.code}")
            if name not in NO_ARGPARSE and "usage:" not in buf.getvalue():
                bad.append((name, "-h で usage が出ない"))
        except Exception as e:                       # noqa: BLE001
            bad.append((name, f"{type(e).__name__}: {str(e)[:150]}"))
    return bad


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else fixture_dir()
    if root is None:
        print("[SKIP] RECON3D_FIXTURE 未設定 (入力ディレクトリを指すこと)")
        return 0
    from recon3d.tools.recon import TOOLS
    bad = failures(root)
    print(f"[{'OK' if not bad else 'NG'}] tool -h: "
          f"失敗 {len(bad)} / 全 {len(TOOLS)}")
    for n, why in bad:
        print(f"    {n}: {why}")
    return bool(bad)


if __name__ == "__main__":
    sys.exit(main())
