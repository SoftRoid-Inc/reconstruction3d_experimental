"""PanoVGGTを複数チャンクまとめて推論する (モデル読み込みを1回に)

`inference.py` をチャンクごとに起動すると、1チャンク22秒のうち
**12.2秒がプロセス起動とモデル読み込み**で占められる (2026-10-14実測)。
16フレームの実推論は9秒しかないので、固定コストが55%の無駄になっていた。

ここでは inference.py を**モジュールとしてimportし、`load_model` を
読み込み済みモデルを返す関数に差し替える**ことで、第三者コードを一切
改変せずにモデル読み込みを1回に抑える。出力形式は変わらない。

呼び出し側は `core.panovggt.run_chunks()` / `run_panovggt()`。
かつては専用 conda env の subprocess だったが、単一イメージ化で
**同一プロセス実行だけ**になった (2026-10-19)。
"""
from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
from typing import Any, Sequence

from recon3d.types import PathLike

_MODEL = {}          # (dir, config, ckpt, device) -> 読み込み済みモデル


@contextlib.contextmanager
def _in_panovggt(panovggt_dir: PathLike):
    """PanoVGGT のディレクトリを cwd と sys.path に入れる

    inference.py は相対パスで重み・設定を引くので chdir が要る。
    **同一プロセスなので必ず戻す** — subprocess 時代は無害だったが、
    戻さないと後続の段 (チェーン・アンカー) が相対パスを見失う。
    """
    cwd = os.getcwd()
    path = str(panovggt_dir)
    added = path not in sys.path
    if added:
        sys.path.insert(0, path)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(cwd)
        if added and sys.path and sys.path[0] == path:
            sys.path.pop(0)


def _load(panovggt_dir: PathLike, config: str, checkpoint: str,
          device: str):
    """モデルを読み込む (同じ組み合わせなら使い回す)"""
    key = (str(panovggt_dir), config, checkpoint, device)
    if key in _MODEL:
        return _MODEL[key]
    # PanoVGGT 側のモジュール。sys.path に入るのは _in_panovggt の中だけ
    import inference as inf  # type: ignore[import-not-found]
    t0 = time.time()
    _MODEL[key] = inf.load_model(config, checkpoint, device)
    print(f"[batch] モデル読み込み {time.time() - t0:.1f}秒", flush=True)
    return _MODEL[key]


def run_spec(chunks: Sequence[dict[str, Any]], panovggt_dir: PathLike,
             config: str, checkpoint: str,
             device: str = "cuda") -> None:
    """チャンク定義のリストを順に推論する

    chunks: [{"image_dir":…, "mask_dir":…|None, "output_dir":…}, …]
    """
    if not chunks:
        print("[batch] 推論すべきチャンクなし")
        return
    with _in_panovggt(panovggt_dir):
        import inference as inf  # type: ignore[import-not-found]
        model = _load(panovggt_dir, config, checkpoint, device)
        orig_load = inf.load_model
        inf.load_model = lambda *a, **k: model
        try:
            for i, c in enumerate(chunks):
                t = time.time()
                inf.main(argparse.Namespace(
                    config=config, checkpoint=checkpoint,
                    image_dir=c["image_dir"], mask_dir=c.get("mask_dir"),
                    output_dir=c["output_dir"], device=device,
                    log_depth=True))
                print(f"[batch] {i + 1}/{len(chunks)} {time.time() - t:.1f}秒",
                      flush=True)
        finally:
            inf.load_model = orig_load
