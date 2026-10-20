"""MapAnythingポーズ条件付けの起動 (コンテナ内なら実行、外なら手順を出す)

`postproc/mapany` 本体は mapanything コンテナでしか動かないので、
オーケストレーション側だけ分けてある (ホストで import しても落ちない)。
"""
from __future__ import annotations

from recon3d.paths import Paths
from recon3d.types import Manifest
import argparse


def run_mapany(paths: Paths, manifest: Manifest) -> None:
    try:
        import mapanything  # type: ignore[import-not-found] # noqa: F401
        in_container = True
    except ImportError:
        in_container = False
    for m in manifest:
        mtag = m["tag"].replace("pv", "macond", 1)
        n = m["b"] - m["a"] + 1
        stride = max(1, -(-n // 56))      # 1チャンク56ビューが16GBの上限
        if in_container:
            from recon3d.postproc.mapany import run_infer
            run_infer(paths, argparse.Namespace(
                tag=mtag, floor=m["floor"], frames=f"{m['a']}:{m['b']}",
                stride=stride, minibatch=1, yaws="0", res=None,
                pose_cond=m["tag"], dump_local=False, nc=False))
        else:
            print(f"[mapany] コンテナ外。次を実行してください:\n"
                  f"  docker compose exec -e RECON_INPUT={paths.root} "
                  f"-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
                  f"mapanything python -m recon3d.postproc.mapany infer "
                  f"--tag {mtag} --floor {m['floor']} "
                  f"--frames {m['a']}:{m['b']} --stride {stride} "
                  f"--minibatch 1 --pose-cond {m['tag']}")
