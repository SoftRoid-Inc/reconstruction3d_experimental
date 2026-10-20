"""入口プロセスが SIGTERM を子孫へ伝える

Kubernetes はポッド終了時 **PID 1 にだけ** SIGTERM を送る。Python の既定では
それを受けた入口が死ぬだけで、子孫は孤児化して走り続ける (実測):

  - `core/panovggt.py` が別 env の python で起こす **GPU推論の subprocess**
  - `postproc/anchor_run.py` の `ProcessPoolExecutor` の fork ワーカー
  - `core/chunking.py` の `docker compose exec`

GPU を掴んだまま孤児が残ると、`terminationGracePeriod` の SIGKILL まで
VRAM が解放されず、Dagster 側のキャンセルは成功したように見える。

対策はシグナルを**プロセスグループ全体**へ中継すること。CLI の入口
(`tools/recon.py` / `pipeline.py`) で1回呼ぶ。ライブラリ側 (`api.run`) では
呼ばない — シグナルハンドラの差し替えは呼び出し元の領分だし、メインスレッド
でしか登録できない。
"""
import os
import signal


def forward_to_process_group(sig=signal.SIGTERM):
    """`sig` を受けたらプロセスグループ全体へ中継してから死ぬ

    グループリーダーのときだけ働く。リーダーでない (別のラッパーに包まれて
    いる) 場合は、無関係な兄弟プロセスまで巻き込むので何もしない。
    ポッドの PID 1 でも、シェルのジョブとして起動したときもリーダーになる。
    """
    if os.getpid() != os.getpgrp():
        return False

    def handler(signum, _frame):
        # 自分も同じグループに居るので、既定動作に戻してから撒く
        signal.signal(signum, signal.SIG_DFL)
        os.killpg(os.getpgrp(), signum)

    signal.signal(sig, handler)
    return True
