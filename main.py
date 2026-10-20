"""recon3d パイプラインの入口 (Docker イメージの ENTRYPOINT)

    python main.py inspect input=/data/reconstruction
    python main.py run     input=/data/reconstruction 'steps=[anchor,export]'
    python main.py steps                        # ステップの一覧と依存
    python main.py tool                         # 開発・調査用CLIの一覧
    python main.py plan2model build input=/data/x

実体は `recon3d/tools/recon.py` (CLI=typer / 設定=hydra)。

**ここで `recon3d.*` をトップレベル import しないこと。** 重い依存 (torch 等)
を `--help` だけで引きずり込まないため、各サブコマンドが必要になった時点で
import する作り。`tests/check_imports.py` の「import衛生」で固定している。
"""


def main():
    from recon3d.tools.recon import main as recon_main
    recon_main()


if __name__ == "__main__":
    main()
