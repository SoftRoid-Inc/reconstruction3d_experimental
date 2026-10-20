"""recon3d パイプラインの入口 (Docker イメージの ENTRYPOINT)

    python main.py inspect input=/data/reconstruction
    python main.py run     input=/data/reconstruction 'steps=[anchor,export]'
    python main.py steps

実体は `recon3d/tools/recon.py` (CLI=typer / 設定=hydra)。
**ここで `recon3d.*` をトップレベル import しないこと** — `recon3d.config`
は最初の import でセッションを凍結するので、`IMG_SESSION` が決まる前に
読み込むと既定の "0828" に固まる。`tools.recon` は config を巻き込まない
作りになっており、`tests/check_imports.py` の「import衛生」で固定している。
"""


def main():
    from recon3d.tools.recon import main as recon_main
    recon_main()


if __name__ == "__main__":
    main()
