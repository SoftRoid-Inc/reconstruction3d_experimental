"""img_align パイプラインの入口 (Docker イメージの ENTRYPOINT)

    python main.py inspect input=/data/reconstruction
    python main.py run     input=/data/reconstruction 'steps=[anchor,export]'
    python main.py steps

実体は `img_align/tools/recon.py` (CLI=typer / 設定=hydra)。
**ここで `img_align.*` をトップレベル import しないこと** — `img_align.config`
は最初の import でセッションを凍結するので、`IMG_SESSION` が決まる前に
読み込むと既定の "0828" に固まる。`tools.recon` は config を巻き込まない
作りになっており、`tests/check_imports.py` の「import衛生」で固定している。
"""


def main():
    from img_align.tools.recon import main as recon_main
    recon_main()


if __name__ == "__main__":
    main()
