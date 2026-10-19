"""VGGT軌跡パイプライン (開発用CLI。本体のステップ定義は img_align/steps.py)

本番・外部データ向けの新インタフェースは `img_align/tools/recon.py`
(typer + hydra、入力ディレクトリを渡す形) を使ってください。こちらは
既存7セッションの開発フローと tools/ext_batch.py の156件が乗っている
互換CLIで、引数を steps.py のオプションへ翻訳するだけ。

段 (steps.py の登録表):
  prep   : 人物マスク確認 + 入力の imu/ を results へ取り込み
  chains : GT/IMU区間が必要とするチャンクのみ推論 (既存スキップ) → チェーン
  eval   : GT軌跡とのフィットATE + compare図 (--no-evalで省略)

任意後処理 (--post で選択、カンマ区切り):
  yaw_fix : チャンク内フレーム回転の修復 (方位ロック撮影のみ有効)
  posegraph: 島をポーズグラフ+ループ拘束で解き直す (→pg_タグ。以降の段が使う)
  mapany  : MapAnythingポーズ条件付け (→macondタグ。要mapanythingコンテナ)
  interp  : macond制御点×pvチェーン形状の全フレーム補間 (→mafタグ)
  bridge  : 橋渡しで島を群れに結合 (以降の後処理は群れを対象にする)
  anchor  : コーナーRANSACで図面アンカー (→anchor_<tag>_corners.json)
            ※--segments imu では階ラベルが要る。backend/floor_assign.json を
              置くと自動で流し込まれる (core/floor_assign --template で雛形)
  hier    : 分割サブ島へ親アンカーを移送+壁ICP (→anchor_<tag>_hier.json)
  export  : アンカー済み点群を図面座標へ書き出し (aligned_house_<sess>_auto.ply)
  viewer  : 確認画像+WebGLビューア生成 (export と同時)

**順序は書いた順ではなく依存で決まる** (steps.py)。`--post anchor,bridge`
と書いても bridge が先に走る。前提の段も自動で補われる。

使い方:
  python -m img_align.pipeline run --session 0616 --max-seg 160 \\
      --post anchor,hier,export,viewer
  python -m img_align.pipeline stages      # 段の説明を表示
"""
import argparse
import os
import sys

# --post の語彙 (yaw_fix / viewer は段ではなくオプション)
POSTS = ("yaw_fix", "posegraph", "mapany", "interp", "bridge", "anchor",
         "hier", "export", "viewer")
OPTION_POSTS = {"yaw_fix": "chunking.yaw_fix", "viewer": "export.viewer"}


def _opts(args):
    """argparse の結果を steps.py のオプション辞書へ翻訳"""
    return {
        "segments": args.segments,
        "islands": args.islands,
        "chunking": {
            "max_seg": args.max_seg, "stride": args.stride,
            "chunk_size": args.chunk_size, "model": args.model,
            "manifest_out": args.manifest_out, "use_idle": not args.no_idle,
            "yaw_fix": "yaw_fix" in args.post,
            "crops": "mapany" in args.post,
        },
        "bridge": {"modes": args.bridge_modes, "thresh": args.bridge_thresh,
                   "skip_cross_floor": args.skip_cross_floor},
        "anchor": {"jobs": args.jobs},
        "export": {"viewer": "viewer" in args.post},
    }


def main():
    from img_align.core.signals import forward_to_process_group
    forward_to_process_group()
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("run")
    sp.add_argument("--session", required=True,
                    help="IMG_SESSION (例 0616)。データセットディレクトリの"
                         "パスも可")
    sp.add_argument("--max-seg", type=int, default=160,
                    help="島の最大フレーム数 (長島分割。0で無効)")
    sp.add_argument("--islands", nargs="*", default=None,
                    help="後処理の対象タグを限定")
    sp.add_argument("--post", default="",
                    help=f"任意後処理 (カンマ区切り): {','.join(POSTS)}")
    sp.add_argument("--segments", default="gt", choices=("gt", "imu"),
                    help="島区間ソース: gt=GT軌跡CSV(開発) / "
                         "imu=IMU階段検出(本番・GTフリー)")
    sp.add_argument("--stride", type=int, default=None,
                    help="チャンクストライド (既定12。8で重なり倍増・接続安定)")
    sp.add_argument("--model", default="pv", choices=("pv", "pi3", "holo"),
                    help="前段モデル: pv=PanoVGGT(既定) / pi3 / "
                         "holo=Pi3-Finetuned-Holo360D (要pi3コンテナ)")
    sp.add_argument("--manifest-out", default=None,
                    help="既定は <model>_manifest.json")
    sp.add_argument("--no-idle", action="store_true",
                    help="静止区間での島切断を無効化 (対照実験用)")
    sp.add_argument("--chunk-size", type=int, default=None,
                    help="1チャンクのフレーム数 (既定16)。大きくすると接続数が"
                         "減り、累積スケール誤差が効きにくくなる")
    sp.add_argument("--bridge-modes", nargs="*", default=None,
                    help="橋の方式 (既定 n8s1 n8s2 n8s4)。先頭が採用する変換")
    sp.add_argument("--bridge-thresh", type=float, default=None,
                    help="方式間の一致度[m]がこれ未満の切れ目だけ繋ぐ")
    sp.add_argument("--skip-cross-floor", action="store_true",
                    help="階をまたぐ切れ目は橋を試さない (推論を約37%% 節約)")
    sp.add_argument("--jobs", type=int, default=None,
                    help="アンカーの並列数 (既定 CPU数/4。島ごとに独立)")
    sp.add_argument("--no-eval", action="store_true")
    sub.add_parser("stages")
    args = ap.parse_args()

    if args.cmd == "stages":
        print(__doc__)
        from img_align.steps import main as steps_main
        print()
        steps_main([])
        return

    args.post = [p for p in args.post.split(",") if p]
    bad = set(args.post) - set(POSTS)
    if bad:
        sys.exit(f"未知の後処理: {bad} (有効: {POSTS})")
    # configはimport時にIMG_SESSIONを凍結するため、必ずimportより先に設定する
    os.environ["IMG_SESSION"] = args.session
    if args.max_seg == 0:
        args.max_seg = None

    steps = ["chains"] + [p for p in args.post if p not in OPTION_POSTS]
    if not args.no_eval:
        steps.append("eval")
    from img_align.steps import execute
    execute(steps, _opts(args))


if __name__ == "__main__":
    main()
