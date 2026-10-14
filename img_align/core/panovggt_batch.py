"""PanoVGGTを複数チャンクまとめて推論する (モデル読み込みを1回に)

`inference.py` をチャンクごとにsubprocess起動すると、1チャンク22秒のうち
**12.2秒がプロセス起動とモデル読み込み**で占められる (2026-10-14実測)。
16フレームの実推論は9秒しかないので、固定コストが55%の無駄になっていた。

ここでは inference.py を**モジュールとしてimportし、`load_model` を
読み込み済みモデルを返す関数に差し替える**ことで、第三者コードを一切
改変せずにモデル読み込みを1回に抑える。`main()` の出力形式は変わらない。

このスクリプトは **panovggt env のpythonで** 実行する (core/panovggt.PANOVGGT_PY)。
呼び出し側は `core.panovggt.run_chunks()`。

使い方 (通常は run_chunks 経由):
  <panovggt_py> -m img_align.core.panovggt_batch --spec <jsonファイル>
spec: [{"image_dir":…, "mask_dir":…|null, "output_dir":…}, …]
"""
import argparse
import json
import os
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help="チャンク定義のJSONファイル")
    ap.add_argument("--panovggt-dir", required=True)
    ap.add_argument("--config", default="training/config/default.yaml")
    ap.add_argument("--checkpoint", default="checkpoints/model.pt")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.chdir(args.panovggt_dir)          # 相対パスの重み・設定を参照するため
    sys.path.insert(0, args.panovggt_dir)
    import inference as inf

    t0 = time.time()
    model = inf.load_model(args.config, args.checkpoint, args.device)
    # 以降 main() が呼ぶ load_model は読み込み済みモデルを返す
    inf.load_model = lambda *a, **k: model
    print(f"[batch] モデル読み込み {time.time() - t0:.1f}秒", flush=True)

    chunks = json.loads(open(args.spec).read())
    for i, c in enumerate(chunks):
        t = time.time()
        ns = argparse.Namespace(
            config=args.config, checkpoint=args.checkpoint,
            image_dir=c["image_dir"], mask_dir=c.get("mask_dir"),
            output_dir=c["output_dir"], device=args.device, log_depth=True)
        inf.main(ns)
        print(f"[batch] {i + 1}/{len(chunks)} {time.time() - t:.1f}秒",
              flush=True)


if __name__ == "__main__":
    main()
