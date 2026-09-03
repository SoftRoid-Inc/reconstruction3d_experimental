# reconstruction_3d プロジェクト

## 開発目標
住宅の平面図(CAD出力のJPEG、1/50・A4横300dpi相当・2480x1753)から**3次元モデルを再構成**する。
フェーズ1: 2D抽出(壁・柱・窓・ドア)の環境と評価基盤は完成。ただし現行CubiCasa5kは
誤検出が多く精度不十分 → 日本図面での再学習を予定(データ収集は別途進行中)。
**CubiCasaの精度改善検討には時間を使わないこと。**
フェーズ2(現在・優先): GTデータを入力に3次元再構成の開発を進める。
再学習済みモデルが来たら prediction.json 互換で差し替える。
詳細は `MODEL_RECONSTRUCTION.md` を参照。

## データ構成
- `28-XXXXX/zumen/` — 物件ごとの図面 (1F.jpg, 2F.jpg) と正解データ (1F.json, 2F.json = labelme形式)
  - 正解ラベル: wall / column / window / door (全て rectangle、元画像ピクセル座標)
  - 28-AAAAA: 正解データあり(整列済み)。28-DDDDD: 図面のみ
- `cubicasa/` — CubiCasa5kによる2D抽出の検討一式
  - `run_cubicasa.py` 推論 / `align_gt.py` GT整列 / `compare_gt.py` GT比較評価
  - `results/` 推論・評価結果 (`results/28-AAAAA/eval_summary.txt` に評価まとめ)

## 実行環境
- Docker: `/home/maren/programming/docker-compose.yml` のサービス `cubicasa`
  (`docker compose exec cubicasa python ...`、リポジトリ類は `/code2/...` にマウント)
- CubiCasa5k本体: `/home/maren/programming/third_party/CubiCasa5k` (無改変。重み `model_best_val_loss_var.pkl` 配置済み)
- 依存ピン(scipy 1.9.3 / shapely 1.8.5 / matplotlib<3.9 / numpy 1.24.4)は動作条件。上げないこと
- fp16推論はNaNになる (fp32 か bf16 を使う)

## 既知の重要事実
- CubiCasa5kに柱クラスは無い(柱は壁に統合して学習済み)→ 柱は別アプローチが必要
- 推論はスケール0.75が最良だが、それでも誤検出多数・窓~30%と精度不十分(再学習待ち)
- コンテナが生成したファイルはroot所有になるため、必要なら `chown -R 1001:1001` する
