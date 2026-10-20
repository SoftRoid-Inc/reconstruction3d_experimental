# SceneAligner 実装調査ノート (2026-09-03)

対象: SceneAligner "3D-Grounded Floorplan Localization in the Wild"
(arXiv:2605.22581, Cornell VAILab)

## 結論サマリ
- **コード・学習済み重みは未公開**(2026-09-03時点。GitHubリポジトリ
  Cornell-VAILab/SceneAligner は assets のみ、"Code and data will be publicly available" 表記)
- フル再現は非現実的: 核心のクロスモーダルマッチャの学習に **約28万サンプル・
  RTX A6000 1枚で約22日** が必要
- ただし**前段(点群→密度マップ)は公開モデルだけで今すぐ再現可能**で、
  本プロジェクトの「ユーザー点群↔図面の統合」にそのまま流用できる(下記プランB)

## 手法の分解(論文から)
1. **3D再構成**: π³ (Pi3) を主に使用(VGGTでも可)。フィードフォワード点群+信頼度+カメラ姿勢
2. **重力整列**: GeoCalib で画像ごとの重力ベクトル推定 → medoid で頑健化 → y軸整列
3. **密度マップ**: 整列済み3D点を水平面に正射影しグリッド集計。
   信頼度フィルタ(ρ_conf=45%)・空間バウンディング(ρ_xz=2.5%)・
   **高さフィルタ [20%, 95%] で床天井を除去し壁を残す**・γ=0.5補正
4. **マッチャ**: DINOv3 ViT-B/16 凍結 + LoRA(rank16)。密度マップ↔ラスタ図面の対応学習
   (InfoNCE+座標回帰+トポロジー/幾何整合、C3+Structured3Dで学習) ← **ここが未公開の壁**
5. **位置合わせ**: 相互最近傍対応 → RANSAC で2D相似変換(スケール+回転+並進)

## 本プロジェクトへの適用パス(推奨順)
- **プランB: SceneAligner-lite(推奨・学習不要)**
  ユーザー提供予定の点群(SphereSfM等)→ 重力整列(GeoCalib or 点群の床面フィット)
  → 壁帯の高さフィルタで密度マップ → **GT図面はクリーンなベクタなので学習済み
  マッチャは不要**、chamfer距離+回転・スケールのグリッドサーチ(localize.pyと同じ発想の
  剛体版)で相似変換を推定。これは位置合わせ後段「点群と図面由来3Dモデルの統合」の
  中核処理そのもの
- **プランC: SPVLoc(コード公開済み)をD検証に使う**
  https://github.com/fraunhoferhhi/spvloc (ECCV'24 Oral)。参照側=「部屋形状+窓・扉の
  無テクスチャ3Dモデル」→ **recon3dの出力がそのまま使える**。クエリ=透視画像
  → 360°フレームからの切り出し(seg_ade20k.py の e2c 流用)。scene-specific training不要を
  謳っており、ドメインギャップ(欧米学習データ)の検証が論点
- **プランA: SceneAligner公開待ち** — リポジトリをウォッチ。公開されたら重みの
  ライセンスと推論要件を再確認

## 補足: 「素のDINOv3では対応が取れない」(論文Fig.3)とプランBの関係
論文は「ノイジーな密度マップ⇔様式化されたラスタ図面の**見た目ギャップ**は
基盤モデルの特徴でも越えられない → fine-tuningが必要」と述べる。これは
**特徴(見た目)マッチングを使う場合**の話。プランBは特徴マッチングを使わず、
両者を同じモダリティ(2D壁ジオメトリ)に還元して幾何照合するため、この警告は該当しない:
- 図面側はベクタGT+校正済みスケール → 図面の「見た目」を解釈する工程が無い
- 探索は相似変換4自由度・直交壁前提で回転候補ほぼ4方向 → グローバル探索が可能
- 傍証: 単フレーム仮想スキャンの幾何照合(localize.py)でも妥当な結果

**警告が刺さるケース**: ベクタ化を経ずに生の図面JPEGと照合する場合
(寸法線・文字・ハッチング・設備記号がchamferを騙す)。GT無し物件(28-DDDDD)に
適用する際は必ず「2D抽出でベクタ化 → 照合」の順を守ること。SceneAlignerの
学習済みマッチャは、このベクタ化工程を省略するための装置と解釈できる。

## 360°(等距円筒)の注意
π³ / VGGT / DINOv3 / SPVLoc はいずれも**透視画像前提**。本データの5376x2688パノラマは
cubemap/透視切り出しが必要(py360convert 実装済み)。点群側はユーザーが
SphereSfM 等の360°対応SfMで作る想定なのでプランBでは問題にならない。

## 明日の正解データ作成(手動アノテーション)の推奨形式
評価スクリプトが読みやすい形式の提案:
- labelme で `28-AAAAA/zumen/1F.jpg` / `2F.jpg` を開き、**point** を打つ
  - label = `frame_000211`(代表16フレームのフレーム名)
  - 向きも分かる場合は追加pointで label = `frame_000211_dir`(その位置から見て
    正面方向にある任意の1点。矢印の代わり)
- 保存先は **`recon3d/gt_pose/1F.json` / `2F.json`**(zumen/ フォルダは無改変を維持:
  `labelme 1F.jpg -O .../recon3d/gt_pose/1F.json` のように出力先を指定)
- 揃い次第、`results/loc/summary.json` と突き合わせる評価スクリプト(位置誤差[m]・
  角度誤差・階正解率)をこちらで用意する

## 実装結果 (2026-09-04)
プランBを実装・実行済み。ただし点群はユーザー提供ではなく **PanoVGGT (CVPR2026,
パノラマ直接入力の公開モデル)** で全454フレームから生成する構成にした
(`third_party/PanoVGGT` + conda env `panovggt`、重み3.7GB取得済み)。
結果と設計上の教訓(チャンク処理必須・階段で3島分割・precision×recallスコア必須・
スケール不安定)は `results/REPORT.md` の「SceneAligner-lite 実装結果」を参照。
3島すべてで正しい階が選択され、仮説はさらに支持された。定量評価はGT撮影位置待ち。

## Sources
- https://arxiv.org/abs/2605.22581 / https://arxiv.org/html/2605.22581v1
- https://cornell-vailab.github.io/SceneAligner/
- https://github.com/Cornell-VAILab/SceneAligner
- https://github.com/fraunhoferhhi/spvloc / https://arxiv.org/abs/2404.10527
