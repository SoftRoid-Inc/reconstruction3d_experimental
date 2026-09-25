# 合併Gaussian Splatting生成 設計スペック (2026-09-25)

## 目的と背景

位置合わせ済み点群 (`aligned_house{_0312}.ply`) と同じ図面メートル座標系で、
家全体を1つに合併した Gaussian Splatting を生成する。

- **主目的**: 見た目品質の高い3D表現(点群・メッシュより写実的)
- **副目的**: 人型ロボット視点(高さ1.3m以上)の対話的な再現。
  撮影視点(自撮り棒 約1.5m)とほぼ同帯のため**内挿**であり、品質リスクは低い
- **対象**: 0828(竣工後)と0312(工事中)の両セッション
- **商用条件**: 実装は gsplat (Apache 2.0) + ビューアは MIT 系のみ。
  INRIA本家実装(非商用)は使わない

## 成功基準

1. `results/backend/gaussian_house.ply`(0828)/`gaussian_house_0312.ply` が
   図面座標(x,y=図面平面m、z=高さm、1F床=0/2F=+2.9)で生成される
2. 単体HTMLビューア(`gs_viewer{SFX}.html`)で一人称移動でき、
   視点高さプリセット(1.3m/1.5m)で屋内を破綻なく見て回れる
3. 保持ビューPSNRが屋内目安(〜25dB)に達する
4. GS中心点の真上ビューが図面壁と整合する(既存compare指標の流用)

## アーキテクチャ

```
img_align/gs_prep.py     equirect→学習用透視ビュー+ポーズ+初期点群 (env: img_align)
Dockerfile.gsplat        gsplat学習環境 (composeサービス gsplat)
img_align/gs_train.py    島ごとの学習ドライバ (コンテナ内実行)
img_align/gs_merge.py    島GS→図面座標へ変換・合併 (env: panovggt)
img_align/gs_viewer.py   単体HTMLビューア生成
```

島レジストリ・ポーズ・図面座標変換は **meshing.py の土台を再利用**
(`ISLANDS` / `resolve_frames` / `drawing_matrix(metric_input=True)`)。

## コンポーネント設計

### gs_prep (データ準備)

- ポーズ: `meshing.resolve_frames()` — メートル島座標のcam2world。
  重複先勝ち・untrusted除外・COLMAP吸着外れ除外を継承
- ビュー合成: **原画equirect(5376×2688)から** ヨー8方向×FOV70°、
  ピッチ0°/−35°の2段 = 1フレーム16ビュー(解像度は学習時間と相談、初期値800px)
- 人物マスク: 黒塗りではなく**per-viewマスク画像**として出力し学習の損失から除外
- 初期点群: 島点群(メートル・RGB)を3cmボクセル間引きしてPLY出力
- 出力形式: gsplat simple_trainer が読める COLMAP形式ディレクトリ
  (cameras/images/points3D) をビュー合成ポーズから直接書く

### gs_train (学習)

- gsplat の標準トレーナを薄くラップ。**ポーズ最適化は無効**(位置合わせ資産を信頼)
- 点群初期化、15k〜30k iterations、SHは3次まで(合併時にDC化)
- 島5つ(0828×3 + 0312×2)を順次学習。RTX 4090 Laptopで1島30分前後を見込む

### gs_merge (合併) — 反射の扱いが肝

- 位置: `drawing_matrix(metric_input=True)` の4x4(検証済み・機械精度・det<0)
- 共分散: Σ=R·S·Sᵀ·Rᵀ を A·Σ·Aᵀ で変換(反射でも正)→ 固有分解で
  quaternion+scale に復元
- 色: **SHをDC(0次)に落としてRGB化**(反射下の高次SH回転は誤りやすい。
  屋内の見た目では0次で十分。必要になったら拡張)
- 不透明度: そのまま
- 島の重なり(0828のAとCは1Fで重複): そのまま重ねる。目視で問題があれば
  島Cの重複領域を刈る(後判断)

### gs_viewer (ビューア)

- antimatter15/splat (MIT) のレンダラを単体HTMLに同梱(既存ビューア群と同じ配布形態)
- `.ply`→`.splat` 量子化で 20〜60MB/セッションを想定
- UI: 階表示切替、視点高さプリセット(1.3m/1.5m)、WASD+マウスの一人称移動

## 検証

1. 学習の保持ビューPSNR(各島)
2. 合併後: GS中心点の真上ビュー+図面壁オーバーレイ(compareの流用)、
   高さヒストグラム(床0m/2F 2.9m)
3. ビューアで1.3m高の一人称移動を目視(0828の家具帯・0312の軸組)

## リスクと対応

| リスク | 対応 |
|---|---|
| PanoVGGT島の内部ドリフト(0828島A 0.2-0.33m)でGSがボケる | まず素で学習して目視。ボケが酷ければ該当島のみポーズ微調整を許可(gsplatのpose-opt)して再学習 |
| 反射変換の誤り | gs_merge単体テスト: 変換前後でレンダリング視点を対応させ、GS中心点群のChamferが0近傍 |
| equirect下端の撮影者残り | 既存segマスクを損失マスクに含める |
| VRAM不足(4090 Laptop 16GB) | ビュー解像度800px・バッチ1で開始。不足時は縮小 |

## 非スコープ

- 2DGS/PGSR系の表面再構成(メッシュは別タスクで完了済み)
- 高次SHを保った反射変換
- リアルタイムロボシミュレータへの直接組込み(ビューア確認まで)
