# img_align — VGGT軌跡パイプライン

360°動画フレームから **PanoVGGT** で軌跡・点群を生成し(メイン)、精度を上げる
**任意後処理**(回転修復・MapAnything条件付け・図面アンカー等)を積む構成。

```bash
# 本番 (GTなし): 推論 → 手動で階を割当 → アンカー以降
python -m img_align.pipeline run --session 0912 --segments imu --max-seg 160 --no-eval
python -m img_align.core.floor_assign --template   # 雛形を編集して階を割り当てる
python -m img_align.pipeline run --session 0912 --segments imu --max-seg 160 \
    --post anchor,hier,export,viewer --no-eval

# メイン一発 (prep → chunks → chains → eval)
python -m img_align.pipeline run --session 0616

# 後処理込み (図面アンカー+階層移送+点群書き出し+ビューア)
python -m img_align.pipeline run --session 0616 --post anchor,hier,export,viewer

# 本番モード: GTなしで島計画 (IMU階段検出) + 評価スキップ
python -m img_align.pipeline run --session 0616 --segments imu --no-eval

python -m img_align.pipeline stages   # 段の説明
```

実行環境: ホスト conda env `img_align`(PanoVGGT推論は内部で env `panovggt` を
subprocess 起動)。`mapany` 後処理のみ docker サービス `mapanything` 内で実行。
セッションは `--session`(内部では環境変数 `IMG_SESSION` → `config.py` が
import時に凍結するため、必ずenv設定後にimportされる)。

---

## メインの仕組み(パイプライン4段)

```
[1 prep] → [2 chunks] → [3 chains] → [4 eval]
```

### 1. prep — 入力準備(`core/crops`)

- **入力**: equirectフレーム列(`<物件>/<セッション>/images/`、5376×2688)、
  人物マスク(`masks/` または seg_ade20k出力)
- **内容**: 人物マスクの有無を確認する。後処理`mapany`を使う場合のみ、
  equirectからピンホールクロップ(FOV130°/pitch10°/518px、人物黒塗り)を生成する
- **出力**: `results/<s>/vidmap/mapany_f130/`(クロップ)+`name_map_*.json`
  (出力名→元フレーム番号の対応表)

### 2. chunks — 前段モデルのチャンク推論(`core/chunking`+`core/panovggt`)

- **入力**: フレーム列+人物マスク、島区間ソース(`--segments`で選択):
  - `gt`(開発・評価用、既定): GT軌跡の階ラベル(`eval/gt_traj.load_gt`)
  - `imu`(本番用・GTフリー): IMU階段検出(`imu/transitions`、学習済みstairnet CNN。
    `transitions.json`が無ければ自動生成)。検出イベントを切断点として区間を切る。
    過検出(はしご・脚立)は島が増えるだけで無害。階ラベルは付けず
    (タグは`pv<sess>_sNN`)、階はアンカー段で決める設計。
    チェーン後に**幾何検算**(`core/stair_check`: 重力整列カメラ軌跡の鉛直変位
    >1.5×カメラ高)が走り、IMU見逃しの階段またぎ島とチェーン破綻島を
    自動再分割する(`stair_resplit`、タグ接尾辞r1/r2)
- **内容**: 全フレームを**16フレーム窓・ストライド12(重なり4)**に区切り、
  窓ごとにPanoVGGTでポーズ+点群を推論する(一括推論は低重なりで破綻するため)。
  島=階の連続訪問区間を計画し(階段区間は除外、`--max-seg`超の長い区間は等分割)、
  **島が必要とするチャンクだけ**推論する。既存チャンクはスキップ(再開可能)
- **出力**: `results/<s>/pcd/chunks/chunk_XX/`(チャンクごとの
  `cameras.npz`・`points.ply`・per-frame点群)+島台帳 `pv_manifest.json`
- **モデル切替** `--model {pv,pi3,holo}`: 既定`pv`=PanoVGGT(ホストのconda env)。
  `pi3`/`holo`(=Pi3-Finetuned-Holo360D)は**pi3コンテナ**で動き、
  `core/chunking.infer_chunks`が`docker compose exec`へ自動で振り分ける
  (欠落チャンクをまとめて1プロセスに渡し、モデル読み込みは1回)。
  チャンク置き場・タグ接頭辞・マニフェストはモデル別に分離
  (`chunks`/`chunks_pi3`/`chunks_holo`、`pv`/`pi3`/`holo`)なので共存できる。
  **モデル比較では島を揃えるため`--segments gt`を使うこと**
  (結果は results/REPORT.md。結論=Pi3系に乗り換える理由は無い)

### 3. chains — 島ごとのチャンク連結(`core/chaining`)

- **入力**: 島台帳と各チャンクの per-frame 点群・ポーズ
- **内容**: 隣接チャンクが共有する4フレームのローカル点群同士を
  **trimmed Sim(3) ICP**(対応の下位30%棄却、スケールは初期値±25%クランプ=
  縮小退化防止)で整合し、チャンク0座標系へ順送りに合成する(VGGT-Long式)。
  島を短く保つほどドリフトが小さい(実測: ≦130フレームで median 0.03〜0.27m)
- **出力**: 島ごとの `results/<s>/frontend_bench/pv<sess>_<島>/`
  (`global_cameras.npz`=全フレームのポーズ・位置、`global_points.ply`=合成点群、
  `chain_log.json`=連結残差ログ)。座標は非メートル(チャンク0ゲージ)

### 4. eval — GT軌跡との比較(`eval/frontend`)

- **入力**: 島タグの `global_cameras.npz`、GT軌跡
  (`<物件>/trajectory_points.csv`。未配備の物件のみ手動登録
  `results/<s>/backend/gt_trajectory.csv` にフォールバック)
- **内容**: 重力整列(カメラup中央値→+y)→XZ平面に落とし、鏡映2択×2D相似フィットで
  GT軌跡へ全フレーム合わせ、残差を測る(**フィットATE**)。図面照合を使わないので
  「前段の相対幾何の正確さ」だけを公平に測れる。
  注意: タグ間比較はフレーム集合を揃えること(階段エッジ混入で数値が振れる)
- **出力**: `frontend_bench/compare_<タグ>.png`(俯瞰比較図)+`compare.json`
  (median/p90等のメトリクス)

---

## 任意後処理(--post、全て入力=frontend_benchタグ)

| 名前 | モジュール | アルゴリズム概要 |
|---|---|---|
| `yaw_fix` | postproc/yaw_fix | 方位ロック撮影ではチャンク内全フレームのヨーが一定のはず。頑健円周中央値から3°超逸脱したフレームを鉛直軸回転で引き戻し、30°超はポーズ全壊として軌跡・点群から除外 |
| `mapany` | postproc/mapany | MapAnything(Apache版)にピンホールクロップ+**既知内参+pvポーズを条件付け**して一括推論。出力はほぼ実寸(メートル)。軌跡改善は小幅(median −7〜15%)だが、メトリックスケールと点群品質(壁残差−19%実証)が価値。16GB VRAMで約55ビュー上限→stride間引き |
| `interp` | postproc/interp | macondの精錬ポーズを制御点、pvチェーンの局所形状を事前として全フレーム補間(制御点近傍窓のSim(3)ブレンド+残差スナップで制御点厳密一致) |
| `colmap_prior` | postproc/colmap_prior | 社内COLMAP大域解へチャンク毎に独立Sim(3)フィット=連結を介さず共通ゲージ化(チェーン破綻セッションの代替。0428/0925向け) |
| `anchor` | postproc/corners (+anchor_score/scale_prior/plan_icp) | **図面アンカーの主役**。密度地図からL/Tコーナーを検出→図面コーナーとRANSAC照合→壁ICP。スケール事前は `postproc/scale_prior`(既定mapany。床スラブ推定は2〜5倍ずれる)で与え、倍率掃引×0.7〜1.4で残差を吸収。候補の順位付けは `postproc/anchor_score` の **f1_feas_scale**(対称被覆F1 × カメラ妥当性 × スケール逸脱罰)。旧wall_coverは片方向被覆で縮小退化があり全周島8本で0/8しか当てられなかった(2026-10-13) |
| `hier` | postproc/hier_anchor | **階層アンカー**: 分割サブ島は親島(全周)のアンカー軌跡と共有するフレームから2D相似を解き、壁ICPのみで仕上げる(RANSAC不使用=偽解の余地なし)。実証: 0912 2F 0.53→0.28m |
| `export` | out/export | アンカー変換(重力整列→z反転→セル→相似→図面px→メートル)で島点群を図面座標へ。鉛直スケールはアンカーの水平スケールから導出(カメラ高さ仮定は使わない)、高さ基準は物理床ピーク。島ごとに高さクロップ指定可(最上階の小屋組は+4.6mまで等) |
| `viewer` | out/viewer | 真上ビュー確認PNG+単体WebGLビューア(uint16量子化・base64埋め込み、高さスライダーは実レンジに動的化) |

### フレーム時刻の規約
`frame_number` は .insv のフレーム索引と **1:1・0始まり**(全セッションで
画像枚数 == insv nb_frames を確認)。フレーム↔IMU時刻の変換は
**`imu/timeline.py` が唯一の窓口**:

1. `imu/frame_times.npz`(insvテレメトリ `Default/Timestamps` 由来の実測時刻。
   `docker compose exec -e IMG_SESSION=<S> imu python -m img_align.imu.frame_times` で生成。
   telemetry_parserとimu_trajはimuコンテナにのみ存在)
2. 無ければ `meta.json` の `duration_s / N` で等間隔

実測では全セッションで間隔は厳密に0.500秒で、等間隔仮定とのずれは末尾1フレーム程度
(stairnet結果は完全一致)。将来の不規則間隔録画への保険として実測時刻を既定にしている。
旧 `imu/sync.py`(FPS2.0+オフセット推定)は**非推奨** — 0828/19176にしか無く誤差も大きい。

### 座標・規約の要点(事故防止)
- **PanoVGGTのworld Y軸は物理的に「下」**(equirect画像座標由来)。水平処理は
  この規約のまま検証済みなので維持し、鉛直は出力段で `floor_phys_y_units`
  (物理床=カメラよりy大側の最大質量ピーク)基準に反転解釈する
- 島座標は非メートル(チャンク0ゲージ)。メートル化はアンカー(図面)または
  MapAnything出力(メトリック)から
- アンカーjsonのスキーマ(R_align/origin/cell_units/s/th/t/floor_phys_y_units…)が
  postproc→out間の契約。手動登録(tools/align_tool)も同じスキーマを書く

---

## パッケージ構成

| パッケージ | 役割 |
|---|---|
| `config.py` | セッション/物件レジストリ・パス・SESSION_CFG(唯一のIMG_SESSION読取り元) |
| `pipeline.py` | メインCLI(上記4段+後処理のオーケストレーション) |
| `core/` | `panovggt`(チャンク推論subprocess) `panovggt_batch`(一括推論=モデル読込1回) `floor_assign`(手動の階割当) `floor_seq`(IMU遷移からの階決定。2/6セッションのみ) `pi3`(Pi3/Pi3-FT推論、PanoVGGT互換出力) `chunking`(島計画) `chaining`(連結) `crops`(e2pクロップ) `plan_grid`(図面グリッド) `geometry`(Sim3/重力/ヨー統計) `io_ply` |
| `postproc/` | 上表の後処理+`anchor_score`(候補スコア) `scale_prior`(m_per_unitの供給源) `island_quality`(島の指紋の濃さ。判別力は無かった) `metric_stitch`(図面なしメートル化。不発) `density_map`(壁密度地図) `plan_match`(FFTグリッド照合) `plan_icp`(point-to-line ICP・座標変換ユーティリティ) `anchor`(島アンカー本体・prepare_island) |
| `eval/` | `anchor_oracle`(候補に正解があるか) `anchor_confidence`(GT無しで当たりを見分ける指標) `anchor_overlay`(アンカー素の出力) `density_view`(密度地図) `outdoor_proxy`(屋外島の暫定判定) `scale_probe` `gt_traj`(GT読込・MM_PER_PX・GTは物件ごとの trajectory_points.csv) `frontend`(フィットATE) `chunk_cmp`(チャンク単体採点+`gt_interp`) `floor_overlay`(階別の図面オーバーレイ比較) `overview`(島別色分け) `island_diag`(島内診断) `depth`(壁面残差で点群品質比較) |
| `out/` | `export` `viewer` |
| `tools/` | `align_tool`(ブラウザ手動登録) `mask_eval_tool` `boundary_mask`(敷地マスク) `model_bench`(3モデル×窓の比較ドライバ) |
| `gs/` `mesh/` | Gaussian Splatting / メッシュ・衝突モデル の各系統 |
| `imu/` | insv抽出(`extract`)・姿勢(`attitude`)・**階段検出(`transitions`=stairnet CNN、島計画の本番ソース)**・**フレーム時刻(`frame_times`→`timeline`)**ほか |
| `legacy/` | 退役スパイク・実験(動態保存。新規参照は禁止) |

## 旧→新モジュール対応(2026-10-07再編)

| 旧 | 新 |
|---|---|
| pointcloud_pano | core.panovggt |
| stitch_chunks(chunk_windows) + mapany_batch | core.chunking |
| chain_chunks | core.chaining |
| vidmap_prep | core.crops |
| localize(PlanGrid部) | core.plan_grid |
| da3_frontend.write_ply / pointcloud_pano.read_ply_xyzrgb | core.io_ply |
| density_map(rotation_to_y等) / chain_chunks(sim3) / yaw_fix(ヨー統計) | core.geometry |
| align_floorplan / refine_align / backend_align | postproc.plan_match / plan_icp / anchor |
| corners / mapany_infer / macond_interp / colmap_prior | postproc.* |
| eval_gt_traj / eval_frontend / mapany_depth_eval | eval.gt_traj / frontend / depth |
| export_aligned2 / backend_viz | out.export / viewer |
| SESSION_CFG・BENCH_DIR・OUT_DIR・CAM_HEIGHT_M | config |
| その他(unbend, opening_*, joint_refine, temporal_fuse, …) | legacy.* |

## 確立済みの運用レシピ(実測の詳細は results/REPORT.md)

- 島は **最大120〜160フレームで分割**(`--max-seg`)→ フィットATE median 0.03〜0.27m
- 図面アンカーは **全周親島でコーナーRANSAC** → **サブ島へ階層移送**(`anchor`→`hier`)
- 本番(GTなし)の島計画は `--segments imu`。stairnetの`offset_sec`は**0**を使う
  (imu_traj既定の−5sは本データでは10〜15フレーム早ずれ)。既知の限界:
  短い2F→3F階段(数秒)を見逃すことがある(0616/0925で実測)→ max-seg分割と
  アンカー段の階選択が保険
- MapAnything条件付けの価値はメトリックスケールと点群品質(軌跡改善は小幅)
- 屋外・基礎段階(0428型)はPanoVGGT不成立 → COLMAP系(`colmap_prior`)
- 後期工程(0925型)は密度コーナーが汚れ自動アンカー未解決 → COLMAPプライア併用が次の一手

## 消してはいけない中間出力

`<results>/pcd/chunks/*/panovggt_out/pointclouds/per_frame/*.ply` は
**チェーン生成の中核**。`chain()` のチャンク間ICPがこのフレーム別点群を使う
(共有カメラ4点のUmeyamaだけでは共線で弱い)。`core/posegraph` も同じ。
容量が欲しくて消すと、既存のチェーンは残るが**再生成できなくなる**
(2026-10-16に外部データで実際に踏んだ)。消してよいのは `depth/` のみ。

## 試して捨てた実装 (再発明しないため、2026-10-16)

実装してA/Bし、**実測で有害だったので消した**もの。詳細は `results/REPORT.md`。

### 点群のボクセル間引き — 13/21 → 9/20 に悪化

`chain_and_save` の `xyz[::8]` を、ボクセルあたり1点 (重心) に置き換えた。
狙いは**順序依存の解消**で、それ自体は達成した (同じ島が両経路で9.99m一致)。
だが精度は大きく落ちた (0912が5/5 → 2/5)。

**理由: 点の多さ自体が「何度も見た = 信頼できる」という情報を持っている。**
`[::8]` は点が多い面ほど多く残すのでこの重みが暗黙に保たれるが、ボクセル化は
一様にするので捨ててしまう。孤立ノイズも1点として等しく生き残る
(`[::8]` なら8分の1に減る)。加えて高さ分布が変わるので `slab_peaks` の
床・天井検出もずれ、スラブ除去の結果が動く。

**「密度地図のセルより細かいボクセルなら可逆」は誤り** — 密度地図そのものは
occupancyなので不変でも、その前段 (スラブ検出・バウンディング・wall_pts の
再間引き) が点数に依存している。

筋の良い方向は**ボクセル内の点数を重みとして残す**こと。順序非依存と信頼度を
両立できるが、密度地図を重み付きに変える必要がある (未実装)。

### 屋外フレームを点群から外す — 効果なし、1島を破壊

`outdoor_classification` のフレーム判定で屋外フレームの点だけを除いた
(姿勢は残す)。0616が0/2 → 2/2 になったように見えたが、**除去の効果ではない**。
`per_frame/*.ply` を使う経路が `merged.ply` と**点の並び順が違う**ため、
`[::8]` で残る点が変わっただけだった (除去なしで per_frame を通すと同じ
0.51/0.69m が出る)。唯一除去が効いた島は 0925_s003r2 で、屋外率3.4% =
3フレームを抜いただけで **0.46m → 8.04m に破綻**した。

**島単位の除外 (`eval/outdoor_label`) は有効** — こちらは残してある。
`outdoor_classification` は囲繞性ベースなので、空では拾えない**外周撮影と
庇つきテラス**を正しく扱う (`eval/outdoor_proxy` は外周島を屋外率0.20、
テラスを0.00 と見て「室内」と誤判定した)。

### ついでに判明した落とし穴
- **`panovggt_out/pointclouds/per_frame/*.ply` はカメラ座標系**。`merged.ply`
  は姿勢を掛けた世界座標。掛け忘れると全島壊滅する (残差0.005で一致する)
- **アンカーは点の並び順に依存する**。同じ点群でも `merged` 経路と
  `per_frame` 経路で 0616 が 1.46/1.62m と 0.69/0.51m に分かれた。
  多くの島が判定の境界線上にいるということで、小さな差が結果を裏返す
