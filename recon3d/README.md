# recon3d — VGGT軌跡パイプライン

360°動画フレームから **PanoVGGT** で軌跡・点群を生成し(メイン)、精度を上げる
**任意後処理**(回転修復・MapAnything条件付け・図面アンカー等)を積む構成。

## 外部の入力ディレクトリから回す (本番)

S3から落とした素材が1つのディレクトリに揃っている前提。リポジトリを
編集せずに新しい物件を回せる。CLIは **typer**、設定は **hydra**。

```
reconstruction/
  images/  frame_number_*.jpg       必須
  masks/   frame_number_*.jpg       人物マスク (白=有効)
  imu/     transitions.json idle_spans.json
  floors/  1F.json 1F.jpg … meta.json   ← meta.json に図面の実寸
  results/                           ← 出力はここだけ
```

入口は**リポジトリ直下の `main.py`** (Docker イメージの ENTRYPOINT)。

```bash
python main.py inspect input=/data/reconstruction          # 点検
python main.py run     input=/data/reconstruction
python main.py run     input=/data/reconstruction \
    'steps=[anchor,export]' chunking.chunk_size=24
python main.py steps                                       # ステップの一覧
```

**ステップの順序は書いた順ではなく依存で決まる** (`steps.py`)。選んだステップの前提は
自動で補われ、能力が足りないステップは理由つきで skip される
(明示的に選んだステップが能力不足のときだけエラーで止まる)。

> `floors/meta.json` の `known_width_mm` (最外壁芯間のx方向スパン[mm]) が
> 無いと図面アンカーは動かない。**壁注釈だけでは px→m が決まらない。**

## 個別モジュールを叩く (開発・調査)

ステップより細かい単位で回したいときは `tool`。各モジュールの argparse を
そのまま使う (`-h` はツール側のヘルプに通る)。

```bash
python main.py tool                            # ツール一覧
python main.py tool posegraph --input /data/x --from-tag pv_s001
python main.py tool floor_assign --input /data/x --template
python main.py tool corners --input /data/x --anchor pv_s001:1F
```

## 間取り図から3Dモデル (plan2model)

撮影とは独立の系統。`floors/*.json` (labelme) と `floors/meta.json` の
`known_width_mm` だけで動く。

```bash
python main.py plan2model verify input=/data/x   # 2D再投影IoU + オーバーレイ
python main.py plan2model build  input=/data/x   # glb / obj / report.json
```

出力は `results/plan2model/`。詳細は [`plan2model/README.md`](plan2model/README.md)。

実行環境: **単一 Docker イメージ** (`reconstruction_3d/Dockerfile`、
compose サービス `recon3d`)。PanoVGGT 推論から図面アンカーまで同一プロセスで走る。

```bash
docker compose build recon3d
docker compose exec recon3d python main.py run input=/data/reconstruction
```

> ローカルでは compose が `working_dir: /code2/reconstruction_3d` を設定して
> いるので**マウントした編集中のコード**が動く。イメージに焼いた
> `/opt/recon3d` は k8s 用 (ENTRYPOINT がそちらを使う)。

> **GPUステップ (`chains`) はイメージの中でだけ動く。** `PANOVGGT_DIR` が未設定だと
> 推論ステップで止まる — ホストの python には torch 2.9.1 が入っていて
> `import inference` が通ってしまうが、PanoVGGT が想定するのは 2.5 系で、
> 「動くが結果が違う」を防ぐため。既存チャンクを使うステップ (posegraph / anchor /
> export / 評価) は GPU も PanoVGGT も要らないのでホストのままで動く。

`mapany` 後処理と `pi3`/`holo` は別コンテナ。`mesh/*` (open3d) と `gs/*`
(gsplat) はこのイメージに入れていない。

**データセットを指すのは `input=` だけ**。所在は `paths.py` の `Paths` に
入り、各ステップへ**引数で配られる** — 環境変数もモジュール凍結も無い。
タグ接頭辞や出力先は入力ディレクトリ名から決まる。

---

> **全体像と動かし方は [`PIPELINE.md`](PIPELINE.md) にまとめた**
> (図つき。構造・実行コマンド・大量実走の設計・落とし穴)。
> ここは各ステップの詳細とモジュール対応表。

## メインの仕組み

```
[prep] → [chains] → [eval]   ＋ 任意後処理 (mapany/bridge/anchor/hier/export)
```

ステップの定義と依存は `steps.py` の登録表。`core/caps.py` が入力の能力を判定し、
できないステップを理由つきで skip する。

### 1. prep — 入力準備(`core/ingest` + `core/crops`)

- **入力**: equirectフレーム列(`images/`、5376×2688)、人物マスク
  (`masks/` または seg_ade20k出力)、IMU派生物(`imu/`)
- **内容**: 入力の `imu/*.json|npz` を `results/imu/` へコピーし、
  取り込んだ名前を `_materialized.json` に記録する(次回はそれを消してから
  張り直す ← `imu_segments` は `exists()` しか見ないので古い残骸が効き続ける)。
  人物マスクの有無を確認する。後処理`mapany`を使う場合のみ、equirect から
  ピンホールクロップ(FOV130°/pitch10°/518px、人物黒塗り)を生成する
- **出力**: `results/imu/`(取り込み分)、
  `results/vidmap/mapany_f130/`(クロップ)+`name_map_*.json`

### 2. chains(前半) — 前段モデルのチャンク推論(`core/chunking`+`core/panovggt`)

- **入力**: フレーム列+人物マスク、島区間ソース(`--segments`で選択):
  - `gt`(開発・評価用、既定): GT軌跡の階ラベル(`eval/gt_traj.load_gt`)
  - `imu`(本番用・GTフリー): IMU階段検出(`imu/transitions`、学習済みstairnet CNN。
    `transitions.json`が無ければ自動生成)。検出イベントを切断点として区間を切る。
    過検出(はしご・脚立)は島が増えるだけで無害。階ラベルは付けず
    (タグは`pv<sess>_sNN`)、階はアンカーステップで決める設計。
    チェーン後に**幾何検算**(`core/stair_check`: 重力整列カメラ軌跡の鉛直変位
    >1.5×カメラ高)が走り、IMU見逃しの階段またぎ島とチェーン破綻島を
    自動再分割する(`stair_resplit`、タグ接尾辞r1/r2)
- **内容**: 全フレームを**16フレーム窓・ストライド12(重なり4)**に区切り、
  窓ごとにPanoVGGTでポーズ+点群を推論する(一括推論は低重なりで破綻するため)。
  島=階の連続訪問区間を計画し(階段区間は除外、`--max-seg`超の長い区間は等分割)、
  **島が必要とするチャンクだけ**推論する。既存チャンクはスキップ(再開可能)
- **出力**: `results/pcd/chunks/chunk_XX/`(チャンクごとの
  `cameras.npz`・`points.ply`・per-frame点群)+島台帳 `pv_manifest.json`
- **モデル切替** `--model {pv,pi3,holo}`: 既定`pv`=PanoVGGT(recon3d イメージ内)。
  `pi3`/`holo`(=Pi3-Finetuned-Holo360D)は**pi3コンテナ**で動き、
  `core/chunking.infer_chunks`が`docker compose exec`へ自動で振り分ける
  (欠落チャンクをまとめて1プロセスに渡し、モデル読み込みは1回)。
  チャンク置き場・タグ接頭辞・マニフェストはモデル別に分離
  (`chunks`/`chunks_pi3`/`chunks_holo`、`pv`/`pi3`/`holo`)なので共存できる。
  **モデル比較では島を揃えるため`--segments gt`を使うこと**
  (結果は results/REPORT.md。結論=Pi3系に乗り換える理由は無い)

### 3. chains(後半) — 島ごとのチャンク連結(`core/chaining`)

- **入力**: 島台帳と各チャンクの per-frame 点群・ポーズ
- **内容**: 隣接チャンクが共有する4フレームのローカル点群同士を
  **trimmed Sim(3) ICP**(対応の下位30%棄却、スケールは初期値±25%クランプ=
  縮小退化防止)で整合し、チャンク0座標系へ順送りに合成する(VGGT-Long式)。
  島を短く保つほどドリフトが小さい(実測: ≦130フレームで median 0.03〜0.27m)
- **出力**: 島ごとの `results/frontend_bench/pv<sess>_<島>/`
  (`global_cameras.npz`=全フレームのポーズ・位置、`global_points.ply`=合成点群、
  `chain_log.json`=連結残差ログ)。座標は非メートル(チャンク0ゲージ)

### 4. eval — GT軌跡との比較(`eval/frontend`)

能力 `gt` と `scale` の両方が要る。片方でも欠ければこのステップは skip される。

- **入力**: 島タグの `global_cameras.npz`、GT軌跡
  (`<物件>/trajectory_points.csv`。未配備の物件のみ手動登録
  `results/backend/gt_trajectory.csv` にフォールバック)
- **内容**: 重力整列(カメラup中央値→+y)→XZ平面に落とし、鏡映2択×2D相似フィットで
  GT軌跡へ全フレーム合わせ、残差を測る(**フィットATE**)。図面照合を使わないので
  「前段の相対幾何の正確さ」だけを公平に測れる。
  注意: タグ間比較はフレーム集合を揃えること(階段エッジ混入で数値が振れる)
- **出力**: `frontend_bench/compare_<タグ>.png`(俯瞰比較図)+`compare.json`
  (median/p90等のメトリクス)

---

## 任意後処理(入力はすべて frontend_benchタグ)

新CLIでは `steps=[...]`、互換CLIでは `--post`。**順序は依存で決まる**ので
書く順は関係ない (`bridge` と `anchor` を両方選べば bridge が先)。

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
   `docker compose exec imu python -m recon3d.imu.frame_times` で生成。
   telemetry_parserとimu_trajはimuコンテナにのみ存在)
2. 無ければ `meta.json` の `duration_s / N` で等間隔

実測では全セッションで間隔は厳密に0.500秒で、等間隔仮定とのずれは末尾1フレーム程度
(stairnet結果は完全一致)。将来の不規則間隔録画への保険として実測時刻を既定にしている。
旧 `imu/sync.py`(FPS2.0+オフセット推定)は**非推奨** — 0828/19176にしか無く誤差も大きい。

### 座標・規約の要点(事故防止)
- **PanoVGGTのworld Y軸は物理的に「下」**(equirect画像座標由来)。水平処理は
  この規約のまま検証済みなので維持し、鉛直は出力ステップで `floor_phys_y_units`
  (物理床=カメラよりy大側の最大質量ピーク)基準に反転解釈する
- 島座標は非メートル(チャンク0ゲージ)。メートル化はアンカー(図面)または
  MapAnything出力(メトリック)から
- アンカーjsonのスキーマ(R_align/origin/cell_units/s/th/t/floor_phys_y_units…)が
  postproc→out間の契約。手動登録(tools/align_tool)も同じスキーマを書く

---

## パッケージ構成

| パッケージ | 役割 |
|---|---|
| `paths.py` | **データセットの所在** (`Paths`)。入力1ディレクトリから in/out を導く凍結 dataclass。引数で引き回す — モジュール変数に固めない |
| `constants.py` | 入力に依らない定数だけ (代表フレーム数・ラベル定義など) |
| `steps.py` | **ステップの登録表と依存解決** (`needs`/`before`/`caps` + graphlib)。順序はここが決める |
| `main.py` (直下) | **ポッドの ENTRYPOINT**。`tools/recon` の薄いラッパ |
| `api.py` | Python から呼ぶ入口 (`run` / `plan`)。**このプロセスで実行する** — 1ポッド1データセットならプロセス分離は要らない |
| `conf/` | hydra設定 (`recon.yaml` = 入力ディレクトリ実行) |
| `core/` | `caps`(入力の能力判定 + 使える CPU 数) `signals`(入口が SIGTERM をプロセスグループへ中継) `ingest`(入力の imu/ を results へ台帳つきコピー) `panovggt`(チャンク推論subprocess) `panovggt_batch`(一括推論=モデル読込1回) `floor_assign`(手動の階割当) `floor_seq`(IMU遷移からの階決定。2/6セッションのみ) `pi3`(Pi3/Pi3-FT推論、PanoVGGT互換出力) `chunking`(島計画) `chaining`(連結) `crops`(e2pクロップ) `plan_grid`(図面グリッド) `geometry`(Sim3/重力/ヨー統計) `io_ply` |
| `postproc/` | 上表の後処理+`anchor_run`(全島を並列アンカー) `mapany_run`(mapany起動) `anchor_score`(候補スコア) `scale_prior`(m_per_unitの供給源) `island_quality`(島の指紋の濃さ。判別力は無かった) `metric_stitch`(図面なしメートル化。不発) `density_map`(壁密度地図) `plan_match`(FFTグリッド照合) `plan_icp`(point-to-line ICP・座標変換ユーティリティ) `anchor`(島アンカー本体・prepare_island) |
| `eval/` | `anchor_oracle`(候補に正解があるか) `anchor_confidence`(GT無しで当たりを見分ける指標) `anchor_overlay`(アンカー素の出力) `density_view`(密度地図) `outdoor_proxy`(屋外島の暫定判定) `scale_probe` `gt_traj`(GT読込・`mm_per_px()`=遅延評価の関数・GTは物件ごとの trajectory_points.csv) `frontend`(フィットATE) `chunk_cmp`(チャンク単体採点+`gt_interp`) `floor_overlay`(階別の図面オーバーレイ比較) `overview`(島別色分け) `island_diag`(島内診断) `depth`(壁面残差で点群品質比較) |
| `out/` | `export` `viewer` |
| `plan2model/` | **間取り図→3Dモデル** (loader/calibrate/geometry/mesh/build/verify)。撮影と独立の系統 |
| `tools/` | `recon`(**入力ディレクトリから回すCLI**。入口は直下の `main.py`) `_hydra`(hydra読込の共通部) `align_tool`(ブラウザ手動登録) `mask_eval_tool` `boundary_mask`(敷地マスク) `model_bench`(3モデル×窓の比較ドライバ) |
| `gs/` `mesh/` | Gaussian Splatting / メッシュ・衝突モデル の各系統 |
| `imu/` | insv抽出(`extract`)・姿勢(`attitude`)・**階段検出(`transitions`=stairnet CNN、島計画の本番ソース)**・**フレーム時刻(`frame_times`→`timeline`)**ほか |
| `tests/` | pytest 無しでも走る検査 (`check_imports`=importスモーク+`paths` 省略の禁止 / `check_paths`=実物の入力で `Paths` を確かめる) |

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

## 旧→新対応(2026-10-19 実行インタフェース再編)

| 旧 | 新 |
|---|---|
| `pipeline.stage_chunks_chains` ほかステップの本体 | `steps.py` の登録表 (`_prep`/`_chains`/…) |
| `pipeline.post_anchor` / `_anchor_one` | `postproc.anchor_run.anchor_islands` |
| `pipeline.post_mapany` | `postproc.mapany_run.run_mapany` |
| `--post` を書いた順に実行 | `steps.resolve` が依存で順序を決める |
| `eval.gt_traj.MM_PER_PX`(import時に評価する定数) | `eval.gt_traj.mm_per_px()`(遅延+キャッシュ) |
| `config._register_ext`(ext/ 専用) | `input=` に任意のディレクトリ |
| 入力の imu/ は手で置く | `core.ingest.materialize_imu`(台帳つき) |
| PanoVGGT は conda env の subprocess | **同一プロセス** (`PANOVGGT_DIR` 必須)。単一 Docker イメージ |
| `python main.py` | `python main.py` (イメージの ENTRYPOINT) |
| `tools/ext_batch`(大量実走) | 撤去。`ext/sfmXXXX` は `input=` にそのまま渡す |

## 確立済みの運用レシピ(実測の詳細は results/REPORT.md)

- 島は **最大120〜160フレームで分割**(`--max-seg`)→ フィットATE median 0.03〜0.27m
- 図面アンカーは **全周親島でコーナーRANSAC** → **サブ島へ階層移送**(`anchor`→`hier`)
- 本番(GTなし)の島計画は `--segments imu`。stairnetの`offset_sec`は**0**を使う
  (imu_traj既定の−5sは本データでは10〜15フレーム早ずれ)。既知の限界:
  短い2F→3F階段(数秒)を見逃すことがある(0616/0925で実測)→ max-seg分割と
  アンカーステップの階選択が保険
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

## 旧→新対応(2026-10-20 recon3d 1モジュール化)

| 旧 | 新 |
|---|---|
| `img_align.*` | `recon3d.*` |
| 旧 `recon3d.loader` / `calibrate` / `build` / `verify` ほか7本 | `recon3d.plan2model.*` |
| `config.py` の物件レジストリ + `IMG_SESSION` | `paths.Paths` を**引数で引き回す** |
| `config.SESSION` / `SESSION_REF` / `SESSION_CFG` | 廃止 (入力ディレクトリがデータセット) |
| `config.BENCH_DIR` / `OUT_DIR` / `ZUMEN_DIR` / `KNOWN_WIDTH_MM` | `paths.bench` / `paths.backend` / `paths.floors` / `paths.known_width_mm` |
| `results/<session>/` (入力名が二重に付いた) | `results/` |
| `python -m recon3d.<module>` (モジュール直叩き) | `python main.py tool <name> --input <dir>` |
| `python -m recon3d.build --floors 1F=... --width-mm 9100` | `python main.py plan2model build input=<dir>` |
| `pipeline.py` / `legacy/` / `gs/` / `seg_ade20k` ほか64本 | 削除 |

> 本番入口から到達しないモジュールは**残さない**方針にした。削除分は
> git 履歴 (`ca5c883` 以前) にある。
