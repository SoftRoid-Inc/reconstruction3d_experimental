# recon3d を pipeline で比較・統合するための設計

2026-10-20

## 意図

recon3d(軌跡生成・3D再構築・図面合わせ)を、`/home/maren/git/pipeline` の
本番基盤に乗せたい。発端は「recon3d が大きくなってきて、開発の進め方が決まらない」
という悩みで、当初の選択肢は「submodule にして `run` を呼ぶだけ」対
「そもそも recon3d を簡素化する」の二択だった。

### 判断軸(依頼者の言葉)

1. **本番で落ちたとき、そのコードを書いていない人が「どこを見ればいいか」に
   辿り着けるか。** 手元に最初に届くのは Dagster が拾った Pipes のログだけ。
2. **これからアルゴリズムを改善するとき、改善と評価がしやすいか。**

### 調査で分かった前提

この二択は、調査の結果**前提から組み直す必要がある**ことが分かった。

- **pipeline には既に `pipes/recon_ff/` がある。** ER フレーム → 軌跡で、
  3段を同一プロセスで回す: ① `pano_chunks`(chunk 単位の feed-forward 再構成、
  GPU) ② `traj_segments`(チャンク連結・重力・セグメント化・壁点抽出、CPU)
  ③ `drawing_fit`(図面へのフィット、CPU、`fit.enabled` のときだけ)。
  `steps/` には `panovggt.py` `mapanything.py` `cubicasa.py` `drawing.py`
  `fit.py` `segmentation.py` `placed.py` が並ぶ。
  **recon3d と同じエンジン・同じ段構成で、ほぼ同じ仕事をしている。**
- **図面フィットの方式だけが違う。** recon_ff は scale × yaw4択 × 全並進の
  総当たり(FFT。cost = proximity + coverage + traj_on_wall)。recon3d は
  コーナーRANSAC + 壁ICP。
- **壁の出どころも違う。** recon_ff は `steps/cubicasa.py` が図面画像から
  壁マスクを推論する。recon3d は人手の labelme 注釈 + `known_width_mm` を要求し、
  無ければ動かない。
- **pipeline の `step()` は Dagster の境界ではない。**
  `common/src/pipes_runtime/step.py` の、同一プロセス内のログ・計測ラッパ。
  `[name] start` / `[name] end (duration, cpu = N cores)` を `pipes.log` に流し、
  `step/timings` テーブルを materialization に載せる。
  1 Pipes プロセス = 1 Dagster アセットで、その中に `step()` が何本も並ぶ。
  **粒度の選択はリトライ/キャッシュの話ではなく、ログの粒度・所要時間・
  pod サイジング(`eff_cores` で cpu requests を決めている)の話。**
- **評価の土俵は pipeline 側の方が広い。** `eval/` に撮影245件のデータセット、
  正解(`bi.csv` / `floor.csv`)、図面画像の取得とキャッシュ、軌跡の採点
  (位置と形)、図面オーバーレイPDF、cairn の台帳(`--engine` × `--variant`)。
  recon3d 側は21島で、ホールドアウトには壁注釈が無く図面アンカーを評価できない。
- **engine の拡張点は小さい。** `eval/src/engine_dfsfm.py` が持つのは実質
  `load(s3, bucket, scan_id, data_prefix) -> Result | None` と
  `tracks_of(result, floor) -> list[Track]` の2つ。`score_recon.py` の
  `_load()` に分岐を足して `--engine` に名前を増やせば採点が通る。
- **dfsfm は図面に載っていない** (`placed=False`、形だけが意味を持つ)。
  recon_ff は図面px で入ってくる。recon3d も図面に載せるので、
  **位置(正答率%)で測れる数少ない engine になる。**

## 目標

1. recon3d と recon_ff を**同じ物差しで比較**できる(pipeline `eval/`、245件)。
2. recon3d 単体でアルゴリズムを**部分評価**し続けられる(島・ステップ単位)。
3. 本番で落ちたとき、書いていない人が Dagster のログから当たりを付けられる。
4. 比較の結果を踏まえて、最終的に勝った方式を recon_ff に統合できる。

## 非目標

- recon3d を**製品の本番経路**に置くこと(運用者に見える成果物を出す段になること)。
  ただし**比較のための投入はデプロイを伴う** — 245件を回す手段が `eval/run_recon.py`
  の Dagster 投入なので、image を焼き、location を再デプロイし、
  `ReconEngineEnum` と `run_recon.py --engine` の選択肢に名前を足す必要がある。
  「デプロイは後」は製品経路の話で、検証環境への投入は前提。
- `plan2model`(図面→3Dモデル、撮影と無関係)の移設。行き先は別途。
- CubiCasa の精度改善。
- recon3d の研究用の足場(`steps.py` の graphlib 解決・`caps`・hydra・
  `tool` 49本・`eval/`)を pipeline に持っていくこと。**これは残す。**

## アンチゴール(技術的に動いても失敗とみなすもの)

- **`api.run()` を1つの `step()` で包んだ黒箱。** `step/timings` に1行しか出ず、
  落ちても `[recon3d] FAILED after 700s` で終わる。目標3がそこで死ぬ。
- **前段(PanoVGGT 推論・チャンク連結)を recon_ff と recon3d で二重に持ち続ける。**
  同じエンジンを同じように回しているだけで何も比較できず、保守だけ倍になる。
- **比較のために作った足場が、統合時に全部捨てになること。**
  「submodule を置いて Dockerfile を書いて entry を書いて、統合時にその全部を剥がす」
  が一番安く見えて一番捨てる量が多い。

## 制約(pipeline の家の規約。`docs/coding-rules.md`)

- 層: `entry/<asset>.py`(薄い。pipes ctx と S3 を知る) / `stages/` /
  `steps/<step>.py`(**純関数**。`dagster_pipes` / `pipes_runtime` / `aws` /
  同じ pipe の `io` を import しない) / `utils/` / `io.py`。**検査あり。**
- **`steps/` は互いを import しない。** step が別の step を要るなら、
  それは下の層なので `utils/` か `schemas/` へ出す。**検査あり。**
- **ミニ DAG は作らない。直列 python 呼び出しがそのまま DAG。**
- 入力の有無による分岐は orchestration(`ScanContext.has_smartphone_data` の形)。
  パイプの中は直列で、走らせるかは `extras` の None 判定だけ。
- vendor は `submodules/<X>` → Dockerfile で `COPY submodules/<X> /pymodules/<X>`
  → `steps/<x>.py` が薄いラッパとして依存を閉じる(`steps/panovggt.py` が前例)。
- `pipe を跨いで同じ計算をするなら common/ へ。判断は「エンジンに依らないか」`。
  ただし**図面合わせは共有の計算ではなく製品の段**なので、`common/` ではなく
  パイプの段として置く(依頼者の指示)。

## 決めたこと

### 1. recon3d はリポジトリとして残し、pipeline の submodule として参照する

両方の実装を残すのは、**性能を比較するため**。統合は比較の後。
リポジトリ名は `reconstruction3d_experimental` に変更済み(本番は pipeline 側、
ここは研究・評価、という役割を名前に出す)。パッケージ名 `recon3d` は据え置き。

### 2. 順番は「簡素化 → submodule → 統合」

当初の二択(submodule か簡素化か)は**両方で、順番がある**というのが結論。

1. **recon3d の芯を I/O から剥がす。**
   いまは `corner_anchor(paths, tag, floor, ...)` が `paths` を受け取り、
   自分で `global_cameras.npz` を読み `anchor_*.json` を書く。これを
   `(密度地図, PlanGrid, スケール事前) -> (変換, 指標)` の純関数と、
   端の I/O に割る。
2. **submodule にして薄いラッパで包む。**
   `submodules/recon3d` → `COPY ... /pymodules/recon3d` →
   `pipes/recon_3d/src/steps/<名>.py` が純関数を呼ぶ薄いラッパ →
   `entry/` が `step()` で直列に並べる。
3. **比較して、勝った方式を `pipes/recon_ff/src/steps/` へ持っていく。**

1 を飛ばして 2 をやると薄いラッパが書けない(`paths` を受けて自分で読み書きする
関数は包めない)。そして 1 をやっておけば、3 の移植は「書き直す」ではなく
**「関数を持っていく」**になり、捨てるのは Dockerfile と entry だけで済む。

### 3. pipeline から `api.run()` は呼ばない

アンチゴールの1つ目。各段を `step()` で個別に包むことで、Dagster の
`step/timings` に recon3d の段が1行ずつ出て、`eff_cores` から pod を
サイジングできる。

### 4. 評価は2層。層ごとに場所が違う

| | どこで | 何を比べる | 粒度 |
|---|---|---|---|
| 全体 | pipeline `eval/` | recon_ff vs recon3d(engine 軸)、recon3d の版同士(variant 軸) | 245件・撮影単位 |
| 部分 | recon3d(このリポジトリ) | アルゴリズムの芯 | 島・ステップ単位 |

新しい評価の仕組みは要らない。cairn が既に `--engine` × `--variant` の2軸を持つ。
**全体評価のために pipeline へ投入する必要がある**(245件を回す手段が
`run_recon.py` の Dagster 投入なので、手元で回して採点だけ繋ぐ形は成立しない)。

### 5. 繋ぎは `eval/src/engine_recon3d.py` の2関数 — ただし採点経路が2つある

`load()` と `tracks_of()` を書き、`score_recon.py` の `_load()` に分岐を足す、
という形自体は正しい。recon3d は図面に載るので `placed=True` 側になり、
**位置(正答率%)でも測れる**。

**ただし当初「2関数で済む」と書いたのは誤りだった。** 実際には採点経路が
2系統ある:

- `eval/score_recon.py` の `--engine` は **`dfsfm` / `dfsfm_legacy` のみ**。
- recon_ff の245件採点は `pipes/recon_ff/eval/`(`harness.py` `report.py`
  `metrics.py` `placement.py` `launch.py`…)の**独自ハーネス**。
- `eval/README.md` 自身が「**今 engine を跨いで比べられるのは `形の正答率%` と
  `形残差` だけ**」「px 系を揃えるには `src/engine_vggt.py` を起こす必要がある」
  と明記している。

つまり recon3d を `score_recon.py` 側に足すだけだと、cairn の同じテーブルに
**別実装で採点された行が並ぶ**。「位置で測れる数少ない engine」という利点も、
recon_ff 側の位置が別コードで出た数字では直接比べられない。
どう揃えるかは未決(下記 D)。

### 6. `ReconEngineEnum.RECON3D` として `reconstruction` asset に足す。前段は自前

依頼者の決定。recon_ff の中間成果物を食う後段パイプにはしない。
`ReconEngineEnum` に `RECON3D` を足して `reconstruction` asset の engine 分岐に
乗せるので、`run_recon.py --engine recon3d` がそのまま投入手段になる。

代償として、チャンクは recon_ff と別に持つことになる(recon_ff 側は
224MB〜4.5GB/件 × 245件が既に S3。recon3d は窓16/stride12 の別形式)。
置き場と GPU 時間を見積もること。

### 7. IMU 由来物は prepro の成果物をそのまま使う

**recon3d の入力契約と一致している。** `prepro_gpu` の `imu_pdr` stage が
S3 に上げるのは3つ:

| S3 キー deriver | recon3d 側 |
|---|---|
| `get_imu_floor_transitions_json_key` | `imu/transitions.json`(島の切断点) |
| `get_imu_idle_spans_json_key` | `imu/idle_spans.json`(静止区間) |
| `get_imu_pdr_csv_key` | (recon3d は未使用。`USE_PDR_BRIDGE` の実験経路のみ) |

> **builder check の訂正**: レビューは「prepro から出るのは `imu_pdr.csv` と
> idle 判定だけなので、telemetry_parser + frame_times + stairnet の移植が別途
> 要る」と報告したが、**誤り**。階段遷移の JSON も prepro が上げている
> (`pipes/prepro_gpu/src/stages/imu_pdr.py:73-77`)。移植は不要。

階の割当は recon3d の `backend/floor_assign.json`(手動)に対し、pipeline 側は
運用者の `FloorAssignment`(floor.csv 由来)。ここは変換を1枚挟む。

### 8. アンカーできなかった島も必ず出す(設計者の判断)

recon3d はアンカー失敗島・階未割当島・36フレーム未満の島を落とす。
採点の分母は全正解フレームなので、黙って落とすと実力より遥かに悪い行になる。
**失敗した島も `placed=False` のトラックとして出す。**
recon_ff の `drawing_fit` が既に同じ方針を採っている — 「フィットできなかった
セグメントも必ず出す(`rescued`)── 無音で消えるとオペレータに原因も
直し方も分からない」。それに揃える。

## 未決(依頼者が決めること)

### 9. マイルストーンを2つに割る(スケールが揃うまで図面合わせは比較できない)

依頼者の回答: `m_per_unit` は**将来 MapAnything で出す**予定、図面の mm/px は
**将来 floor.csv のようにアノテーションする**予定。どちらも今は無い。

したがって:

| | 比べるもの | 要るもの | いつ |
|---|---|---|---|
| **M1** | 軌跡の**形**(`形の正答率%` / `形残差`) | フレーム + prepro の IMU 成果物のみ。**図面もスケールも不要** | いますぐ |
| **M2** | 図面合わせ(位置) | mm/px アノテーション + MapAnything 由来の `m_per_unit` | 両方が揃ってから |

M1 が今すぐ成立するのは、recon3d が図面なしでも軌跡生成まで走るから
(`core/caps` が `anchor` を能力不足で skip する)。そして
`eval/README.md` が「**今 engine を跨いで比べられるのは `形の正答率%` と
`形残差` だけ**」と言っているので、M1 の指標は既存の土俵にそのまま乗る。

**M2 までは recon3d を「落ちずに完走して当たらない」状態で回さないこと。**
スケールが無いまま図面アンカーを走らせると `slab`(誤差97〜383%)に落ち、
`match_corners` の ±40% 探索窓に正解が入らない。

### 10. M1 の採点は `score_recon.py` に足す(設計者の判断)

`eval/src/engine_recon3d.py`(`load` / `tracks_of`)を書き、`_load()` に分岐を
足し、`--engine` に名前を増やす。M1 の指標は形だけで、形は README が
「engine を跨いで比べられる」と保証している範囲なので、recon_ff の独自ハーネスと
突き合わせる必要が無い。

**位置(M2)で recon_ff と直接比べる段になったら、採点経路の二重を解消する
必要がある**(`engine_recon_ff.py` を書いて `score_recon.py` 側へ移すか、
recon3d を `pipes/recon_ff/eval/` 側に入れるか)。M2 の課題として残す。

## 未決(依頼者が決めること)

### B. 図面合わせの入力(M2 の前提)

**どちらも「将来出す」段階で、今は245件側に存在しない。**

- **(a) 図面の mm/px** — `floors/meta.json` の `known_width_mm`。pipeline 側の
  図面情報は `floor.csv` の `floor_id` + `image_url` だけで、正解も
  `captured_x_percent`(%)。**実寸は一次資料に一切無い。** recon_ff の fit は
  scale を総当たりするので不要だが、recon3d は `core/plan_grid.PlanGrid` が
  これを要求し、無ければ `SystemExit`。
- **(b) 島の `m_per_unit`** — `postproc/scale_prior.py`。供給源は
  `calib`(同一物件の既アンカー済み `scale_bias.json`)/ `mapany`(MapAnything
  推論 = 追加の GPU パス)/ `slab`(誤差97〜383%)。`match_corners` は事前の
  **±40% しか探索しない**ので、`slab` に落ちると正解の仮説が生成されない。
  245件には較正履歴も macond タグも無い。

依頼者の回答: (b) は MapAnything で出す予定、(a) は floor.csv のように
アノテーションする予定。**残る問いは「いつ・何件」**で、それが M2 の開始条件と
対象件数を決める。

加えて壁の出どころ: recon3d は人手の labelme 注釈、recon_ff は
`steps/cubicasa.py`。CubiCasa は精度不足で再学習待ち。
M2 で recon3d の壁を CubiCasa にするのか、注釈のある物件に限るのかも未決。

### C. M2 の勝敗は何で決めるのか。対象は何件か

(M1 の指標は上記 9 で決まった — 形の正答率% と形残差。)

目標4は「勝った方式を recon_ff に統合」だが判定基準が無い。しかも両者は
差し替え可能な同型ではない — **recon_ff はスケールも壁注釈も要らず全件回る、
recon3d は両方要る**。手で注釈した少数で recon3d が勝っても、その結果は
245件へは移らない。

指標(位置の正答率? 形残差? セグメント数?)・対象サブセット・「勝ち」と呼ぶ差を
決めないと、(a) 何件に注釈を発注するか(B)が決まらず、(b)「芯を剥がす」投資の
是非も判断できない。

### E. `reconstruction3d_experimental` の公開先

**git remote が無い。** submodule には URL が要り、pipeline の CI
(`scripts/ci/images.py` が Dockerfile の COPY 行から submodule を init する)が
fetch できる必要がある。公開先の org・可視性・権限が未定。
あわせて、研究用の pipe を本番リポジトリの常設 image 群に置くことを
pipeline 側のオーナーが許すかも確認が要る。

## builder に委ねること

- 芯を剥がす具体(どの関数から、どう割るか)。`postproc/corners.corner_anchor`
  と `postproc/anchor.prepare_island` が起点になるはず。
- `pipes/recon_3d/` の step 分割と S3 レイアウト。
- `engine_recon3d.py` が読む成果物の形。
- 既存の回帰(0912 の5島アンカー12 JSON のバイト一致、chains の決定性)を
  剥がす作業の金床として使うこと。

## 参照

- `recon3d/PIPELINE.md` — 現状の段構成・設定・落とし穴
- `recon3d/results/REPORT.md` — 実測の記録(2026-10-20 節にポーズグラフの
  スケール境界)
- pipeline `docs/coding-rules.md#pipe` / `docs/pipes.md` / `eval/README.md`
