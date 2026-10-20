# recon3d.plan2model — 平面図JSON → 3次元モデル再構成

GT(labelme)または CubiCasa の prediction.json から、壁・柱・開口(窓/ドア)・スラブで
構成される3Dモデル(glTF/OBJ)を生成する。詳細な経緯は `../MODEL_RECONSTRUCTION.md` と
`../docs/superpowers/plans/2026-09-03-recon3d-gt-to-3d.md` を参照。

## 使い方

入口は**リポジトリ直下の `main.py`** (compose サービス `recon3d`)。
入力は他の段と同じ固定構造のディレクトリで、必要なのは `floors/` だけ。

```
<input>/
  floors/  1F.json 1F.jpg 2F.json 2F.jpg  meta.json
  results/plan2model/                       ← 出力
```

`floors/meta.json` に図面の実寸を書く (最外壁芯間のx方向スパン[mm])。
階でスパンが違うなら階名→mm の辞書にする。

```json
{"known_width_mm": 9100.0}
```

```bash
cd /home/maren/programming

# 2D再投影検証 (IoU + オーバーレイPNG) — まずこれで図面の読みを確かめる
docker compose exec -T recon3d python main.py plan2model verify \
  input=/code2/reconstruction_3d/28-AAAAA/recon_11177

# モデル生成 (glb / obj / report.json)
docker compose exec -T recon3d python main.py plan2model build \
  input=/code2/reconstruction_3d/28-AAAAA/recon_11177

# コンテナ生成ファイルの所有権修正
docker compose exec -T recon3d chown -R 1001:1001 \
  /code2/reconstruction_3d/28-AAAAA/recon_11177/results
```

階は `floors/*.json` から**下階から順に**自動で拾う (`1F` `2F` `B1F` `RF` …、
`meta.json` は除外)。PATHはlabelme GTでもprediction.jsonでもよい
(スキーマ自動判別)。再学習済みモデルが来たら JSON を差し替えるだけ。

### テスト

```bash
docker compose exec -T recon3d python -m pytest \
  /code2/reconstruction_3d/recon3d/plan2model/tests -v
```

## モジュール構成

| ファイル | 責務 |
|---|---|
| `loader.py` | labelme GT / prediction.json → 共通スキーマ `FloorPlan`(軸平行`Rect`のリスト) |
| `calibrate.py` | px→mm校正。外周垂直壁の中心線xスパン ↔ 通り芯寸法(`known_width_mm`)の比。y方向で交差検証 |
| `config.py` | 高さ仮定値(階高2900 / 窓 腰900+H1100 / ドアH2000 / スラブ150)。全てCLI・コードから上書き可 |
| `geometry.py` | 開口の振り分けとボックス生成(下記) |
| `mesh.py` | Box3D列+スラブ → trimesh.Scene → glb/obj |
| `build.py` | `build_from_paths`。複数階の組み立て(階iのbase_z = i×階高)と report.json |
| `verify.py` | `verify_from_paths`。壁+柱の2D再投影 vs GT の pixel IoU、オーバーレイPNG |

## 重要な設計事実(28-AAAAAで実測)

1. **スケールは理論値4.233mm/pxではない**。GT座標系では1F通り芯横全幅9100mm=760.3px
   → **11.968mm/px**。必ず寸法線(通り芯スパン)から動的に校正する。
   2F幅も 609px×11.968≈7288mm ≈ 8×910mmモジュールで整合、y方向誤差0.27%。
2. **GTの壁は開口位置で途切れており、窓/ドア矩形は壁の隙間を埋める**(壁との重なり最大20%)。
   よって開口は壁からの引き算ではなく、開口矩形への充填(腰壁/垂れ壁=`wall_infill`+建具=`pane_*`)
   で処理する。prediction.json系で開口が壁に重畳する場合(重なり≥50%)のみ壁から引き算する。
3. **スラブのフットプリントは壁+柱+開口のunion**で計算する(壁だけでは外周が閉じない)。
   中庭等の穴は埋め、離れた断片は最大要素のみ採用。
4. 1F/2FはGTが同一CAD座標系(右端通り芯cx差0.5px)のため平面位置合わせは恒等変換。
5. 全要素が軸平行矩形なのでブーリアンエンジン不要(1D区間分割で完結)。

## 検証結果 (28-AAAAA GT)

- 2D再投影IoU: **1F=1.0 / 2F=1.0**(`results/plan2model/overlay_{1F,2F}.png` で目視確認可)
- スケール交差検証誤差: 0.28%(y方向9555mm比)
- 生成物: `results/plan2model/model.glb`(カテゴリ別色分け・窓は半透明)、`model.obj`、
  `report.json`(スケール・仮定値・要素数)、`render_iso.png`(簡易等角図)

## 既知の制限 / 次の課題

- 高さは全て仮定値(図面注記 2392H 等の自動読み取りは未実装)
- 屋根は最上階の平スラブ(勾配屋根なし)。バルコニー・ポーチ屋根なし
- 開口の建具区別(引違/FIX/勝手口等)なし。壁厚の実寸正規化(105/120mm)は未適用
- prediction.json入力は動作するが、現行モデルの精度が低いため実用は再学習後
