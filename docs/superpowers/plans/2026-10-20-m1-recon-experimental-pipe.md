# M1: recon_experimental パイプ (軌跡の形で比較できるようにする) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** recon3d の軌跡生成を pipeline の `reconstruction` asset に `engine=EXPERIMENTAL` として載せ、245件の評価台帳で recon_ff / dfsfm と**形**で比較できるようにする。

**Architecture:** 新しい pipe `pipes/recon_experimental/` が、recon3d を submodule として内包する。entry は薄く、`step()` で「作業ディレクトリ構築 → 島計画 → チャンク推論(GPU) → チェーン → 軌跡変換」を直列に呼ぶ。出力は**エンジン非依存の seam** `trajectory/{start}_{end}.json` (`list[TransformedRefinedTrajectory]`) 1本だけで、下流は既存の `recon_result` がそのまま拾う。

**Tech Stack:** Python 3.11 / pydantic (contracts) / dagster-pipes / PanoVGGT (torch 2.5, bf16) / numpy。pipeline 側は uv、recon3d 側は既存の Docker イメージ。

**Spec:** `docs/superpowers/specs/2026-10-20-recon3d-pipeline-integration-design.md`

> **設計書の訂正 (この計画の前提)。** 仕様 §5/§10 は「`eval/src/engine_recon3d.py` を書いて `score_recon.py` に分岐を足す」としているが、**不要**だった。`get_trajectory_json_key` の docstring が「**エンジン非依存**。dfsfm / opensfm / vslam のどれが作っても同じキーへ書き、`recon_result` はここだけを読む」と明言しており、`eval/src/engine_dfsfm.load()` が読むのはその `recon_result` が出す `results/traj_result_*.json` である。したがって seam さえ正しく書けば、既存の `score_recon.py --engine dfsfm` で採点できる (loader 名が engine 名に見えるだけで、中身はエンジン非依存)。この計画は eval 側に**何も足さない**。

## Global Constraints

- `steps/` は**純関数**。`dagster_pipes` / `pipes_runtime` / `aws` / `boto3` / 同じ pipe の `io` を import しない。`pipes_runtime.errors` は可。**[検査]**
- `steps/` は互いを import しない。共有は `utils/` か `schemas/` へ。**[検査]**
- `src/__init__.py`、`src/entry/`(`__init__.py` + `<asset>.py`)、`src/io.py` は必須。`io/` ディレクトリ・`src/main.py`・`stages.py` は禁止。**[検査]**
- entry module は `if __name__ == "__main__": main()` を持つ。**[検査]**
- image 定数は `RECON_EXPERIMENTAL_IMAGE`(`<PIPE_DIR大文字>_IMAGE`)。起動は `python -m src.entry.reconstruction`。
- ミニ DAG を作らない。直列 python 呼び出しがそのまま DAG。
- vendor は `COPY submodules/<X> /pymodules/<X>`。`src` を PYTHONPATH に載せない。**[検査]**
- image は `PYTHONUNBUFFERED=1`。
- PanoVGGT は **bf16**(fp16 は NaN)。`xformers==0.0.28.post3` をピンする (入れないと attention 実装が変わり既存チャンクと数値が揃わない)。
- seam の byte 仕様は `contracts.pipes._shared.recon.dump_trajectories` を通す。各 pipe で `dumps_list` を呼ばない。
- recon3d 側は `paths` を第1引数に取り**既定値を置かない**(`recon3d/tests/check_imports.py` が検査)。

## Review Focus

1. **フレームが1枚も無い / 範囲外** — `recon_range` がフレーム集合と交わらないとき。島が0本になり `n_segment=0` → `ResultStatus.EMPTY` を返すべきで、例外で落ちてはいけない。→ Task 8。
2. **IMU 成果物が無い撮影** — prepro が `transitions.json` / `idle_spans.json` を上げていない場合。recon3d の `imu_segments` は `detect()` を呼びに行きコンテナ専用パスで落ちる。無ければ**単一区間**に縮退すべき。→ Task 4。
3. **移動量ゼロの軌跡** — 静止したままの撮影。`_normalize_position_by_long_side` は `(0.5, 0.5, z)` を返す契約なので、ここで ZeroDivision を起こさないこと。→ Task 7。
4. **島が1本しか立たない** — チャンクが2個未満だとチェーンできない。その島を黙って捨てず、捨てた事実をログと metadata に出す。→ Task 6。
5. **同じ scan を2回流す** — チャンクが既にある状態での再実行。`infer_chunks` は `cameras.npz` の有無で飛ばす設計なので、2回目が推論を飛ばして同じ seam を出すこと。→ Task 5。

---

### Task 1: recon3d — チェーン出力を素のフレーム姿勢に変える

pipeline 側が recon3d の npz レイアウトを知らずに済むよう、recon3d が素の値で返す。pipeline の contract 型には依存しない (recon3d は pipeline を知らない)。

**Files:**
- Create: `recon3d/out/trajectory.py` (リポジトリ: `reconstruction3d_experimental`)
- Test: `recon3d/tests/test_trajectory.py`

**Interfaces:**
- Consumes: なし
- Produces: `FramePose` (frozen dataclass: `frame: int`, `pos: tuple[float, float, float]`, `quat: tuple[float, float, float, float]` = (x, y, z, w)) と `frame_poses(cameras_npz: Path) -> list[FramePose]` (frame 昇順)

- [ ] **Step 1: 失敗するテストを書く**

```python
def test_frame_poses_reads_names_positions_poses(tmp_path):
    npz = tmp_path / "global_cameras.npz"
    np.savez(npz,
             names=np.array(["frame_000002", "frame_000000"]),
             positions=np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]),
             poses=np.stack([np.eye(4), np.eye(4)]))
    out = frame_poses(npz)
    assert [p.frame for p in out] == [0, 2]          # 昇順に並べ替える
    assert out[1].pos == (1.0, 2.0, 3.0)
    assert out[0].quat == (0.0, 0.0, 0.0, 1.0)       # 単位回転

def test_frame_poses_rejects_missing_file(tmp_path):
    with pytest.raises(SystemExit):
        frame_poses(tmp_path / "nope.npz")
```

- [ ] **Step 2: テストが落ちることを確認**

Run: `python -m pytest recon3d/tests/test_trajectory.py -v`
Expected: FAIL (`ModuleNotFoundError: recon3d.out.trajectory`)

- [ ] **Step 3: `frame_poses(cameras_npz: Path) -> list[FramePose]` を `recon3d/out/trajectory.py` に実装**

`poses` は (N,4,4) の同次変換。回転部 `[:3,:3]` を quaternion (x,y,z,w) に直す。変換は `scipy.spatial.transform.Rotation` ではなく、既に依存している numpy だけで書くこと (`recon3d/core/geometry.py` に quaternion→回転の `quat_to_rot` があるので、その逆変換をここに置く)。位置は `positions` をそのまま使う (`poses[:, :3, 3]` ではない — `chain_and_save` が重心などで補正した値が `positions` に入る)。

- [ ] **Step 4: テストが通ることを確認**

Run: `python -m pytest recon3d/tests/test_trajectory.py -v`
Expected: PASS

- [ ] **Step 5: コミット**

```bash
git add recon3d/out/trajectory.py recon3d/tests/test_trajectory.py
git commit -m "feat(out): チェーン出力を素のフレーム姿勢で返す frame_poses"
```

---

### Task 2: pipeline — contracts (extras と engine enum)

**Files:**
- Create: `common/src/contracts/pipes/recon_experimental/__init__.py`
- Create: `common/src/contracts/pipes/recon_experimental/extras.py`
- Modify: `common/src/contracts/enums.py` (`ReconEngineEnum` に `EXPERIMENTAL = "EXPERIMENTAL"`)
- Test: `common/tests/contracts/pipes/test_recon_experimental_extras.py`

**Interfaces:**
- Consumes: `contracts.pipes._shared.base.PipesExtras`、`contracts.pipes._shared.recon.ReconRange`
- Produces: `ReconExperimentalExtras` — フィールドは `scan_meta_data_s3_key: str` / `polish_pipeline_bucket_name: str` / `frames_s3_prefix: str` / `recon_range: ReconRange | None = None` / `max_seg: int = 160` / `stride: int | None = None` / `chunk_size: int | None = None` / `use_idle: bool = True` / `num_parallel: int | None = None`

- [ ] **Step 1: 失敗するテストを書く**

```python
def test_extras_defaults_match_recon3d():
    e = ReconExperimentalExtras(scan_meta_data_s3_key="k", polish_pipeline_bucket_name="b",
                                frames_s3_prefix="p")
    assert (e.max_seg, e.use_idle, e.recon_range) == (160, True, None)

def test_engine_enum_has_experimental():
    assert ReconEngineEnum("EXPERIMENTAL") is ReconEngineEnum.EXPERIMENTAL
```

- [ ] **Step 2: テストが落ちることを確認**

Run: `uv run --package common pytest common/tests/contracts/pipes/test_recon_experimental_extras.py -v`
Expected: FAIL (import error)

- [ ] **Step 3: `ReconExperimentalExtras` と enum メンバを実装**

既定値 160 / True は recon3d の `conf/recon.yaml` の `chunking.max_seg` と `chunking.use_idle` に一致させること (2箇所に別の既定があると「同じ設定のつもりで違う結果」になる)。

- [ ] **Step 4: テストが通ることを確認**

Run: `uv run --package common pytest common/tests/contracts/pipes/test_recon_experimental_extras.py -v`
Expected: PASS

- [ ] **Step 5: コミット**

---

### Task 3: pipe — 作業ディレクトリを recon3d の入力契約に組む

recon3d は「固定構造の1ディレクトリ」を入力に取る。S3 から落としたフレームと IMU をその形に並べる。

**Files:**
- Create: `pipes/recon_experimental/src/steps/workdir.py`
- Test: `pipes/recon_experimental/tests/test_workdir.py`

**Interfaces:**
- Consumes: Task 1 なし
- Produces: `build_workdir(root: Path, frame_paths: Sequence[Path], imu_paths: Mapping[str, Path]) -> Path` — `root/{images,imu,results}` を作り、フレームを `images/frame_number_%06d.jpg` の名前で **symlink**、IMU を `imu/<名前>` に置き、`root` を返す

- [ ] **Step 1: 失敗するテストを書く**

```python
def test_build_workdir_lays_out_recon3d_contract(tmp_path):
    src = tmp_path / "src"; src.mkdir()
    frames = []
    for i in (0, 1, 2):
        p = src / f"{i}.jpg"; p.write_bytes(b"x"); frames.append(p)
    imu = src / "transitions.json"; imu.write_text("{}")
    root = build_workdir(tmp_path / "wd", frames, {"transitions.json": imu})
    assert (root / "images" / "frame_number_000000.jpg").is_symlink()
    assert sorted(p.name for p in (root / "images").iterdir())[-1] == "frame_number_000002.jpg"
    assert (root / "imu" / "transitions.json").read_text() == "{}"
    assert (root / "results").is_dir()

def test_build_workdir_is_idempotent(tmp_path):
    src = tmp_path / "a.jpg"; src.write_bytes(b"x")
    build_workdir(tmp_path / "wd", [src], {})
    build_workdir(tmp_path / "wd", [src], {})      # 2回目で落ちない
```

- [ ] **Step 2: テストが落ちることを確認**

Run: `uv run --package recon_experimental pytest pipes/recon_experimental/tests/test_workdir.py -v`
Expected: FAIL

- [ ] **Step 3: `build_workdir` を実装**

フレーム番号は `frame_paths` の並び順ではなく、**ファイル名の数字**から取る (prepro の出力名を保つ。並び順に振り直すと `recon_range` のフレーム番号と一致しなくなる)。symlink にするのはコピーで GB を二重に持たないため。

- [ ] **Step 4-5: テスト PASS を確認してコミット**

---

### Task 4: pipe — 島の計画 (IMU 区間 + マニフェスト)

**Files:**
- Create: `pipes/recon_experimental/src/steps/islands.py`
- Test: `pipes/recon_experimental/tests/test_islands.py`

**Interfaces:**
- Consumes: Task 3 の `root`
- Produces: `plan_islands(root: Path, *, max_seg: int, stride: int | None, chunk_size: int | None, use_idle: bool) -> IslandPlan`。`IslandPlan` は frozen dataclass: `manifest: list[dict]` / `windows: list[tuple[int, int]]` / `entries: list[dict]` / `chunks_dir: str`

- [ ] **Step 1: 失敗するテストを書く**

```python
def test_plan_islands_falls_back_to_single_segment_without_imu(tmp_path, monkeypatch):
    root = _workdir_with_frames(tmp_path, n=40)          # imu/ を置かない
    plan = plan_islands(root, max_seg=160, stride=None, chunk_size=None, use_idle=True)
    assert len(plan.manifest) == 1                        # 区間を切らずに1島
    assert plan.manifest[0]["a"] == 0

def test_plan_islands_uses_imu_cuts_when_present(tmp_path):
    root = _workdir_with_frames(tmp_path, n=120)
    (root / "imu" / "transitions.json").write_text(
        json.dumps({"events": [{"t_start": 20.0, "t_end": 24.0,
                                "frame_start": 40, "frame_end": 48}]}))
    plan = plan_islands(root, max_seg=160, stride=None, chunk_size=None, use_idle=False)
    assert len(plan.manifest) == 2                        # 切断点で2島
```

- [ ] **Step 2: テストが落ちることを確認**

- [ ] **Step 3: `plan_islands` を実装**

`recon3d.paths.from_input` で `Paths` を作り、`recon3d.core.chunking` の `chunk_windows` / `imu_segments` / `build_manifest` と `recon3d.core.panovggt.select_entries` を呼ぶ。**`imu/transitions.json` が無ければ `imu_segments` を呼ばず `segments=None` で `build_manifest` する** — recon3d の `imu_segments` は無いと `detect()` を呼び、コンテナ専用パス決め打ちで原因の分かりにくい `ModuleNotFoundError` になる (Review Focus 2)。

- [ ] **Step 4-5: テスト PASS を確認してコミット**

---

### Task 5: pipe — チャンク推論 (GPU)

**Files:**
- Create: `pipes/recon_experimental/src/steps/inference.py`
- Test: `pipes/recon_experimental/tests/test_inference.py`

**Interfaces:**
- Consumes: Task 4 の `IslandPlan`
- Produces: `infer_chunks(root: Path, plan: IslandPlan) -> int` — 推論した**新規**チャンク数を返す (既存は飛ばす)

- [ ] **Step 1: 失敗するテストを書く**

torch を要求しないよう、recon3d 側の `run_chunks` を monkeypatch して「何が渡ったか」だけを見る。

```python
def test_infer_chunks_skips_existing(tmp_path, monkeypatch):
    root, plan = _plan_with_chunks(tmp_path, needed=[0, 1])
    _make_chunk(root, 0)                                   # chunk_00/cameras.npz を置く
    seen = {}
    monkeypatch.setattr("recon3d.core.panovggt.run_chunks",
                        lambda *a, **kw: seen.update(miss=a[1]))
    n = infer_chunks(root, plan)
    assert n == 1 and [k for k, *_ in seen["miss"]] == [1]   # 既存の 0 は渡らない

def test_infer_chunks_returns_zero_when_all_present(tmp_path, monkeypatch):
    root, plan = _plan_with_chunks(tmp_path, needed=[0])
    _make_chunk(root, 0)
    monkeypatch.setattr("recon3d.core.panovggt.run_chunks",
                        lambda *a, **kw: pytest.fail("推論してはいけない"))
    assert infer_chunks(root, plan) == 0
```

- [ ] **Step 2: テストが落ちることを確認**

- [ ] **Step 3: `infer_chunks` を実装**

`recon3d.core.chunking.infer_chunks(paths, windows, needed, chunks_dir, entries)` に委譲し、戻り値のために「欠けているチャンク数」を自分でも数える。GPU と torch の依存はこの step に閉じる (`steps/panovggt.py` が recon_ff でやっているのと同じ役)。

- [ ] **Step 4-5: テスト PASS を確認してコミット**

---

### Task 6: pipe — 島ごとのチェーン

**Files:**
- Create: `pipes/recon_experimental/src/steps/chaining.py`
- Test: `pipes/recon_experimental/tests/test_chaining.py`

**Interfaces:**
- Consumes: Task 4 の `IslandPlan`、Task 5 の副作用 (チャンク)
- Produces: `chain_islands(root: Path, plan: IslandPlan) -> ChainResult`。`ChainResult` は frozen dataclass: `tags: list[str]` / `skipped: list[tuple[str, str]]` (タグ, 理由)

- [ ] **Step 1: 失敗するテストを書く**

```python
def test_chain_islands_skips_island_with_one_chunk(tmp_path, monkeypatch):
    root, plan = _plan_with_manifest(tmp_path, islands=[("a", [0, 1]), ("b", [2])])
    monkeypatch.setattr("recon3d.core.chaining.chain_and_save", lambda *a, **kw: None)
    res = chain_islands(root, plan)
    assert res.tags == ["a"]
    assert res.skipped == [("b", "チャンクが2個未満")]      # 黙って捨てない
```

- [ ] **Step 2: テストが落ちることを確認**

- [ ] **Step 3: `chain_islands` を実装**

マニフェストの島ごとに `recon3d.core.chaining.chain_and_save(paths, cdirs, tag)` を呼ぶ。チャンクが2個未満の島は理由つきで `skipped` に積む (Review Focus 4)。1島の失敗で全体を落とさない。

- [ ] **Step 4-5: テスト PASS を確認してコミット**

---

### Task 7: pipe — 島 → seam の軌跡

**Files:**
- Create: `pipes/recon_experimental/src/steps/trajectory.py`
- Test: `pipes/recon_experimental/tests/test_trajectory.py`

**Interfaces:**
- Consumes: Task 1 の `frame_poses`、Task 6 の `ChainResult`
- Produces: `build_trajectories(root: Path, tags: Sequence[str]) -> list[TransformedRefinedTrajectory]` — 島1本 = 軌跡1本

- [ ] **Step 1: 失敗するテストを書く**

```python
def test_build_trajectories_one_per_island(tmp_path):
    root = _bench_with(tmp_path, {"a": [(0, (0, 0, 0)), (1, (1, 0, 0))],
                                  "b": [(5, (0, 0, 0))]})
    out = build_trajectories(root, ["a", "b"])
    assert len(out) == 2
    assert [p.frame_number for p in out[0].transformed_frame_points] == [0, 1]
    assert all(t.auto_adjusted is False for t in out)

def test_build_trajectories_handles_zero_extent(tmp_path):
    root = _bench_with(tmp_path, {"a": [(0, (0, 0, 0)), (1, (0, 0, 0))]})
    pt = build_trajectories(root, ["a"])[0].transformed_frame_points[0].transformed_point
    assert (pt.width_normalized_drawing_x, pt.width_normalized_drawing_y) == (0.5, 0.5)
```

- [ ] **Step 2: テストが落ちることを確認**

- [ ] **Step 3: `build_trajectories` を実装**

`frame_poses` の返りを `contracts.pipes._shared.recon` の `FramePoint` / `Point` に詰めて `RefinedTrajectory` を作り、**`common.recon.transform.transform_refined_trajectories` に渡す**。正規化と紙面座標への回転は自前で書かない (3エンジン共通の1本なので、ここで別実装を作ると比較の土俵がずれる)。移動量0の縮退は `transform` 側の契約 (Review Focus 3) なので、テストはその契約を確かめるだけでよい。

- [ ] **Step 4-5: テスト PASS を確認してコミット**

---

### Task 8: pipe — io.py と entry (直列オーケストレーション)

**Files:**
- Create: `pipes/recon_experimental/src/__init__.py`, `src/config.py`, `src/io.py`
- Create: `pipes/recon_experimental/src/entry/__init__.py`, `src/entry/reconstruction.py`
- Test: `pipes/recon_experimental/tests/test_entry.py`

**Interfaces:**
- Consumes: Task 2〜7 すべて
- Produces: `main()` (起動点)、`run(pipes: PipesContext, cfg: ReconExperimentalExtras) -> None`
- `io.py`: `download_frames(s3, prefix, dest, rng) -> list[Path]` / `download_imu(s3, scan_key, dest) -> dict[str, Path]` / `upload_trajectories(s3, key, trajectories) -> None`

- [ ] **Step 1: 失敗するテストを書く**

```python
def test_run_reports_empty_when_no_island(monkeypatch, fake_pipes):
    _stub_steps(monkeypatch, tags=[], skipped=[])
    run(fake_pipes, _extras())
    meta = fake_pipes.materializations[-1]
    assert meta["result_status"] == "EMPTY" and meta["result_size"] == 0

def test_run_writes_seam_only_with_recon_range(monkeypatch, fake_pipes):
    uploaded = []
    _stub_steps(monkeypatch, tags=["a"], uploaded=uploaded)
    run(fake_pipes, _extras(recon_range=None))
    assert uploaded == []                                  # 範囲が無ければ seam を書かない
    run(fake_pipes, _extras(recon_range=ReconRange(start_frame_number=0, end_frame_number=9)))
    assert uploaded[0].endswith("/trajectory/0_9.json")
```

- [ ] **Step 2: テストが落ちることを確認**

- [ ] **Step 3: `io.py` と `entry/reconstruction.py` を実装**

entry は `open_pipes()` → `ReconExperimentalExtras.model_validate(pipes.get_extra(EXTRAS_KEY))` → `step(pipes, …)` で Task 3〜7 を順に呼ぶ → `io.upload_trajectories` → `report_asset_materialization`。metadata は `timings_metadata()` + `resource_metadata()` + `AssetMetadata(input_unit=InputUnit.FRAMES, …)`。島が0本なら `ResultStatus.EMPTY` (Review Focus 1)。S3 キーは `contracts.s3_layout` の deriver 経由 (文字列直組み禁止)。アップロードは `dump_trajectories` を通す。

- [ ] **Step 4-5: テスト PASS を確認してコミット**

---

### Task 9: pipe — image と submodule

**Files:**
- Create: `pipes/recon_experimental/Dockerfile`, `pyproject.toml`, `uv.lock`, `download_large_files.sh`
- Modify: `.gitmodules` (`submodules/reconstruction3d_experimental` を追加)
- Modify: `docker-compose.yml` (CI の build plan が `platform` をここから読む)

**Interfaces:**
- Consumes: Task 8 の `src/`
- Produces: image `pipes_recon_experimental`

- [ ] **Step 1: submodule を足す**

```bash
git submodule add https://github.com/SoftRoid-Inc/reconstruction3d_experimental.git \
  submodules/reconstruction3d_experimental
```

- [ ] **Step 2: Dockerfile を書く**

`pipes/recon_ff/Dockerfile` を下敷きにする。`COPY submodules/PanoVGGT /pymodules/PanoVGGT` と `COPY submodules/reconstruction3d_experimental /pymodules/reconstruction3d_experimental`、重みは `download_large_files.sh` で HF から取り `PANOVGGT_MODEL_PATH` で指す。`xformers==0.0.28.post3` をピン。`PYTHONUNBUFFERED=1`。`CMD ["python", "-m", "src.entry.reconstruction"]`。

- [ ] **Step 3: build が通ることを確認**

Run: `docker build -f pipes/recon_experimental/Dockerfile -t pipes_recon_experimental:dev .`
Expected: 成功し、build 最後の smoke (`python -c "import recon3d, torch"`) が通る

- [ ] **Step 4: 規約検査が通ることを確認**

Run: `uv run python scripts/ci/check_pipe_layout.py` (実名はリポジトリの検査スクリプトに合わせる)
Expected: `recon_experimental` が合格

- [ ] **Step 5: コミット**

---

### Task 10: orchestration — asset 分岐と投入経路

**Files:**
- Modify: `locations/polish_pipeline/polish_pipeline/defs/assets/recon/_common.py` (`RECON_EXPERIMENTAL_IMAGE = pipe_image("recon_experimental")`)
- Modify: `locations/polish_pipeline/polish_pipeline/defs/assets/recon/reconstruction.py` (`_run_experimental` と分岐)
- Modify: `locations/polish_pipeline/polish_pipeline/defs/utils/recon_engine.py` (enum 検査)
- Modify: `eval/run_recon.py` (`--engine` の choices に `experimental`)
- Test: `locations/polish_pipeline/tests/assets/recon/test_reconstruction_experimental.py`

**Interfaces:**
- Consumes: Task 2 の `ReconExperimentalExtras` / `ReconEngineEnum.EXPERIMENTAL`、Task 9 の image
- Produces: `_run_experimental(context, config, recon_inputs, pipes_client) -> dg.MaterializeResult`

- [ ] **Step 1: 失敗するテストを書く**

```python
def test_dispatch_experimental_uses_experimental_image(fake_pipes_client):
    _materialize(engine=ReconEngineEnum.EXPERIMENTAL, client=fake_pipes_client)
    assert fake_pipes_client.calls[-1]["image"] == RECON_EXPERIMENTAL_IMAGE
    assert fake_pipes_client.calls[-1]["asset"] == "reconstruction"

def test_experimental_payload_carries_recon_range(fake_pipes_client):
    _materialize(engine=ReconEngineEnum.EXPERIMENTAL, client=fake_pipes_client)
    assert fake_pipes_client.calls[-1]["payload"].recon_range is not None
```

- [ ] **Step 2: テストが落ちることを確認**

- [ ] **Step 3: `_run_experimental` と分岐を実装**

`_run_ff` を下敷きに、図面関係 (`floors` / `fit`) を持たない版にする。GPU を使うので `use_gpu=True` と GPU 用の `pod_spec`。pod のサイズは最初は `RECON_FF_GPU_POD_SPEC` と同じにし、初回実行の `step/timings` の `eff_cores` を見て調整する。

- [ ] **Step 4-5: テスト PASS を確認してコミット**

---

### Task 11: 1件を端から端まで通し、採点する

**Files:**
- Modify: なし (実行と記録のみ)
- Create: `pipes/recon_experimental/README.md` (単体での動かし方)

**Interfaces:**
- Consumes: Task 1〜10 すべて

- [ ] **Step 1: pipe を単体で動かす (Dagster 無し)**

Run: `docs/pipes.md#pipe-単体で動かす-dagster-無し` の手順で、IMU 成果物のある scan を1件
Expected: `trajectory/{start}_{end}.json` が書かれ、`step/timings` に workdir / plan / infer / chain / trajectory の5行が出る

- [ ] **Step 2: 同じ scan をもう一度動かす**

Expected: `infer_chunks` が 0 を返し、推論を飛ばして同じ seam を出す (Review Focus 5)

- [ ] **Step 3: test 環境へ1件投入する**

Run: `AWS_PROFILE=dtsu python eval/run_recon.py launch --engine experimental --scan-id <1件> --tag m1-smoke`
Expected: `reconstruction` → `recon_result` が成功し、`results/traj_result_{start}_{end}.json` ができる

- [ ] **Step 4: 既存の採点経路で採点する**

Run: `AWS_PROFILE=dtsu EVAL_DRAWING_AWS_PROFILE=default python eval/score_recon.py --tag m1-smoke --scan-id <1件> --engine dfsfm`
Expected: 形の指標 (`形の正答率%` / `形残差`) が出る。**`engine_recon3d.py` は書かない** — seam がエンジン非依存なので既存の loader で読める

- [ ] **Step 5: README を書いてコミット**

単体実行の手順、`PLAN2MODEL_GT_DIR` のような環境変数、既知の限界 (M1 は形だけ。図面フィットは M2) を書く。

---

## 自己レビュー

**仕様の網羅**: M1 の範囲 (仕様 §9 の表の上段) は Task 1〜11 で埋まる。仕様 §5/§10 の `engine_recon3d.py` は、調査の結果**不要**と分かったので冒頭に訂正を置いた (seam がエンジン非依存)。仕様 §2 の「芯を I/O から剥がす」は M1 では Task 1 の範囲だけ必要 (図面合わせの剥がしは M2)。仕様 §7 の IMU 成果物は Task 3/4 が使う。仕様 §8 (失敗島も出す) は Task 6 の `skipped` と Task 8 の metadata。

**未着手として残すもの (M2)**: 図面フィット、`m_per_unit`、mm/px 注釈、採点経路の二重解消。いずれも仕様の未決 B/C に紐づく。

**型の一貫性**: `IslandPlan` (Task 4) は Task 5/6 が消費、`ChainResult.tags` (Task 6) は Task 7 が消費、`FramePose` (Task 1) は Task 7 が消費。`ReconExperimentalExtras` (Task 2) は Task 8/10 が消費。名前は上の Interfaces ブロックで統一した。

**Review Focus の割り当て**: 1→Task 8、2→Task 4、3→Task 7、4→Task 6、5→Task 11 Step 2。すべて該当タスクのテスト手順に入っている。
