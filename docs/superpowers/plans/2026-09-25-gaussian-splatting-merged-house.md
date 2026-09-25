# 合併Gaussian Splatting生成 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 0828/0312両セッションの家全体を図面メートル座標で1つに合併したGaussian Splatting(.ply+単体HTMLビューア)を生成する。

**Architecture:** equirect原画+既知ポーズ(meshing.pyの土台)から透視ビューを合成し、gsplatで島ごとに学習。学習済みガウシアンを検証済み4x4(det<0)で図面座標へ変換・合併し、antimatter15/splat(MIT)ベースの単体HTMLで一人称閲覧する。

**Tech Stack:** gsplat (Apache 2.0) / PyTorch CUDA (Dockerサービス gsplat) / numpy+cv2 (env img_align) / antimatter15 splat renderer (MIT, vendor)

**Spec:** `docs/superpowers/specs/2026-09-25-gaussian-splatting-merged-house-design.md`

## Global Constraints

- 商用可能ライセンスのみ: gsplat (Apache 2.0)、antimatter15/splat (MIT)。INRIA本家3DGS実装のコード・派生は一切使わない
- 新しい実行環境はDockerで作る(conda env追加は不可)。パターンは `Dockerfile.vidmap` 参照
- セッション切替は `IMG_SESSION` 環境変数(`config.SESSION`)、出力命名は `SFX = "" if SESSION=="0828" else f"_{SESSION}"`(meshing.py と同一)
- ポーズ・島構成・図面座標変換は `img_align/meshing.py` の `ISLANDS` / `resolve_frames()` / `drawing_matrix()` を再利用(再実装しない)
- 数値検証はCLIコマンド+期待値で行う(このリポジトリにpytest基盤は無い)
- コードコメントは既存モジュールに合わせ日本語

## Review Focus

- 原画jpgが無いフレーム(0828はstride3抽出のため欠番があり得る)→ gs_prepは警告してスキップし、クラッシュしない(Task 2に確認ステップ)
- ビューがほぼ全面マスク(撮影者が正面に写る方向)→ 有効画素率<40%のビューは書き出さない(Task 2)
- 将来 `mirror_col` 付きアンカーで det>0 になっても gs_merge が正しく動く → 変換の往復テストは符号両方を通す(Task 5)
- 島のチェックポイント欠落(学習失敗)→ gs_merge は欠けている島名を列挙してエラー終了(Task 5)
- 合併GSが巨大(>2百万個)だとビューアが固まる → gs_viewer は不透明度×スケールの寄与順で上限数に間引く(Task 6)

---

### Task 1: Docker環境 (Dockerfile.gsplat + composeサービス)

**Files:**
- Create: `/home/maren/programming/Dockerfile.gsplat`
- Modify: `/home/maren/programming/docker-compose.yml`(vidmapサービスの並びに gsplat を追加)

**Interfaces:**
- Produces: composeサービス `gsplat`。コンテナ内で `python -c "import gsplat"` が成功し、`torch.cuda.is_available()==True`。リポジトリは `/code2/reconstruction_3d` にマウント(既存common踏襲)

- [ ] **Step 1: Dockerfile.gsplat を書く**

```dockerfile
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST=8.6
# conda旧libstdc++がJITビルドの.soロードを壊す定番対策 (Dockerfile.vidmap⑦と同じ)
RUN ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /opt/conda/lib/libstdc++.so.6

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential ninja-build libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# gsplat (Apache 2.0) はソースビルド (CUDA拡張はarch 8.6のみ)
RUN pip install --no-cache-dir --retries 10 --timeout 120 \
    gsplat==1.4.0 pytorch-msssim opencv-python-headless plyfile

ENV MPLBACKEND=Agg
WORKDIR /code2/reconstruction_3d
```

- [ ] **Step 2: docker-compose.yml にサービス追加**

vidmapサービスのブロックをコピーし、名前を `gsplat`、`dockerfile: Dockerfile.gsplat` に変えるだけ(extends/common・GPU・マウントは同一)。

- [ ] **Step 3: ビルド(バックグラウンド、ログをファイルへ)**

Run: `cd /home/maren/programming && docker compose build gsplat --progress=plain > /tmp/gsplat_build.log 2>&1; echo EXIT=$?`
Expected: EXIT=0(gsplatのCUDAビルドで10〜20分)

- [ ] **Step 4: 動作検証**

Run: `docker compose up -d gsplat && docker compose exec -T gsplat python -c "import torch, gsplat; print(gsplat.__version__, torch.cuda.is_available())"`
Expected: `1.4.0 True`

- [ ] **Step 5: Commit**

```bash
git add /home/maren/programming/Dockerfile.gsplat /home/maren/programming/docker-compose.yml
git commit -m "feat: gsplat学習用Dockerサービス"
```
(注: docker-compose.yml がreconstruction_3dリポジトリ外なら、このcommitはDockerfileの置き場のリポジトリで行うか、コミット対象から外して口頭報告のみ)

### Task 2: gs_prep.py — 透視ビュー合成+学習データセット書き出し

**Files:**
- Create: `img_align/gs_prep.py`
- Modify: `img_align/vidmap_prep.py`(personマスク読込を関数として切り出し・再利用)
- 出力: `img_align/results/gs/<tag>/{images,masks}/*.jpg|png`, `views.npz`, `init_points.npz`

**Interfaces:**
- Consumes: `meshing.ISLANDS[SESSION]`, `meshing.resolve_frames(isl)` → `[{"name", "R"(3,3), "t_m"(3,), "depth_scale", "chunk_dir"}]`, `config.FRAMES_DIR`(原画5376×2688), `config.SEG_DIR`
- Produces: 島ごとのデータセットdir。`views.npz` は `names`(V,), `K`(3,3) float64, `cam2world`(V,4,4) float64(メートル島座標・カメラ系x右/y下/z前)。`init_points.npz` は `xyz`(N,3) float32(メートル島座標), `rgb`(N,3) uint8

- [ ] **Step 1: vidmap_prep.py のpersonマスク読込を関数化**

vidmap_prep.py内の「SEG_DIRからpersonラベルを読みequirectのboolマスクにする」処理を
`load_person_mask(frame_name) -> np.ndarray(H,W) bool | None` として切り出し
(既存の呼び出し元も同関数を使うようリファクタ。挙動不変)。

- [ ] **Step 2: ビューLUTとポーズ合成を実装**

```python
"""GS学習用データセット生成: equirect原画 → 透視ビュー+ポーズ+初期点群

ビュー構成: ヨー8方向(45°刻み) × ピッチ{0°, 35°下} = 16ビュー/フレーム
カメラ系は meshing.py と同一 (x右/y下/z前)。Rx(+35°)が下向き35°。
"""
VIEW_SIZE = 800
FOV_DEG = 70.0
YAWS = list(range(0, 360, 45))
PITCH_DOWN = [0.0, 35.0]
MIN_VALID = 0.40          # 有効画素率がこれ未満のビューは捨てる

def view_rotations():
    from .meshing import _rot
    return [(f"y{y:03d}p{int(p):02d}", _rot("y", y) @ _rot("x", p))
            for y in YAWS for p in PITCH_DOWN]

def build_view_lut(W, H):
    """原画equirect(W,H)へのサンプリングLUT (meshing.build_face_luts と同型)"""
    f = 0.5 * VIEW_SIZE / math.tan(math.radians(FOV_DEG / 2))
    c = VIEW_SIZE / 2.0
    i, j = np.meshgrid(np.arange(VIEW_SIZE), np.arange(VIEW_SIZE), indexing="xy")
    d = np.stack([(i - c) / f, (j - c) / f, np.ones_like(i, float)], -1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    K = np.array([[f, 0, c], [0, f, c], [0, 0, 1.0]])
    return d, K   # 方向はビューローカル。面ごとの回転は呼び出し側で d @ R.T

def sample_equirect(img, d_world, W, H):
    lon = np.arctan2(d_world[..., 0], d_world[..., 2])
    lat = np.arcsin(np.clip(-d_world[..., 1], -1, 1))
    col = np.rint((lon / (2 * math.pi) + 0.5) * W - 0.5).astype(np.int32) % W
    row = np.clip(np.rint((0.5 - lat / math.pi) * H - 0.5), 0, H - 1).astype(np.int32)
    return img[row, col]
```

各フレーム: 原画 `FRAMES_DIR/<name>.jpg` を読み(**無ければ警告してcontinue**)、
personマスク(あれば)をequirect boolで取得。ビュー毎に
`d_world = d @ R_view.T` でRGBとマスクをサンプル、**有効率<MIN_VALIDのビューは捨てる**。
`cam2world = [[R_frame @ R_view, t_m],[0,1]]`。
初期点群: `frontend_bench/<tag>/global_points.ply` を読み `island_scale`倍でメートル化
→ 3cmボクセル間引き(numpyで `np.unique(floor(xyz/0.03))` の代表点)。

- [ ] **Step 3: CLI**

`python -m img_align.gs_prep`(IMG_SESSIONで両島/全島を処理)。tagごとに
`results/gs/<tag>/` へ書き出し、最後に集計行(フレーム数/ビュー数/スキップ数)を表示。

- [ ] **Step 4: 幾何の検証ゲート (ポーズ整合)**

Run: `python -m img_align.gs_prep --gate`
実装: 島の先頭フレームの1ビューについて、init_points をそのビューへ投影
(`p_cam = (xyz - t) @ R_view_world`, `uv = K @ (p_cam/z)`)し、zバッファで
手前の点の色と、書き出したビュー画像の同画素の色の**中央値絶対差**を出す。
Expected: 差 < 40(8bit値。テクスチャずれがなければ照明差程度に収まる)。
これが放射→方向・ポーズ・LUTの取り違えを一発検出する。

- [ ] **Step 5: 両セッション実行**

Run: `IMG_SESSION=0828 python -m img_align.gs_prep && IMG_SESSION=0312 python -m img_align.gs_prep`
Expected: 5島ぶんのデータセットdir(計 ~7000ビュー、~1.5GB)

- [ ] **Step 6: Commit**

```bash
git add img_align/gs_prep.py img_align/vidmap_prep.py
git commit -m "feat: GS学習用の透視ビュー合成とデータセット書き出し"
```

### Task 3: gs_train.py — gsplat学習ドライバ

**Files:**
- Create: `img_align/gs_train.py`
- 出力: `results/gs/<tag>/ckpt.npz`(means/quats/log_scales/logit_opacities/sh0/shN)+ `train_log.json`

**Interfaces:**
- Consumes: Task 2の `views.npz` / `images/` / `masks/` / `init_points.npz`
- Produces: `ckpt.npz` — keys: `means`(N,3) f32, `quats`(N,4) f32(wxyz・正規化済み), `log_scales`(N,3) f32, `logit_opacities`(N,) f32, `sh0`(N,1,3) f32, `shN`(N,15,3) f32。`train_log.json` に `{"psnr_holdout": float, "iters": int, "n_gs": int}`

- [ ] **Step 1: 学習ループを実装(gsplat rasterization + DefaultStrategy)**

```python
"""島1つのGS学習 (コンテナgsplat内で実行)
docker compose exec -T gsplat python -m img_align.gs_train --tag <tag> [--iters 20000]
"""
import torch, torch.nn.functional as F
from gsplat import rasterization
from gsplat.strategy import DefaultStrategy
from pytorch_msssim import SSIM

HOLDOUT_EVERY = 20      # 20ビューに1つを評価用に除外

def init_params(xyz, rgb, dev):
    n = len(xyz)
    means = torch.tensor(xyz, dtype=torch.float32, device=dev)
    # 近傍距離から初期スケール (kNNの簡略: 3cmボクセル由来なので固定0.03でも可)
    log_scales = torch.full((n, 3), math.log(0.03), device=dev)
    quats = torch.zeros(n, 4, device=dev); quats[:, 0] = 1.0
    opac = torch.logit(torch.full((n,), 0.1, device=dev))
    sh0 = ((torch.tensor(rgb, device=dev) / 255.0 - 0.5) / 0.28209479).unsqueeze(1)
    shN = torch.zeros(n, 15, 3, device=dev)
    ps = {"means": means, "quats": quats, "scales": log_scales,
          "opacities": opac, "sh0": sh0, "shN": shN}
    return {k: torch.nn.Parameter(v) for k, v in ps.items()}

def train(tag, iters):
    dev = "cuda"
    data = load_dataset(tag)          # views.npz + 画像/マスクの遅延読込リスト
    params = init_params(*load_init(tag), dev)
    lrs = {"means": 1.6e-4 * data.scene_scale, "quats": 1e-3, "scales": 5e-3,
           "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    opts = {k: torch.optim.Adam([p], lr=lrs[k]) for k, p in params.items()}
    strategy = DefaultStrategy(verbose=False)
    strategy.check_sanity(params, opts)
    state = strategy.initialize_state(scene_scale=data.scene_scale)
    ssim = SSIM(data_range=1.0, channel=3)
    for it in range(iters):
        view = data.random_train_view()
        sh_deg = min(it // 1000, 3)
        colors = torch.cat([params["sh0"], params["shN"]], 1)
        render, alpha, info = rasterization(
            params["means"], F.normalize(params["quats"], dim=-1),
            torch.exp(params["scales"]), torch.sigmoid(params["opacities"]),
            colors, view.w2c[None], view.K[None], view.W, view.H,
            sh_degree=sh_deg, packed=False)
        strategy.step_pre_backward(params, opts, state, it, info)
        img, m = view.image_cuda(), view.mask_cuda()   # m: 有効=1
        l1 = (torch.abs(render[0] - img) * m).sum() / (m.sum() * 3 + 1)
        ls = 1 - ssim((render[0] * m).permute(2,0,1)[None],
                      (img * m).permute(2,0,1)[None])
        loss = 0.8 * l1 + 0.2 * ls
        loss.backward()
        strategy.step_post_backward(params, opts, state, it, info, packed=False)
        for o in opts.values(): o.step(); o.zero_grad(set_to_none=True)
        if it % 2000 == 0:
            print(f"{it}: loss {loss.item():.4f} n_gs {len(params['means']):,}")
    psnr = eval_holdout(params, data)
    save_ckpt(tag, params, psnr, iters)
```

(load_dataset/eval_holdout/save_ckpt は同ファイル内の素直な実装:
評価は保持ビューのマスク内PSNR平均。)

- [ ] **Step 2: スモークテスト(最小島・短時間)**

Run: `docker compose exec -T gsplat env IMG_SESSION=0828 PYTHONPATH=/code2/reconstruction_3d python -m img_align.gs_train --tag panovggt_C_full --iters 1500`
Expected: lossが単調に下がり、holdout PSNR > 18dB、ckpt.npz生成。
落ちる/PSNRが上がらない場合はポーズかスケールの取り違え → Task 2のゲート再確認。

- [ ] **Step 3: Commit**

```bash
git add img_align/gs_train.py
git commit -m "feat: gsplatによる島ごとのGS学習ドライバ"
```

### Task 4: 全5島の本学習(長時間・バックグラウンド)

**Files:** なし(Task 3の実行のみ)

**Interfaces:**
- Produces: 5島すべての `results/gs/<tag>/ckpt.npz` + `train_log.json`

- [ ] **Step 1: 5島を順次実行するスクリプト行を投げる(バックグラウンド)**

Run:
```bash
docker compose exec -T gsplat bash -c '
for spec in "0828 panovggt_A_yawfix" "0828 panovggt_Bcore_yawfix" "0828 panovggt_C_full" \
            "0312 cm0312_1F" "0312 cm0312_2F"; do
  set -- $spec
  IMG_SESSION=$1 PYTHONPATH=/code2/reconstruction_3d python -m img_align.gs_train --tag $2 --iters 20000 || exit 1
done'
```
Expected: 各島30分前後、計2〜3時間。PSNRをtrain_log.jsonで確認
(目安: 25dB前後。島A(ドリフト持ち)だけ低め=22dB台でも先へ進み、目視で判断)

- [ ] **Step 2: PSNR集計を報告**

Run: `grep -H psnr img_align/results/gs/*/train_log.json`
Expected: 5行。極端に低い島(<20dB)があれば、その島のみ `--pose-opt` 追加を検討(スペックのリスク対応)

### Task 5: gs_merge.py — 図面座標へ変換・合併

**Files:**
- Create: `img_align/gs_merge.py`
- 出力: `results/backend/gaussian_house{SFX}.ply`(3DGS標準ply)

**Interfaces:**
- Consumes: `results/gs/<tag>/ckpt.npz`(Task 3の形式)、`meshing.drawing_matrix(anchor, grid, z_off, metric_input=True)`
- Produces: 3DGS標準plyフォーマット(x,y,z,f_dc_0..2,opacity,scale_0..2,rot_0..3)。色はDC化(shN破棄)

- [ ] **Step 1: 変換コア(共分散のA·Σ·Aᵀ)を実装**

```python
def transform_gaussians(means, quats, log_scales, M):
    """4x4相似(反射可)でガウシアンを写す。返り値は同形式"""
    L = M[:3, :3]
    means2 = means @ L.T + M[:3, 3]
    R = quat_to_rot(quats)                         # (N,3,3)
    S = np.exp(log_scales)                         # (N,3)
    A = L[None] @ R * S[:, None, :]                # L·R·S  (N,3,3)
    Sigma = A @ A.transpose(0, 2, 1)               # L·Σ·Lᵀ
    w, V = np.linalg.eigh(Sigma)                   # 昇順固有値
    w = np.clip(w, 1e-12, None)
    scales2 = np.log(np.sqrt(w))
    det = np.linalg.det(V)
    V[det < 0, :, 2] *= -1                         # 右手系に直す
    quats2 = rot_to_quat(V)
    return means2, quats2, scales2
```

- [ ] **Step 2: 変換の往復テスト(det両符号)**

Run: `python -m img_align.gs_merge --selftest`
実装: 乱数ガウシアン100個を (a) det<0 の実アンカー4x4、(b) それに追加のx反転を
掛けたdet>0行列、の両方で変換し、**変換前後の共分散を直接計算して
`max|L·Σ·Lᵀ − Σ'| < 1e-9`** を確認。
Expected: `selftest OK (det<0, det>0)`

- [ ] **Step 3: 合併本体**

島ごとに ckpt.npz を読み(**無い島は名前を列挙してエラー終了**)、
`drawing_matrix(metric_input=True)` で変換、`z_off−0.3〜+3.0` の高さクロップ
(meshingと同じ理由: 階間の漏れ)、全島を連結して3DGS標準plyで書く。
色: `rgb = clip(0.5 + 0.28209479 * sh0, 0, 1)` をf_dcに入れ、shNは捨てる。

- [ ] **Step 4: 幾何検証(真上ビュー)**

Run: `IMG_SESSION=0828 python -m img_align.gs_merge --check`(0312も)
実装: GS中心点を `meshing.wall_distance_stats` と同じ壁帯(z_off+0.3〜2.2)で評価し、
`mesh_compare` と同形式の真上ビューPNG(`gs_check{SFX}.png`)を出す。
Expected: median が点群評価(0828: 1F 0.078/2F 0.046)から大きく悪化しない(+3cm以内)

- [ ] **Step 5: Commit**

```bash
git add img_align/gs_merge.py
git commit -m "feat: 島GSを図面座標へ変換・合併 (反射対応の共分散変換)"
```

### Task 6: gs_viewer.py — 単体HTMLビューア

**Files:**
- Create: `img_align/vendor/splat_main.js`(antimatter15/splat の main.js、MITヘッダ付きでvendor)
- Create: `img_align/gs_viewer.py`
- 出力: `results/backend/gs_viewer{SFX}.html`

**Interfaces:**
- Consumes: `gaussian_house{SFX}.ply`
- Produces: 単体HTML(データ同梱・オフラインで開ける)。UI: 一人称WASD+マウス、視点高さプリセット1.3m/1.5m、階フィルタ(1F/2F/全部)

- [ ] **Step 1: レンダラをvendor**

Run: `curl -L https://raw.githubusercontent.com/antimatter15/splat/master/main.js -o img_align/vendor/splat_main.js`
先頭にMITライセンスヘッダ(Copyright (c) 2023 Kevin Kwok / MIT)を追記。

- [ ] **Step 2: .splat量子化と埋め込みHTML生成**

```python
def to_splat_bytes(means, rgb, opac, log_scales, quats, max_n=1_200_000):
    """antimatter15 .splat 形式 (32B/個)。寄与順(不透明度×体積)で間引き"""
    w = opac * np.exp(log_scales).prod(1) ** (1 / 3)
    idx = np.argsort(-w)[:max_n]
    buf = np.zeros(len(idx), dtype=[("pos", "f4", 3), ("scale", "f4", 3),
                                    ("rgba", "u1", 4), ("quat", "u1", 4)])
    buf["pos"] = means[idx]
    buf["scale"] = np.exp(log_scales[idx])
    buf["rgba"][:, :3] = np.clip(rgb[idx] * 255, 0, 255)
    buf["rgba"][:, 3] = np.clip(opac[idx] * 255, 0, 255)
    q = quats[idx] / np.linalg.norm(quats[idx], axis=1, keepdims=True)
    buf["quat"] = np.clip(q * 128 + 128, 0, 255)
    return buf.tobytes()
```

HTML: vendorのJSを `<script>` に同梱し、`fetch(url)` 部分を
「同梱base64をデコードしてUint8Array」に差し替え。カメラ初期位置=1F中央・高さ1.5m。
高さプリセットボタン(1.3/1.5)はviewMatrixのy成分を固定するトグルとして実装。
階フィルタは事前に1F/2F/全部の3つのsplatバッファを持ち切替(高さクロップ済みなので単純)。

- [ ] **Step 3: 生成と目視確認**

Run: `IMG_SESSION=0828 python -m img_align.gs_viewer && IMG_SESSION=0312 python -m img_align.gs_viewer`
→ `python -m http.server` 経由でブラウザ確認(file://不可の既知事実)。
Expected: 60MB級HTML、一人称で屋内を歩けて破綻がない。SendUserFileで両HTMLを送付

- [ ] **Step 4: Commit**

```bash
git add img_align/vendor/splat_main.js img_align/gs_viewer.py
git commit -m "feat: 合併GSの単体HTMLビューア (一人称・高さプリセット)"
```

### Task 7: 記録

**Files:**
- Modify: `img_align/results/REPORT.md`(冒頭に成果ブロック追記)
- Modify: メモリ `project-roadmap.md`

- [ ] **Step 1: REPORT追記** — PSNR表(5島)、gs_check指標、成果物一覧、ライセンス確認(gsplat Apache/ビューアMIT)、既知の弱点(島Aドリフトの見え方)
- [ ] **Step 2: メモリ更新** — パイプライン(gs_prep→gs_train→gs_merge→gs_viewer)と再実行コマンド
- [ ] **Step 3: Commit**(REPORTはimg_align配下=untracked運用なのでコミット不要。specリポジトリ整合のみ確認)
