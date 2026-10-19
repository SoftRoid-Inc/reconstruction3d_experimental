# img_align パイプラインの実行イメージ (1ポッド=1データセット)
#
# PanoVGGT 推論からアンカー・書き出しまで**同一プロセス**で走る。
# 以前は PanoVGGT だけ専用 conda env の subprocess だったが、
# k8s では Docker 外で実行しないので1環境に寄せた (2026-10-19)。
#
#   docker compose build img_align
#   docker compose exec img_align python main.py run input=/data/reconstruction
#
# 重みは**焼かない** (3.9GB)。既定の置き場は
#   /opt/panovggt/checkpoints/model.pt
# ここへボリュームをマウントする (k8s なら PVC / initContainer で S3 から)。
# 別の場所に置くなら `PANOVGGT_CKPT` で指す — ローカルの compose は
# /code2/third_party/PanoVGGT/checkpoints/model.pt を向けている
#
# 入っていないもの (意図的):
#   seg_ade20k (py360convert) … 人物マスクは外部供給
#   mesh/*     (open3d)       … 400MB級。本番経路は numpy/scipy だけ
#   gs/*       (gsplat)       … Gaussian Splatting は別コンテナ
#   pi3/holo                  … 別コンテナ。--model pv を使う

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git libgl1 libglib2.0-0 libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# PanoVGGT (MIT, YijingGuo-June/PanoVGGT) を sha 固定で。
# requirements.txt は torch==2.5.0 をピンしているので**流さない** —
# base の 2.5.1 を再インストールで壊してしまう。必要な分だけ明示で入れる
RUN git clone https://github.com/YijingGuo-June/PanoVGGT /opt/panovggt \
 && git -C /opt/panovggt checkout 556bb7d2ec2d02bd3ee4ed535542e74290ba22cf \
 && rm -rf /opt/panovggt/.git

# PanoVGGT が実際に import するもの + img_align の本番経路。
# **xformers は torch に合わせてピンする** — 参照 env (conda panovggt) に
# 入っていて `XFORMERS_DISABLED` も未設定なので、推論は memory_efficient_attention
# を通っていた。入れないと attention の実装が変わり、既存チャンクと
# 数値が揃わなくなる (panovggt/layers/attention.py は try/except で黙って落ちる)
RUN pip install --no-cache-dir \
    xformers==0.0.28.post3 \
    huggingface_hub safetensors requests \
    opencv-python-headless einops omegaconf hydra-core yacs kornia \
    imageio plyfile trimesh shapely scikit-learn scipy matplotlib \
    pillow tqdm typer

COPY img_align /opt/img_align/img_align
COPY recon3d   /opt/img_align/recon3d
COPY main.py   /opt/img_align/main.py

# **焼いたコードが動く。** `python main.py` は sys.path[0] にスクリプトの
# ディレクトリを置くので、PYTHONPATH の順で上書きすることはできない。
# ローカルで編集中のコードを使いたいときは、マウントした側で動かす:
#   docker compose exec -w /code2/reconstruction_3d img_align python main.py ...
# (compose の img_align サービスは working_dir と PYTHONPATH をそう設定ずみ)
ENV PYTHONPATH=/opt/img_align:/opt/panovggt
ENV PANOVGGT_DIR=/opt/panovggt
ENV MPLBACKEND=Agg
ENV PYTHONUNBUFFERED=1
WORKDIR /opt/img_align

# 本番経路の依存漏れを build 時に落とす (実行時に気づくと遅い)。
# PanoVGGT も import して、重み以外が揃っていることを確かめる
RUN python -c "import img_align.api, img_align.steps, img_align.tools.recon" \
 && python main.py steps > /dev/null \
 && (cd /opt/panovggt && python -c "import inference")

ENTRYPOINT ["python", "main.py"]
