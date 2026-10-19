"""PanoVGGT / Pi3 / Pi3-Finetuned-Holo360D を同一フレーム窓で走らせて比較する

比較は**全区間平均ではなく局所破綻の窓**で行う。平常時のPanoVGGTは既にチャンク
単体ATE 0.009mなので、全区間平均では差が出ないため。窓の定義は
`eval/chunk_cmp.BREAKDOWN` (破綻) と `CONTROL` (同セッションの健全区間)。

PanoVGGTはimg_alignイメージ、Pi3系はpi3コンテナと実行先が違うので、
このドライバはホストで起動し、Pi3側は `docker compose exec` で投げる。
既に出力があるタグはスキップするので中断・再開できる。

使い方 (ホスト):
  python -m img_align.tools.model_bench --preset breakdown
  python -m img_align.tools.model_bench --preset all --models pv holo
出力: <物件>/results/<sess>/pcd/{pv,pi3,holo}_<sess>_c<start>/ と
      28-BBBBB/results/model_bench/model_bench.json
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

COMPOSE_DIR = "/home/maren/programming"
MODELS = ("pv", "pi3", "holo")
# configはimport時にIMG_SESSIONを凍結する。このドライバは全セッションを跨ぐので
# 実際の推論・採点はセッション毎のsubprocessに任せ、ここでは物件ルートしか使わない
os.environ.setdefault("IMG_SESSION", "0925")


def tag_of(model, sess, a):
    return f"{model}_{sess}_c{a}"


def pcd_dir(sess, tag):
    """セッションは物件をまたぐ (0828=28-AAAAA / 他=28-BBBBB) ので
    configの凍結済みPROPではなくPROPERTY表から引く"""
    from img_align.config import PROPERTY
    return PROPERTY[sess].root / "results" / sess / "pcd" / tag


def out_exists(sess, tag):
    return (pcd_dir(sess, tag) / "cameras.npz").exists()


def run_pv(sess, a, b, tag):
    """PanoVGGTはimg_alignイメージ内で動く (core.panovggt が面倒を見る)"""
    subprocess.run(["python", "-m", "img_align.core.panovggt",
                    "--tag", tag, "--frames", f"{a}:{b + 1}:1"],
                   check=True, env={**_env(sess)})


def run_pi3(sess, a, b, tag, ckpt):
    subprocess.run(
        ["docker", "compose", "exec", "-T",
         "-e", f"IMG_SESSION={sess}",
         "-e", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", "pi3",
         "python", "-m", "img_align.core.pi3",
         "--tag", tag, "--frames", f"{a}:{b + 1}:1", "--ckpt", ckpt],
        check=True, cwd=COMPOSE_DIR)
    # コンテナ生成物はroot所有になるのでホスト側から触れるよう戻す
    subprocess.run(["docker", "compose", "exec", "-T", "pi3", "chown", "-R",
                    "1001:1001", f"/code2/reconstruction_3d/"
                    f"{pcd_dir(sess, tag).relative_to(_repo_root())}"],
                   cwd=COMPOSE_DIR, capture_output=True)


def bench_dir():
    """セッションは2物件にまたがるので、主物件(28-BBBBB)の results 配下に
    まとめて置く。img_align/ はgit管理対象なので生成物を入れないこと"""
    from img_align.config import PROPERTY
    d = PROPERTY["0925"].root / "results" / "model_bench"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _repo_root():
    from img_align.config import PROPERTY
    return PROPERTY["0828"].root.parent


def _env(sess):
    return {**os.environ, "IMG_SESSION": sess}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="breakdown",
                    choices=("breakdown", "control", "all"))
    ap.add_argument("--models", nargs="*", default=list(MODELS), choices=MODELS)
    ap.add_argument("--force", action="store_true", help="既存出力も再実行")
    args = ap.parse_args()

    from img_align.eval.chunk_cmp import BREAKDOWN, CONTROL
    wins = {"breakdown": BREAKDOWN, "control": CONTROL,
            "all": BREAKDOWN + CONTROL}[args.preset]

    timings = []
    for sess, a, b, note in wins:
        for model in args.models:
            tag = tag_of(model, sess, a)
            if not args.force and out_exists(sess, tag):
                print(f"[skip] {tag}")
                continue
            print(f"\n===== {tag}  frames {a}-{b}  ({note})", flush=True)
            t0 = time.time()
            try:
                if model == "pv":
                    run_pv(sess, a, b, tag)
                else:
                    run_pi3(sess, a, b, tag,
                            "pi3" if model == "pi3" else "holo")
            except subprocess.CalledProcessError as e:
                print(f"[fail] {tag}: {e}")
                continue
            timings.append({"tag": tag, "sec": round(time.time() - t0, 1)})
            print(f"[done] {tag} {timings[-1]['sec']}s", flush=True)

    # 採点はセッション毎にまとめて (GT読み込みがセッション依存のため)
    rows = []
    for sess in sorted({w[0] for w in wins}):
        tags = [tag_of(m, sess, a) for s, a, _, _ in wins if s == sess
                for m in args.models if out_exists(sess, tag_of(m, sess, a))]
        if not tags:
            continue
        out = subprocess.run(
            ["python", "-m", "img_align.eval.chunk_cmp", "--tags", *tags,
             "--out", f"/tmp/mb_{sess}.json"],
            env=_env(sess), capture_output=True, text=True)
        print(out.stdout, out.stderr[-500:] if out.returncode else "")
        p = Path(f"/tmp/mb_{sess}.json")
        if p.exists():
            notes = {str(w[1]): w[3] for w in wins if w[0] == sess}
            for r in json.loads(p.read_text()):
                rows.append({"session": sess,
                             "note": notes.get(r["tag"].split("_c")[-1], ""),
                             **r})

    dst = bench_dir() / "model_bench.json"
    dst.write_text(json.dumps({"rows": rows, "timings": timings},
                              indent=1, ensure_ascii=False))
    print(f"\n-> {dst}")
    summarize(rows, args.models)


def summarize(rows, models):
    """窓 × モデル の median ATE 表"""
    by = {}
    for r in rows:
        if "ate_median_m" not in r:
            continue
        model, rest = r["tag"].split("_", 1)
        by.setdefault(rest, {})[model] = r["ate_median_m"]
    if not by:
        return
    print(f"\n{'window':20} " + " ".join(f"{m:>8}" for m in models))
    for win in sorted(by):
        print(f"{win:20} " + " ".join(
            f"{by[win].get(m, float('nan')):8.3f}" for m in models))


if __name__ == "__main__":
    main()
