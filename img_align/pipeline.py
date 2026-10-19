"""VGGT軌跡パイプライン: メイン=PanoVGGT処理、精度は任意後処理で積む

メイン(常に実行):
  1 prep   : 人物マスク確認 (+mapany指定時はピンホールクロップ生成)
  2 chunks : GT階区間が必要とするチャンクのみPanoVGGT推論 (既存スキップ)
  3 chains : 区間別チェーン → frontend_bench/pv<sess>_<島>/ + pv_manifest.json
  4 eval   : GT軌跡とのフィットATE + compare図 (--no-evalで省略)

任意後処理 (--post で選択、カンマ区切り):
  yaw_fix : チャンク内フレーム回転の修復 (方位ロック撮影のみ有効。chunksの後段)
  mapany  : MapAnythingポーズ条件付け (→macondタグ。要mapanythingコンテナ)
  interp  : macond制御点×pvチェーン形状の全フレーム補間 (→mafタグ)
  anchor  : コーナーRANSACで図面アンカー (全周島向け。→anchor_<tag>_corners.json)
            ※--segments imu では階ラベルが要る。backend/floor_assign.json を
              置くと自動で流し込まれる (core/floor_assign --template で雛形)
  hier    : 分割サブ島へ親アンカーを移送+壁ICP (→anchor_<tag>_hier.json)
  export  : アンカー済み点群を図面座標へ書き出し (aligned_house_<sess>_auto.ply)
  viewer  : 確認画像+WebGLビューア生成

使い方:
  python -m img_align.pipeline run --session 0616 --max-seg 160 \\
      --post anchor,hier,export,viewer
  python -m img_align.pipeline stages      # 段の説明を表示
"""
import argparse
import json
import os
import sys


def _setup_session(session):
    """configはimport時にIMG_SESSIONを凍結するため、必ずimportより先に設定する"""
    os.environ["IMG_SESSION"] = session


def stage_prep(args):
    from img_align.config import PROP, SEG_DIR
    posts = args.post
    if PROP.person_mask_dir is None and not SEG_DIR.exists():
        print("[prep] 人物マスクなし (person_mask_dir/seg未生成) — "
              "マスク無しで続行。必要なら seg_ade20k を実行")
    if "mapany" in posts:
        from img_align.core.crops import generate
        generate(tag="mapany_f130")


def stage_chunks_chains(args):
    from img_align.core.chunking import run_batch
    from img_align.core.chunking import MODEL_DIRS
    base, prefix = MODEL_DIRS[args.model]
    # stride / chunk-size を変えたら**チャンク置き場もタグも**分ける。
    # タグを分けないと frontend_bench/<タグ>/ を上書きして既定の結果が消える
    var = ""
    if args.stride:
        var += f"_s{args.stride}"
    if getattr(args, "chunk_size", None):
        var += f"_c{args.chunk_size}"
    chunks_dir = base + var
    # 後段(resplit/remerge)も同じマニフェストを読み書きするのでここで確定させる
    args.manifest_out = args.manifest_out or f"{prefix}_manifest.json"
    manifest = run_batch(max_seg=args.max_seg, chunks_dir=chunks_dir,
                         segments_source=args.segments, stride=args.stride,
                         manifest_out=args.manifest_out,
                         use_idle=not args.no_idle, model=args.model,
                         chunk_size=getattr(args, "chunk_size", None),
                         tag_suffix=var)
    if "yaw_fix" in args.post:
        from img_align.config import RESULTS_DIR
        from img_align.postproc.yaw_fix import fix_chunk
        src = RESULTS_DIR / "pcd" / chunks_dir
        out = RESULTS_DIR / "pcd" / f"{chunks_dir}_yawfix"
        out.mkdir(parents=True, exist_ok=True)
        for c in sorted(src.glob("chunk_*")):
            fixed, _ = fix_chunk(c.name, src, out)
            if not fixed and not (out / c.name).exists():
                (out / c.name).symlink_to(c)
        print("[yaw_fix] 済 → チェーンを修正チャンクで再実行")
        chunks_dir = f"{chunks_dir}_yawfix"
        manifest = run_batch(max_seg=args.max_seg, chunks_dir=chunks_dir,
                             segments_source=args.segments)
    if args.segments == "imu":
        # IMU見逃し(短い階段)の保険: 幾何検査で跨ぎ島を自動再分割
        from img_align.core.chunking import stair_remerge, stair_resplit
        manifest = stair_resplit(manifest, chunks_dir=chunks_dir,
                                 manifest_out=args.manifest_out,
                                 stride=args.stride)
        # IMU過検出(はしご等)の偽切断を癒合してカバーを回復
        manifest = stair_remerge(manifest, chunks_dir=chunks_dir,
                                 manifest_out=args.manifest_out,
                                 stride=args.stride, max_seg=args.max_seg,
                                 model=args.model)
        # IMU区間は階ラベルを持たない。手動割当 (backend/floor_assign.json) が
        # あれば流し込む。無ければ floor=None のままで後段がskipする
        from img_align.core.floor_assign import apply as apply_floors
        manifest = apply_floors(manifest)
        from img_align.config import BENCH_DIR
        (BENCH_DIR / args.manifest_out).write_text(json.dumps(manifest, indent=1))
    return manifest


def stage_eval(args, tags):
    from img_align.eval.frontend import evaluate
    from img_align.eval.gt_traj import load_gt
    gt = load_gt()
    results = [evaluate(t, gt) for t in tags]
    from img_align.config import BENCH_DIR
    (BENCH_DIR / "compare.json").write_text(json.dumps(results, indent=1))


def post_mapany(args, manifest):
    try:
        import mapanything  # noqa: F401
        in_container = True
    except ImportError:
        in_container = False
    from img_align.postproc.mapany import run_infer
    for m in manifest:
        mtag = m["tag"].replace("pv", "macond", 1)
        n = m["b"] - m["a"] + 1
        stride = max(1, -(-n // 56))
        if in_container:
            ns = argparse.Namespace(
                tag=mtag, floor=m["floor"], frames=f"{m['a']}:{m['b']}",
                stride=stride, minibatch=1, yaws="0", res=None,
                pose_cond=m["tag"], dump_local=False, nc=False)
            run_infer(ns)
        else:
            print(f"[mapany] コンテナ外。次を実行してください:\n"
                  f"  docker compose exec -e IMG_SESSION={args.session} "
                  f"-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
                  f"mapanything python -m img_align.postproc.mapany infer "
                  f"--tag {mtag} --floor {m['floor']} "
                  f"--frames {m['a']}:{m['b']} --stride {stride} "
                  f"--minibatch 1 --pose-cond {m['tag']}")


def post_interp(manifest):
    from img_align.config import BENCH_DIR
    from img_align.postproc.interp import interp
    for m in manifest:
        ma = m["tag"].replace("pv", "macond", 1)
        if (BENCH_DIR / ma / "global_cameras.npz").exists():
            interp(m["tag"], ma, m["tag"].replace("pv", "maf", 1))
        else:
            print(f"[interp] {ma} なし -> skip")


def _anchor_one(args):
    """1島をアンカー (プロセス並列のワーカー。例外は文字列で返す)"""
    tag, floor, session, suffix, cont = (*args, False)[:5]
    os.environ["IMG_SESSION"] = session      # 子プロセスでconfigを凍結させる
    from img_align.postproc.corners import corner_anchor
    try:
        _, info = corner_anchor(tag, floor, s_prior=1.0, suffix=suffix,
                                return_info=True, continuity_ref=cont)
        return tag, None, info.get("score", 0.0)
    except Exception as e:
        return tag, f"{type(e).__name__}: {e}", 0.0


# 2段アンカー: 1段目で全島を独立に解き、2段目はその結果を時系列の基準にして
# **全島を1本ずつ順に解き直す**。人は連続して歩くので、隣の島から離れすぎた
# 解は物理的に有り得ない…という筋だが、**実測では有害だった** (2026-10-15:
# 0616が1/3→0/3)。1段目が間違っている島が基準になると、正しかった島まで
# そこへ引き寄せられる。基準にする島を選別できない限り使えないので既定はOFF。
# 選別にスコアの絶対値を使う案も失敗済み (最悪島 8.13m の score 0.132 が
# 成功島 0.64m の 0.027 を上回る。スコアは島ごとの条件に強く依存する)。
CONTINUITY_STAGE = False


def post_anchor(manifest, jobs=None):
    """全島を直接アンカーする (島は互いに独立なのでプロセス並列)

    かつては分割サブ島を除外して hier (親からの移送) に回していたが、
    (1) `--max-seg` 分割では**親island自体が作られない**ので移送元が無い、
    (2) 新スコアではサブ島も単独でアンカーできる (0912で7/7成功、0.26〜0.96m)
    の2点から、直接アンカーを既定にした (2026-10-14)。
    hier は直接アンカーが失敗した島の保険として残す。
    """
    import concurrent.futures as cf
    no_floor = [m["tag"] for m in manifest if not m.get("floor")]
    if no_floor:
        print(f"[anchor] 階が未割当の島をskip: {no_floor}\n"
              f"  → python -m img_align.core.floor_assign --template "
              f"で雛形を作り、backend/floor_assign.json を編集してください")
    todo = [(m["tag"], m["floor"], os.environ["IMG_SESSION"])
            for m in manifest if m.get("floor")]
    if not todo:
        return
    jobs = jobs or min(len(todo), max(1, (os.cpu_count() or 4) // 4))

    def run(batch, label):
        if len(batch) <= 1 or jobs <= 1:
            return [_anchor_one(t) for t in batch]
        print(f"[anchor] {label}: {len(batch)}島を{jobs}並列で処理")
        with cf.ProcessPoolExecutor(max_workers=jobs) as ex:
            return list(ex.map(_anchor_one, batch))

    # 1段目: 全島を独立に解く (並列。継続性の基準がまだ無い状態)
    res1 = run([(t, f, s, "_corners") for t, f, s in todo], "1段目")
    for tag, err, _ in res1:
        if err:
            print(f"[anchor] {tag}: 解なし ({err})")
    if len(todo) < 2 or not CONTINUITY_STAGE:
        return
    # 2段目: 1段目の全島を基準に、**1本ずつ順番に**解き直す。
    # 自分以外のアンカーが基準になるので、相互に拘束し合う。逐次なのは
    # 解き直した結果を次の島の基準に反映させるため
    print(f"[anchor] 2段目: 継続性拘束で{len(todo)}島を再試行")
    import shutil
    from img_align.config import OUT_DIR
    for t, f, sess in todo:
        _, err, _ = _anchor_one((t, f, sess, "_cont", True))
        if err:
            print(f"[anchor] {t}: 2段目失敗 ({err}) -> 1段目を維持")
            continue
        src = OUT_DIR / f"anchor_{t}_cont.json"
        if src.exists():
            shutil.copy(str(src), str(OUT_DIR / f"anchor_{t}_corners.json"))


def post_bridge(args):
    """橋渡しで島を群れにまとめ、結合チェーンを書いてマニフェストを返す

    以降の anchor / export / viewer はこの群れマニフェストを食う。島を長く
    するのが狙い (律速は候補生成なので、指紋が増えるほど当たりやすい)。
    """
    from img_align.core.island_group import MODES, build_groups, group_manifest
    from img_align.core.island_group import CONSENSUS_OK_M
    groups = build_groups(list(args.bridge_modes or MODES),
                          args.bridge_thresh or CONSENSUS_OK_M,
                          args.skip_cross_floor)
    print(f"[bridge] {len(groups)}群れ " +
          " / ".join(f"{len(g)}島" for g, _ in groups))
    man = group_manifest(groups)
    from img_align.config import BENCH_DIR
    (BENCH_DIR / "pv_group_manifest.json").write_text(
        json.dumps(man, ensure_ascii=False, indent=1))
    return man


def post_hier(manifest):
    from img_align.config import OUT_DIR
    from img_align.postproc.hier_anchor import hier_anchor
    for m in manifest:
        parent = m.get("parent")
        if not parent:
            continue
        if (OUT_DIR / f"anchor_{m['tag']}_corners.json").exists():
            continue        # 直接アンカーが成功した島は触らない
        tp = OUT_DIR / f"corner_anchor_{parent}.json"
        if not tp.exists():
            print(f"[hier] {m['tag']}: 親 {parent} のアンカーなし -> skip")
            continue
        try:
            hier_anchor(m["tag"], m["floor"],
                        json.loads(tp.read_text()))
        except Exception as e:
            print(f"[hier] {m['tag']}: 失敗 ({e})")


def post_export_viewer(args, manifest, do_viewer):
    from img_align.config import OUT_DIR, SESSION
    from img_align.out.export import export
    z_off = {"1F": 0.0, "2F": 2.9, "3F": 5.8}
    islands = []
    for m in manifest:
        for sfx in ("_corners", "_hier", ""):
            if (OUT_DIR / f"anchor_{m['tag']}{sfx}.json").exists():
                islands.append((m["tag"], z_off.get(m["floor"], 0.0), sfx))
                break
    if not islands:
        print("[export] アンカー済み島なし -> skip")
        return
    out_name = f"aligned_house_{SESSION}_auto.ply"
    export(islands=islands, out_name=out_name)
    if do_viewer:
        from img_align.out.viewer import main as viz
        viz(f"_{SESSION}_auto")


POSTS = ("yaw_fix", "mapany", "interp", "bridge", "anchor", "hier", "export",
         "viewer")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("run")
    sp.add_argument("--session", required=True, help="IMG_SESSION (例 0616)")
    sp.add_argument("--max-seg", type=int, default=160,
                    help="島の最大フレーム数 (長島分割。0で無効)")
    sp.add_argument("--islands", nargs="*", default=None,
                    help="後処理の対象タグを限定")
    sp.add_argument("--post", default="",
                    help=f"任意後処理 (カンマ区切り): {','.join(POSTS)}")
    sp.add_argument("--segments", default="gt", choices=("gt", "imu"),
                    help="島区間ソース: gt=GT軌跡CSV(開発) / imu=IMU階段検出(本番・GTフリー)")
    sp.add_argument("--stride", type=int, default=None,
                    help="チャンクストライド (既定12。8で重なり倍増・接続安定)")
    sp.add_argument("--model", default="pv", choices=("pv", "pi3", "holo"),
                    help="前段モデル: pv=PanoVGGT(既定) / pi3 / "
                         "holo=Pi3-Finetuned-Holo360D (要pi3コンテナ)")
    sp.add_argument("--manifest-out", default=None,
                    help="既定は <model>_manifest.json")
    sp.add_argument("--no-idle", action="store_true",
                    help="静止区間での島切断を無効化 (対照実験用)")
    sp.add_argument("--chunk-size", type=int, default=None,
                    help="1チャンクのフレーム数 (既定16)。大きくすると接続数が"
                         "減り、累積スケール誤差が効きにくくなる")
    sp.add_argument("--bridge-modes", nargs="*", default=None,
                    help="橋の方式 (既定 n8s1 n8s2 n8s4)。先頭が採用する変換")
    sp.add_argument("--bridge-thresh", type=float, default=None,
                    help="方式間の一致度[m]がこれ未満の切れ目だけ繋ぐ")
    sp.add_argument("--skip-cross-floor", action="store_true",
                    help="階をまたぐ切れ目は橋を試さない (推論を約37%% 節約)")
    sp.add_argument("--jobs", type=int, default=None,
                    help="アンカーの並列数 (既定 CPU数/4。島ごとに独立)")
    sp.add_argument("--no-eval", action="store_true")
    sub.add_parser("stages")
    args = ap.parse_args()

    if args.cmd == "stages":
        print(__doc__)
        return
    args.post = [p for p in args.post.split(",") if p]
    bad = set(args.post) - set(POSTS)
    if bad:
        sys.exit(f"未知の後処理: {bad} (有効: {POSTS})")
    _setup_session(args.session)
    if args.max_seg == 0:
        args.max_seg = None

    print(f"===== [1/4] prep (session {args.session})")
    stage_prep(args)
    print("===== [2-3/4] chunks + chains")
    manifest = stage_chunks_chains(args)
    if args.islands:
        manifest = [m for m in manifest if m["tag"] in args.islands]
    if not args.no_eval:
        print("===== [4/4] eval")
        stage_eval(args, [m["tag"] for m in manifest])
    for p in args.post:
        if p in ("yaw_fix", "viewer"):
            continue        # yaw_fixはchains内 / viewerはexportと同時
        print(f"===== post: {p}")
        if p == "mapany":
            post_mapany(args, manifest)
        elif p == "interp":
            post_interp(manifest)
        elif p == "bridge":
            # 以降のpostは**群れ**を対象にする (島単独ではなく)
            manifest = post_bridge(args)
        elif p == "anchor":
            post_anchor(manifest, jobs=args.jobs)
        elif p == "hier":
            post_hier(manifest)
        elif p == "export":
            post_export_viewer(args, manifest, "viewer" in args.post)
    print("pipeline done")


if __name__ == "__main__":
    main()
