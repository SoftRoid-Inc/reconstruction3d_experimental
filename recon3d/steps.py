"""パイプラインのステップ登録と依存解決

**順序はここが決める**。旧 `pipeline.py --post a,b,c` は書いた順に実行して
いたので、`--post anchor,bridge` と書くと群れ化の**前**に anchor が走って
いた (島単独のマニフェストを食う = 指定した意味が出ない)。
ここでは各ステップが `needs` を宣言し、`graphlib` で順序を決める。
選んだステップの前提は**自動で補う** (`steps=[anchor]` -> prep, chains, anchor)。

能力 (`core/caps.py`) が足りないステップは理由つきで skip する。明示的に
指定されたステップが能力不足のときだけ、黙らずにエラーで止める。

  IMG_SESSION=/data/reconstruction python -m recon3d.steps      # 一覧
"""
import graphlib
import json
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple


@dataclass
class Ctx:
    """ステップ間で受け渡す状態

    `paths` が「どのデータセットか」を持つ唯一の場所。環境変数も
    モジュール凍結も使わず、ここから各段へ引数で配る。
    """
    paths: object = None
    opt: dict = field(default_factory=dict)
    manifest: Optional[list] = None     # 島 or 群れのマニフェスト
    chunks_dir: Optional[str] = None
    tag_suffix: str = ""
    caps: object = None
    order: tuple = ()              # この実行で走る段 (前提の補完ずみ)
    skipped: dict = field(default_factory=dict)   # step -> 理由

    def get(self, path, default=None):
        """opt["chunking"]["stride"] を opt.get("chunking.stride") で引く"""
        cur = self.opt
        for k in path.split("."):
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return default if cur is None else cur


@dataclass(frozen=True)
class Step:
    name: str
    doc: str
    run: Callable                  # (Ctx) -> None
    needs: Tuple[str, ...] = ()    # 無ければ自動で足す前提ステップ
    before: Tuple[str, ...] = ()   # 両方選ばれたときだけ効く順序制約
    caps: Tuple[str, ...] = ()     # 欠けたら skip する能力


# --------------------------------------------------------------- 各ステップ
def _prep(ctx):
    from recon3d.core.ingest import materialize_imu
    materialize_imu(ctx.paths)
    if ctx.paths.masks is None and not ctx.paths.seg.exists():
        print("[prep] 人物マスクなし (person_mask_dir/seg未生成) — "
              "マスク無しで続行。必要なら seg_ade20k を実行")
    # mapany はピンホールクロップを食うので、選ばれていれば自動で作る
    if ctx.get("chunking.crops") or "mapany" in ctx.order:
        from recon3d.core.crops import generate
        generate(ctx.paths, tag="mapany_f130")


def _chains(ctx):
    """チャンク推論 + 区間別チェーン (既存チャンクはスキップされる)"""
    from recon3d.core.chunking import MODEL_DIRS, run_batch
    model = ctx.get("chunking.model", "pv")
    stride = ctx.get("chunking.stride")
    chunk_size = ctx.get("chunking.chunk_size")
    max_seg = ctx.get("chunking.max_seg", 160) or None
    base, prefix = MODEL_DIRS[model]
    # stride / chunk_size を変えたら**チャンク置き場もタグも**分ける。
    # タグを分けないと frontend_bench/<タグ>/ を上書きして既定の結果が消える
    var = (f"_s{stride}" if stride else "") + \
          (f"_c{chunk_size}" if chunk_size else "")
    ctx.chunks_dir, ctx.tag_suffix = base + var, var
    manifest_out = ctx.get("chunking.manifest_out") or f"{prefix}_manifest.json"
    ctx.opt.setdefault("chunking", {})["manifest_out"] = manifest_out
    segments = ctx.get("segments", "gt")
    # 区間ソースは段ではなくオプションなので、能力検査はここで。
    # 黙って segments=gt へ落とさない (GTの無い物件で別セッションのGTを
    # 読んでしまう経路がある)。chunking.imu_segments は transitions.json が
    # 無いと detect() を呼び、その中がコンテナ専用パス決め打ちで
    # 原因の分かりにくい ModuleNotFoundError になる
    need = {"imu": "imu_segments", "gt": "gt"}[segments]
    if ctx.caps is not None and need not in ctx.caps:
        raise SystemExit(f"segments={segments} は使えない: "
                         f"{ctx.caps.reason(need)}")

    common = dict(max_seg=max_seg, segments_source=segments, stride=stride,
                  manifest_out=manifest_out,
                  use_idle=ctx.get("chunking.use_idle", True),
                  model=model, chunk_size=chunk_size)
    manifest = run_batch(ctx.paths, chunks_dir=ctx.chunks_dir,
                         tag_suffix=var, **common)

    if ctx.get("chunking.yaw_fix"):
        manifest = _yaw_fix_rechain(ctx, common, var)

    if segments == "imu":
        from recon3d.core.chunking import stair_remerge, stair_resplit
        # IMU見逃し(短い階段)の保険: 幾何検査で跨ぎ島を自動再分割
        manifest = stair_resplit(ctx.paths, manifest, chunks_dir=ctx.chunks_dir,
                                 manifest_out=manifest_out, stride=stride)
        # IMU過検出(はしご等)の偽切断を癒合してカバーを回復
        manifest = stair_remerge(ctx.paths, manifest, chunks_dir=ctx.chunks_dir,
                                 manifest_out=manifest_out, stride=stride,
                                 max_seg=max_seg, model=model)
        # IMU区間は階ラベルを持たない。手動割当があれば流し込む
        from recon3d.core.floor_assign import apply as apply_floors
        manifest = apply_floors(ctx.paths, manifest)
        (ctx.paths.bench / manifest_out).write_text(json.dumps(manifest, indent=1))
    ctx.manifest = manifest


def _yaw_fix_rechain(ctx, common, var):
    """方位ロック撮影のチャンク内回転を直して**同じ条件で**チェーンし直す

    旧実装は再チェーンに max_seg と segments しか渡しておらず、
    stride / chunk_size / model / manifest_out / タグ接尾辞が全部抜けて
    いた (既定条件のチェーンを上書きしてしまう)。
    """
    from recon3d.core.chunking import run_batch
    from recon3d.postproc.yaw_fix import fix_chunk
    src = ctx.paths.results / "pcd" / ctx.chunks_dir
    out = ctx.paths.results / "pcd" / f"{ctx.chunks_dir}_yawfix"
    out.mkdir(parents=True, exist_ok=True)
    for c in sorted(src.glob("chunk_*")):
        fixed, _ = fix_chunk(c.name, src, out)
        if not fixed and not (out / c.name).exists():
            (out / c.name).symlink_to(c)
    print("[yaw_fix] 済 → チェーンを修正チャンクで再実行")
    ctx.chunks_dir = f"{ctx.chunks_dir}_yawfix"
    ctx.tag_suffix = f"{var}_yawfix"
    return run_batch(ctx.paths, chunks_dir=ctx.chunks_dir,
                     tag_suffix=ctx.tag_suffix, **common)


def _posegraph(ctx):
    """島ごとにポーズグラフ+ループ拘束で解き直し、以降の段の対象を差し替える

    逐次連結は誤差を一本道に積むが、全チャンクを同時に解くと解消する
    (34島の実測で形状残差 0.0059 -> 0.0046、−22%)。**逐次エッジだけの
    ポーズグラフは逐次連結と同じ解になる**ので、効くのは再訪ループ拘束が
    入ったときだけ。採用ループ数と改善量は相関しない。
    """
    import json as _json

    from recon3d.core.posegraph import YAW_W, run as pg_run
    pcd = ctx.paths.results / "pcd" / ctx.chunks_dir
    prefix = ctx.get("posegraph.prefix", "pg_")
    yaw_w = ctx.get("posegraph.yaw_w", YAW_W)
    loops = ctx.get("posegraph.loops", True)
    out = []
    for m in ctx.manifest:
        tag = prefix + m["tag"]
        cdirs = [pcd / f"chunk_{k:02d}" for k in m["chunks"]]
        try:
            pg_run(ctx.paths, cdirs, tag, yaw_w, loops=loops)
        except Exception as e:                      # noqa: BLE001
            print(f"[posegraph] {m['tag']}: 失敗 ({type(e).__name__}: {e})"
                  " -> 逐次連結のまま")
            out.append(m)
            continue
        out.append({**m, "tag": tag, "from_tag": m["tag"]})
    name = "pg_manifest.json"
    (ctx.paths.bench / name).write_text(_json.dumps(out, ensure_ascii=False,
                                              indent=1))
    ctx.manifest = out
    # 以降の段 (bridge の islands_at 等も含む) が同じマニフェストを読むよう
    # 差し替える。揃えないと「群れ化はPG、橋の両側判定は逐次」になる
    ctx.opt.setdefault("chunking", {})["manifest_out"] = name
    print(f"[posegraph] {len(out)}島 -> {ctx.paths.bench / name}")


def _eval(ctx):
    from recon3d.eval.frontend import evaluate
    from recon3d.eval.gt_traj import load_gt
    gt = load_gt(ctx.paths)
    res = [evaluate(ctx.paths, m["tag"], gt) for m in ctx.manifest]
    (ctx.paths.bench / "compare.json").write_text(json.dumps(res, indent=1))


def _mapany(ctx):
    from recon3d.postproc.mapany_run import run_mapany
    run_mapany(ctx.paths, ctx.manifest)


def _interp(ctx):
    from recon3d.postproc.interp import interp
    for m in ctx.manifest:
        ma = m["tag"].replace("pv", "macond", 1)
        if (ctx.paths.bench / ma / "global_cameras.npz").exists():
            interp(ctx.paths, m["tag"], ma, m["tag"].replace("pv", "maf", 1))
        else:
            print(f"[interp] {ma} なし -> skip")


def _bridge(ctx):
    """橋渡しで島を群れにまとめる。以降のステップは**群れ**を対象にする"""
    from recon3d.core.island_group import (CONSENSUS_OK_M, MODES,
                                             build_groups, group_manifest)
    groups = build_groups(ctx.paths, list(ctx.get("bridge.modes") or MODES),
                          ctx.get("bridge.thresh", CONSENSUS_OK_M),
                          ctx.get("bridge.skip_cross_floor", False),
                          manifest_name=ctx.get("chunking.manifest_out"))
    print(f"[bridge] {len(groups)}群れ " +
          " / ".join(f"{len(g)}島" for g, _ in groups))
    ctx.manifest = group_manifest(ctx.paths, groups)
    (ctx.paths.bench / "pv_group_manifest.json").write_text(
        json.dumps(ctx.manifest, ensure_ascii=False, indent=1))


def _anchor(ctx):
    from recon3d.postproc.anchor_run import anchor_islands
    anchor_islands(ctx.paths, ctx.manifest, jobs=ctx.get("anchor.jobs"))


def _hier(ctx):
    """直接アンカーに失敗した**分割サブ島**へ親から移送する保険

    親島は `--segments gt` の階区間分割でしか作られない。`--max-seg` の
    長島分割は親 island を残さないので、本番経路 (segments=imu) では
    対象ゼロになるのが正常。
    """
    from recon3d.postproc.hier_anchor import hier_anchor
    targets = [m for m in ctx.manifest if m.get("parent")]
    if not targets:
        print("[hier] parent を持つ島が無い -> 適用対象なし "
              "(--max-seg 分割では親島が作られないので正常)")
        return
    for m in targets:
        if (ctx.paths.backend / f"anchor_{m['tag']}_corners.json").exists():
            continue        # 直接アンカーが成功した島は触らない
        tp = ctx.paths.backend / f"corner_anchor_{m['parent']}.json"
        if not tp.exists():
            print(f"[hier] {m['tag']}: 親 {m['parent']} のアンカーなし -> skip")
            continue
        try:
            hier_anchor(ctx.paths, m["tag"], m["floor"],
                        json.loads(tp.read_text()))
        except Exception as e:                      # noqa: BLE001
            print(f"[hier] {m['tag']}: 失敗 ({e})")


def _export(ctx):
    from recon3d.out.export import export
    z_off = {"1F": 0.0, "2F": 2.9, "3F": 5.8}
    islands = []
    for m in ctx.manifest:
        for sfx in ("_corners", "_hier", ""):
            if (ctx.paths.backend / f"anchor_{m['tag']}{sfx}.json").exists():
                islands.append((m["tag"], z_off.get(m["floor"], 0.0), sfx))
                break
    if not islands:
        print("[export] アンカー済み島なし -> skip")
        return
    export(ctx.paths, islands, f"aligned_house_{ctx.paths.name}_auto.ply")
    if ctx.get("export.viewer"):
        from recon3d.out.viewer import main as viz
        viz(ctx.paths, f"_{ctx.paths.name}_auto")


# ------------------------------------------------------------------ 登録表
STEPS = {s.name: s for s in (
    Step("prep", "人物マスク確認 + 入力の imu/ を results へ取り込み",
         _prep, caps=("frames",)),
    Step("chains", "PanoVGGT推論 (既存チャンクはskip) → 区間別チェーン",
         _chains, needs=("prep",), caps=("frames",)),
    Step("posegraph", "島ごとにポーズグラフ+ループ拘束で解き直す (→pg_タグ)",
         _posegraph, needs=("chains",),
         before=("eval", "mapany", "bridge", "anchor", "hier", "export")),
    Step("eval", "GT軌跡とのATE + compare図", _eval, needs=("chains",),
         caps=("gt", "scale")),
    Step("mapany", "MapAnythingポーズ条件付け (→macondタグ)", _mapany,
         needs=("chains",)),
    Step("interp", "macond制御点×pvチェーン形状の全フレーム補間 (→mafタグ)",
         _interp, needs=("mapany",)),
    Step("bridge", "橋渡しで島を群れに結合 (以降は群れを対象にする)",
         _bridge, needs=("chains",), before=("anchor", "hier", "export")),
    Step("anchor", "コーナーRANSACで図面アンカー", _anchor, needs=("chains",),
         caps=("anchor",)),
    Step("hier", "直接アンカー失敗島へ親から移送 (分割サブ島のみ)", _hier,
         needs=("anchor",), caps=("anchor",)),
    Step("export", "アンカー済み点群を図面座標へ書き出し (+viewer)", _export,
         needs=("anchor",), caps=("anchor",)),
)}

DEFAULT_STEPS = ("prep", "chains", "eval")


# ------------------------------------------------------------------ 依存解決
def resolve(selected):
    """選んだステップ -> 前提を補った実行順。未知の名前は SystemExit"""
    bad = [s for s in selected if s not in STEPS]
    if bad:
        raise SystemExit(f"未知のステップ: {bad}\n  有効: {list(STEPS)}")
    want, stack = set(), list(selected)
    while stack:                                  # needs を推移的に閉じる
        s = stack.pop()
        if s in want:
            continue
        want.add(s)
        stack.extend(STEPS[s].needs)
    g = {s: {n for n in STEPS[s].needs if n in want} for s in want}
    for s in want:                                # before は両方選ばれたら効く
        for t in STEPS[s].before:
            if t in want:
                g[t].add(s)
    try:
        order = list(graphlib.TopologicalSorter(g).static_order())
    except graphlib.CycleError as e:
        raise SystemExit(f"ステップ依存が循環している: {e}")
    return order


def plan(paths, selected, caps=None, strict=True):
    """実行順と、能力不足で skip するステップの理由を返す

    明示的に選ばれたステップが能力不足なら**止める** (strict=True)。自動で
    補われただけの前提ステップは skip して先へ進む (評価が無くてもアンカーは
    動く等)。`strict=False` は点検用 — 止めずに skip 扱いで報告する。
    """
    from recon3d.core import caps as caps_mod
    caps = caps if caps is not None else caps_mod.detect(paths)
    order = resolve(selected)
    run, skip = [], {}
    for s in order:
        miss = [c for c in STEPS[s].caps if c not in caps]
        # 前提が skip されたステップは、自分の能力が足りていても走れない
        dead = [n for n in STEPS[s].needs if n in skip]
        if not miss and not dead:
            run.append(s)
            continue
        if miss:
            why = " / ".join(caps.reason(c) for c in miss)
        else:
            why = f"前提の {','.join(dead)} が実行できない"
        if strict and s in selected:
            raise SystemExit(f"ステップ '{s}' は実行できない: {why}")
        skip[s] = why
    return run, skip


def execute(paths, selected, opt=None):
    """前提を補って順に実行し、Ctx を返す"""
    from recon3d.core import caps as caps_mod
    caps = caps_mod.detect(paths)
    order, skip = plan(paths, selected, caps)
    ctx = Ctx(paths=paths, opt=dict(opt or {}), caps=caps,
              order=tuple(order),
              skipped=skip)
    for s, why in skip.items():
        print(f"[skip] {s}: {why}")
    print(f"[plan] {' -> '.join(order)}")
    for i, s in enumerate(order, 1):
        print(f"===== [{i}/{len(order)}] {s}")
        STEPS[s].run(ctx)
        if s == "chains" and ctx.get("islands"):
            keep = set(ctx.get("islands"))
            ctx.manifest = [m for m in ctx.manifest if m["tag"] in keep]
    print("pipeline done")
    return ctx


def describe():
    """段の一覧 (name / doc / 依存 / 必要な能力) を行のリストで返す"""
    w = max(len(s) for s in STEPS)
    out = []
    for s in STEPS.values():
        dep = f"  <- {','.join(s.needs)}" if s.needs else ""
        cap = f"  [要 {','.join(s.caps)}]" if s.caps else ""
        out.append(f"{s.name:{w}}  {s.doc}{dep}{cap}")
    out.append(f"\n既定: {' '.join(DEFAULT_STEPS)}")
    return out
