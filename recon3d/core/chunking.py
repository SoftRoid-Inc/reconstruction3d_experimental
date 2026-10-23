"""セッション一括前段: GT階区間 → PanoVGGTチャンク推論 → 区間別チェーン

gt_trajectory.csv の階ラベルから連続区間を切り出し、各区間を覆うチャンクだけ
PanoVGGTで推論(既存はスキップ=再開可能)して chain_chunks で連結する。
階段またぎを避けるため、区間内に完全に収まるチャンクのみ使う(不足時は重なり許容)。

出力: frontend_bench/pv<sess>_<fl>[a,b,…]/ と pv_manifest.json
使い方: python main.py run input=/data/reconstruction 'steps=[chains]'
"""
from __future__ import annotations

import json
from typing import Iterable, Sequence, TypeAlias

from recon3d.core.chaining import chain_and_save
from recon3d.core.panovggt import select_entries
from recon3d.paths import Paths
from recon3d.types import Manifest

#: 区間 (階名 or None, 開始フレーム, 終了フレーム)
Seg: TypeAlias = "tuple[str | None, int, int]"

CHUNK_SIZE = 16
CHUNK_STRIDE = 12          # 重なり = CHUNK_SIZE - CHUNK_STRIDE = 4フレーム


def chunk_windows(n: int, stride: int | None = None,
                  size: int | None = None) -> list[tuple[int, int]]:
    """チャンク窓列。strideを小さくすると重なりが増える

    **重なりを増やしても島の精度は上がらない** (2026-10-16実測、外部データ
    10島): stride 12→10 で接続1本あたりの |log s| は 0.195→0.154 (−21%) と
    改善するが、チャンクが27%増えるので掛け合わせる回数も増え、島の形状残差は
    0.0060→0.0071 (+19%) と**悪化**した。スケール誤差は乗算で累積するので、
    1本あたりを良くしても本数が増えれば相殺される。
    → 効くとすれば **size を大きくして本数を減らす**方向。
    """
    size = size or CHUNK_SIZE
    stride = stride or CHUNK_STRIDE
    starts = list(range(0, max(n - size, 0) + 1, stride))
    if starts and starts[-1] + size < n:
        starts.append(n - size)
    return [(s, min(s + size, n)) for s in starts] or [(0, n)]


MIN_SEG_FRAMES = 24   # これ未満の訪問区間は島にしない (チャンク2本に満たない)
GAP_FRAMES = 12       # 同一階ラベルの小ギャップはつなぐ
EDGE_SLACK = 8        # 端チャンクの区間外はみ出し許容 (フレーム。端の取りこぼし対策)
MIN_RESCUE_FRAMES = 12  # 切除後の端切れをミニ島として救済する下限
# **アンカー可能な最小長**。短い島は図面に対して一意に決まるだけの指紋を
# 持たない。全5セッションの短島を長さ順に並べると境界がはっきり出る
# (2026-10-15実測、数字はGT残差):
#   12/16/16/31f=解なし  20f=4.29  23f=2.18  27f=1.01  32f=5.30  (0/8)
#   40f=0.15 ✓           41f=5.90                              (1/2)
# 32と40の間が境目。40ちょうどで成功例があるので**36**で切る
# (0828の39フレーム区間を残すため。40にすると0.15mの成功島を落とす)。
# 短い島を作るくらいなら、階段を跨ぐリスクを取って隣と繋いだ方が価値がある
MIN_ANCHOR_FRAMES = 36


def gt_segments(paths: Paths) -> list[Seg]:
    """GT軌跡 → [(floor, a, b)] (連続訪問区間)。開発・評価用 (GT必要)

    GTの出どころは物件で違う (28-BBBBB=手動登録CSV / 28-AAAAA=メタCSV) ので
    直接CSVを開かず load_gt に任せる。
    """
    from recon3d.eval.gt_traj import load_gt
    rows = [(f, v["floor"]) for f, v in sorted(load_gt(paths).items())]
    segs, cur = [], None
    for fr, fl in rows:
        if cur and fl == cur[0] and fr - cur[2] <= GAP_FRAMES:
            cur[2] = fr
        else:
            if cur:
                segs.append(tuple(cur))
            cur = [fl, fr, fr]
    if cur:
        segs.append(tuple(cur))
    return [s for s in segs if s[2] - s[1] + 1 >= MIN_SEG_FRAMES]


IMU_STAIR_PAD = 4     # IMUイベント前後の除外マージン (フレーム。検出の±ずれ吸収)
IMU_EDGE_SEC = 12.0   # 録画端のイベントはカメラ上げ下ろしアーチファクト
IDLE_PAD = 2          # 静止区間の前後マージン (検出は真の静止より内側に出るため)
MIN_IDLE_FRAMES = 4   # これ未満の静止は切断しない (島を無駄に刻まない)


def idle_cuts(paths: Paths, n_frames: int) -> list[tuple[int, int]]:
    """静止区間 (imu.idle) → 切断点 [(a, b)]。無ければ空

    並進ゼロでは三角測量が成立せず、その区間を含むチャンク接続が島全体を壊す
    (0828 f300-312=撮影者が電話中、並進はカメラ高の0.08倍)。階段と同じく
    「島に入れる価値が無い区間」として切断する。
    """
    from recon3d.imu.idle import detect, spans
    if not (paths.imu / "idle_spans.json").exists():
        try:
            detect(paths)
        except Exception as e:
            print(f"[idle] 検出失敗 ({type(e).__name__}) -> 静止切断なしで続行")
            return []
    out = []
    for a, b in spans(paths):
        if b - a + 1 < MIN_IDLE_FRAMES:
            continue
        out.append((max(0, a - IDLE_PAD), min(n_frames - 1, b + IDLE_PAD)))
    if out:
        print(f"[idle] 静止による切断 {len(out)} 箇所: {out[:8]}")
    return out


def imu_segments(paths: Paths, n_frames: int, use_idle: bool = True
                 ) -> tuple[list[Seg], list[tuple[int, int]]]:
    """IMU階段検出 (imu.transitions, stairnet CNN) → ([(None, a, b)], 切断点)

    検出イベント(昇降)を切断点として区間を切る。stairnetは再現率が高い一方、
    工事現場のはしご・脚立で過検出するが、余分な切断は島が増えるだけで無害
    (見逃しによる階段またぎ島だけが有害)。階ラベルは付けない
    (昇降の積算は過検出で壊れるため、アンカーステップで全フロア照合して決める)。
    """
    tr_path = paths.imu / "transitions.json"
    if not tr_path.exists():
        from recon3d.imu.transitions import detect
        detect(paths)
    doc = json.loads(tr_path.read_text())
    ev = doc["events"]
    # 録画長は **meta.frame_dt から** 出す。以前は max(t_end)/n_frames で逆算していたが、
    # これは `n_frames * fdt == max(t_end)` と恒等になるので、下の端判定が
    # 「t_start > max(t_end) - 12s」に潰れていた。すなわち**最終イベントは長さが
    # 12秒未満なら必ず捨てられる**。検出イベントの64%が12秒未満で、実データ487件の
    # 18%がこれで消えていた。区間が少ないほど影響が大きく、人手入力では1区間しか
    # 無い撮影の切断点が丸ごと消える。
    # meta が無い旧ファイルだけ従来の逆算に落とす (挙動を変えない)。
    fdt = (doc.get("meta") or {}).get("frame_dt")
    if not fdt:
        fdt = max(e["t_end"] for e in ev) / max(n_frames, 1) if ev else 0.5
    rec_sec = n_frames * fdt
    cuts = []
    for e in ev:
        if e["t_end"] < IMU_EDGE_SEC or e["t_start"] > rec_sec - IMU_EDGE_SEC:
            continue    # 録画端アーチファクト
        a = max(0, int(e["frame_start"]) - IMU_STAIR_PAD)
        b = min(n_frames - 1, int(e["frame_end"]) + IMU_STAIR_PAD)
        if cuts and a <= cuts[-1][1] + GAP_FRAMES:
            cuts[-1] = (cuts[-1][0], max(cuts[-1][1], b))
        else:
            cuts.append((a, b))
    if use_idle:        # 静止区間も同じ扱いの切断点として合流
        cuts = sorted(cuts + idle_cuts(paths, n_frames))
        merged = []
        for a, b in cuts:
            if merged and a <= merged[-1][1] + GAP_FRAMES:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))
            else:
                merged.append((a, b))
        cuts = merged
    segs, prev = [], 0
    for a, b in cuts:
        if a - 1 - prev + 1 >= MIN_ANCHOR_FRAMES:
            segs.append((None, prev, a - 1))
        prev = b + 1
    if n_frames - 1 - prev + 1 >= MIN_ANCHOR_FRAMES:
        segs.append((None, prev, n_frames - 1))
    # 切断点も返す。build_manifest の fallback が「切ってよい場所」を知るため
    # (以前はモジュール変数に置いていたが、1プロセスで2データセットを回すと
    #  前の回の切断点が残る。所在と同じく引数で配る)
    return segs, cuts


def build_manifest(paths: Paths, windows: Sequence[tuple[int, int]],
                   max_seg: int | None = None,
                   segments: Sequence[Seg] | None = None,
                   prefix: str = "pv", tag_suffix: str = "",
                   cuts: Sequence[tuple[int, int]] = ()) -> Manifest:
    """max_seg指定時: それより長い訪問区間を等分割する (長島ドリフト対策。
    タグは pv0925_1Fa1, 1Fa2… のように数字接尾辞)。
    segments: [(floor|None, a, b)] を外部注入可 (imu_segments等)。
    cuts: 切ってはいけないフレーム域 (imu_segments の第2返り値)。
    floor=None の区間はタグ pv<sess>_sNN (階はアンカーステップで決める)"""
    segs = segments if segments is not None else gt_segments(paths)
    by_floor = {}
    for fl, a, b in segs:
        by_floor.setdefault(fl, []).append((a, b))
    manifest = []
    for fl, lst in by_floor.items():
        for i, (a, b) in enumerate(lst):
            if fl is None:
                fl_name, sfx = f"s{i:02d}", ""
            else:
                fl_name = fl
                sfx = "" if len(lst) == 1 else chr(97 + i)
            n = b - a + 1
            parts = max(1, -(-n // max_seg)) if max_seg else 1
            bounds = [a + round(n * j / parts) for j in range(parts + 1)]
            for j in range(parts):
                a2, b2 = bounds[j], bounds[j + 1] - (0 if j == parts - 1 else 1)
                dsfx = str(j + 1) if parts > 1 else ""
                # 完全内包+はみ出しEDGE_SLACK以内の端チャンク (端取りこぼし対策)
                ks_in = [k for k, (wa, wb) in enumerate(windows)
                         if wa < b2 + 1 and wb > a2
                         and max(0, a2 - wa) + max(0, wb - 1 - b2)
                         <= EDGE_SLACK]
                if len(ks_in) >= 2:
                    use = ks_in
                else:   # 小さい区間: 重なるチャンクまで許容 (切断域は回避)
                    use = [k for k, (wa, wb) in enumerate(windows)
                           if wa < b2 + 1 and wb > a2
                           and not any(wa <= cb_ and wb - 1 >= ca_
                                       for ca_, cb_ in cuts)]
                    if not use:
                        continue    # 切断域を避けられない小区間は破棄
                ent = {"tag": f"{prefix}{paths.name}_{fl_name}{sfx}{dsfx}"
                              f"{tag_suffix}",
                       "floor": fl, "a": a2, "b": b2, "chunks": use}
                if parts > 1:
                    # 分割サブ島は親を明示する。タグ末尾の数字で判定すると
                    # IMU区間のタグ (pv<sess>_sNN) を誤ってサブ島扱いしてしまう
                    # (2026-10-14: 本番で6島中5島がskipされていた)
                    ent["parent"] = (f"{prefix}{paths.name}_{fl_name}{sfx}"
                                     f"{tag_suffix}")
                manifest.append(ent)
    return manifest


# モデル別の既定: (チャンク置き場, タグ接頭辞)。pvだけ歴史的に無印
# 前段モデルは PanoVGGT のみ。Pi3 / Holo360D は別コンテナで、モデル比較の
# 結論も「乗り換える理由は無い」だったので 2026-10-19 に経路ごと撤去した
MODEL_DIRS = {"pv": ("chunks", "pv")}


def model_dirs(model: str) -> tuple[str, str]:
    """モデル名 -> (チャンク置き場, タグ接頭辞)。未知なら実行可能な形で落とす

    レジストリを直接引くと `KeyError: 'pi3'` になって、設定のどのキーが
    悪いのか分からない。門番はここ1箇所。
    """
    if model not in MODEL_DIRS:
        raise SystemExit(
            f"未知の chunking.model {model!r} (有効: {', '.join(MODEL_DIRS)})")
    return MODEL_DIRS[model]


def infer_chunks(paths: Paths, windows: Sequence[tuple[int, int]],
                 needed: Iterable[int], chunks_dir: str,
                 entries: Sequence[dict]) -> None:
    """欠けているチャンクだけ推論する (既存はスキップ=再開可能)

    1チャンクずつモデルを読み直すと固定コスト12.2秒が毎回かかる
    (実推論は16フレームで9秒)。まとめて1プロセスに渡して1回に抑える。

    `stair_remerge` からも呼ぶ。あちらは隣接2島を隙間ごと仮連結するので、
    `imu_segments` が切り落とした域のチャンク (どの島にも属さない) を
    ここで初めて要求する — 0912 で63本中19本がそれに当たる。
    """
    miss = [(k, *windows[k]) for k in needed
            if not (paths.pcd / chunks_dir / f"chunk_{k:02d}"
                    / "cameras.npz").exists()]
    if not miss:
        return
    from recon3d.core.panovggt import run_chunks
    print(f"[pv] {len(miss)} チャンクを一括推論", flush=True)
    run_chunks(paths, miss, entries, chunks_dir, use_mask=True)

def run_batch(paths: Paths, max_seg: int | None = None,
              manifest_out: str | None = None,
              chunks_dir: str | None = None,
              segments_source: str = "gt", stride: int | None = None,
              use_idle: bool = True, model: str = "pv",
              chunk_size: int | None = None,
              tag_suffix: str = "") -> Manifest:
    """前段一括: 区間計画 → チャンク推論(既存スキップ) → 区間別チェーン

    segments_source: "gt"=gt_trajectory.csv(開発・評価用) / "imu"=stairnet検出
    (GTフリー、本番用。階ラベルなし→アンカーステップで決定)
    model: "pv"=PanoVGGT のみ。
    チャンク置き場・タグ接頭辞・マニフェストをモデル別に分けるので共存できる。

    **tag_suffix は stride/chunk_size を変えるとき必須**。チャンク置き場と
    マニフェストを分けてもタグが同名だと `frontend_bench/<タグ>/` を上書きして
    しまう (2026-10-16に重なり6枚の検証で重なり4枚のチェーンを消した)。"""
    dflt_dir, prefix = model_dirs(model)
    chunks_dir = chunks_dir or dflt_dir
    manifest_out = manifest_out or f"{prefix}_manifest.json"
    n = len(paths.frame_indices)
    entries = select_entries(paths, f"0:{n}:1")
    windows = chunk_windows(n, stride, chunk_size)
    segs, cuts = (imu_segments(paths, n, use_idle)
                  if segments_source == "imu" else (None, []))
    manifest = build_manifest(paths, windows, max_seg=max_seg, segments=segs,
                              prefix=prefix, tag_suffix=tag_suffix, cuts=cuts)
    for m in manifest:
        print(f"{m['tag']}: frames {m['a']}-{m['b']} chunks {m['chunks'][0]}"
              f"..{m['chunks'][-1]}")

    needed = sorted({k for m in manifest for k in m["chunks"]})
    infer_chunks(paths, windows, needed, chunks_dir, entries)

    for m in manifest:
        cdirs = [paths.pcd / chunks_dir / f"chunk_{k:02d}" for k in m["chunks"]]
        print(f"=== chain {m['tag']}", flush=True)
        chain_and_save(paths, cdirs, m["tag"])
    paths.bench.mkdir(parents=True, exist_ok=True)
    (paths.bench / manifest_out).write_text(json.dumps(manifest, indent=1))
    print(f"-> {paths.bench / manifest_out} ({len(manifest)} islands)")
    return manifest


def stair_resplit(paths: Paths, manifest: Manifest,
                  chunks_dir: str = "chunks",
                  manifest_out: str = "pv_manifest.json",
                  max_depth: int = 2,
                  stride: int | None = None) -> Manifest:
    """チェーン済み島を幾何検査し、階段またぎ島を自動再分割して再チェーンする

    stairnet(IMU)の見逃し対策 (core/stair_check)。再分割島のタグは <元>r1/r2。
    分割後の半島は「区間内に完全に収まるチャンク」のみ使う (階段チャンク排除)。
    """
    from recon3d.core.stair_check import find_stair, island_vertical
    n = len(paths.frame_indices)
    windows = chunk_windows(n, stride)
    out = []
    queue = [(m, 0) for m in manifest]
    while queue:
        m, depth = queue.pop(0)
        try:
            s = find_stair(*island_vertical(paths, m["tag"]))
        except Exception as e:
            print(f"[stair_check] {m['tag']}: 検査失敗 ({e}) -> そのまま採用")
            s = None
        if s is None or depth >= max_depth:
            out.append(m)
            continue
        ca, cb, ratio, kind = s
        # 分割するとアンカー不能な断片しか残らないなら、分割しない方がまし
        halves = [(m["a"], ca - 1), (cb + 1, m["b"])]
        long_halves = [h for h in halves if h[1] - h[0] + 1 >= MIN_ANCHOR_FRAMES]
        if not long_halves:
            print(f"[stair_check] {m['tag']}: 鉛直異常[{kind}] だが両半島とも "
                  f"{MIN_ANCHOR_FRAMES}f未満 -> 分割せず採用")
            out.append(m)
            continue
        print(f"[stair_check] {m['tag']}: 鉛直異常[{kind}] frames {ca}-{cb} "
              f"(変位 {ratio:.2f}×カメラ高) -> 再分割")
        subs = []
        for i, (a2, b2) in enumerate([(m["a"], ca - 1), (cb + 1, m["b"])]):
            if b2 - a2 + 1 < MIN_ANCHOR_FRAMES:
                print(f"  半島 {a2}-{b2} ({b2 - a2 + 1}f): "
                      f"{MIN_ANCHOR_FRAMES}f未満でアンカー不能 -> 破棄")
                continue
            use = [k for k, (wa, wb) in enumerate(windows)
                   if wa >= a2 and wb <= b2 + 1]
            if len(use) < 2:   # 救済: 切除域[ca,cb]と重ならない重なりチャンク
                use = [k for k, (wa, wb) in enumerate(windows)
                       if wa < b2 + 1 and wb > a2
                       and not (wa <= cb and wb - 1 >= ca)]
            if not use:
                print(f"  半島 {a2}-{b2}: 収まるチャンク不足 -> 破棄")
                continue
            sub = {"tag": f"{m['tag']}r{i + 1}", "floor": m["floor"],
                   "a": a2, "b": b2, "chunks": use}
            cdirs = [paths.pcd / chunks_dir / f"chunk_{k:02d}" for k in use]
            print(f"=== chain {sub['tag']}", flush=True)
            try:
                chain_and_save(paths, cdirs, sub["tag"])
            except Exception as e:
                print(f"  {sub['tag']}: チェーン失敗 ({e}) -> 破棄")
                continue
            subs.append(sub)
        if not subs:     # 両半島が立たなければ元の島を残す (消失防止)
            print(f"  再分割不成立 -> {m['tag']} をそのまま採用")
            out.append(m)
        for sub in subs:
            queue.append((sub, depth + 1))
    (paths.bench / manifest_out).write_text(json.dumps(out, indent=1))
    print(f"[stair_check] {len(manifest)} -> {len(out)} islands "
          f"(-> {manifest_out})")
    return out


def stair_remerge(paths: Paths, manifest: Manifest,
                  chunks_dir: str = "chunks",
                  manifest_out: str = "pv_manifest.json",
                  max_gap: int = 120, max_len: int | None = None,
                  stride: int | None = None,
                  max_seg: int | None = None) -> Manifest:
    """偽切断の癒合: 隣接島を隙間ごと仮連結し、幾何検査がcleanなら統合する

    stairnetの過検出 (はしご・脚立) は島を無駄に分断し、間の端切れが
    カバー損失になる。隣接島+隙間の仮チェーンに鉛直異常が無ければ
    「偽切断」と断定して統合する (stair_resplitの対操作)。
    真の階段は仮チェーンに異常が出るため統合されない。"""
    from recon3d.core.stair_check import find_stair, island_vertical
    # 癒合後も島長の上限を守る (守らないと 148f×2 -> 296f の巨大島ができ、
    # 長島ドリフトで精度が落ちる。0428で 0.25m -> 0.99m の実例)
    if max_len is None:
        max_len = int(max_seg) if max_seg else 160
    n = len(paths.frame_indices)
    windows = chunk_windows(n, stride)
    # フレーム1枚ごとに exists() を見るので、ループの中で作り直さない
    entries = select_entries(paths, f"0:{n}:1")
    manifest = sorted(manifest, key=lambda m: m["a"])
    tried = set()
    changed = True
    while changed:
        changed = False
        for i in range(len(manifest) - 1):
            m1, m2 = manifest[i], manifest[i + 1]
            key = (m1["tag"], m2["tag"])
            gap = m2["a"] - m1["b"] - 1
            length = m2["b"] - m1["a"] + 1
            if key in tried or gap < 0 or gap > max_gap or length > max_len:
                continue
            tried.add(key)
            a, b = m1["a"], m2["b"]
            use = [k for k, (wa, wb) in enumerate(windows)
                   if wa < b + 1 and wb > a
                   and max(0, a - wa) + max(0, wb - 1 - b) <= EDGE_SLACK]
            if len(use) < 2:
                continue
            tag = f"{m1['tag']}m"
            cdirs = [paths.pcd / chunks_dir / f"chunk_{k:02d}" for k in use]
            print(f"[remerge] {m1['tag']}+{m2['tag']} (gap {gap}f) を試行",
                  flush=True)
            try:
                infer_chunks(paths, windows, use, chunks_dir, entries)
                chain_and_save(paths, cdirs, tag)
                s = find_stair(*island_vertical(paths, tag))
            except Exception as e:
                print(f"  失敗 ({e}) -> 統合せず")
                continue
            if s is not None and s[3] == "excursion":
                # はしご等の往復: レベルは変わらない -> 偽切断とみなし統合
                print(f"  往復(excursion, {s[2]:.2f}×) -> 統合 ({tag}: {a}-{b})")
                s = None
            short = (m1["b"] - m1["a"] + 1 < MIN_ANCHOR_FRAMES
                     or m2["b"] - m2["a"] + 1 < MIN_ANCHOR_FRAMES)
            if s is not None and short:
                # 片方がアンカー不能な短さなら、階段を跨ぐリスクを取ってでも繋ぐ。
                # 使えない島が2本残るより、怪しい島が1本ある方が価値がある
                # (2026-10-15実測: 40フレーム未満の島は0/4で全滅)
                print(f"  短島あり ({m1['b']-m1['a']+1}f / {m2['b']-m2['a']+1}f) "
                      f"-> 鉛直異常[{s[3]}]を許容して統合")
                s = None
            if s is None:
                print(f"  clean -> 統合 ({tag}: {a}-{b})")
                manifest[i] = {"tag": tag, "floor": m1["floor"],
                               "a": a, "b": b, "chunks": use}
                del manifest[i + 1]
                changed = True
                break
            print(f"  鉛直異常あり ({s[2]:.2f}×カメラ高) -> 切断は本物、維持")
    (paths.bench / manifest_out).write_text(json.dumps(manifest, indent=1))
    print(f"[remerge] -> {len(manifest)} islands")
    return manifest


def main(paths: Paths, argv: list[str] | None = None) -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-seg", type=int, default=None,
                    help="これより長い訪問区間を等分割 (長島ドリフト対策)")
    ap.add_argument("--manifest-out", default="pv_manifest.json")
    ap.add_argument("--chunks-dir", default="chunks")
    ap.add_argument("--segments", default="gt", choices=("gt", "imu"),
                    help="区間ソース: gt=GT軌跡CSV / imu=stairnet検出(GTフリー)")
    ap.add_argument("--stride", type=int, default=None,
                    help="チャンクストライド (既定12=重なり4f。8で重なり8f)")
    ap.add_argument("--no-idle", action="store_true",
                    help="静止区間での切断を無効化 (対照実験用)")
    args = ap.parse_args(argv)
    run_batch(paths, args.max_seg, args.manifest_out, args.chunks_dir,
              args.segments,
              args.stride, use_idle=not args.no_idle)

