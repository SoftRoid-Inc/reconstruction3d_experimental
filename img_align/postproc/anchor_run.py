"""全島をまとめて図面アンカーする (島は互いに独立なのでプロセス並列)

かつては分割サブ島を除外して hier (親からの移送) に回していたが、
(1) `--max-seg` 分割では**親island自体が作られない**ので移送元が無い、
(2) 新スコアではサブ島も単独でアンカーできる (0912で7/7成功、0.26〜0.96m)
の2点から、直接アンカーを既定にした (2026-10-14)。
hier は直接アンカーが失敗した島の保険として残す。
"""
import concurrent.futures as cf
import os
import shutil

# 2段アンカー: 1段目で全島を独立に解き、2段目はその結果を時系列の基準にして
# **全島を1本ずつ順に解き直す**。人は連続して歩くので、隣の島から離れすぎた
# 解は物理的に有り得ない…という筋だが、**実測では有害だった** (2026-10-15:
# 0616が1/3→0/3)。1段目が間違っている島が基準になると、正しかった島まで
# そこへ引き寄せられる。基準にする島を選別できない限り使えないので既定はOFF。
# 選別にスコアの絶対値を使う案も失敗済み (最悪島 8.13m の score 0.132 が
# 成功島 0.64m の 0.027 を上回る。スコアは島ごとの条件に強く依存する)。
CONTINUITY_STAGE = False


def _anchor_one(args):
    """1島をアンカー (プロセス並列のワーカー。例外は文字列で返す)"""
    tag, floor, session_ref, suffix, cont = (*args, False)[:5]
    # 子プロセスでconfigを凍結させる。**SESSION ではなく SESSION_REF** を
    # 渡すこと (SESSION はデータセット登録後の派生名で、env に戻すと
    # レジストリに無くて落ちる)。fork では親の config がコピーされるので
    # 実質無効だが、spawn/forkserver へ切り替わったときの保険
    os.environ["IMG_SESSION"] = session_ref
    from img_align.postproc.corners import corner_anchor
    try:
        _, info = corner_anchor(tag, floor, s_prior=1.0, suffix=suffix,
                                return_info=True, continuity_ref=cont)
        return tag, None, info.get("score", 0.0)
    except Exception as e:                          # noqa: BLE001
        return tag, f"{type(e).__name__}: {e}", 0.0


def anchor_islands(manifest, jobs=None):
    from img_align.config import SESSION_REF
    no_floor = [m["tag"] for m in manifest if not m.get("floor")]
    if no_floor:
        print(f"[anchor] 階が未割当の島をskip: {no_floor}\n"
              f"  → python -m img_align.core.floor_assign --template "
              f"で雛形を作り、backend/floor_assign.json を編集してください")
    todo = [(m["tag"], m["floor"], SESSION_REF)
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
    # 2段目: 1段目の全島を基準に、**1本ずつ順番に**解き直す
    print(f"[anchor] 2段目: 継続性拘束で{len(todo)}島を再試行")
    from img_align.config import OUT_DIR
    for t, f, sess in todo:
        _, err, _ = _anchor_one((t, f, sess, "_cont", True))
        if err:
            print(f"[anchor] {t}: 2段目失敗 ({err}) -> 1段目を維持")
            continue
        src = OUT_DIR / f"anchor_{t}_cont.json"
        if src.exists():
            shutil.copy(str(src), str(OUT_DIR / f"anchor_{t}_corners.json"))
