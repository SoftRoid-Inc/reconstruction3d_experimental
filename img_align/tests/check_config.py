"""config の凍結値が全セッションで不変であることを固定する

リファクタで一番怖いのは「例外で落ちる」ではなく「**黙って別の物件を読む**」。
`config.py` の既定 `"0828"` へ落ちると `eval/gt_traj.gt_dir()` の 28-AAAAA
フォールバックと組んで、新しい建物を 28-AAAAA の GT で評価して完走する
(2026-10-09 に実際に踏んだ)。それを検出するための金床。

  python -m img_align.tests.check_config --update   # ベースライン作成
  python -m img_align.tests.check_config            # 差分検査 (0=一致)
"""
import json
import sys

from img_align.tests.conftest import SNAP_DIR, known_sessions, run_json

SNAP = SNAP_DIR / "config.json"
# ベースライン更新の履歴 (意図した変更だけ。ここに書けないズレは退行):
#   2026-10-19 person_mask_dir: masks/ が無いセッションは None になった
#              (旧: 実在しないパスを持つ。新: seg_ade20k へのフォールバックが
#               素直に効く)。ext/ は評価後に masks/ を消すので5件が該当

# mm/px は図面GTから導出される実測値。アンカーの全座標がこれに乗るので、
# 凍結値とは別枠で固定する (0828 の 11.968 は実測。理論値 4.233 は誤り)
MM_PER_PX_EXPECT = {
    "0828": 11.968,
    "0912": 12.021969168662748,
    "e11196": None,          # 外部データセットは図面の壁注釈が無いので None
}

_PROBE = "from img_align.tests._probe import main; main()"
_MMPX = ("import json; from img_align.eval.gt_traj import mm_per_px; "
         "print(json.dumps(mm_per_px()))")


def collect():
    out = {}
    for s in known_sessions():
        try:
            out[s] = run_json(_PROBE, session=s)
        except Exception as e:                   # noqa: BLE001
            out[s] = {"error": str(e)[:1500]}
    return out


def diff_snapshot():
    """ベースラインとの差分 {session: (want, got)}。ベースライン不在は None"""
    if not SNAP.exists():
        return None
    want = json.loads(SNAP.read_text())
    got = collect()
    return {s: (want.get(s), got.get(s))
            for s in set(want) | set(got) if want.get(s) != got.get(s)}


def mm_per_px_mismatch():
    """MM_PER_PX_EXPECT との食い違い {session: (want, got_or_error)}"""
    bad, avail = {}, known_sessions()
    for s, want in MM_PER_PX_EXPECT.items():
        if s not in avail:
            continue
        try:
            got = run_json(_MMPX, session=s)
        except Exception as e:                   # noqa: BLE001
            bad[s] = (want, f"ERROR {str(e)[:500]}")
            continue
        ok = (got is None) if want is None else (
            got is not None and abs(got - want) < 1e-9)
        if not ok:
            bad[s] = (want, got)
    return bad


def main():
    if "--update" in sys.argv:
        SNAP_DIR.mkdir(parents=True, exist_ok=True)
        snap = collect()
        SNAP.write_text(json.dumps(snap, ensure_ascii=False, indent=1,
                                   sort_keys=True) + "\n")
        err = [s for s, v in snap.items() if "error" in v]
        print(f"{SNAP}: {len(snap)}セッション記録" +
              (f" / エラー {err}" if err else ""))
        return 0

    d = diff_snapshot()
    if d is None:
        print(f"ベースライン {SNAP} が無い。--update で作成してください")
        return 2
    rc = 0
    if d:
        rc = 1
        print(f"[NG] config の凍結値が変わった ({len(d)}セッション)")
        print(json.dumps(d, ensure_ascii=False, indent=1))
    else:
        print(f"[OK] config 凍結値 一致 ({len(json.loads(SNAP.read_text()))}"
              "セッション)")
    m = mm_per_px_mismatch()
    if m:
        rc = 1
        print(f"[NG] mm/px: {json.dumps(m, ensure_ascii=False)}")
    else:
        print("[OK] mm/px 一致")
    return rc


if __name__ == "__main__":
    sys.exit(main())
