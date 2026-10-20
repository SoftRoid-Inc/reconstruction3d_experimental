"""手動で決めた階を島に割り当てる (本番運用の入力経路)

本番モード `--segments imu` の島は階ラベルを持たない (floor=None)。後段の
図面アンカー・hier・書き出しは全て階を要求するので、**割当が入らないと
後半が動かない**。階の自動決定 (C1) は未解決 (core/floor_seq は6中2セッション
でしか成立しない) ため、運用としては**PanoVGGT推論の裏で人が割り当てる**。

割当ファイル `backend/floor_assign.json` は2通りの書き方を受け付ける:

  {"islands": {"pv0912_s001": "1F", "pv0912_s002": "2F"}}
  {"ranges":  [[0, 256, "1F"], [257, 510, "2F"], [511, 727, "3F"]]}

`ranges` 形式は島の切り方に依存しないので、**島が再生成されても使い回せる**
(推論の裏で割り当てる運用に向く)。島の階はその範囲との重なりの多数決で決まる。

割当作業の材料は `eval/density_view` (島ごとの密度地図+軌跡) と
`core/floor_seq` (IMU階段検出からの相対階。絶対階が不定でも**順序**は正しい)
を併用すると速い。

使い方:
  IMG_SESSION=0912 python -m recon3d.core.floor_assign --template
      → 現在の島を並べた雛形を出力 (これを編集して保存する)
  IMG_SESSION=0912 python -m recon3d.core.floor_assign --check
      → 割当の妥当性を確認 (未割当の島、図面に無い階名)
  pipeline run --session 0912 --segments imu --post anchor,hier,export
      → マニフェスト読み込み時に自動適用される
"""
import argparse
import json


ASSIGN_NAME = "floor_assign.json"


def floor_names(paths):
    """図面がある階を下から順に

    B1F は地下なので負、RF/PH のように数字を持たない階名は末尾に送る
    (int() 決め打ちだと ValueError で階の列挙ごと落ちていた)。
    """
    def key(s):
        d = "".join(c for c in s if c.isdigit())
        if not d:
            return (1, 0, s)        # 数字なし (RF, PH, ...) は最後
        n = int(d)
        return (0, -n if s.upper().startswith("B") else n, s)

    return sorted((p.stem for p in paths.floors.glob("*.jpg")), key=key)


def load(paths, name=ASSIGN_NAME):
    """割当ファイルを読む。無ければ None"""
    p = paths.backend / name
    return json.loads(p.read_text()) if p.exists() else None


def floor_of(info, assign):
    """島1本の階を決める。islands指定が優先、無ければ ranges の多数決"""
    if not assign:
        return info.get("floor")
    direct = assign.get("islands", {}).get(info["tag"])
    if direct:
        return direct
    best, best_n = info.get("floor"), 0
    for a, b, fl in assign.get("ranges", []):
        n = min(info["b"], b) - max(info["a"], a) + 1
        if n > best_n:
            best, best_n = fl, n
    return best


def apply(paths, manifest, name=ASSIGN_NAME, quiet=False):
    """マニフェストに階を流し込む (割当が無ければ素通し)"""
    assign = load(paths, name)
    if not assign:
        return manifest
    for m in manifest:
        fl = floor_of(m, assign)
        if fl != m.get("floor"):
            m["floor"] = fl
    if not quiet:
        miss = [m["tag"] for m in manifest if not m.get("floor")]
        print(f"[floor_assign] {len(manifest) - len(miss)}/{len(manifest)}島に"
              f"階を割当" + (f" (未割当: {miss})" if miss else ""))
    return manifest


def template(paths, manifest, name=ASSIGN_NAME):
    """現在の島を並べた雛形を書く (floor_seq があれば初期値に使う)"""
    seq = paths.imu / "floor_seq.json"
    hint = {}
    if seq.exists():
        for s in json.loads(seq.read_text())["segments"]:
            for m in manifest:
                n = min(m["b"], s["b"]) - max(m["a"], s["a"]) + 1
                if n > hint.get(m["tag"], (None, 0))[1]:
                    hint[m["tag"]] = (s["floor"], n)
    out = {"_floors": floor_names(paths),
           "_note": "islands を編集するか、ranges に [開始, 終了, 階名] を並べる",
           "islands": {m["tag"]: hint.get(m["tag"], (None,))[0]
                       for m in manifest},
           "ranges": []}
    p = paths.backend / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"-> {p}")
    for m in manifest:
        print(f"  {m['tag']:18} f{m['a']:4d}-{m['b']:<4d} "
              f"({m['b'] - m['a'] + 1:3d}f)  候補 {out['islands'][m['tag']]}")
    return out


def check(paths, manifest, name=ASSIGN_NAME):
    """割当の妥当性を確認する"""
    assign = load(paths, name)
    if not assign:
        print(f"[floor_assign] {paths.backend / name} がありません (--template で作成)")
        return False
    names = floor_names(paths)
    ok = True
    for m in manifest:
        fl = floor_of(m, assign)
        if not fl:
            print(f"  ✗ {m['tag']}: 未割当")
            ok = False
        elif fl not in names:
            print(f"  ✗ {m['tag']}: 階名 '{fl}' の図面がない (有効: {names})")
            ok = False
        else:
            print(f"  ✓ {m['tag']:18} -> {fl}")
    return ok


def main(paths, argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="pv_manifest.json")
    ap.add_argument("--name", default=ASSIGN_NAME)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--template", action="store_true", help="雛形を出力")
    g.add_argument("--check", action="store_true", help="割当を確認")
    args = ap.parse_args(argv)
    manifest = json.loads((paths.bench / args.manifest).read_text())
    if args.template:
        template(paths, manifest, args.name)
    else:
        raise SystemExit(0 if check(paths, manifest, args.name) else 1)

