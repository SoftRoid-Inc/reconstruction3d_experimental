"""橋で繋がる島を「群れ」にまとめ、結合チェーンを島と同じ形式で書き出す

図面アンカーの律速は**候補生成**だとオラクルで確定している (失敗島は候補480本の
最良そのものが1.34〜2.77m = 正解が候補に無い)。島を長くすれば指紋が増えるので、
群れをそのままアンカーに掛けるのが橋渡しの本来の狙い。

出力は `frontend_bench/<群れタグ>/global_{cameras.npz,points.ply}` で島チェーンと
同形式。既存の anchor / export / viewer が無改造で動く。

**採否は方式間の一致度**で決める (`CONSENSUS_OK_M`)。橋の残差は当てにならない
(2026-10-15実測: 残差0.003で0.16m、0.128で**4.30m**)。橋が「両側のチェーンには
合うが相対関係は誤り」の状態を残差は検出できないため。

**階またぎは繋がらない** (2026-10-16実測、19切れ目):
  同一階   一致度 中央 0.76m  / GT≤0.5m 8/12
  階またぎ 一致度 中央 10.62m / GT≤0.5m 1/7
難しいのは階段の動きでなく**階が変わること** (同じ階段イベントでも階が変わらない
3件は中央0.67mで繋がる)。両側が別の階だと視覚的な重なりが消えるため。
ただし唯一通った1件 (0925 1F→2F) はGT 0.33mの良質な橋なので、**既定では
またぎも試して一致度に任せる**。`--skip-cross-floor` で37%の推論を節約できる。

使い方:
  IMG_SESSION=0925 python -m recon3d.core.island_group            # 群れを書く
  IMG_SESSION=0925 python -m recon3d.core.island_group --dry-run  # 判定だけ
"""
import argparse
import json

import numpy as np

from recon3d.core.bridge import (bridge_transform, consensus, junction_kind,
                                   manifest_islands)
from recon3d.core.geometry import apply, compose
from recon3d.core.io_ply import read_ply_xyzrgb, write_ply

CONSENSUS_OK_M = 1.2    # 一致度[m]がこれ未満の切れ目だけ繋ぐ
                        # 一致度は **m_per_unit でメートルに正規化済み**
                        # (島ゲージ単位のままだと物件で尺度が3倍違い、同じ
                        #  しきい値が1.2m〜4.0mに化ける。2026-10-16に外部
                        #  データセットで発覚)。旧値2.5は 28-BBBBB の尺度
                        #  0.49 m/unit 換算で約1.2m だったので同値に揃えた
MODES = ("n8s1", "n8s2", "n8s4")   # 一致度を測る方式。先頭が採用する変換
MAX_PLY_POINTS = 4_000_000         # 点群出力の上限 (ディスク逼迫のため必須)
GROUP_PREFIX = "grp"


def _eye():
    return (1.0, np.eye(3), np.zeros(3))


def build_groups(paths, modes=MODES, thresh=CONSENSUS_OK_M,
                 skip_cross_floor=False,
                 verbose=True, manifest_name=None):
    """島を群れにまとめる。[(島リスト, {tag: 群れ先頭ゲージへの変換})]

    `manifest_name` は `--model pi3` / `--manifest-out` で既定から外れる。
    islands_at 等が間接的に同じファイルを読むので bridge 側の既定も揃える
    (揃えないと「群れ化は新マニフェスト、橋の両側判定は旧マニフェスト」と
    いう食い違いが起きる)。
    """
    from recon3d.core.bridge import set_manifest_name
    man = manifest_islands(paths, set_manifest_name(manifest_name))
    if not man:
        return []
    groups, cur, T = [], [man[0]], {man[0]["tag"]: _eye()}
    for A, B in zip(man, man[1:]):
        cross = A["floor"] != B["floor"]
        kind = junction_kind(paths, A["b"], B["a"])
        if skip_cross_floor and cross:
            c, Tba, why = None, None, "階またぎはskip指定"
        else:
            c = consensus(paths, A["b"], B["a"], modes)
            Tba, _ = bridge_transform(paths, A["b"], B["a"], modes[0])
            why = (f"一致度 {c:.2f}m" if c is not None else "一致度 測定不能")
        ok = c is not None and c < thresh and Tba is not None
        if verbose:
            print(f"  {A['tag']} -> {B['tag']} [{kind}/"
                  f"{'またぎ' if cross else '同一'}] {why}"
                  f"  -> {'繋ぐ' if ok else '切る'}")
        if ok:
            T[B["tag"]] = compose(T[A["tag"]], Tba)
            cur.append(B)
        else:
            groups.append((cur, T))
            cur, T = [B], {B["tag"]: _eye()}
    groups.append((cur, T))
    return groups


def group_tag(islands):
    """群れのタグ。1島なら島タグのまま (無駄な別名を作らない)"""
    if len(islands) == 1:
        return islands[0]["tag"]
    sess = islands[0]["tag"].split("_")[0].replace("pv", "")
    return f"{GROUP_PREFIX}{sess}_{islands[0]['a']}_{islands[-1]['b']}"


def write_group(paths, islands, T, with_points=True):
    """群れの結合チェーンを frontend_bench/<群れタグ>/ に島と同形式で書く"""
    tag = group_tag(islands)
    if len(islands) == 1:
        return tag, paths.bench / tag      # 単独島は既存の出力をそのまま使う
    out = paths.bench / tag
    out.mkdir(parents=True, exist_ok=True)
    # 階で割ると先頭が変わるので、**この群れの先頭島**を基準に取り直す
    from recon3d.core.geometry import compose as _c
    s0, R0, t0 = T[islands[0]["tag"]]
    inv0 = (1.0 / s0, R0.T, -(R0.T @ t0) / s0)
    T = {m["tag"]: _c(inv0, T[m["tag"]]) for m in islands}
    names, positions, poses, pts, rgbs = [], [], [], [], []
    for m in islands:
        src = paths.bench / m["tag"]
        cam = np.load(src / "global_cameras.npz")
        s, R, t = T[m["tag"]]
        names.extend(str(n) for n in cam["names"])
        positions.append(apply((s, R, t), cam["positions"]))
        g = np.repeat(np.eye(4)[None], len(cam["poses"]), axis=0)
        g[:, :3, :3] = np.einsum("ij,njk->nik", R, cam["poses"][:, :3, :3])
        g[:, :3, 3] = apply((s, R, t), cam["poses"][:, :3, 3])
        poses.append(g)
        ply = src / "global_points.ply"
        if with_points and ply.exists():
            xyz, rgb = read_ply_xyzrgb(ply)
            pts.append(apply((s, R, t), xyz).astype(np.float32))
            rgbs.append(rgb)
    order = np.argsort(names)
    np.savez_compressed(
        out / "global_cameras.npz",
        names=np.array(names)[order],
        positions=np.vstack(positions)[order],
        poses=np.concatenate(poses)[order])
    if pts:
        xyz, rgb = np.concatenate(pts), np.concatenate(rgbs)
        if len(xyz) > MAX_PLY_POINTS:     # ディスク逼迫のため上限は必須
            k = max(1, len(xyz) // MAX_PLY_POINTS)
            xyz, rgb = xyz[::k], rgb[::k]
        write_ply(out / "global_points.ply", xyz, rgb)
    (out / "group_log.json").write_text(json.dumps(
        {"members": [m["tag"] for m in islands],
         "floor": islands[0]["floor"],
         "frames": [islands[0]["a"], islands[-1]["b"]],
         "transforms": {k: [float(v[0]), v[1].tolist(), v[2].tolist()]
                        for k, v in T.items()}}, ensure_ascii=False, indent=1))
    print(f"-> {out} ({len(names)} frames, {len(islands)}島)")
    return tag, out


def split_by_floor(groups):
    """群れを階の変わり目で割る (橋の変換は保ったまま)

    **アンカーは単一フロアを前提**にしているので、階をまたぐ群れをそのまま
    掛けると大半の島が別の階の図面に当てられる。一方で階またぎの橋そのものは
    ゲージを繋ぐ情報として価値があるので、変換 T は捨てずに持ち回る
    (group_log.json に残るので、後段で階間の関係を使える)。
    """
    out = []
    for islands, T in groups:
        cur = [islands[0]]
        for m in islands[1:]:
            if m["floor"] == cur[-1]["floor"]:
                cur.append(m)
            else:
                out.append((cur, T))
                cur = [m]
        out.append((cur, T))
    return out


def group_manifest(paths, groups, with_points=True):
    """群れを階で割って書き出し、アンカーが食えるマニフェスト形式で返す"""
    out = []
    for islands, T in split_by_floor(groups):
        tag, _ = write_group(paths, islands, T, with_points)
        out.append({"tag": tag, "floor": islands[0]["floor"],
                    "a": islands[0]["a"], "b": islands[-1]["b"],
                    "members": [m["tag"] for m in islands]})
    return out


def main(paths, argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="*", default=list(MODES))
    ap.add_argument("--thresh", type=float, default=CONSENSUS_OK_M)
    ap.add_argument("--skip-cross-floor", action="store_true",
                    help="階をまたぐ切れ目は橋を試さない (推論を約37%%節約)")
    ap.add_argument("--dry-run", action="store_true", help="判定だけして書かない")
    ap.add_argument("--no-points", action="store_true", help="点群を書かない")
    args = ap.parse_args(argv)
    groups = build_groups(paths, args.modes, args.thresh,
                          args.skip_cross_floor)
    print(f"\n{len(groups)}群れ " +
          " / ".join(f"{len(g)}島" for g, _ in groups))
    if args.dry_run:
        return
    man = group_manifest(paths, groups, with_points=not args.no_points)
    p = paths.bench / "pv_group_manifest.json"
    p.write_text(json.dumps(man, ensure_ascii=False, indent=1))
    print(f"-> {p}")

