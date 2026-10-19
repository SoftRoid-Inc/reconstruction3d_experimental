"""位置合わせ済み点群のメッシュ化: TSDF融合 と Poisson表面再構成

2系統を同じ土台(島レジストリ+ポーズ解決+図面座標への4x4)の上に実装する。

- tsdf: PanoVGGTのper-frame深度(equirect・放射距離・メートル)をキューブマップ6面の
  ピンホール深度に変換し、Open3DのScalableTSDFVolumeで島座標(メートル)に融合。
  空間の自由空間カービングが効くのでPoissonの浮遊面が出ない。
- poisson: 図面座標の統合点群からPoisson再構成。法線の向きは軌跡カメラで決める。

反射(図面座標への写像は常にdet<0)を扱うため、TSDFは島座標で積分し、
最後にメッシュごと図面座標へ写して三角形の巻き順を反転する。

実行 (env=panovggt):
  IMG_SESSION=0312 conda run -n panovggt python -m img_align.mesh.meshing verify
  IMG_SESSION=0312 conda run -n panovggt python -m img_align.mesh.meshing tsdf
  IMG_SESSION=0312 conda run -n panovggt python -m img_align.mesh.meshing poisson
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from img_align.config import BENCH_DIR
from img_align.core.chaining import chain
from img_align.core.geometry import apply
from img_align.postproc.colmap_prior import colmap_centers, fit_chunk_to_colmap
from img_align.config import PROJECT_ROOT, RESULTS_DIR, SESSION
from img_align.eval.gt_traj import mm_per_px
from img_align.core.plan_grid import RES_M, PlanGrid
from img_align.postproc.plan_icp import transform_pts

OUT_DIR = BENCH_DIR.parent / "backend"
SFX = "" if SESSION == "0828" else f"_{SESSION}"

# 島レジストリ: 図面座標へのアンカーと、その島を構成するチャンク群。
# mode: "colmap"=チャンクごとに独立してCOLMAPへSim(3)フィット (colmap_prior由来)
#       "chain"=隣接チャンクの重なりICPで連結 (chain_chunks由来)
ISLANDS = {
    "0312": [
        {"tag": "cm0312_1F", "floor": "1F", "z_off": 0.0, "mode": "colmap",
         "root": "pcd/chunks",
         "chunks": list(range(0, 10)) + [20, 21],
         "anchor": "anchor_cm0312_1F_face.json"},
        {"tag": "cm0312_2F", "floor": "2F", "z_off": 2.9, "mode": "colmap",
         "root": "pcd/chunks", "chunks": list(range(11, 20)),
         "anchor": "anchor_cm0312_2F_face.json"},
    ],
    "19176": [
        # アンカー島 (colmap_prior) は屋内チャンクのみで構築。
        # マスク生成用の chunks は階段・屋外も含む全カバレッジ
        # (境界判定は水平位置のみなので、階段フレームの island 帰属は任意)
        {"tag": "cm19176_1F", "floor": "1F", "z_off": 0.0, "mode": "colmap",
         "root": "pcd/chunks",
         "chunks": list(range(0, 7)) + list(range(13, 18)),
         "anchor": "anchor_cm19176_1F.json"},
        {"tag": "cm19176_2F", "floor": "2F", "z_off": 2.9, "mode": "colmap",
         "root": "pcd/chunks",
         "chunks": list(range(7, 13)),
         "anchor": "anchor_cm19176_2F.json"},
    ],
    "0428": [
        # 28-BBBBB 基礎工事段階: 建物なし・単一レベル。アンカーは軌跡外周↔境界
        # リングのフィット (boundary_mask anchor-traj)
        {"tag": "cm0428_ALL", "floor": "1F", "z_off": 0.0, "mode": "colmap",
         "root": "pcd/chunks", "chunks": list(range(0, 31)),
         "anchor": "anchor_cm0428_ALL.json"},
    ],
    "0616": [
        # 28-BBBBB 建方後 (3フロア)。アンカー島は屋内クリーンチャンクのみ、
        # マスク用chunksは階段・出入り込みの全カバレッジ
        {"tag": "cm0616_1F", "floor": "1F", "z_off": 0.0, "mode": "colmap",
         "root": "pcd/chunks",
         "chunks": list(range(0, 13)) + list(range(31, 34)),
         "anchor": "anchor_cm0616_1F.json"},
        {"tag": "cm0616_2F", "floor": "2F", "z_off": 2.9, "mode": "colmap",
         "root": "pcd/chunks", "chunks": list(range(13, 24)),
         "anchor": "anchor_cm0616_2F.json"},
        {"tag": "cm0616_3F", "floor": "3F", "z_off": 5.8, "mode": "colmap",
         "root": "pcd/chunks", "chunks": list(range(24, 31)),
         "anchor": "anchor_cm0616_3F.json"},
    ],
    "0715": [
        {"tag": "cm0715_1F", "floor": "1F", "z_off": 0.0, "mode": "colmap",
         "root": "pcd/chunks",
         "chunks": list(range(0, 11)) + list(range(29, 35)),
         "anchor": "anchor_cm0715_1F.json"},
        {"tag": "cm0715_2F", "floor": "2F", "z_off": 2.9, "mode": "colmap",
         "root": "pcd/chunks", "chunks": list(range(11, 21)),
         "anchor": "anchor_cm0715_2F.json"},
        {"tag": "cm0715_3F", "floor": "3F", "z_off": 5.8, "mode": "colmap",
         "root": "pcd/chunks", "chunks": list(range(21, 29)),
         "anchor": "anchor_cm0715_3F.json"},
    ],
    "0912": [
        {"tag": "cm0912_1F", "floor": "1F", "z_off": 0.0, "mode": "colmap",
         "root": "pcd/chunks",
         "chunks": list(range(0, 22)) + [61, 62],
         "anchor": "anchor_cm0912_1F_corners.json"},
        {"tag": "cm0912_2F", "floor": "2F", "z_off": 2.9, "mode": "colmap",
         "root": "pcd/chunks", "chunks": list(range(22, 44)),
         "anchor": "anchor_cm0912_2F.json"},
        {"tag": "cm0912_3F", "floor": "3F", "z_off": 5.8, "mode": "colmap",
         "root": "pcd/chunks", "chunks": list(range(44, 61)),
         "anchor": "anchor_cm0912_3F.json"},
    ],
    # 0925: 竣工間際1378枚。生COLMAP中心を直接使う (大域BA済みで形状が正しい)。
    # metric-rescale (ステップ再積分) は形状を歪めるため不採用 (131-280の外周ループが
    # 室内へ引き込まれた)。COLMAPのスケールドリフトが残る場合は island を短く分割して対処
    # 0925: 竣工間際1378枚。COLMAP自体に実スケールドリフト (~2倍、屋内前半vs後半で実測)
    # + PanoVGGT屋外スケール誤り (~3倍) があり、単一相似では合わない。
    # → スケールがほぼ一定の区間でサブ島分割し、align_tool で島ごとに手動登録する。
    # metric-rescale (ステップ再積分) は屋外でPanoVGGT誤りを取り込み形状を歪めるため不採用
    "0925": [
        {"tag": "cm0925_1Fa", "floor": "1F", "z_off": 0.0, "mode": "colmap",
         "root": "pcd/chunks", "chunks": list(range(0, 10)),
         "anchor": "anchor_cm0925_1Fa.json"},
        {"tag": "cm0925_1Fout", "floor": "1F", "z_off": 0.0, "mode": "colmap",
         "root": "pcd/chunks", "chunks": list(range(10, 24)),
         "anchor": "anchor_cm0925_1Fout.json"},
        {"tag": "cm0925_1Fb", "floor": "1F", "z_off": 0.0, "mode": "colmap",
         "root": "pcd/chunks", "chunks": list(range(24, 40)),
         "anchor": "anchor_cm0925_1Fb.json"},
        {"tag": "cm0925_1Fc", "floor": "1F", "z_off": 0.0, "mode": "colmap",
         "root": "pcd/chunks", "chunks": list(range(40, 60)),
         "anchor": "anchor_cm0925_1Fc.json"},
        {"tag": "cm0925_2Fa", "floor": "2F", "z_off": 2.9, "mode": "colmap",
         "root": "pcd/chunks", "chunks": list(range(60, 76)),
         "anchor": "anchor_cm0925_2Fa.json"},
        {"tag": "cm0925_2Fb", "floor": "2F", "z_off": 2.9, "mode": "colmap",
         "root": "pcd/chunks", "chunks": list(range(76, 92)),
         "anchor": "anchor_cm0925_2Fb.json"},
        {"tag": "cm0925_3F", "floor": "3F", "z_off": 5.8, "mode": "colmap",
         "root": "pcd/chunks", "chunks": list(range(92, 107)),
         "anchor": "anchor_cm0925_3F.json"},
        {"tag": "cm0925_1Fend", "floor": "1F", "z_off": 0.0, "mode": "colmap",
         "root": "pcd/chunks", "chunks": list(range(107, 115)),
         "anchor": "anchor_cm0925_1Fend.json"},
    ],
    "0828": [
        {"tag": "panovggt_A_yawfix", "floor": "1F", "z_off": 0.0,
         "mode": "chain", "root": "pcd/chunks_yawfix",
         "chunks": list(range(0, 6)),
         "anchor": "anchor_panovggt_A_yawfix_rays.json"},
        {"tag": "panovggt_Bcore_yawfix", "floor": "2F", "z_off": 2.9,
         "mode": "chain", "root": "pcd/chunks_yawfix",
         "chunks": list(range(6, 11)),
         "anchor": "anchor_panovggt_Bcore_yawfix_rays.json"},
        {"tag": "panovggt_C_full", "floor": "1F", "z_off": 0.0,
         "mode": "chain", "root": "pcd/chunks_full_yawfix",
         "chunks": [35, 36, 37],
         "anchor": "anchor_panovggt_C_full_rays.json"},
    ],
}
COLMAP_MODEL = {
    "0312": PROJECT_ROOT / "28-AAAAA" / "video_data" / "0312"
    / "colmap_coarse" / "0",
    "19176": PROJECT_ROOT / "28-CCCCC" / "sfm" / "colmap_coarse" / "0",
    "0428": PROJECT_ROOT / "28-BBBBB" / "0428" / "sfm" / "colmap_coarse" / "0",
    "0616": PROJECT_ROOT / "28-BBBBB" / "0616" / "sfm" / "colmap_coarse" / "0",
    "0715": PROJECT_ROOT / "28-BBBBB" / "0715" / "sfm" / "colmap_coarse" / "0",
    "0912": PROJECT_ROOT / "28-BBBBB" / "0912" / "sfm" / "colmap_coarse" / "0",
    "0925": PROJECT_ROOT / "28-BBBBB" / "0925" / "sfm" / "colmap_coarse" / "0",
}
SNAP_MAX_M = 0.25      # COLMAP中心への吸着ずれ上限 (超えたらそのフレームを捨てる)


# ---------------------------------------------------------------- 島の土台

def islands_for(session):
    """セッションの島定義。results/<sess>/backend/islands.json があれば
    それを優先する (動的編集用の逃げ道)。無ければ静的な ISLANDS レジストリ"""
    from img_align.config import PROPERTY
    prop = PROPERTY.get(session)
    if prop is not None:
        p = prop.root / "results" / session / "backend" / "islands.json"
        if p.exists():
            return json.loads(p.read_text())
    if session not in ISLANDS:
        raise SystemExit(
            f"セッション '{session}' の島定義が無い。"
            f"results/{session}/backend/islands.json を置くか、"
            "meshing.ISLANDS に足してください "
            f"(静的レジストリ: {sorted(ISLANDS)})")
    return ISLANDS[session]


def islands_expanded(session):
    """islands_for の各島を、アンカーの "segments" で仮想島に展開する。

    "segments" は align_tool の区間微調整が書く: チャンク範囲ごとに別の
    (A, t) を持つ独立アンカー (base情報は親と同一)。仮想島は通常の島と
    同じ形の辞書なので resolve_frames / drawing_matrix はそのまま使え、
    セグメントごとに正しいスケールkと図面変換Mが出る。
    アンカー未存在 (未登録) や mode!=colmap の島は展開せずそのまま返す"""
    from img_align.config import PROPERTY
    backend = PROPERTY[session].root / "results" / session / "backend"
    out = []
    for isl in islands_for(session):
        segs = None
        p = backend / isl["anchor"]
        if isl.get("mode") == "colmap" and p.exists():
            segs = json.loads(p.read_text()).get("segments")
        if not segs:
            out.append(isl)
            continue
        for k, sg in enumerate(segs):
            out.append({**isl, "tag": f"{isl['tag']}__seg{k}",
                        "chunks": sg["chunks"], "anchor": sg["anchor"]})
    return out


def anchor_scale(anchor):
    """アンカーの実効一様スケール。手動ツールの非等方アフィン "A" があれば
    sqrt(|det A|) (面積保存の等価一様スケール)、無ければ従来の s"""
    if anchor.get("A") is not None:
        return float(math.sqrt(abs(np.linalg.det(np.asarray(anchor["A"])))))
    return anchor["s"]


def anchor_transform_cells(cells, anchor):
    """島セル → グリッドセル。"A"(2x2) があればアフィン、無ければ s*R(th)"""
    if anchor.get("A") is not None:
        return cells @ np.asarray(anchor["A"]).T + np.asarray(anchor["t"])
    return transform_pts(cells, anchor["s"], anchor["th"],
                         np.array(anchor["t"]))


def island_scale(anchor, grid):
    """島world単位 → メートル の一様スケール k (水平=鉛直)。

    1セルの実メートルは階ごとに異なる (2Fは図面のmm/pxが違い0.04m/セル)ので
    RES_M固定ではなく grid.cell_px から実測値を使う
    """
    m_per_cell = grid.cell_px * mm_per_px() / 1000
    return anchor_scale(anchor) * m_per_cell / anchor["cell_units"]


def load_anchor(isl):
    return json.loads((OUT_DIR / isl["anchor"]).read_text())


def load_chunk_light(cdir):
    """cameras.npz と untrusted.json だけを読む (per-frame点群は読まない)"""
    cams = np.load(cdir / "cameras.npz")
    names = [str(n) for n in cams["names"]]
    bad = cdir / "untrusted.json"
    untrusted = set(json.loads(bad.read_text())) if bad.exists() else set()
    return names, cams["positions"], cams["poses"], untrusted


def island_to_drawing_pts(pts, anchor, grid, z_off):
    """島world座標 → 図面メートル座標 (export_aligned2.export と同一の手順)"""
    p = np.asarray(pts, dtype=np.float64) @ np.array(anchor["R_align"]).T
    p = p.copy()
    p[:, 2] *= -1
    cells = np.stack([(p[:, 2] - anchor["origin"][1]) / anchor["cell_units"],
                      (p[:, 0] - anchor["origin"][0]) / anchor["cell_units"]],
                     axis=1)
    if anchor["mirror_col"] is not None:
        cells[:, 1] = anchor["mirror_col"] - cells[:, 1]
    pc = anchor_transform_cells(cells, anchor)
    xm = (grid.x0_px + pc[:, 1] * grid.cell_px) * mm_per_px() / 1000
    ym = (grid.y0_px + pc[:, 0] * grid.cell_px) * mm_per_px() / 1000
    mpu_z = island_scale(anchor, grid)
    floor_ref = anchor.get("floor_phys_y_units", anchor["floor_y_units"])
    h = (floor_ref - p[:, 1]) * mpu_z + z_off
    return np.stack([xm, ym, h], axis=1)


def drawing_matrix(anchor, grid, z_off, metric_input=False):
    """島world → 図面メートル の4x4を数値的に求める (手計算せず実変換にフィット)。

    metric_input=True なら入力が「島座標をkでメートル化したもの」であるとして、
    スケール1の剛体+反射行列を返す (TSDFのメッシュ変換用)。
    """
    rng = np.random.default_rng(0)
    src = rng.normal(scale=5.0, size=(12, 3))
    dst = island_to_drawing_pts(src, anchor, grid, z_off)
    if metric_input:
        src = src * island_scale(anchor, grid)
    A = np.hstack([src, np.ones((len(src), 1))])
    sol, *_ = np.linalg.lstsq(A, dst, rcond=None)   # (4,3)
    M = np.eye(4)
    M[:3, :3] = sol[:3].T
    M[:3, 3] = sol[3]
    resid = np.abs(A @ sol - dst).max()
    L = M[:3, :3]
    lam2 = (L.T @ L)
    scale = math.sqrt(lam2[0, 0])
    ortho_err = np.abs(lam2 / scale ** 2 - np.eye(3)).max()
    return M, {"fit_resid_m": float(resid), "scale": float(scale),
               "ortho_err": float(ortho_err), "det": float(np.linalg.det(L))}


def apply_matrix(M, pts):
    return np.asarray(pts) @ M[:3, :3].T + M[:3, 3]


# ---------------------------------------------------------------- ポーズ解決

def resolve_frames(isl, verbose=True):
    """島を構成する全フレームの (chunk_dir, name, R_world, t_world_m, depth_scale)

    - R_world/t_world_m は「島座標をメートル化した系」でのcam2world
    - depth_scale は生の深度(単位)→メートル の係数 (= s_k * k)
    - 重複フレームは先勝ち、untrustedは除外、COLMAP吸着の外れフレームも除外
    """
    root = RESULTS_DIR / isl["root"]
    cdirs = [root / f"chunk_{k:02d}" for k in isl["chunks"]]
    anchor = load_anchor(isl)
    k = island_scale(anchor, PlanGrid(isl["floor"]))

    out, seen = [], set()
    n_skip_dup = n_skip_bad = n_skip_snap = 0
    if isl["mode"] in ("colmap", "island"):
        if isl["mode"] == "island":
            # 島自身の補正済みゲージ (colmap_prior --metric-rescale の出力) に
            # チャンクをフィットする。COLMAPスケールドリフトの大きい長尺用
            d = np.load(RESULTS_DIR / "frontend_bench" / isl["tag"]
                        / "global_cameras.npz")
            centers = {int(str(n).replace("frame_", "")): p
                       for n, p in zip(d["names"], d["positions"])}
        else:
            centers, _ = colmap_centers(COLMAP_MODEL[SESSION])
        for cdir in cdirs:
            names, pos, poses, untrusted = load_chunk_light(cdir)
            T, resid, n_match = fit_chunk_to_colmap(names, pos, centers)
            if T is None:
                if verbose:
                    print(f"  {cdir.name}: COLMAP対応不足 -> skip")
                continue
            s_k, R_k, t_k = T
            for i, n in enumerate(names):
                if n in untrusted:
                    n_skip_bad += 1
                    continue
                if n in seen:
                    n_skip_dup += 1
                    continue
                t_world = apply(T, pos[i][None])[0]
                fr = int(n.replace("frame_", ""))
                if fr in centers:
                    snap = float(np.linalg.norm(t_world - centers[fr])) * k
                    if snap > SNAP_MAX_M:
                        n_skip_snap += 1
                        continue
                seen.add(n)
                out.append({"chunk_dir": cdir, "name": n,
                            "R": R_k @ poses[i][:3, :3],
                            "t_m": t_world * k, "depth_scale": s_k * k})
    else:
        from img_align.core.chaining import load_chunk
        chunks, T_list, _ = chain(cdirs)
        for ci, cdir in enumerate(cdirs):
            names, pos, poses, _, untrusted = chunks[ci]
            s_k, R_k, t_k = T_list[ci]
            for i, n in enumerate(names):
                if n in untrusted:
                    n_skip_bad += 1
                    continue
                if n in seen:
                    n_skip_dup += 1
                    continue
                seen.add(n)
                out.append({"chunk_dir": cdir, "name": n,
                            "R": R_k @ poses[i][:3, :3],
                            "t_m": apply(T_list[ci], pos[i][None])[0] * k,
                            "depth_scale": s_k * k})
    if verbose:
        print(f"{isl['tag']}: {len(out)} frames "
              f"(dup {n_skip_dup}, untrusted {n_skip_bad}, snap {n_skip_snap}), "
              f"k={k:.4f} m/unit")
    return out, anchor


# ---------------------------------------------------------------- キューブマップ

FACE_S = 272          # 面の解像度 (equirectの0.347°/pxに整合)
FACE_F = 128.0        # 焦点距離 → ±46.7° (90°+パディング。継ぎ目の穴を防ぐ)
FACE_C = 136.0
DEPTH_TRUNC = {"0828": 6.0, "0312": 7.0, "19176": 7.0}
DEPTH_MIN_M = 0.3
VOXEL = 0.025
SDF_TRUNC = 0.075     # 間仕切り壁(厚0.10-0.12m)の表裏が融合しない範囲で最大


def _rot(axis, deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


# カメラ系は x右 / y下 / z前
FACES = {
    "front": np.eye(3),
    "right": _rot("y", 90),
    "back": _rot("y", 180),
    "left": _rot("y", -90),
    "up": _rot("x", -90),
    "down": _rot("x", 90),
}


def build_face_luts(w, h):
    """各面について equirect の (row, col) インデックスと 放射→z 係数を事前計算"""
    i, j = np.meshgrid(np.arange(FACE_S), np.arange(FACE_S), indexing="xy")
    u = (i - FACE_C) / FACE_F
    v = (j - FACE_C) / FACE_F
    norm = np.sqrt(1.0 + u ** 2 + v ** 2)
    d_face = np.stack([u, v, np.ones_like(u)], axis=-1) / norm[..., None]
    luts = {}
    for name, R in FACES.items():
        d = d_face @ R.T                      # カメラ系の方向
        lon = np.arctan2(d[..., 0], d[..., 2])
        lat = np.arcsin(np.clip(-d[..., 1], -1, 1))
        col = np.rint((lon / (2 * math.pi) + 0.5) * w - 0.5).astype(np.int32) % w
        row = np.clip(np.rint((0.5 - lat / math.pi) * h - 0.5), 0,
                      h - 1).astype(np.int32)
        luts[name] = (row, col, (1.0 / norm).astype(np.float32), R)
    return luts


def clean_equirect_depth(d_m):
    """深度不連続をまたぐ「膜」を防ぐ: 段差の大きい画素を無効化し、無効を膨張

    フライングピクセルや開口枠の混合画素を落とさないと、TSDFがドア枠に
    膜を張って融合してしまう
    """
    valid = d_m > 0
    thr = np.maximum(0.05, 0.03 * d_m)
    bad = np.zeros_like(valid)
    for ax, sh in ((1, 1), (1, -1), (0, 1), (0, -1)):
        nb = np.roll(d_m, sh, axis=ax)
        nb_v = np.roll(valid, sh, axis=ax)
        if ax == 0:      # 行方向はラップしない
            if sh == 1:
                nb_v[0] = False
            else:
                nb_v[-1] = False
        bad |= valid & nb_v & (np.abs(d_m - nb) > thr)
    # 無効マスクを1px膨張 (混合画素とマスク境界を除去)
    inv = (~valid) | bad
    grow = inv.copy()
    for ax, sh in ((1, 1), (1, -1), (0, 1), (0, -1)):
        grow |= np.roll(inv, sh, axis=ax)
    out = d_m.copy()
    out[grow] = 0.0
    return out


def frame_faces(fr, luts, depth_trunc):
    """1フレーム分の (面名, depth(S,S)float32[m], color(S,S,3)uint8) を返す"""
    from img_align.core.io_ply import read_ply_xyzrgb
    cdir = fr["chunk_dir"]
    name = fr["name"]
    d_png = cdir / "panovggt_out" / "depth" / f"{name}_depth_raw.png"
    import cv2
    raw = cv2.imread(str(d_png), cv2.IMREAD_UNCHANGED)
    if raw is None:
        return []
    h, w = raw.shape
    d_m = raw.astype(np.float32) / 1000.0 * fr["depth_scale"]
    d_m[d_m < DEPTH_MIN_M] = 0.0
    d_m[d_m > depth_trunc] = 0.0
    d_m = clean_equirect_depth(d_m)

    # 色: per-frame PLY の RGB を equirect へ散布 (非ゼロ深度画素と行優先で1:1)
    ply = cdir / "panovggt_out" / "pointclouds" / "per_frame" / f"{name}.ply"
    color = np.zeros((h, w, 3), np.uint8)
    if ply.exists():
        _, rgb = read_ply_xyzrgb(ply)
        rows, cols = np.nonzero(raw)
        n = min(len(rgb), len(rows))
        color[rows[:n], cols[:n]] = rgb[:n]

    out = []
    for fname, (row, col, zmul, R) in luts.items():
        dep = d_m[row, col] * zmul
        if not np.any(dep > 0):
            continue
        col_img = color[row, col]
        out.append((fname, np.ascontiguousarray(dep),
                    np.ascontiguousarray(col_img), R))
    return out


def integrate_island(isl, limit=None, verbose=True):
    """島のTSDFボリュームを構築して (mesh, anchor, grid, stats) を返す"""
    import open3d as o3d
    frames, anchor = resolve_frames(isl, verbose=verbose)
    if limit:
        frames = frames[:limit]
    grid = PlanGrid(isl["floor"])
    luts = build_face_luts(1036, 518)
    intr = o3d.camera.PinholeCameraIntrinsic(FACE_S, FACE_S, FACE_F, FACE_F,
                                             FACE_C, FACE_C)
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=VOXEL, sdf_trunc=SDF_TRUNC,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    trunc = DEPTH_TRUNC[SESSION]
    n_int = 0
    for idx, fr in enumerate(frames):
        cam2world = np.eye(4)
        for fname, dep, col_img, R in frame_faces(fr, luts, trunc):
            cam2world[:3, :3] = fr["R"] @ R
            cam2world[:3, 3] = fr["t_m"]
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(col_img), o3d.geometry.Image(dep),
                depth_scale=1.0, depth_trunc=trunc,
                convert_rgb_to_intensity=False)
            vol.integrate(rgbd, intr,
                          np.ascontiguousarray(np.linalg.inv(cam2world)))
            n_int += 1
        if verbose and idx % 25 == 0:
            print(f"  {idx}/{len(frames)} frames")
    mesh = vol.extract_triangle_mesh()
    if verbose:
        print(f"  integrated {n_int} faces -> {len(mesh.vertices):,} verts / "
              f"{len(mesh.triangles):,} tris")
    return mesh, anchor, grid, {"frames": len(frames), "faces": n_int}


def mesh_to_drawing(mesh, anchor, grid, z_off):
    """島メートル座標のメッシュを図面座標へ (反射なら巻き順反転)"""
    M, info = drawing_matrix(anchor, grid, z_off, metric_input=True)
    v = np.asarray(mesh.vertices)
    mesh.vertices = type(mesh.vertices)(apply_matrix(M, v))
    if info["det"] < 0:
        tri = np.asarray(mesh.triangles)[:, ::-1]
        mesh.triangles = type(mesh.triangles)(np.ascontiguousarray(tri))
    mesh.compute_vertex_normals()
    return mesh, info


def clean_mesh(mesh, min_tris=500, min_diag=0.30):
    """小さな連結成分を除去"""
    import open3d as o3d
    idx, counts, _ = mesh.cluster_connected_triangles()
    idx = np.asarray(idx)
    counts = np.asarray(counts)
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    drop = np.zeros(len(tris), bool)
    for c in range(len(counts)):
        sel = idx == c
        if counts[c] < min_tris:
            drop |= sel
            continue
        pts = verts[tris[sel].ravel()]
        if np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)) < min_diag:
            drop |= sel
    mesh.remove_triangles_by_mask(drop)
    mesh.remove_unreferenced_vertices()
    return mesh, int(drop.sum())


# ---------------------------------------------------------------- 検証

def cmd_verify(args):
    """土台の健全性チェック: 4x4の一致と、ポーズ解決の既存実装との突き合わせ"""
    for isl in islands_expanded(SESSION):
        anchor = load_anchor(isl)
        grid = PlanGrid(isl["floor"])
        M, info = drawing_matrix(anchor, grid, isl["z_off"])
        Mm, info_m = drawing_matrix(anchor, grid, isl["z_off"],
                                    metric_input=True)
        print(f"\n--- {isl['tag']} ({isl['floor']}, z_off={isl['z_off']})")
        print(f"  4x4(units): resid {info['fit_resid_m']:.2e}m "
              f"scale {info['scale']:.4f} ortho_err {info['ortho_err']:.2e} "
              f"det {info['det']:+.4f}")
        print(f"  4x4(metric): scale {info_m['scale']:.4f} "
              f"det {info_m['det']:+.4f}  -> 巻き順反転 "
              f"{'必要' if info_m['det'] < 0 else '不要'}")
        # 既存の点群書き出しと一致するか (ランダム点で)
        rng = np.random.default_rng(1)
        pts = rng.normal(scale=4.0, size=(5, 3))
        ref = island_to_drawing_pts(pts, anchor, grid, isl["z_off"])
        got = apply_matrix(M, pts)
        print(f"  export_aligned2 整合: max {np.abs(ref-got).max():.2e} m")

        frames, _ = resolve_frames(isl)
        # 既存の軌跡(アンカー適用後のカメラ位置)と一致するか
        cams_m = np.stack([f["t_m"] for f in frames])
        cams_draw = apply_matrix(Mm, cams_m)
        print(f"  camera z range: {cams_draw[:,2].min():.2f} .. "
              f"{cams_draw[:,2].max():.2f} m (期待: 床上~1.0-1.7m)")
        ds = np.array([f["depth_scale"] for f in frames])
        print(f"  depth_scale: {ds.min():.4f} .. {ds.max():.4f} "
              f"(チャンク毎に変動)")


def cmd_poisson(args):
    """Poisson表面再構成 (精製版): 法線の向きを軌跡カメラで決めてから再構成"""
    import open3d as o3d
    from scipy.spatial import cKDTree

    from img_align.mesh.collision_mesh import CURTAIN_COLORS, PLY_NAME
    from img_align.core.io_ply import read_ply_xyzrgb
    xyz, rgb = read_ply_xyzrgb(OUT_DIR / PLY_NAME[SESSION])
    keep = np.ones(len(xyz), bool)
    for c in CURTAIN_COLORS:          # export時の合成カーテン点を除去
        keep &= ~np.all(rgb == np.array(c, np.uint8), axis=1)
    print(f"points {len(xyz):,} -> {int(keep.sum()):,} (カーテン除去)")

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz[keep])
    pc.colors = o3d.utility.Vector3dVector(rgb[keep] / 255.0)
    pc = pc.voxel_down_sample(args.voxel)
    pc, _ = pc.remove_statistical_outlier(nb_neighbors=16, std_ratio=2.0)
    print(f"downsample({args.voxel}m)+外れ値除去 -> {len(pc.points):,}")

    # 法線: 推定後、最近傍カメラの方向へ向きを揃える
    # (consistent_tangent_plane は屋内の閉空間で向きが破綻しやすい)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(0.10, 30))
    cams = []
    for isl in islands_expanded(SESSION):
        frames, anchor = resolve_frames(isl, verbose=False)
        grid = PlanGrid(isl["floor"])
        M, _ = drawing_matrix(anchor, grid, isl["z_off"], metric_input=True)
        cams.append(apply_matrix(M, np.stack([f["t_m"] for f in frames])))
    cams = np.concatenate(cams)
    pts = np.asarray(pc.points)
    nrm = np.asarray(pc.normals)
    _, ci = cKDTree(cams).query(pts, k=1)
    to_cam = cams[ci] - pts
    flip = (nrm * to_cam).sum(axis=1) < 0
    nrm[flip] *= -1
    pc.normals = o3d.utility.Vector3dVector(nrm)
    print(f"法線をカメラ{len(cams)}台の方向へ整列 ({int(flip.sum()):,} 反転)")

    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pc, depth=args.depth, linear_fit=True)
    dens = np.asarray(dens)
    mesh.remove_vertices_by_mask(dens < np.quantile(dens, 0.05))
    bb = pc.get_axis_aligned_bounding_box()
    mesh = mesh.crop(bb)
    mesh, n_drop = clean_mesh(mesh)
    if args.target and len(mesh.triangles) > args.target:
        mesh = mesh.simplify_quadric_decimation(args.target)
    mesh.compute_vertex_normals()
    out = OUT_DIR / f"mesh_poisson{SFX}.ply"
    o3d.io.write_triangle_mesh(str(out), mesh, write_ascii=False,
                               compressed=True)
    (OUT_DIR / f"mesh_poisson{SFX}.json").write_text(json.dumps(
        {"session": SESSION, "voxel": args.voxel, "depth": args.depth,
         "verts": len(mesh.vertices), "tris": len(mesh.triangles)}, indent=1))
    print(f"-> {out} ({len(mesh.vertices):,} verts / "
          f"{len(mesh.triangles):,} tris, 小成分 {n_drop:,} 除去)")


def cmd_gate1(args):
    """ゲート1: 1フレームだけ積分し、そのフレームの点群(メートル島座標)と照合する。

    放射→z変換・面回転・LUTのラップ・チャンク毎スケールの誤りを一発で検出する
    """
    import open3d as o3d
    from img_align.core.io_ply import read_ply_xyzrgb
    isl = islands_expanded(SESSION)[0]
    mesh, anchor, grid, _ = integrate_island(isl, limit=1, verbose=True)
    fr = resolve_frames(isl, verbose=False)[0][0]
    xyz, _ = read_ply_xyzrgb(fr["chunk_dir"] / "panovggt_out" / "pointclouds"
                             / "per_frame" / f"{fr['name']}.ply")
    # カメラローカル点群 → メートル島座標
    pts = (xyz * fr["depth_scale"]) @ fr["R"].T + fr["t_m"]
    pc_ref = o3d.geometry.PointCloud()
    pc_ref.points = o3d.utility.Vector3dVector(pts[::7])
    d = np.asarray(pc_ref.compute_point_cloud_distance(
        o3d.geometry.PointCloud(mesh.vertices)))
    print(f"\nGATE1 {fr['name']}: 点群→メッシュ距離 "
          f"median {np.median(d):.4f}m p90 {np.percentile(d, 90):.4f}m "
          f"(閾値 voxel={VOXEL}m)")
    print("PASS" if np.median(d) < VOXEL else "FAIL")


def cmd_tsdf(args):
    import open3d as o3d
    meshes, stats = [], []
    for isl in islands_expanded(SESSION):
        print(f"\n=== {isl['tag']}")
        mesh, anchor, grid, st = integrate_island(isl, limit=args.limit)
        mesh, info = mesh_to_drawing(mesh, anchor, grid, isl["z_off"])
        # 高さクロップ (これが無いと隣接階のジオメトリが漏れる)
        v = np.asarray(mesh.vertices)
        lo, hi = isl["z_off"] - 0.3, isl["z_off"] + 3.0
        bad_v = (v[:, 2] < lo) | (v[:, 2] > hi)
        if bad_v.any():
            mesh.remove_vertices_by_mask(bad_v)
        mesh, n_drop = clean_mesh(mesh, min_tris=args.min_tris,
                                  min_diag=args.min_diag)
        print(f"  -> {len(mesh.vertices):,} verts / {len(mesh.triangles):,} tris "
              f"(小成分 {n_drop:,} tris除去, det {info['det']:+.3f})")
        o3d.io.write_triangle_mesh(
            str(OUT_DIR / f"mesh_tsdf_{isl['tag']}.ply"), mesh,
            write_ascii=False, compressed=True)
        meshes.append(mesh)
        st.update({"tag": isl["tag"], "verts": len(mesh.vertices),
                   "tris": len(mesh.triangles)})
        stats.append(st)
    merged = meshes[0]
    for m in meshes[1:]:
        merged += m
    if args.target and len(merged.triangles) > args.target:
        merged = merged.simplify_quadric_decimation(args.target)
    merged.compute_vertex_normals()
    out = OUT_DIR / f"mesh_tsdf{SFX}.ply"
    o3d.io.write_triangle_mesh(str(out), merged, write_ascii=False,
                               compressed=True)
    (OUT_DIR / f"mesh_tsdf{SFX}.json").write_text(json.dumps(
        {"session": SESSION, "islands": stats, "voxel": VOXEL,
         "sdf_trunc": SDF_TRUNC, "verts": len(merged.vertices),
         "tris": len(merged.triangles)}, indent=1))
    print(f"\n-> {out} ({len(merged.vertices):,} verts / "
          f"{len(merged.triangles):,} tris)")


def wall_distance_stats(pts, floor):
    """点群/メッシュ頂点 → 図面GT壁の面ライン への2D距離統計 (メートル)"""
    from img_align.postproc.plan_icp import wall_face_segments_px
    segs = wall_face_segments_px(floor) * mm_per_px() / 1000
    a, b = segs[:, :2], segs[:, 2:]
    ab = b - a
    ab2 = (ab ** 2).sum(axis=1).clip(min=1e-9)
    p = np.asarray(pts)[:, :2]
    best = np.full(len(p), np.inf)
    for i in range(0, len(p), 20000):        # メモリ節約のため分割
        q = p[i:i + 20000]
        t = np.clip(((q[:, None, :] - a[None]) * ab[None]).sum(-1)
                    / ab2[None], 0, 1)
        foot = a[None] + t[..., None] * ab[None]
        best[i:i + 20000] = np.linalg.norm(q[:, None, :] - foot,
                                           axis=-1).min(axis=1)
    # 壁被覆率: 壁線を0.1m刻みでサンプルし、近傍に点があるか
    samp = []
    for x1, y1, x2, y2 in segs:
        n = max(int(math.hypot(x2 - x1, y2 - y1) / 0.1), 2)
        tt = np.linspace(0, 1, n)
        samp.append(np.stack([x1 + tt * (x2 - x1), y1 + tt * (y2 - y1)], 1))
    samp = np.concatenate(samp)
    from scipy.spatial import cKDTree
    d_s, _ = cKDTree(p).query(samp, k=1)
    return {"median_m": round(float(np.median(best)), 4),
            "p90_m": round(float(np.percentile(best, 90)), 4),
            "within_10cm": round(float((best < 0.10).mean()), 3),
            "wall_coverage": round(float((d_s < 0.25).mean()), 3),
            "n": int(len(p))}


def cmd_compare(args):
    """Poisson / TSDF / 元点群 を図面壁への距離で横並び比較 + 真上ビュー"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import open3d as o3d

    from img_align.mesh.collision_mesh import CURTAIN_COLORS, PLY_NAME
    from img_align.core.io_ply import read_ply_xyzrgb
    from img_align.postproc.plan_icp import wall_face_segments_px

    items = {}
    xyz, rgb = read_ply_xyzrgb(OUT_DIR / PLY_NAME[SESSION])
    keep = np.ones(len(xyz), bool)
    for c in CURTAIN_COLORS:
        keep &= ~np.all(rgb == np.array(c, np.uint8), axis=1)
    items["pointcloud"] = xyz[keep]
    for kind in ("tsdf", "poisson"):
        p = OUT_DIR / f"mesh_{kind}{SFX}.ply"
        if p.exists():
            m = o3d.io.read_triangle_mesh(str(p))
            items[kind] = np.asarray(m.vertices)

    res = {}
    for name, pts in items.items():
        for floor, z_off in (("1F", 0.0), ("2F", 2.9)):
            band = pts[(pts[:, 2] > z_off + 0.3) & (pts[:, 2] < z_off + 2.2)]
            if len(band) < 1000:
                continue
            res[f"{name}/{floor}"] = wall_distance_stats(band, floor)
    print(json.dumps(res, indent=1, ensure_ascii=False))
    (OUT_DIR / f"mesh_compare{SFX}.json").write_text(
        json.dumps(res, indent=1, ensure_ascii=False))

    # 真上ビュー (壁高さ帯のみ)
    names = [n for n in ("pointcloud", "tsdf", "poisson") if n in items]
    fig, axes = plt.subplots(len(names), 2,
                             figsize=(20, 5.5 * len(names)), squeeze=False)
    for r, name in enumerate(names):
        for c, (floor, z_off) in enumerate((("1F", 0.0), ("2F", 2.9))):
            ax = axes[r][c]
            pts = items[name]
            band = pts[(pts[:, 2] > z_off + 0.3) & (pts[:, 2] < z_off + 2.2)]
            ax.scatter(band[::3, 0], band[::3, 1], s=0.3, c="orange" if
                       name != "pointcloud" else "white", alpha=0.5)
            for x1, y1, x2, y2 in wall_face_segments_px(floor) * mm_per_px() / 1000:
                ax.plot([x1, x2], [y1, y2], "-", color="cyan", lw=1.0)
            ax.set_aspect("equal")
            ax.invert_yaxis()
            ax.set_facecolor("#101014")
            k = f"{name}/{floor}"
            t = res.get(k, {})
            ax.set_title(f"{name} {floor}: med {t.get('median_m','-')}m "
                         f"cover {t.get('wall_coverage','-')}", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"mesh_compare{SFX}.png", dpi=100,
                facecolor="#181820")
    plt.close(fig)
    print(f"-> {OUT_DIR / f'mesh_compare{SFX}.png'}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("verify")
    sub.add_parser("gate1")
    p = sub.add_parser("tsdf")
    p.add_argument("--limit", type=int, default=None, help="使用フレーム数上限")
    p.add_argument("--target", type=int, default=1_500_000,
                   help="統合メッシュの目標三角形数 (0で無効)")
    # 工事中の軸組は細い断片になるため小成分除去は控えめに
    p.add_argument("--min-tris", type=int, default=150)
    p.add_argument("--min-diag", type=float, default=0.15)
    q = sub.add_parser("poisson")
    q.add_argument("--voxel", type=float, default=0.02)
    q.add_argument("--depth", type=int, default=10)
    q.add_argument("--target", type=int, default=1_500_000)
    sub.add_parser("compare")
    args = ap.parse_args()
    {"verify": cmd_verify, "gate1": cmd_gate1, "tsdf": cmd_tsdf,
     "poisson": cmd_poisson, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    main()
