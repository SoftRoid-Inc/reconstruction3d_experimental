"""PanoRecon(geniesim_world)の単画像GSを実測ポーズで配置・合併して家全体のGSを作る

- prep : 軌跡2m間隔の候補を最遠点サンプリング順に並べ、2048x1024パノラマを書き出す
         (リストのどこで打ち切っても、その枚数での最良カバレッジになる)
- merge: フレームGSを 深度比スケール→ポーズ→図面座標 で変換し、
         撮影者カット+Voronoiデデュープの上で1つのplyに合併する

使い方:
  IMG_SESSION=0828 conda run -n img_align   python -m img_align.gs.gs_world prep
  (バッチは prep が出力する run_batch.sh を実行)
  IMG_SESSION=0828 conda run -n panovggt    python -m img_align.gs.gs_world merge
出力: results/geniesim_world/<sess>/panos/*.png, runlist.json
      results/backend/gaussian_house_pr{SFX}.ply
"""
import argparse
import json
import math
import os

import numpy as np

from img_align.config import (FRAMES_DIR, PROJECT_ROOT, RESULTS_DIR, SESSION,
                              frame_name)
from img_align.core.plan_grid import PlanGrid
from img_align.mesh.meshing import ISLANDS, OUT_DIR, SFX, apply_matrix, drawing_matrix, islands_expanded, islands_for, island_scale, load_anchor, resolve_frames

GW_DIR = RESULTS_DIR / "geniesim_world"
STEP_M = 2.0
PANO_W, PANO_H = 2048, 1024
SHARP_CKPT = "/code2/third_party/weights/sharp_2572gikvuh.pt"
DA360_CKPT = ("/code2/third_party/genie_sim/source/external/DA360/"
              "checkpoints/DA360_large.pth")
PERSON_CUT_R = 2.0        # 撮影者カットの半径 [m]
OPAC_MIN = 0.05
MAX_LOCAL_R_M = 7.0       # カメラからこの距離を超えるガウシアンは捨てる (窓越し屋外)
THIN_VOXEL = 0.01         # フレーム毎のボクセル間引き (不透明度優先でTop-2/ボクセル)


# ---------------------------------------------------------------- 共通

def frame_poses_drawing():
    """全島 (セグメント微調整があれば仮想島に展開) のフレームを図面座標ポーズ付きで返す
    [{name, fr, isl, R_w(3,3 island-m), t_m, depth_scale, M(4,4 island-m→図面), pos_draw}]
    チャンクのオーバーラップで隣接セグメント両方に出るフレームは先勝ちで1つに"""
    out, seen = [], set()
    for isl in islands_expanded(SESSION):
        frames, _ = resolve_frames(isl, verbose=False)
        M, _ = drawing_matrix(load_anchor(isl), PlanGrid(isl["floor"]),
                              isl["z_off"], metric_input=True)
        for f in frames:
            fr = int(f["name"].replace("frame_", ""))
            if fr in seen:
                continue
            seen.add(fr)
            out.append({**f, "isl": isl, "M": M, "fr": fr,
                        "pos_draw": apply_matrix(M, f["t_m"][None])[0]})
    return out


def select_candidates(poses):
    """軌跡順で2m間隔に間引く"""
    poses = sorted(poses, key=lambda p: p["fr"])
    sel, acc, prev = [], None, None
    for p in poses:
        if prev is None:
            sel.append(p)
            acc = 0.0
        else:
            acc += float(np.linalg.norm(p["pos_draw"][:2] - prev["pos_draw"][:2]))
            if acc >= STEP_M:
                sel.append(p)
                acc = 0.0
        prev = p
    return sel


def fps_order(cands):
    """最遠点サンプリング順 (先頭=重心から最遠)"""
    pts = np.stack([c["pos_draw"] for c in cands])
    order = [int(np.argmax(np.linalg.norm(pts - pts.mean(0), axis=1)))]
    d = np.linalg.norm(pts - pts[order[0]], axis=1)
    while len(order) < len(cands):
        i = int(np.argmax(d))
        order.append(i)
        d = np.minimum(d, np.linalg.norm(pts - pts[i], axis=1))
    return [cands[i] for i in order]


# ---------------------------------------------------------------- prep

def cmd_prep(args):
    import cv2
    poses = frame_poses_drawing()
    cands = fps_order(select_candidates(poses))
    out = GW_DIR / SESSION
    (out / "panos").mkdir(parents=True, exist_ok=True)
    runlist = []
    for c in cands:
        src = FRAMES_DIR / frame_name(c['fr'])
        dst = out / "panos" / f"{c['name']}.png"
        if not src.exists():
            print(f"  warn: {src.name} なし -> skip")
            continue
        if not dst.exists():
            img = cv2.imread(str(src))
            cv2.imwrite(str(dst), cv2.resize(img, (PANO_W, PANO_H),
                                             interpolation=cv2.INTER_AREA))
        runlist.append(c["name"])
    (out / "runlist.json").write_text(json.dumps(runlist, indent=1))
    # バッチスクリプト (既存出力スキップ・BUDGET_MIN分で打ち切り)
    # コンテナ内パス: リポジトリは /code2 にマウント (compose: .:/code2)
    gw_c = "/code2/" + str((GW_DIR / SESSION).relative_to(PROJECT_ROOT.parent))
    sh = ["#!/bin/bash", "# PanoReconバッチ (FPS順・時間予算打ち切り)",
          'BUDGET_MIN="${BUDGET_MIN:-60}"', "t0=$(date +%s)", ""]
    for name in runlist:
        sh += [f'if [ ! -f {gw_c}/{name}/merged_gaussians.ply ]; then',
               f'  el=$(( ($(date +%s) - t0) / 60 ))',
               f'  if [ $el -ge $BUDGET_MIN ]; then echo "BUDGET reached (${{el}}min)"; exit 0; fi',
               f'  echo "=== {SESSION}/{name} (${{el}}min)"',
               f'  geniesim_world create '
               f'-p {gw_c}/panos/{name}.png '
               f'-o {gw_c} '
               f'-c {SHARP_CKPT} --da360-checkpoint {DA360_CKPT} || exit 1',
               # 中間物(sharp/cubes ~1GB/枚)は即削除 (ディスク満杯の再発防止)
               f'  rm -rf {gw_c}/{name}/{{sharp,cubes}}',
               'fi']
    (out / "run_batch.sh").write_text("\n".join(sh) + "\n")
    os.chmod(out / "run_batch.sh", 0o755)
    print(f"{SESSION}: 候補 {len(runlist)} 枚 (FPS順) -> {out}/runlist.json, run_batch.sh")


# ---------------------------------------------------------------- merge

def da360_scale(name, isl_dir, depth_scale):
    """DA360深度(相対) vs PanoVGGT深度(メートル) の比 -> ガウシアン→メートル係数"""
    import cv2
    exr = GW_DIR / SESSION / name / "input_depth.exr"
    da = cv2.imread(str(exr), cv2.IMREAD_UNCHANGED)
    if da is None:
        return None
    if da.ndim == 3:
        da = da[..., 0]
    vg_p = isl_dir / "panovggt_out" / "depth" / f"{name}_depth_raw.png"
    vg = cv2.imread(str(vg_p), cv2.IMREAD_UNCHANGED)
    if vg is None:
        return None
    vg_m = vg.astype(np.float64) / 1000.0 * depth_scale     # メートル
    da_r = cv2.resize(da, (vg.shape[1], vg.shape[0]),
                      interpolation=cv2.INTER_NEAREST).astype(np.float64)
    valid = (vg_m > 0.3) & (da_r > 1e-6) & np.isfinite(da_r)
    if valid.sum() < 10000:
        return None
    return float(np.median(vg_m[valid] / da_r[valid]))


def person_dirs_mask(name):
    from img_align.core.crops import load_person_mask
    return load_person_mask(name)          # (1024,2048) bool or None


def cut_person(xyz_local_conv, mask):
    """VGGT規約ローカル座標(x右/y下/z前)のガウシアンからpersonマスク方向を除外"""
    if mask is None:
        return np.ones(len(xyz_local_conv), bool)
    r = np.linalg.norm(xyz_local_conv, axis=1).clip(min=1e-9)
    lon = np.arctan2(xyz_local_conv[:, 0], xyz_local_conv[:, 2])
    lat = np.arcsin(np.clip(-xyz_local_conv[:, 1] / r, -1, 1))
    H, W = mask.shape
    col = np.clip(((lon / (2 * math.pi) + 0.5) * W - 0.5).round().astype(int),
                  0, W - 1)
    row = np.clip(((0.5 - lat / math.pi) * H - 0.5).round().astype(int),
                  0, H - 1)
    hit = mask[row, col] & (r < PERSON_CUT_R)
    return ~hit


RY180 = np.diag([-1.0, 1.0, -1.0])


def plan_bbox_m():
    """図面GT壁のbbox±マージン [m] (窓越し屋外の飛散クロップ用)"""
    from img_align.eval.gt_traj import mm_per_px
    from img_align.postproc.plan_icp import wall_segments_px
    lo = np.array([np.inf, np.inf])
    hi = -lo
    for isl in islands_for(SESSION):
        segs = wall_segments_px(isl["floor"]) * mm_per_px() / 1000
        pts = segs.reshape(-1, 2)
        lo = np.minimum(lo, pts.min(0))
        hi = np.maximum(hi, pts.max(0))
    return lo - 1.0, hi + 1.0


def cmd_merge(args):
    from img_align.gs.gs_merge import transform_gaussians, write_3dgs_ply
    from img_align.legacy.merge_gs import estimate_conv_and_scale, read_gaussians
    from img_align.core.io_ply import read_ply_xyzrgb

    poses = {p["name"]: p for p in frame_poses_drawing()}
    done = sorted(d.name for d in (GW_DIR / SESSION).iterdir()
                  if d.is_dir() and (d / "merged_gaussians.ply").exists()
                  and d.name in poses)
    if not done:
        raise SystemExit("PanoRecon出力が無い (run_batch.sh を先に)")
    print(f"{SESSION}: {len(done)} フレームを合併")
    cam_draw = np.stack([poses[n]["pos_draw"] for n in done])
    from scipy.spatial import cKDTree
    cam_tree = cKDTree(cam_draw)

    SH_C0 = 0.28209479
    bbox_lo, bbox_hi = plan_bbox_m()
    cache_p = GW_DIR / SESSION / "scale_cache.json"
    cache = json.loads(cache_p.read_text()) if cache_p.exists() else {}
    parts = []
    rng = np.random.default_rng(0)
    for fi, name in enumerate(done):
        p = poses[name]
        data, props = read_gaussians(GW_DIR / SESSION / name
                                     / "merged_gaussians.ply")
        xyz = np.stack([data["x"], data["y"], data["z"]], 1).astype(np.float64)

        if name in cache and not args.verify:
            s_m = float(cache[name])
            keep = cut_person(xyz @ RY180.T, person_dirs_mask(name))
            print(f"  {name}: s={s_m:.3f} (cache)")
        else:
          s_da = da360_scale(name, p["chunk_dir"], p["depth_scale"])
          if fi == 0 or s_da is None or args.verify:
            # クロスチェック/フォールバック: PanoVGGTローカル点群とのNN探索
            ref_xyz, _ = read_ply_xyzrgb(
                p["chunk_dir"] / "panovggt_out" / "pointclouds" / "per_frame"
                / f"{name}.ply")
            R_nn, s_nn = estimate_conv_and_scale(xyz, ref_xyz, rng)
            assert np.allclose(R_nn, RY180), f"規約がRy180でない: {name}"
            s_nn_m = s_nn * p["depth_scale"]
            if s_da is None:
                s_m = s_nn_m
                print(f"  {name}: depth-ratio不可 -> NN探索 s={s_m:.3f}")
            else:
                s_m = s_da * p["depth_scale"]
                dev = abs(s_m / s_nn_m - 1)
                print(f"  {name}: s(depth)={s_m:.3f} s(NN)={s_nn_m:.3f} "
                      f"差{dev*100:.1f}%")
                assert dev < 0.15, "スケール推定の不一致"
          else:
            s_m = s_da * p["depth_scale"]
          cache[name] = s_m
          cache_p.write_text(json.dumps(cache, indent=1))
          # 撮影者カット (Ry180でVGGT規約ローカルに直してから方向判定)
          keep = cut_person(xyz @ RY180.T, person_dirs_mask(name))

        # 4x4: 図面 ← 島メートル ← ポーズ ← (s_m·Ry180)
        T_pose = np.eye(4)
        T_pose[:3, :3] = p["R"]
        T_pose[:3, 3] = p["t_m"]
        S = np.eye(4)
        S[:3, :3] = s_m * RY180
        M = p["M"] @ T_pose @ S
        means, quats0, ls0 = (xyz[keep],
                              np.stack([data[f"rot_{i}"] for i in range(4)],
                                       1)[keep].astype(np.float64),
                              np.stack([data[f"scale_{i}"] for i in range(3)],
                                       1)[keep].astype(np.float64))
        means, quats, log_scales = transform_gaussians(means, quats0, ls0, M)

        # 高さ/図面bbox/距離クロップ + 不透明度 + Voronoi(最寄りカメラが自分)
        z_off = p["isl"]["z_off"]
        opac = 1 / (1 + np.exp(-data["opacity"][keep].astype(np.float64)))
        r_local = np.linalg.norm(xyz[keep], axis=1) * s_m
        ok = ((means[:, 2] > z_off - 0.3) & (means[:, 2] < z_off + 3.0)
              & (opac > OPAC_MIN) & (r_local < MAX_LOCAL_R_M)
              & (means[:, 0] > bbox_lo[0]) & (means[:, 0] < bbox_hi[0])
              & (means[:, 1] > bbox_lo[1]) & (means[:, 1] < bbox_hi[1]))
        # Voronoi緩和: 最寄り2カメラのどちらかが自分なら残す
        # (硬い単一所有だと「所有フレームから見えない領域」が黒抜けする)
        k = min(2, len(done))
        _, nn = cam_tree.query(means[ok], k=k)
        nn = nn.reshape(len(nn), -1)
        ok_idx = np.where(ok)[0][(nn == fi).any(axis=1)]
        # ボクセル間引き (不透明度優先でTop-2/ボクセル。1個だと点状の穴が出る)
        order = ok_idx[np.argsort(-opac[ok_idx])]
        cell = np.floor(means[order] / THIN_VOXEL).astype(np.int64)
        _, inv = np.unique(cell, axis=0, return_inverse=True)
        srt = np.argsort(inv, kind="stable")       # グループ順・opac降順を保持
        starts = np.r_[0, np.flatnonzero(np.diff(inv[srt])) + 1]
        first_of_group = np.zeros(len(srt), np.int64)
        first_of_group[starts] = starts
        np.maximum.accumulate(first_of_group, out=first_of_group)
        rank = np.arange(len(srt)) - first_of_group
        ok_idx = order[srt[rank < 2]]
        rgb = np.clip(0.5 + SH_C0 * np.stack(
            [data[f"f_dc_{i}"] for i in range(3)], 1)[keep], 0, 1)
        parts.append((means[ok_idx], quats[ok_idx], log_scales[ok_idx],
                      rgb[ok_idx], opac[ok_idx]))
        print(f"  {name}: {len(xyz):,} -> {len(ok_idx):,} "
              f"(person cut {int((~keep).sum()):,})")

    merged = [np.concatenate([q[i] for q in parts]) for i in range(5)]
    out = OUT_DIR / f"gaussian_house_pr{SFX}.ply"
    write_3dgs_ply(out, *merged)
    print(f"-> {out} ({len(merged[0]):,} gaussians)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prep")
    m = sub.add_parser("merge")
    m.add_argument("--verify", action="store_true",
                   help="全フレームでNN探索とのクロスチェック")
    args = ap.parse_args()
    {"prep": cmd_prep, "merge": cmd_merge}[args.cmd](args)


if __name__ == "__main__":
    main()
