"""敷地境界線ベースのプライバシーマスク生成

図面に注釈された敷地境界線ポリゴン (labelme, label="boundary") を基準に、
各360°フレームの画素を「写っている表面が敷地内なら白 / 敷地外なら黒」に
分類したマスクを出力する。用途はプライバシー(敷地外全部を隠す)なので判定は黒側に倒す:
白は「敷地内かつ深度がD_MAX_M以内の確実な表面」のみ。敷地外・遠方(>D_MAX_M)・
深度なし・ポーズ不信は全て黒。白領域は境界から MARGIN_M 内側へ収縮させる。

パイプライン位置: colmap_prior (グローバルポーズ) + backend_align (図面アンカー)
の後段。深度は PanoVGGT の equirect 深度 (uint16 mm 放射距離)。

サブコマンド:
  anchor    島ごとに図面アンカーJSONを生成 (GT軌跡CSVなしの薄いドライバ)
  generate  全フレームのマスクPNGを <物件>/masks_boundary/ に出力
  viz       図面トップダウン軌跡図 + サンプルフレームのオーバーレイ
"""
import argparse
import json
import math

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from img_align.config import PROP, RESULTS_DIR, SESSION

BOUNDARY_LABEL = "boundary"
CELL_M = 0.02        # 境界ラスタの解像度
MARGIN_M = 0.15      # 白領域の内側収縮 (メートル)
D_MAX_M = 30.0       # これ以遠の深度は不信 (深度なし扱い)
INVALID_DEPTH_WHITE = False  # 深度なし/しきい値超え画素: False=黒(隠す) / True=白
                             # 用途=プライバシー: 白は「敷地内かつD_MAX以内の確実な表面」のみ。
                             # 敷地外・遠方(>D_MAX)・深度なしは全て黒で隠す
NADIR_WHITE = True   # 下端の自撮り棒帯: カメラが敷地内なら白 (カメラ直下=敷地内地面)
# レイ透過テスト (表面の先が敷地外なら黒、薄物透け対策) は不成立で無効化:
# 玄関前はカメラ→境界が3m超で1.5m延長では足場ネット(1.2m)の透けを検出できず、
# 延長を伸ばすと室内の窓際面が誤黒になる (実測: frame100白率85→72%)。0=無効
SEE_THROUGH_M = 0.0
ERODE_PX = 11        # フル解像度での白収縮 (NN拡大境界の補償)
SPECKLE_CLOSE_ITER = 3  # 白領域内の小黒斑 (深度不連続除去由来) をクロージングで埋める
                        # (低解像度で~6px幅まで。窓越し屋外などの大きな黒領域は残る)
FULL_W, FULL_H = 5376, 2688

OUT_DIR = RESULTS_DIR / "boundary_mask"
MASK_OUT_DIR = PROP.frames_dir.parent / "masks_boundary"


# ---------------------------------------------------------------- 境界ポリゴン

def load_boundary_px(floor):
    """labelme JSON から境界の頂点列 (図面px, (N,2)) を読む。
    polygon はそのまま、rectangle (2点=対角) は4隅に展開する"""
    data = json.loads((PROP.zumen_dir / f"{floor}.json").read_text())
    for sh in data["shapes"]:
        if sh["label"] != BOUNDARY_LABEL:
            continue
        pts = np.asarray(sh["points"], dtype=np.float64)
        if sh.get("shape_type") == "rectangle" or len(pts) == 2:
            (x1, y1), (x2, y2) = pts
            return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        if len(pts) < 3:
            raise ValueError(f"{floor}: boundaryの頂点が3未満")
        return pts
    raise FileNotFoundError(
        f"{PROP.zumen_dir / f'{floor}.json'} に label='{BOUNDARY_LABEL}' が"
        "ない (labelmeで敷地境界線を注釈すること)")


class BoundaryLookup:
    """図面メートル座標の点 → 敷地内判定 (境界からmargin以上内側なら True)

    ポリゴンを CELL_M ラスタに描き、内側距離場と比較する
    (collision_mesh.structure_dist_raster と同じ方式)
    """

    def __init__(self, floor, mm_per_px, margin_m=MARGIN_M):
        poly_m = load_boundary_px(floor) * mm_per_px / 1000.0
        pad = margin_m + 1.0
        self.x0 = poly_m[:, 0].min() - pad
        self.y0 = poly_m[:, 1].min() - pad
        nx = int(np.ceil((poly_m[:, 0].max() + pad - self.x0) / CELL_M))
        ny = int(np.ceil((poly_m[:, 1].max() + pad - self.y0) / CELL_M))
        img = Image.new("L", (nx, ny), 0)
        cells = [((x - self.x0) / CELL_M, (y - self.y0) / CELL_M)
                 for x, y in poly_m]
        ImageDraw.Draw(img).polygon(cells, fill=255)
        inside = np.asarray(img) > 0
        dist_in = ndimage.distance_transform_edt(inside) * CELL_M
        self.ok = dist_in > margin_m
        self.area_m2 = float(inside.sum()) * CELL_M ** 2

    def __call__(self, xy_m):
        """(N,2) 図面メートル → (N,) bool。ラスタ外は False"""
        col = ((xy_m[:, 0] - self.x0) / CELL_M).astype(np.int64)
        row = ((xy_m[:, 1] - self.y0) / CELL_M).astype(np.int64)
        h, w = self.ok.shape
        inb = (col >= 0) & (col < w) & (row >= 0) & (row < h)
        res = np.zeros(len(xy_m), dtype=bool)
        res[inb] = self.ok[row[inb], col[inb]]
        return res


# ---------------------------------------------------------------- マスク生成

def equirect_dirs(w, h):
    """equirect画素中心の方向ベクトル (h,w,3)。カメラ系 x右/y下/z前
    (gs_prep.equirect_lut の逆変換)"""
    col, row = np.meshgrid(np.arange(w), np.arange(h))
    lon = ((col + 0.5) / w - 0.5) * 2 * np.pi
    lat = (0.5 - (row + 0.5) / h) * np.pi
    return np.stack([np.sin(lon) * np.cos(lat), -np.sin(lat),
                     np.cos(lon) * np.cos(lat)], axis=-1)


def _inside_test(d_m, dirs, fr, lookup):
    """深度マップの有効画素を図面座標に投影し (valid, inside) を返す"""
    from img_align.mesh.meshing import apply_matrix
    h, w = d_m.shape
    pts_cam = dirs[:h, :w] * d_m[..., None]
    p_isl = pts_cam.reshape(-1, 3) @ fr["R"].T + fr["t_m"]
    p_draw = apply_matrix(fr["M"], p_isl)
    return (d_m > 0).reshape(-1), lookup(p_draw[:, :2])


def frame_white_mask(fr, dirs, lookup):
    """1フレームの低解像度 白=敷地内 bool マスク。深度が読めなければ None"""
    from img_align.mesh.meshing import DEPTH_MIN_M, clean_equirect_depth
    p = fr["chunk_dir"] / "panovggt_out" / "depth" / f"{fr['name']}_depth_raw.png"
    if not p.exists():
        return None
    raw = np.asarray(Image.open(p)).astype(np.float32)
    d_m = raw / 1000.0 * fr["depth_scale"]
    d_m[(d_m < DEPTH_MIN_M) | (d_m > D_MAX_M)] = 0.0
    d_clean = clean_equirect_depth(d_m)
    if INVALID_DEPTH_WHITE:
        # 深度なし画素は白 (=敷地内扱い)。黒になるのは「深度が取れて、その表面が
        # 敷地外」と確信できた画素のみ
        valid, inside = _inside_test(d_clean, dirs, fr, lookup)
        white = ~valid | inside
        return white.reshape(d_m.shape)
    # 判定に使う深度: クリーニング後を基本に、以下を実深度で復元する
    #  - クリーニングで捨てられた画素 (部材エッジの深度不連続): 飛び画素の深度は
    #    前景と背景の中間なので、室内ならどちらも敷地内=白
    #  - 近すぎ(<DEPTH_MIN_M)クリップ (カメラ直近の柱・枠): カメラ位置≒敷地内
    # 遠すぎ(>D_MAX_M)は復元しない (黒のまま=方針)
    d_full = raw / 1000.0 * fr["depth_scale"]
    retest = ((d_m > 0) & (d_clean == 0)) \
        | ((d_full > 0) & (d_full < DEPTH_MIN_M))
    d_used = np.where(retest, d_full, d_clean)
    valid, inside = _inside_test(d_used, dirs, fr, lookup)
    white = valid & inside
    if SEE_THROUGH_M > 0:
        # レイ透過テスト: 表面から視線方向に SEE_THROUGH_M 延長した点が敷地外に
        # 出るなら、敷地際の薄物 (足場ネット等) 越しに外が見えている疑い → 黒。
        # 建物は境界から数mセットバックしているので屋内の壁面は影響を受けない
        _, inside_ext = _inside_test(
            np.where(d_used > 0, d_used + SEE_THROUGH_M, 0.0),
            dirs, fr, lookup)
        white &= inside_ext
    h, w = d_m.shape
    white = white.reshape(h, w)
    if NADIR_WHITE:
        # 下端の自撮り棒帯 (PanoVGGT入力でマスク済み=深度なし) はカメラ直下の
        # 敷地内地面なので、カメラ位置が敷地内なら白にする
        from img_align.mesh.meshing import apply_matrix
        from img_align.core.panovggt import NADIR_INVALID_FRAC
        cam_xy = apply_matrix(fr["M"], fr["t_m"][None])[:, :2]
        if lookup(cam_xy)[0]:
            white[int(h * (1 - NADIR_INVALID_FRAC)) - 1:, :] = True
    return white


def upsample_and_erode(white_low):
    """低解像度bool → フル解像度 0/255 uint8。白を ERODE_PX 収縮 (黒側へ膨張)。

    先にクロージングで白領域内の小黒斑を埋める: 深度不連続除去 (clean_equirect_depth)
    で梁・枠エッジの画素が深度なし→黒になるが、周囲が敷地内なら実質敷地内。
    大きな黒領域 (窓越し屋外・遠方) は埋まらず残る"""
    if SPECKLE_CLOSE_ITER > 0:
        # binary_closing 相当だが収縮側は border_value=1 で画像縁の白を保護する
        # (既定の境界=黒だと縁が iterations px 黒く食われ、拡大+白収縮で
        # ~27px の黒枠になる)
        s = np.ones((3, 3), bool)
        grown = ndimage.binary_dilation(white_low, s,
                                        iterations=SPECKLE_CLOSE_ITER)
        white_low = ndimage.binary_erosion(grown, s,
                                           iterations=SPECKLE_CLOSE_ITER,
                                           border_value=1)
    img = Image.fromarray((white_low * 255).astype(np.uint8))
    full = np.asarray(img.resize((FULL_W, FULL_H), Image.NEAREST))
    eroded = ndimage.minimum_filter(full, size=2 * ERODE_PX + 1)
    return np.where(eroded > 127, 255, 0).astype(np.uint8)


def all_frame_numbers():
    return sorted(int(p.stem.replace("frame_number_", ""))
                  for p in PROP.frames_dir.glob("frame_number_*.jpg"))


def colmap_fallback_poses(missing_frs):
    """チャンクポーズがCOLMAP吸着外れ (resolve_framesでスキップ) のフレームを、
    COLMAPポーズ直用で救済する。COLMAPは全216フレーム登録済みかつ大域BA済みで、
    健全フレームでのチャンクfit回転との差は0.4-2.2°と実測確認済み。
    depth_scale はチャンクfitのスケール (チャンク単位で頑健) を使う"""
    from img_align.postproc.colmap_prior import colmap_centers, fit_chunk_to_colmap
    from img_align.core.plan_grid import PlanGrid
    from img_align.mesh.meshing import COLMAP_MODEL, drawing_matrix, island_scale, islands_expanded, load_anchor, load_chunk_light
    centers, Rw2c = colmap_centers(COLMAP_MODEL[SESSION])
    out = {}
    for isl in islands_expanded(SESSION):
        anchor = load_anchor(isl)
        grid = PlanGrid(isl["floor"])
        k = island_scale(anchor, grid)
        M, _ = drawing_matrix(anchor, grid, isl["z_off"], metric_input=True)
        root = RESULTS_DIR / isl["root"]
        for kk in isl["chunks"]:
            cdir = root / f"chunk_{kk:02d}"
            names, pos, _, _ = load_chunk_light(cdir)
            nums = [int(n.replace("frame_", "")) for n in names]
            hit = [f for f in missing_frs
                   if f in nums and f in centers and f not in out]
            if not hit:
                continue
            T, _, _ = fit_chunk_to_colmap(names, pos, centers)
            if T is None:
                continue
            for f in hit:
                out[f] = {"chunk_dir": cdir, "name": f"frame_{f:06d}",
                          "isl": isl, "M": M, "fr": f,
                          "R": Rw2c[f].T, "t_m": centers[f] * k,
                          "depth_scale": T[0] * k}
    return out


def cmd_generate(args):
    from img_align.eval.gt_traj import MM_PER_PX
    from img_align.gs.gs_world import frame_poses_drawing
    assert MM_PER_PX is not None, "図面GT (1F.json) が未注釈: スケールを導出できない"
    MASK_OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    poses = {p["fr"]: p for p in frame_poses_drawing()}
    missing = [f for f in all_frame_numbers() if f not in poses]
    if missing:
        fb = colmap_fallback_poses(missing)
        print(f"COLMAPフォールバック: {sorted(fb)} (救済不可: "
              f"{[f for f in missing if f not in fb]})")
        poses.update(fb)
    lookups = {}
    dirs = None
    frames = all_frame_numbers()
    if args.frames:
        a, b = (int(v) for v in args.frames.split(":"))
        frames = [f for f in frames if a <= f < b]

    meta = {"session": SESSION, "margin_m": MARGIN_M, "d_max_m": D_MAX_M,
            "erode_px": ERODE_PX, "cell_m": CELL_M,
            "mm_per_px": MM_PER_PX, "frames": {}, "skipped": []}
    for i, num in enumerate(frames):
        name = f"frame_number_{num:06d}"
        fr = poses.get(num)
        white_low = None
        if fr is not None:
            floor = fr["isl"]["floor"]
            if floor not in lookups:
                lookups[floor] = BoundaryLookup(floor, MM_PER_PX)
                print(f"boundary[{floor}]: {lookups[floor].area_m2:.0f} m^2")
            if dirs is None:
                p0 = (fr["chunk_dir"] / "panovggt_out" / "depth"
                      / f"{fr['name']}_depth_raw.png")
                h0, w0 = np.asarray(Image.open(p0)).shape
                dirs = equirect_dirs(w0, h0)
            white_low = frame_white_mask(fr, dirs, lookups[floor])
        if white_low is None:
            mask = np.zeros((FULL_H, FULL_W), np.uint8)
            meta["skipped"].append(num)
        else:
            mask = upsample_and_erode(white_low)
        Image.fromarray(mask).save(MASK_OUT_DIR / f"{name}.png")
        wf = float((mask == 255).mean())
        meta["frames"][num] = round(wf, 4)
        if i % 20 == 0 or white_low is None:
            note = " (ポーズなし→全黒)" if white_low is None else ""
            print(f"{name}: white={wf:.2%}{note}")

    (OUT_DIR / "run_meta.json").write_text(json.dumps(meta, indent=1))
    n_black = len(meta["skipped"])
    print(f"done: {len(frames)}枚 -> {MASK_OUT_DIR} (全黒 {n_black}枚: "
          f"{meta['skipped'][:20]}{'...' if n_black > 20 else ''})")


# ---------------------------------------------------------------- アンカー

def cmd_anchor(args):
    """島ごとに図面アンカーJSONを生成 (GT軌跡なし物件用の薄いドライバ)"""
    from img_align.config import OUT_DIR as BK_DIR
    from img_align.postproc.anchor import anchor_island
    from img_align.mesh.meshing import islands_for
    for isl in islands_for(SESSION):
        if args.tag and isl["tag"] != args.tag:
            continue
        print(f"=== {isl['tag']} ({isl['floor']})")
        # align_tool の区間微調整 ("segments") は自動アンカーの上書きで
        # 消えるため退避→復元する (セグメントアンカーファイル自体は残る)
        seg_bak = {}
        for name in (isl["anchor"], f"anchor_{isl['tag']}.json"):
            p = BK_DIR / name
            if p.exists():
                segs = json.loads(p.read_text()).get("segments")
                if segs:
                    seg_bak[p] = segs
        thetas = None
        if args.near_theta is not None:
            c = int(round(args.near_theta))
            thetas = [t % 360 for t in range(c - 10, c + 11, 2)]
        _, resid, _ = anchor_island(isl["tag"], isl["floor"], gt=None,
                                    use_openings=args.openings,
                                    allow_mirror=not args.no_mirror,
                                    thetas=thetas)
        for p, segs in seg_bak.items():
            if not p.exists():
                continue
            a = json.loads(p.read_text())
            if a.get("segments") != segs:
                a["segments"] = segs
                p.write_text(json.dumps(a, indent=1))
                print(f"    [警告] {p.name} の区間微調整(segments)を引き継いだ"
                      " — 自動アンカーと区間変換が不整合の可能性。align_tool"
                      "で要確認")
        print(f"    wall_resid_m={resid:.3f}"
              + ("  [警告: >0.15m — corners.py 等の代替を検討]"
                 if resid > 0.15 else ""))


WALK_OFF_M = 0.7   # 外周歩行時の仮囲いからの想定距離 (境界リングの内側収縮量)


def _ring_points(poly, step):
    """閉ポリゴンの辺を step 間隔で密サンプル"""
    pts = []
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        n = max(int(np.linalg.norm(b - a) / step), 1)
        for t in np.arange(n) / n:
            pts.append(a + (b - a) * t)
    return np.asarray(pts)


def _offset_convex(poly, off):
    """凸ポリゴンを off だけ内側へオフセット (辺を法線方向に平行移動して交点)"""
    n = len(poly)
    # 符号付き面積で回り方向を判定し、内側法線を決める
    area = sum(np.cross(poly[i], poly[(i + 1) % n]) for i in range(n))
    sgn = 1.0 if area > 0 else -1.0
    lines = []
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        d = (b - a) / np.linalg.norm(b - a)
        nrm = sgn * np.array([-d[1], d[0]])   # 内側法線
        lines.append((a + nrm * off, d))
    out = []
    for i in range(n):
        (p1, d1), (p2, d2) = lines[i - 1], lines[i]
        A = np.stack([d1, -d2], axis=1)
        t = np.linalg.solve(A, p2 - p1)
        out.append(p1 + d1 * t[0])
    return np.asarray(out)


def cmd_anchor_traj(args):
    """壁が実在しないセッション (基礎工事段階など) 用のアンカー:
    カメラ軌跡の外周リング (凸包) を、敷地境界ポリゴンを歩行オフセット分
    内側へ収縮したリングへ相似フィットする。4回転x鏡映の8仮説を
    リングICPで精密化し、残差最小を採用してアンカーJSONを書き出す"""
    from scipy.spatial import ConvexHull

    from img_align.config import OUT_DIR
    from img_align.postproc.anchor import prepare_island
    from img_align.core.plan_grid import PlanGrid
    from img_align.postproc.plan_icp import icp_to_walls, transform_pts
    tag, floor = args.tag, args.floor
    d = prepare_island(tag)
    grid = PlanGrid(floor)
    cam_rc0 = np.stack(
        [(d["cam_pos_aligned"][:, 2] - d["origin"][1]) / d["cell_units"],
         (d["cam_pos_aligned"][:, 0] - d["origin"][0]) / d["cell_units"]],
        axis=1)

    # ターゲット: 境界ポリゴン (grid cell座標) を WALK_OFF_M 内側へ収縮したリング
    from img_align.eval.gt_traj import MM_PER_PX
    poly_px = load_boundary_px(floor)
    poly_cell = np.stack([(poly_px[:, 1] - grid.y0_px) / grid.cell_px,
                          (poly_px[:, 0] - grid.x0_px) / grid.cell_px], axis=1)
    m_per_cell = grid.cell_px * MM_PER_PX / 1000
    ring_tgt = _offset_convex(poly_cell, WALK_OFF_M / m_per_cell)
    segs = np.concatenate([ring_tgt, np.roll(ring_tgt, -1, axis=0)], axis=1)

    # ターゲット矩形の主軸
    qt = ring_tgt - ring_tgt.mean(0)
    _, _, vt_t = np.linalg.svd(qt, full_matrices=False)
    th_t = math.atan2(vt_t[0, 1], vt_t[0, 0])
    pt = qt @ vt_t.T
    ext_t = np.array([np.ptp(pt[:, 0]), np.ptp(pt[:, 1])])

    # リングは対称性が高く 0/180° と鏡映が残差で同点になりがちなので、
    # --rot / --mirror で仮説を固定できる (軌跡図の開始点が道路側かで目視判定)
    rot_force = None if args.rot == "auto" else int(args.rot) // 90
    mir_force = (None if args.mirror == "auto"
                 else args.mirror in ("1", "true", "True"))
    best = None
    for mirror in (False, True):
        if mir_force is not None and mirror != mir_force:
            continue
        cam_rc = cam_rc0.copy()
        mirror_col = None
        if mirror:
            mirror_col = d["density"].shape[1] - 1
            cam_rc[:, 1] = mirror_col - cam_rc[:, 1]
        hull = ConvexHull(cam_rc)
        ring_src = _ring_points(cam_rc[hull.vertices], step=1.0)
        qs = ring_src - ring_src.mean(0)
        _, _, vt_s = np.linalg.svd(qs, full_matrices=False)
        th_s = math.atan2(vt_s[0, 1], vt_s[0, 0])
        ps = qs @ vt_s.T
        ext_s = np.array([np.ptp(ps[:, 0]), np.ptp(ps[:, 1])])
        for k in range(4):
            if rot_force is not None and k != rot_force:
                continue
            th0 = th_t - th_s + k * math.pi / 2
            e = ext_s if k % 2 == 0 else ext_s[::-1]
            s0 = float(np.mean(ext_t / e))
            R = np.array([[math.cos(th0), -math.sin(th0)],
                          [math.sin(th0), math.cos(th0)]])
            t0 = ring_tgt.mean(0) - s0 * (R @ ring_src.mean(0))
            s1, th1, t1, resid = icp_to_walls(ring_src, segs, s0, th0, t0)
            if best is None or resid < best["resid"]:
                best = dict(s=s1, th=th1, t=t1, resid=resid,
                            mirror_col=mirror_col, k=k)
            print(f"  mirror={mirror} rot{k*90:3d}: resid {resid:.3f}m")

    print(f"{tag}: best rot{best['k']*90} mirror={best['mirror_col'] is not None} "
          f"resid {best['resid']:.3f}m")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"anchor_{tag}.json").write_text(json.dumps({
        "tag": tag, "floor": floor,
        "R_align": d["R_align"].tolist(),
        "floor_y_units": d["floor_y_units"],
        "floor_phys_y_units": d["floor_phys_y_units"],
        "m_per_unit": d["m_per_unit"],
        "origin": d["origin"].tolist(),
        "cell_units": float(d["cell_units"]),
        "mirror_col": best["mirror_col"],
        "s": best["s"], "th": best["th"],
        "t": list(map(float, best["t"])),
        "wall_resid_m": best["resid"]}, indent=1))
    print(f"-> {OUT_DIR / f'anchor_{tag}.json'}")


def cmd_anchor_transfer(args):
    """基準島のアンカーを同一COLMAPゲージの別島へ転写し、壁ICPで微調整する。

    全島が colmap_prior の共通ワールドに載っている場合、R_align(重力)と
    z反転はほぼ共通なので、島間の違いは origin/cell_units の平行移動・
    スケールのみ。基準島の相似変換 (s,th,t) を合成し直して転写できる。
    (単独グリッド照合は足場ノイズ等で階ごとに偽解を出しやすい)"""
    from img_align.config import OUT_DIR
    from img_align.postproc.anchor import prepare_island
    from img_align.eval.gt_traj import MM_PER_PX
    from img_align.core.plan_grid import PlanGrid
    from img_align.postproc.plan_icp import icp_to_walls, segs_to_cells, transform_pts, wall_segments_px
    from img_align.config import BENCH_DIR
    src = json.loads((OUT_DIR / f"anchor_{args.src}.json").read_text())
    assert src["mirror_col"] is None, "mirror付き基準は未対応"
    sA, thA = src["s"], src["th"]

    def to_cell(w, Ra, o, cu):
        p = w @ np.asarray(Ra).T
        p = p.copy()
        p[:, 2] *= -1
        return np.stack([(p[:, 2] - o[1]) / cu, (p[:, 0] - o[0]) / cu], axis=1)

    for spec in args.to:
        tag, floor = spec.split(":")
        d = prepare_island(tag)
        # 共通COLMAPワールド経由の厳密転写: 対象島のカメラ位置を
        # (a) 基準島の整列系→図面セル、(b) 対象島の整列系セル に写し、
        # (b)→(a) の2D相似を最小二乗で解く (島ごとの重力軸ヨー差も吸収)
        W = np.load(BENCH_DIR / tag / "global_cameras.npz")["positions"]
        gA = transform_pts(
            to_cell(W, src["R_align"], src["origin"], src["cell_units"]),
            sA, thA, np.array(src["t"]))
        cB = to_cell(W, d["R_align"], d["origin"], d["cell_units"])
        muB, muA = cB.mean(0), gA.mean(0)
        X, Y = cB - muB, gA - muA
        # 2D相似 (反射なし): 複素数最小二乗 y = a*x
        zx = X[:, 0] + 1j * X[:, 1]
        zy = Y[:, 0] + 1j * Y[:, 1]
        a = (np.conj(zx) @ zy) / (np.conj(zx) @ zx)
        s_B = float(abs(a))
        thB = float(np.angle(a))
        Rm = np.array([[math.cos(thB), -math.sin(thB)],
                       [math.sin(thB), math.cos(thB)]])
        t_B = muA - s_B * (Rm @ muB)
        fit_err = float(np.median(np.linalg.norm(
            transform_pts(cB, s_B, thB, t_B) - gA, axis=1)))
        print(f"  {tag}: world-fit err {fit_err:.2f}cell "
              f"(dth vs src {math.degrees(thB-thA):+.1f}°)")
        grid = PlanGrid(floor)
        zz, xx = np.nonzero(d["density"] > 0.15)
        pts = np.stack([zz, xx], axis=1).astype(np.float64)
        if len(pts) > 20000:
            pts = pts[:: len(pts) // 20000 + 1]
        segs = segs_to_cells(wall_segments_px(floor), grid)
        s2, th2, t2, resid = icp_to_walls(pts, segs, s_B, thB, t_B)
        # 対応の取れない点群で相似ICPがスケール縮退することがある。
        # 転写スケールはゲージ不変 (s/cu保存) で信頼できるため、大きく
        # 動いたら回転・スケール固定の並進のみICPにフォールバックする
        # (島ごとの重力軸の微差で origin 変換に平行移動誤差が乗るため、
        # 並進だけは推定し直す価値がある)
        if not (0.85 < s2 / s_B < 1.18):
            print(f"{tag}: ICPスケール逸脱 (x{s2/s_B:.2f}) → 並進のみICP")
            s2, th2, t2, resid = s_B, thB, t_B.copy(), float("nan")
            a, b = segs[:, :2], segs[:, 2:]
            ab = b - a
            ab2 = (ab ** 2).sum(1).clip(min=1e-9)
            for it in range(6):
                P = transform_pts(pts, s2, th2, t2)
                tt = np.clip(((P[:, None] - a) * ab).sum(-1) / ab2, 0, 1)
                fp = a + tt[..., None] * ab
                dv = fp - P[:, None]
                dd = np.linalg.norm(dv, axis=-1)
                j = dd.argmin(1)
                dmin = dd[np.arange(len(P)), j]
                cut = np.percentile(dmin, 60)
                sel = dmin <= cut
                t2 = t2 + np.median(dv[np.arange(len(P)), j][sel], axis=0)
            m_per_cell = grid.cell_px * MM_PER_PX / 1000
            resid = float(np.median(dmin[sel])) * m_per_cell
        print(f"{tag} ({floor}): 転写 th={math.degrees(thB)%360:.0f}° "
              f"→ ICP resid {resid:.3f}m (dth {math.degrees(th2-thB):+.1f}°)")
        (OUT_DIR / f"anchor_{tag}.json").write_text(json.dumps({
            "tag": tag, "floor": floor,
            "R_align": d["R_align"].tolist(),
            "floor_y_units": d["floor_y_units"],
            "floor_phys_y_units": d["floor_phys_y_units"],
            "m_per_unit": d["m_per_unit"],
            "origin": d["origin"].tolist(),
            "cell_units": float(d["cell_units"]),
            "mirror_col": None,
            "s": s2, "th": th2, "t": list(map(float, t2)),
            "wall_resid_m": resid}, indent=1))


# ---------------------------------------------------------------- 可視化

def cmd_viz(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from img_align.eval.gt_traj import MM_PER_PX
    from img_align.gs.gs_world import frame_poses_drawing
    from img_align.core.plan_grid import ZUMEN_DIR
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    poses = sorted(frame_poses_drawing(), key=lambda p: p["fr"])
    meta_p = OUT_DIR / "run_meta.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else None

    # (a) 図面トップダウン: 境界ポリゴン + 軌跡 (フレーム番号で色付け)
    floors = sorted({p["isl"]["floor"] for p in poses})
    fig, axes = plt.subplots(1, len(floors), figsize=(11 * len(floors), 8))
    axes = np.atleast_1d(axes)
    for ax, floor in zip(axes, floors):
        bg = Image.open(ZUMEN_DIR / f"{floor}.jpg").convert("L")
        ext = (0, bg.width * MM_PER_PX / 1000, bg.height * MM_PER_PX / 1000, 0)
        ax.imshow(bg, cmap="gray", extent=ext, alpha=0.6)
        poly = load_boundary_px(floor) * MM_PER_PX / 1000
        ax.plot(np.append(poly[:, 0], poly[0, 0]),
                np.append(poly[:, 1], poly[0, 1]), "g-", lw=2, label="boundary")
        pf = [p for p in poses if p["isl"]["floor"] == floor]
        if pf:
            xy = np.array([p["pos_draw"][:2] for p in pf])
            sc = ax.scatter(xy[:, 0], xy[:, 1], c=[p["fr"] for p in pf],
                            cmap="viridis", s=8)
            fig.colorbar(sc, ax=ax, label="frame")
        ax.set_title(f"{floor} ({len(pf)} frames)")
        ax.legend()
    fig.savefig(OUT_DIR / "trajectory_boundary.png", dpi=110,
                bbox_inches="tight")
    print(f"-> {OUT_DIR / 'trajectory_boundary.png'}")

    # (b) サンプルフレームのオーバーレイ (黒領域を暗く)
    have = {p["fr"] for p in poses}
    default = [f for f in (0, 5, 10, 15, 30, 60, 90, 120, 150, 180, 200, 215)]
    sel = [int(v) for v in args.samples.split(",")] if args.samples else default
    for num in sel:
        mp = MASK_OUT_DIR / f"frame_number_{num:06d}.png"
        ip = PROP.frames_dir / f"frame_number_{num:06d}.jpg"
        if not (mp.exists() and ip.exists()):
            continue
        img = np.asarray(Image.open(ip).resize((1344, 672))).astype(np.float32)
        m = np.asarray(Image.open(mp).resize((1344, 672), Image.NEAREST))
        ov = img * np.where(m[..., None] > 127, 1.0, 0.25)
        tag = "" if num in have else " (no pose)"
        Image.fromarray(ov.astype(np.uint8)).save(
            OUT_DIR / f"overlay_{num:06d}.jpg", quality=90)
        print(f"overlay_{num:06d}.jpg{tag}")

    # (c) フレーム毎白率
    if meta:
        frs = sorted(int(k) for k in meta["frames"])
        wf = [meta["frames"][str(f)] for f in frs]
        fig, ax = plt.subplots(figsize=(11, 3))
        ax.plot(frs, wf, ".-", ms=3)
        ax.axhline(0.9, color="r", ls="--", lw=0.8, label="0.9 (漏れ疑い)")
        ax.set_xlabel("frame")
        ax.set_ylabel("white frac")
        ax.legend()
        fig.savefig(OUT_DIR / "white_fraction.png", dpi=110,
                    bbox_inches="tight")
        print(f"-> {OUT_DIR / 'white_fraction.png'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("anchor", help="島ごとの図面アンカーJSON生成")
    p.add_argument("--tag", default=None, help="特定の島タグのみ")
    p.add_argument("--openings", action="store_true", help="開口ICPも使う")
    p.add_argument("--no-mirror", action="store_true",
                   help="鏡映仮説を禁止 (z反転規約では非鏡映が正。偽解対策)")
    p.add_argument("--near-theta", type=float, default=None,
                   help="回転探索をこの角度±10°に拘束 (基準階の向きに合わせる)")
    p = sub.add_parser("anchor-traj",
                       help="軌跡外周↔境界リングのフィット (壁なしセッション用)")
    p.add_argument("--tag", required=True, help="frontend_bench の島タグ")
    p.add_argument("--floor", default="1F")
    p.add_argument("--rot", default="auto", choices=["auto", "0", "90", "180", "270"],
                   help="回転仮説の固定 (対称タイの手動解決用)")
    p.add_argument("--mirror", default="auto", choices=["auto", "0", "1"],
                   help="鏡映仮説の固定")
    p = sub.add_parser("anchor-transfer",
                       help="基準島のアンカーを他島へ転写+壁ICP微調整")
    p.add_argument("--src", required=True, help="基準島タグ (アンカー既存)")
    p.add_argument("--to", nargs="+", required=True, help="'タグ:階' の列")
    p = sub.add_parser("generate", help="全フレームのマスクPNG生成")
    p.add_argument("--frames", default=None, help="'a:b' フレーム番号範囲")
    p = sub.add_parser("viz", help="軌跡図とサンプルオーバーレイ")
    p.add_argument("--samples", default=None, help="カンマ区切りフレーム番号")
    args = ap.parse_args()
    {"anchor": cmd_anchor, "anchor-traj": cmd_anchor_traj,
     "anchor-transfer": cmd_anchor_transfer,
     "generate": cmd_generate, "viz": cmd_viz}[args.cmd](args)


if __name__ == "__main__":
    main()
