"""MapAnything軌跡フロントエンド: equirect由来クロップの一括推論 → frontend_bench

MapAnything (facebookresearch/map-anything, Apacheモデル) は全ビュー一括の
フィードフォワード推論でメトリックスケールのポーズ・点群を返す。
チャンク連結 (chain_chunks) が不要になる代わりにビュー数がGPUメモリで制限される
(16GBで目安100〜200ビュー) ため、フロア別 (--floor) やstride間引きで回す。

入力クロップは vidmap_prep (FOV130/pitch10/518px, tag mapany_f130) を流用し、
内参が厳密に既知なので intrinsics を条件付けする (MapAnythingの精度が上がる)。
出力は chain_chunks と同じ frontend_bench 契約なので eval_frontend がそのまま使える。
ポーズはクロップ回転 (yaw,pitch) の逆を掛けて元equirectリグ姿勢に直して保存する。

使い方 (mapanythingコンテナ内):
  python main.py tool mapany --input /data/x infer --tag mapany0616_1F --floor 1F
  python main.py tool mapany --input /data/x infer --tag mapany0428_ALL --stride 2
  python main.py tool mapany --input /data/x baseline --tag cmbase0616_1F --floor 1F
"""
import argparse
import json
import math

import numpy as np

from recon3d.core.io_ply import write_ply

CROP_TAG = "mapany_f130"
FOV_DEG = 130.0
PITCH_DEG = 10.0
YAW_DEG = 0.0
PLY_STRIDE = 6          # 点群出力の画素間引き


def crop2rig(yaw_deg=YAW_DEG, pitch_deg=PITCH_DEG):
    """クロップカメラ→equirectリグの回転 (OpenCV系: x右/y下/z前)

    py360convert.e2p(u_deg=yaw, v_deg=pitch) は方位yaw・仰角pitchを向く
    ロール無しの透視カメラ: R_crop2rig = Ry(yaw) @ Rx(pitch)。
    """
    u, v = math.radians(yaw_deg), math.radians(pitch_deg)
    ry = np.array([[math.cos(u), 0, math.sin(u)],
                   [0, 1, 0],
                   [-math.sin(u), 0, math.cos(u)]])
    rx = np.array([[1, 0, 0],
                   [0, math.cos(v), -math.sin(v)],
                   [0, math.sin(v), math.cos(v)]])
    return ry @ rx


def select_frames(paths, floor=None, stride=1, frange=None):
    """対象フレーム番号列。--floor指定時はGTの階割当で絞る (階段区間を除外)。
    frange=(a,b) でフレーム範囲も絞る (同一階の複数訪問区間の分離用)

    GTの出どころは物件で違う (28-BBBBB=手動登録CSV / 28-AAAAA=メタCSV) ので
    直接CSVを開かず load_gt に任せる。"""
    name_map = json.loads(
        (paths.results / "vidmap" / f"name_map_{CROP_TAG}.json").read_text())
    fr2file = {fr: name for name, fr in name_map.items()}
    frames = sorted(fr2file)
    if floor:
        from recon3d.eval.gt_traj import load_gt
        keep = {f for f, v in load_gt(paths).items() if v["floor"] == floor}
        frames = [fr for fr in frames if fr in keep]
    if frange:
        frames = [fr for fr in frames if frange[0] <= fr <= frange[1]]
    return frames[::stride], fr2file




def save_bench(paths, tag, names, positions, poses, pts, rgb, log):
    out = paths.bench / tag
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "global_cameras.npz",
                        names=np.array(names), positions=positions, poses=poses)
    if pts is not None and len(pts):
        write_ply(out / "global_points.ply", pts, rgb)
    (out / "chain_log.json").write_text(json.dumps([log], indent=1))
    print(f"-> {out} ({len(names)} frames"
          + (f", {len(pts):,} pts" if pts is not None else "") + ")")


def run_infer(paths, args):
    import torch
    from PIL import Image
    from mapanything.models import MapAnything
    from mapanything.utils.image import preprocess_inputs

    frames, fr2file = select_frames(paths, args.floor, args.stride, _frange(args))
    yaws = [float(y) for y in args.yaws.split(",")]
    crop_dirs = {y: paths.results / "vidmap"
                 / (CROP_TAG if y == 0 else f"{CROP_TAG}_y{int(y)}")
                 for y in yaws}
    print(f"{len(frames)} frames x {len(yaws)} yaws "
          f"(floor={args.floor}, stride={args.stride}, res={args.res})")

    # 案1: ポーズ条件付け (既存frontend_benchタグのリグ姿勢をクロップ姿勢に直して入力)
    cond_pose = {}
    if args.pose_cond:
        d = np.load(paths.bench / args.pose_cond / "global_cameras.npz")
        cond_pose = {str(n): p for n, p in zip(d["names"], d["poses"])}
        n_before = len(frames)
        frames = [fr for fr in frames if f"frame_{fr:06d}" in cond_pose]
        print(f"pose-cond {args.pose_cond}: {len(frames)}/{n_before} frames")

    views, meta = [], []           # meta: (frame, yaw) をviewsと同順で
    for fr in frames:
        for y in yaws:
            im = Image.open(crop_dirs[y] / fr2file[fr])
            if args.res and args.res != im.size[0]:
                im = im.resize((args.res, args.res), Image.LANCZOS)
            img = np.asarray(im)
            s = img.shape[0]
            f = (s / 2) / math.tan(math.radians(FOV_DEG) / 2)
            k = np.array([[f, 0, s / 2], [0, f, s / 2], [0, 0, 1]], np.float32)
            v = {"img": torch.from_numpy(img.copy()),
                 "intrinsics": torch.from_numpy(k)}
            if cond_pose:
                g = cond_pose[f"frame_{fr:06d}"].copy()
                g[:3, :3] = g[:3, :3] @ crop2rig(y)            # リグ→クロップ姿勢
                v["camera_poses"] = torch.from_numpy(g.astype(np.float32))
                # チェーン座標はメートルでない (チャンク単位) ため非メトリック扱い
                v["is_metric_scale"] = torch.tensor([False])
            views.append(v)
            meta.append((fr, y))
    # 既定のfixed_mappingは518へ戻してしまうため、--res時はsquare指定で維持する
    views = (preprocess_inputs(views, resize_mode="square", size=args.res)
             if args.res else preprocess_inputs(views))

    model_name = ("facebook/map-anything" if args.nc
                  else "facebook/map-anything-apache")
    model = MapAnything.from_pretrained(model_name).to("cuda")
    preds = model.infer(views, memory_efficient_inference=True,
                        minibatch_size=args.minibatch,
                        use_amp=True, amp_dtype="bf16",
                        apply_mask=True, mask_edges=True)

    names, positions, poses = [], [], []
    all_pts, all_rgb = [], []
    local = {}                     # 案2: frame別のリグ座標点群 (メートル)
    for (fr, y), pred in zip(meta, preds):
        p44 = pred["camera_poses"][0].float().cpu().numpy()   # crop cam2world
        if y == yaws[0]:           # ポーズ・位置は第1方位のビューから取る
            g = np.eye(4)
            g[:3, :3] = p44[:3, :3] @ crop2rig(y).T            # リグ姿勢へ
            g[:3, 3] = p44[:3, 3]
            names.append(f"frame_{fr:06d}")
            positions.append(g[:3, 3])
            poses.append(g)
        pts = pred["pts3d"][0].float().cpu().numpy()[::PLY_STRIDE, ::PLY_STRIDE]
        msk = pred["mask"][0].cpu().numpy()[::PLY_STRIDE, ::PLY_STRIDE]
        col = pred["img_no_norm"][0].float().cpu().numpy()[
            ::PLY_STRIDE, ::PLY_STRIDE]
        m = msk.reshape(-1, msk.shape[-1])[:, 0] > 0
        pts_w = pts.reshape(-1, 3)[m]
        rgb = (col.reshape(-1, 3)[m] * 255).clip(0, 255)
        all_pts.append(pts_w)
        all_rgb.append(rgb)
        if args.dump_local:        # world→crop cam→リグ座標 (深度はメートルのまま)
            p_cam = (pts_w - p44[:3, 3]) @ p44[:3, :3]
            p_rig = p_cam @ crop2rig(y).T
            key = f"frame_{fr:06d}"
            local.setdefault(key, []).append(
                np.concatenate([p_rig, rgb], axis=1).astype(np.float32))

    log = {"mode": "mapanything", "model": model_name, "crop_tag": CROP_TAG,
           "fov": FOV_DEG, "pitch": PITCH_DEG, "floor": args.floor,
           "stride": args.stride, "yaws": yaws, "res": args.res,
           "pose_cond": args.pose_cond, "n_views": len(views),
           "unit": "metric_m"}
    save_bench(paths, args.tag, names, np.stack(positions), np.stack(poses),
               np.concatenate(all_pts), np.concatenate(all_rgb), log)
    if args.dump_local:
        out = paths.bench / args.tag / "local_points.npz"
        np.savez_compressed(out, **{k: np.concatenate(v)
                                    for k, v in local.items()})
        print(f"-> {out} ({len(local)} frames, rig-frame metric xyz+rgb)")


def run_baseline(paths, args):
    """社内COLMAP coarse (equirect) の軌跡を同じ契約で書き出す (比較用)"""
    from recon3d.postproc.colmap_prior import colmap_centers

    frames, _ = select_frames(paths, args.floor, args.stride, _frange(args))
    centers, rots = colmap_centers(paths.root / "sfm" / "colmap_coarse" / "0")
    names, positions, poses = [], [], []
    for fr in frames:
        if fr not in centers:
            continue
        g = np.eye(4)
        g[:3, :3] = rots[fr].T                                 # world→cam の逆
        g[:3, 3] = centers[fr]
        names.append(f"frame_{fr:06d}")
        positions.append(g[:3, 3])
        poses.append(g)
    print(f"colmap coarse: {len(names)}/{len(frames)} frames registered")
    log = {"mode": "colmap_coarse",
           "floor": args.floor, "unit": "colmap_units"}
    save_bench(paths, args.tag, names, np.stack(positions), np.stack(poses),
               None, None, log)


def _frange(args):
    return tuple(int(v) for v in args.frames.split(":")) if args.frames else None


def main(paths, argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("infer", "baseline"):
        sp = sub.add_parser(name)
        sp.add_argument("--tag", required=True)
        sp.add_argument("--floor", default=None, help="GT CSVの階で絞る (1F/2F/3F)")
        sp.add_argument("--frames", default=None,
                        help="'a:b' フレーム範囲で絞る (同一階の複数訪問の分離用)")
        sp.add_argument("--stride", type=int, default=1)
        if name == "infer":
            sp.add_argument("--minibatch", type=int, default=None)
            sp.add_argument("--yaws", default="0",
                            help="使用するクロップ方位のカンマ列 (例 '0,180'。"
                                 "0以外は vidmap_prep --yaw Y --tag "
                                 f"{CROP_TAG}_yY で事前生成)")
            sp.add_argument("--res", type=int, default=None,
                            help="推論解像度 (クロップを縮小。メモリは約2乗で減り"
                                 "ビュー数を増やせる)")
            sp.add_argument("--pose-cond", default=None,
                            help="条件付けに使うfrontend_benchタグ "
                                 "(リグ姿勢を非メトリックで入力)")
            sp.add_argument("--dump-local", action="store_true",
                            help="frame別リグ座標点群(メートル)を"
                                 "local_points.npzに保存 (深度のみ利用=案2用)")
            sp.add_argument("--nc", action="store_true",
                            help="非商用フルモデル (facebook/map-anything) を使う")
    args = ap.parse_args(argv)
    (run_infer if args.cmd == "infer" else run_baseline)(args)

