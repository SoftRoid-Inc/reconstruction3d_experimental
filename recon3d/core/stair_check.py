"""島の階段またぎ検査: チェーン済み島のカメラ鉛直変位から階段を検出する

stairnet(IMU)の見逃し(短い2F→3F等、0616/0925で実測)への保険。
重力整列したカメラ軌跡の鉛直成分に「カメラ高相当の1.5倍を超える変位」が
あれば階段またぎと断定し、変位が集中する区間を切断範囲として返す。

チャンク座標は非メートルのため、スケールは島点群のスラブ(床/天井)ピークと
カメラ中央高の距離=カメラ高相当(units)で正規化する。階高2.9m ≈ カメラ高
1.3〜1.5mの約2倍なので、閾値1.5×カメラ高は歩行の上下動(±0.1h)とは桁で分かれる。
"""
import numpy as np

from recon3d.config import BENCH_DIR
from recon3d.core.geometry import rotation_to_y
from recon3d.core.io_ply import read_ply_xyzrgb
from recon3d.postproc.density_map import slab_peaks

WIN_FRAMES = (12, 24, 40)   # 変位を測る窓の候補 (階段は10〜40フレームで昇りきる)
DISP_FACTOR = 1.5     # カメラ高に対する跨ぎ判定閾値
EDGE_KEEP = 0.7       # 切断範囲 = |変位|がピークのこの倍率を超える連結区間
SMOOTH_K = 5


def island_vertical(tag):
    """島タグ → (フレーム昇順, 平滑化済み鉛直カメラ位置, カメラ高相当units)"""
    d = np.load(BENCH_DIR / tag / "global_cameras.npz")
    names = [str(n) for n in d["names"]]
    frames = np.array([int(n.replace("frame_", "")) for n in names])
    order = np.argsort(frames)
    frames = frames[order]
    poses = d["poses"][order]
    pos = d["positions"][order]
    ups = poses[:, :3, 1]
    up = np.median(ups / np.linalg.norm(ups, axis=1, keepdims=True), axis=0)
    rg = rotation_to_y(up / np.linalg.norm(up))
    y = (pos @ rg.T)[:, 1]
    xyz, _ = read_ply_xyzrgb(BENCH_DIR / tag / "global_points.ply")
    py = (xyz[::7] @ rg.T)[:, 1]
    peaks, _, _ = slab_peaks(py)
    cam = float(np.median(y))
    h = float(np.min(np.abs(peaks - cam))) if len(peaks) else 1.0
    h = max(h, 0.4)   # ピーク縮退時のガード (カメラ高が0.4units未満はあり得ない)
    k = SMOOTH_K
    ys = np.convolve(np.pad(y, (k // 2, k - k // 2 - 1), mode="edge"),
                     np.ones(k) / k, mode="valid")
    return frames, ys, h


def find_stair(frames, ys, h):
    """鉛直異常の検出。無ければNone、あれば
    (切断開始fr, 切断終了fr, 変位/カメラ高, kind)。
    kind: "transition"=レベルが変わる(階段) / "excursion"=元のレベルへ戻る
    (はしご・脚立の往復。前後の中央レベル差 <0.7×カメラ高で判定)

    昇りの所要フレーム数は10〜40と幅があるため、複数の窓幅で変位を測り
    最大のものを採用する (固定窓だと長い昇りが薄まって漏れる)。"""
    n = len(ys)
    best = None
    for w in WIN_FRAMES:
        w = min(w, max(4, n - 1))
        if n <= w:
            continue
        disp = ys[w:] - ys[:-w]
        j = int(np.argmax(np.abs(disp)))
        m = float(abs(disp[j]))
        if best is None or m > best[0]:
            best = (m, w, j, disp)
    if best is None or best[0] < DISP_FACTOR * h:
        return None
    m, w, j, disp = best
    lo, hi = j, j
    while lo > 0 and abs(disp[lo - 1]) > EDGE_KEEP * m:
        lo -= 1
    while hi < len(disp) - 1 and abs(disp[hi + 1]) > EDGE_KEEP * m:
        hi += 1
    # 候補域 [lo, hi+w] は窓幅ぶん広い。実際に鉛直移動している「コア区間」
    # (進行度5〜95%) に絞る — 切除しすぎによるカバー損失対策
    a_i, b_i = lo, min(hi + w, n - 1)
    seg = ys[a_i:b_i + 1]
    denom = seg[-1] - seg[0]
    if abs(denom) > 1e-9:
        prog = (seg - seg[0]) / denom
        core = np.where((prog > 0.05) & (prog < 0.95))[0]
        if len(core):
            a_i, b_i = a_i + int(core[0]), a_i + int(core[-1])
    pre = ys[max(0, a_i - 30):a_i]
    post = ys[b_i + 1:b_i + 31]
    kind = "transition"
    if len(pre) >= 5 and len(post) >= 5 and             abs(float(np.median(post)) - float(np.median(pre))) < 0.7 * h:
        kind = "excursion"
    return int(frames[a_i]), int(frames[b_i]), m / h, kind


def check_island(tag):
    """便宜関数: タグを検査して結果を表示。跨ぎがあれば切断範囲を返す"""
    frames, ys, h = island_vertical(tag)
    s = find_stair(frames, ys, h)
    span = float(ys.max() - ys.min())
    if s is None:
        print(f"{tag}: clean (鉛直スパン {span / h:.2f}×カメラ高)")
    else:
        print(f"{tag}: 鉛直異常[{s[3]}] frames {s[0]}-{s[1]} "
              f"(変位 {s[2]:.2f}×カメラ高)")
    return s


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    args = ap.parse_args()
    for t in args.tags:
        check_island(t)
