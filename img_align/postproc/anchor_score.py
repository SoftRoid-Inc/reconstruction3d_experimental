"""図面アンカー候補のスコアリング (どの仮説が正しいかを決める)

従来は `wall_cover`(島の壁点のうち図面壁の近くに載った割合)だけで順位付け
していたが、これは**片方向=precisionのみ**で、図面側の壁が説明されたか
(recall)を見ていない。ICPがスケールを自由に動かせるため、
**点群を縮めて壁の密集部に押し込むほど値が上がる退化最適**がある。

2026-10-13の実測 (全周島8本、正しいスケールを与えた条件):
正解が候補に入っているのは6/8だが、wall_coverが選べたのは**0/8**。
しかも正解のwall_coverは選ばれた偽解より**8本中7本で低く、高かった例はゼロ**
(例: 0828_1Fb 正解0.326 vs 偽解0.803)。無情報ではなく逆相関している。

ここでは次を計算して組み合わせる:
  precision : 島の壁点 → 図面壁    (従来の wall_cover)
  recall    : 図面壁 → 島の壁点。ただし**島が到達した範囲の壁だけ**を分母にする
              (島はフロアの一部しか回らないので全体を分母にすると部分島が不当に沈む)
  f1        : 上2つの調和平均。縮小退化を直接潰す (縮めればPは上がるがRが落ちる)
  cam_feas  : カメラが屋内自由空間に載る割合 (従来は0.5の足切りにしか使っていない)
  scale_dev : |log(最終スケール / 事前スケール)|。ICPがどれだけ尺度を動かしたか

使い方:
  sc = AnchorScorer(grid, wall_pts, segs, cam_rc)
  v = sc(s, th, t)        # -> {"precision":…, "recall":…, "f1":…, …}
  v["f1_feas"]            # 既定の総合スコア
"""
import numpy as np
from scipy.spatial import cKDTree

from img_align.postproc.plan_icp import transform_pts

CUT_CELLS = 2.0        # 壁に載ったとみなす距離 (セル。1セル=0.05m → 10cm)
SENSE_CELLS = 120.0    # 島が「到達した」とみなすカメラからの距離 (=6m)
SAMPLE_CELLS = 1.0     # 図面壁線分を点列にする間隔
SCALE_K = 2.0          # scale_dev のペナルティ強度


def sample_segments(segs, step=SAMPLE_CELLS):
    """壁線分 [(r0,c0,r1,c1)] を等間隔の点列にする"""
    a, b = segs[:, :2], segs[:, 2:]
    ln = np.linalg.norm(b - a, axis=1)
    pts = []
    for i in range(len(segs)):
        n = max(2, int(ln[i] / step) + 1)
        u = np.linspace(0, 1, n)[:, None]
        pts.append(a[i] + u * (b[i] - a[i]))
    return np.concatenate(pts) if pts else np.zeros((0, 2))


class AnchorScorer:
    """候補変換 (s, th, t) → スコア一式。図面側のKD木は1度だけ作る"""

    def __init__(self, grid, wall_pts, segs, cam_rc, s_prior=1.0,
                 cut=CUT_CELLS, sense=SENSE_CELLS):
        self.grid = grid
        self.wall_pts = wall_pts
        self.cam_rc = cam_rc
        self.s_prior = s_prior
        self.cut = cut
        self.sense = sense
        self.plan_pts = sample_segments(segs)
        self.plan_tree = cKDTree(self.plan_pts)

    def __call__(self, s, th, t):
        wp = transform_pts(self.wall_pts, s, th, t)
        cam = transform_pts(self.cam_rc, s, th, t)

        # precision: 島の壁点 → 図面壁
        d_w, _ = self.plan_tree.query(wp, k=1)
        precision = float((d_w < self.cut).mean())

        # recall: 図面壁 → 島の壁点。分母は島が到達した範囲の壁のみ
        cam_tree = cKDTree(cam)
        d_cam, _ = cam_tree.query(self.plan_pts, k=1)
        near = d_cam < self.sense
        if near.sum() < 10:
            recall = 0.0
        else:
            d_p, _ = cKDTree(wp).query(self.plan_pts[near], k=1)
            recall = float((d_p < self.cut).mean())

        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall > 0 else 0.0)

        # cam_feas: カメラが屋内自由空間セルに載る割合
        pc = np.round(cam).astype(int)
        ok = ((pc[:, 0] >= 0) & (pc[:, 0] < self.grid.ny)
              & (pc[:, 1] >= 0) & (pc[:, 1] < self.grid.nx))
        feas = (float(self.grid.valid_pose[pc[ok, 0], pc[ok, 1]].mean()
                      * ok.mean()) if ok.any() else 0.0)

        dev = abs(np.log(max(s, 1e-6) / max(self.s_prior, 1e-6)))
        pen = float(np.exp(-SCALE_K * dev))
        return {
            "precision": precision, "recall": recall, "f1": f1,
            "cam_feas": feas, "scale_dev": float(dev),
            # 比較用の名前付きスコア (eval/anchor_oracle が順位を測る)
            "cover": precision,                  # 従来の基準
            "f1_feas": f1 * feas,
            "f1_feas_scale": f1 * feas * pen,
            "cover_feas": precision * feas,
        }


SCORERS = ("cover", "f1", "f1_feas", "f1_feas_scale", "cover_feas")
