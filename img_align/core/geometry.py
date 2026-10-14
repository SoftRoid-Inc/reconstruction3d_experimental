"""共通幾何: Sim(3)・重力整列・ヨーの頑健統計 (全系統共通)"""
import math

import numpy as np

CORE_DEG = 5.0    # 頑健ヨー中央値の再計算で除外する逸脱 (deg)


# ---------------------------------------------------------------- Sim(3)

def sim3_umeyama(src, dst):
    """dst ≈ s R src + t (3D)"""
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    sc, dc = src - mu_s, dst - mu_d
    cov = dc.T @ sc / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var = (sc ** 2).sum() / len(src)
    s = float(np.trace(np.diag(D) @ S) / var)
    t = mu_d - s * R @ mu_s
    return s, R, t


def apply(T, pts):
    s, R, t = T
    return (s * (pts @ R.T)) + t


def compose(T2, T1):
    """T2∘T1 (先にT1)"""
    s1, R1, t1 = T1
    s2, R2, t2 = T2
    return (s2 * s1, R2 @ R1, s2 * (R2 @ t1) + t2)


# ---------------------------------------------------------------- 重力整列

def rotation_to_y(up):
    """ベクトルupを+yへ回す回転行列 (Rodrigues)"""
    up = up / np.linalg.norm(up)
    axis = np.cross(up, [0, 1, 0])
    s = np.linalg.norm(axis)
    c = float(up[1])
    if s < 1e-8:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    k = axis / s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s * K + (1 - c) * K @ K


def gravity_from_cameras(poses):
    """カメラup方向の中央値から重力(world up)を推定する

    ローカル点群で床が y<0 にあるため、カメラ座標の+yが上。
    pose が cam2world / world2cam のどちらの規約かは、フレーム間の
    up方向のばらつきが小さい方を採用して自動判定する。
    """
    cands = {
        "cam2world": poses[:, :3, 1],          # R[:,1] = カメラy軸のworld表現
        "world2cam": poses[:, 1, :3],          # R^T[:,1]
    }
    best = None
    for conv, ups in cands.items():
        u = ups / np.linalg.norm(ups, axis=1, keepdims=True)
        med = np.median(u, axis=0)
        med /= np.linalg.norm(med)
        spread = np.degrees(np.arccos(np.clip(u @ med, -1, 1))).mean()
        print(f"  up-convention {conv}: spread {spread:.1f} deg, "
              f"up={np.round(med, 3)}")
        if best is None or spread < best[2]:
            best = (conv, med, spread)
    conv, up, _ = best
    return conv, up


def camera_centers(poses, conv):
    if conv == "cam2world":
        return poses[:, :3, 3]
    return -np.einsum("nij,nj->ni", poses[:, :3, :3].transpose(0, 2, 1),
                      poses[:, :3, 3])


# ---------------------------------------------------------------- ヨー統計

def rot_y(th):
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def robust_yaw_median(phi):
    """円周中央値 (逸脱大フレームを除外して2回反復)"""
    c = np.exp(1j * phi)
    med = math.atan2(np.median(np.imag(c)), np.median(np.real(c)))
    for _ in range(2):
        dev = np.degrees(np.angle(c * np.exp(-1j * med)))
        good = np.abs(dev) <= CORE_DEG
        if good.sum() < 3:
            break
        med = math.atan2(np.median(np.imag(c[good])),
                         np.median(np.real(c[good])))
    dev = np.degrees(np.angle(c * np.exp(-1j * med)))
    return med, dev
