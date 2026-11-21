"""チェッカーボードによる魚眼キャリブレーション(自前実装、numpy + OpenCV のみ)。

入力は「ボード座標の点群」と「検出した画素座標」の組のリストだけで、画像の入力元には依存しない。
入射角 90° 以上の点もそのまま使える(model.project が atan2 ベースのため)。
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from .model import FisheyeCamera, project, rodrigues, unproject


@dataclass
class CalibResult:
    camera: FisheyeCamera
    rvecs: np.ndarray        # (N,3)
    tvecs: np.ndarray        # (N,3)
    rms: float               # 全点の再投影誤差 RMS [px](点ごとの距離の二乗平均平方根)
    view_rms: np.ndarray     # ビューごとの RMS [px]
    intr_std: np.ndarray     # 内部パラメータ 8 個の推定標準偏差(固定したものは 0)
    iterations: int
    elapsed_s: float
    n_points: int
    views: list = None       # 推定に使えたビューの番号(入力リストの添字)


def init_pose(intr, obj, img):
    """内部パラメータの初期値を使って 1 ビューのボード姿勢の初期値を求める。

    検出点を視線ベクトルに直し、その平均方向が光軸になるよう仮想的にカメラを回してから
    平面 PnP を解く。こうすると入射角 90° 前後のボードでも透視投影として扱える。
    """
    rays, _ = unproject(intr, img)
    m = rays.mean(0)
    m /= np.linalg.norm(m)
    ax = np.cross(m, [0.0, 0.0, 1.0])
    s = np.linalg.norm(ax)
    R0 = np.eye(3) if s < 1e-9 else rodrigues(ax / s * np.arctan2(s, m[2]))
    r = rays @ R0.T
    ok = r[:, 2] > 0.1
    if ok.sum() < 6:
        return None
    pts = (r[ok, :2] / r[ok, 2:3]).astype(np.float64)
    n, rv, tv, _ = cv2.solvePnPGeneric(obj[ok].astype(np.float64), pts, np.eye(3), None,
                                       flags=cv2.SOLVEPNP_IPPE)
    best = None
    for i in range(n):
        R = R0.T @ rodrigues(rv[i].ravel())
        t = R0.T @ tv[i].ravel()
        e = np.sqrt(np.mean(np.sum((project(intr, obj @ R.T + t) - img) ** 2, -1)))
        if best is None or e < best[0]:
            best = (e, cv2.Rodrigues(R)[0].ravel(), t)
    return best[1], best[2]


def _residuals(intr, poses, obj, img, vid):
    R = rodrigues(poses[:, :3])[vid]
    P = np.einsum("nij,nj->ni", R, obj) + poses[vid, 3:]
    return project(intr, P) - img


def _lm(intr, poses, obj, img, vid, free, tie_aspect, max_iter=100):
    """Levenberg-Marquardt。free: 動かす内部パラメータの添字。tie_aspect: fy = fx に拘束。"""
    free = np.asarray(free, int)
    N, M, ni = len(poses), len(obj), len(free)
    rows = np.arange(M)

    def full(i):
        i = i.copy()
        if tie_aspect:
            i[1] = i[0]
        return i

    def jac(intr, poses):
        J = np.zeros((2 * M, ni + 6 * N))
        for c, p in enumerate(free):
            h = 1e-6 * max(1.0, abs(intr[p]))
            a, b = intr.copy(), intr.copy()
            a[p] += h
            b[p] -= h
            J[:, c] = ((_residuals(full(a), poses, obj, img, vid)
                        - _residuals(full(b), poses, obj, img, vid)) / (2 * h)).ravel()
        for j in range(6):
            h = 1e-7
            a, b = poses.copy(), poses.copy()
            a[:, j] += h
            b[:, j] -= h
            d = (_residuals(intr, a, obj, img, vid) - _residuals(intr, b, obj, img, vid)) / (2 * h)
            col = ni + 6 * vid + j
            J[2 * rows, col] = d[:, 0]
            J[2 * rows + 1, col] = d[:, 1]
        return J

    r = _residuals(intr, poses, obj, img, vid).ravel()
    cost = r @ r
    lam, it = 1e-3, 0
    JTJ = None
    for it in range(1, max_iter + 1):
        J = jac(intr, poses)
        JTJ, g = J.T @ J, J.T @ r
        improved = False
        while lam < 1e12:
            try:
                dx = np.linalg.solve(JTJ + lam * np.diag(np.diag(JTJ) + 1e-12), -g)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            i2 = intr.copy()
            i2[free] += dx[:ni]
            i2 = full(i2)
            p2 = poses + dx[ni:].reshape(N, 6)
            r2 = _residuals(i2, p2, obj, img, vid).ravel()
            c2 = r2 @ r2
            if np.isfinite(c2) and c2 < cost:
                rel = (cost - c2) / max(cost, 1e-300)
                intr, poses, r, cost = i2, p2, r2, c2
                lam = max(lam / 10, 1e-9)
                improved = True
                break
            lam *= 10
        if not improved or rel < 1e-10 or cost < 1e-18:
            break
    return intr, poses, r, JTJ, it


def calibrate(obj_points, img_points, width, height, f_init, fov_deg=202.0, n_k=4,
              tie_aspect=False, fix_principal_point=False) -> CalibResult:
    """obj_points / img_points: ビューごとの (n,3) / (n,2) 配列のリスト。

    f_init: 焦点距離の初期値 [px](公称値でよい)。n_k: 推定する歪み係数の個数(0..4、残りは 0 固定)。
    """
    t0 = time.perf_counter()
    intr = np.array([f_init, f_init, (width - 1) / 2.0, (height - 1) / 2.0, 0, 0, 0, 0], float)
    keep, poses = [], []
    for i, (o, p) in enumerate(zip(obj_points, img_points)):
        init = init_pose(intr, np.asarray(o, float), np.asarray(p, float))
        if init is not None:
            keep.append(i)
            poses.append(np.concatenate(init))
    if len(keep) < 3:
        raise ValueError("姿勢の初期値が求まったビューが 3 未満")
    poses = np.array(poses)
    obj = np.concatenate([np.asarray(obj_points[i], float) for i in keep])
    img = np.concatenate([np.asarray(img_points[i], float) for i in keep])
    vid = np.concatenate([np.full(len(obj_points[i]), j) for j, i in enumerate(keep)])

    base = [0] if tie_aspect else [0, 1]
    if not fix_principal_point:
        base += [2, 3]
    # 段階的に自由度を増やす: 姿勢+f,c -> 歪み係数も解放
    intr, poses, r, JTJ, it = _lm(intr, poses, obj, img, vid, base, tie_aspect)
    free = base
    if n_k > 0:
        free = base + list(range(4, 4 + n_k))
        intr, poses, r, JTJ, it2 = _lm(intr, poses, obj, img, vid, free, tie_aspect)
        it += it2

    err2 = (r.reshape(-1, 2) ** 2).sum(-1)
    dof = max(r.size - JTJ.shape[0], 1)
    std = np.zeros(8)
    try:
        cov = np.linalg.inv(JTJ) * (r @ r / dof)
        std[free] = np.sqrt(np.maximum(np.diag(cov)[:len(free)], 0))
        if tie_aspect:
            std[1] = std[0]
    except np.linalg.LinAlgError:
        std[:] = np.nan
    view_rms = np.array([np.sqrt(err2[vid == j].mean()) for j in range(len(keep))])
    return CalibResult(FisheyeCamera.from_intr(width, height, intr, fov_deg),
                       poses[:, :3], poses[:, 3:], float(np.sqrt(err2.mean())), view_rms, std,
                       it, time.perf_counter() - t0, len(obj), keep)


def calibrate_auto(obj_points, img_points, width, height, f_init=None, **kw):
    """焦点距離の初期値が分からなくても使える版。候補を順に試し、再投影誤差が最小の結果を返す。

    戻り値: (CalibResult, 採用した初期値)。f_init を与えた場合はそれだけを使う。
    """
    cands = [f_init] if f_init else [width * s for s in (0.20, 0.26, 0.33, 0.42, 0.54)]
    results = []
    for f in cands:
        try:
            results.append((calibrate(obj_points, img_points, width, height, f, **kw), f))
        except (ValueError, np.linalg.LinAlgError):
            pass
    if not results:
        raise ValueError("どの初期値でも推定できなかった")
    most = max(len(r.views) for r, _ in results)
    return min((x for x in results if len(x[0].views) >= 0.8 * most), key=lambda x: x[0].rms)
