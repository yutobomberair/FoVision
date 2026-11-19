"""チェッカーボードを使わない自己校正(自然特徴点の対応+バンドル調整)。

入力は「どのフレームでどの特徴点がどの画素に見えたか」の対応だけ。既知形状のターゲットも
カメラ姿勢も 3 次元点も使わない。内部パラメータの初期値には公称値を使う。

手順:
  1. 公称パラメータで画素を視線ベクトルに直す(入射角 90° 超もそのまま扱える)
  2. 2 フレームの視線同士からエッセンシャル行列(8 点法)-> 相対姿勢 -> 三角測量
  3. 残りのフレームは視線ベースの DLT で姿勢を求める
  4. 姿勢・3 次元点・内部パラメータを同時にバンドル調整(Schur 補行列を使う LM)

スケールは決まらない(内部パラメータには影響しない)。外れ値除去は未実装。
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from .model import FisheyeCamera, project, rodrigues, unproject


@dataclass
class SelfCalResult:
    camera: FisheyeCamera
    rms: float
    n_frames: int
    n_points: int
    n_obs: int
    iterations: int
    elapsed_s: float


def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0.0]])


def _triangulate(R, t, fi, pi, rays, n_pts):
    """各点について、観測した全視線への距離二乗和が最小になる位置(中点法の多視点版)。"""
    d = np.einsum("kji,kj->ki", R[fi], rays)            # R^T r
    C = -np.einsum("kji,kj->ki", R[fi], t[fi])          # カメラ中心
    A = np.eye(3) - d[:, :, None] * d[:, None, :]
    sA = np.zeros((n_pts, 3, 3))
    sb = np.zeros((n_pts, 3))
    np.add.at(sA, pi, A)
    np.add.at(sb, pi, np.einsum("kij,kj->ki", A, C))
    cnt = np.bincount(pi, minlength=n_pts)
    ok = (cnt >= 2) & (np.linalg.cond(sA + 1e-12 * np.eye(3)) < 1e6)
    X = np.zeros((n_pts, 3))
    X[ok] = np.linalg.solve(sA[ok], sb[ok][..., None])[..., 0]
    return X, ok


def _two_view(r1, r2):
    """視線 r1, r2 (n,3) から相対姿勢 (R, t)(P2 = R P1 + t)と、視差角の中央値を返す。"""
    A = np.einsum("ni,nj->nij", r2, r1).reshape(-1, 9)
    E = np.linalg.svd(A)[2][-1].reshape(3, 3)
    U, _, Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0:
        U = -U
    if np.linalg.det(Vt) < 0:
        Vt = -Vt
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    best = None
    n = len(r1)
    fi = np.r_[np.zeros(n, int), np.ones(n, int)]
    pi = np.r_[np.arange(n), np.arange(n)]
    rr = np.r_[r1, r2]
    for R in (U @ W @ Vt, U @ W.T @ Vt):
        for t in (U[:, 2], -U[:, 2]):
            Rs, ts = np.stack([np.eye(3), R]), np.stack([np.zeros(3), t])
            X, ok = _triangulate(Rs, ts, fi, pi, rr, n)
            front = ok & ((X * r1).sum(1) > 0) & (((X @ R.T + t) * r2).sum(1) > 0)
            if best is None or front.sum() > best[0]:
                d2 = r2 @ R  # フレーム 1 の座標系に戻した視線
                par = np.median(np.arccos(np.clip((r1 * d2).sum(1), -1, 1))[front]) if front.any() else 0.0
                best = (front.sum(), R, t, par)
    return best[1], best[2], best[3], best[0] / n


def _pnp_rays(X, r):
    """3 次元点 X と視線 r から姿勢 (R, t) を求める(DLT。r x (R X + t) = 0)。"""
    m, s = X.mean(0), X.std() + 1e-12
    Xh = np.c_[(X - m) / s, np.ones(len(X))]
    Sk = np.zeros((len(r), 3, 3))
    Sk[:, 0, 1], Sk[:, 0, 2], Sk[:, 1, 0] = -r[:, 2], r[:, 1], r[:, 2]
    Sk[:, 1, 2], Sk[:, 2, 0], Sk[:, 2, 1] = -r[:, 0], -r[:, 1], r[:, 0]
    A = np.einsum("nij,nk->nijk", Sk, Xh).reshape(-1, 12)
    P = np.linalg.svd(A, full_matrices=False)[2][-1].reshape(3, 4)
    if np.sum(r * (Xh @ P.T)) < 0:
        P = -P
    U, S, Vt = np.linalg.svd(P[:, :3])
    R = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    return R, s * P[:, 3] / S.mean() - R @ m


def _res(intr, poses, pts, fi, pi, uv):
    R = rodrigues(poses[:, :3])[fi]
    return project(intr, np.einsum("kij,kj->ki", R, pts[pi]) + poses[fi, 3:]) - uv


def _ba(intr, poses, pts, fi, pi, uv, free, max_iter=60):
    """バンドル調整。フレーム 0 の姿勢は固定。free: 動かす内部パラメータの添字。"""
    free = np.asarray(free, int)
    K, N, M, ni = len(uv), len(poses), len(pts), len(free)
    nc = ni + 6 * (N - 1)
    ar = np.arange(K)
    mov = fi > 0

    def jac():
        Jc = np.zeros((K, 2, nc))
        Jp = np.zeros((K, 2, 3))
        for c, p in enumerate(free):
            h = 1e-6 * max(1.0, abs(intr[p]))
            a, b = intr.copy(), intr.copy()
            a[p] += h
            b[p] -= h
            Jc[:, :, c] = (_res(a, poses, pts, fi, pi, uv) - _res(b, poses, pts, fi, pi, uv)) / (2 * h)
        for j in range(6):
            a, b = poses.copy(), poses.copy()
            a[:, j] += 1e-7
            b[:, j] -= 1e-7
            d = (_res(intr, a, pts, fi, pi, uv) - _res(intr, b, pts, fi, pi, uv)) / 2e-7
            col = ni + 6 * (fi[mov] - 1) + j
            Jc[ar[mov], 0, col] = d[mov, 0]
            Jc[ar[mov], 1, col] = d[mov, 1]
        for j in range(3):
            a, b = pts.copy(), pts.copy()
            a[:, j] += 1e-7
            b[:, j] -= 1e-7
            Jp[:, :, j] = (_res(intr, poses, a, fi, pi, uv) - _res(intr, poses, b, fi, pi, uv)) / 2e-7
        return Jc, Jp

    r = _res(intr, poses, pts, fi, pi, uv)
    cost = float((r * r).sum())
    lam, it = 1e-3, 0
    for it in range(1, max_iter + 1):
        Jc, Jp = jac()
        Jc2 = Jc.reshape(2 * K, nc)
        U = Jc2.T @ Jc2
        gc = Jc2.T @ r.ravel()
        V = np.zeros((M, 3, 3))
        np.add.at(V, pi, np.einsum("kai,kaj->kij", Jp, Jp))
        gp = np.zeros((M, 3))
        np.add.at(gp, pi, np.einsum("kai,ka->ki", Jp, r))
        Wm = np.zeros((M, nc, 3))
        np.add.at(Wm, pi, np.einsum("kan,kab->knb", Jc, Jp))
        improved = False
        while lam < 1e12:
            Vd = V + lam * (np.eye(3) * (np.einsum("mii->mi", V)[:, :, None] + 1e-12))
            Vi = np.linalg.inv(Vd)
            WVi = np.einsum("mnb,mbc->mnc", Wm, Vi)
            S = U + lam * np.diag(np.diag(U) + 1e-12) - np.einsum("mnc,mkc->nk", WVi, Wm)
            rhs = -gc + np.einsum("mnc,mc->n", WVi, gp)
            try:
                dc = np.linalg.solve(S, rhs)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            dp = np.einsum("mbc,mc->mb", Vi, -gp - np.einsum("mnc,n->mc", Wm, dc))
            i2 = intr.copy()
            i2[free] += dc[:ni]
            p2 = poses.copy()
            p2[1:] += dc[ni:].reshape(N - 1, 6)
            x2 = pts + dp
            r2 = _res(i2, p2, x2, fi, pi, uv)
            c2 = float((r2 * r2).sum())
            if np.isfinite(c2) and c2 < cost:
                rel = (cost - c2) / max(cost, 1e-300)
                intr, poses, pts, r, cost = i2, p2, x2, r2, c2
                lam = max(lam / 10, 1e-9)
                improved = True
                break
            lam *= 10
        if not improved or rel < 1e-9:
            break
    return intr, poses, pts, r, it


def self_calibrate(fi, pi, uv, n_frames, n_points, width, height, f_init, fov_deg=202.0, n_k=4):
    """fi, pi: 観測ごとのフレーム番号と特徴点番号 (K,)。uv: 画素座標 (K,2)。"""
    t0 = time.perf_counter()
    fi, pi, uv = np.asarray(fi), np.asarray(pi), np.asarray(uv, float)
    intr = np.array([f_init, f_init, (width - 1) / 2.0, (height - 1) / 2.0, 0, 0, 0, 0], float)
    rays, _ = unproject(intr, uv)

    # 2 フレームでの初期化: フレーム 0 と、視差角が最も大きく取れる相手を選ぶ
    idx0 = {p: k for k, p in zip(np.flatnonzero(fi == 0), pi[fi == 0])}
    best = None
    for f in range(1, n_frames):
        kk = np.flatnonzero(fi == f)
        com = [(idx0[p], k) for k, p in zip(kk, pi[kk]) if p in idx0]
        if len(com) < 30:
            continue
        a, b = np.array(com).T
        R, t, par, frac = _two_view(rays[a], rays[b])
        if frac > 0.8 and (best is None or par > best[0]):
            best = (par, f, R, t)
    if best is None:
        raise ValueError("2 フレーム初期化に失敗(共通点または視差が不足)")
    _, f1, R1, t1 = best
    Rs = np.tile(np.eye(3), (n_frames, 1, 1))
    ts = np.zeros((n_frames, 3))
    Rs[f1], ts[f1] = R1, t1
    sel = (fi == 0) | (fi == f1)
    X, ok = _triangulate(Rs, ts, fi[sel], pi[sel], rays[sel], n_points)
    for f in range(1, n_frames):
        if f == f1:
            continue
        kk = np.flatnonzero((fi == f) & ok[pi])
        if len(kk) < 12:
            raise ValueError(f"フレーム {f} の姿勢初期化に使える点が不足")
        Rs[f], ts[f] = _pnp_rays(X[pi[kk]], rays[kk])
    X, ok = _triangulate(Rs, ts, fi, pi, rays, n_points)

    keep = ok[pi]
    remap = -np.ones(n_points, int)
    remap[ok] = np.arange(ok.sum())
    fi, pi, uv = fi[keep], remap[pi[keep]], uv[keep]
    pts = X[ok]
    poses = np.array([np.r_[cv2.Rodrigues(Rs[f])[0].ravel(), ts[f]] for f in range(n_frames)])

    # 段階的に自由度を増やす: 構造のみ -> f と主点 -> 歪み係数
    intr, poses, pts, r, it = _ba(intr, poses, pts, fi, pi, uv, [], max_iter=20)
    intr, poses, pts, r, it2 = _ba(intr, poses, pts, fi, pi, uv, [0, 1, 2, 3])
    it += it2
    if n_k > 0:
        intr, poses, pts, r, it3 = _ba(intr, poses, pts, fi, pi, uv, [0, 1, 2, 3] + list(range(4, 4 + n_k)))
        it += it3
    rms = float(np.sqrt((r * r).sum(1).mean()))
    return SelfCalResult(FisheyeCamera.from_intr(width, height, intr, fov_deg), rms, n_frames,
                         len(pts), len(uv), it, time.perf_counter() - t0)
