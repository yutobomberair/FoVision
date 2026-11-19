"""シミュレーション: ボード姿勢の生成、理想コーナーの投影、合成画像の描画、個体差の付与。"""
from __future__ import annotations

import cv2
import numpy as np

from .board import Board
from .model import FisheyeCamera, rodrigues


def sample_pose(cam: FisheyeCamera, board: Board, rng, dist=(0.15, 0.40), tilt_deg=35.0,
                min_visible=1.0, max_tries=200):
    """ボード中心が画像内の一様ランダムな位置に写るような姿勢 (R, t) を返す。

    min_visible: 内側コーナーのうち画像内に写っている必要がある割合(1.0 = 全点)。
    """
    obj = board.object_points
    for _ in range(max_tries):
        uv = np.array([rng.uniform(0, cam.width - 1), rng.uniform(0, cam.height - 1)])
        ray, theta = cam.unproject(uv)
        if theta > cam.theta_max:
            continue
        d = rng.uniform(*dist)
        # ボード Z 軸をカメラから遠ざかる向き(OpenCV の検出器の並び順と整合する向き)にしてから、
        # ランダムに傾けて面内回転を与える
        z = ray
        x = np.cross([0.0, 1.0, 0.0], z)
        if np.linalg.norm(x) < 1e-6:
            x = np.array([1.0, 0.0, 0.0])
        x /= np.linalg.norm(x)
        R0 = np.stack([x, np.cross(z, x), z], axis=1)
        tilt = np.deg2rad(tilt_deg) * np.sqrt(rng.uniform())
        az = rng.uniform(0, 2 * np.pi)
        Rt = rodrigues(tilt * np.array([np.cos(az), np.sin(az), 0.0]))
        Rz = rodrigues(np.array([0.0, 0.0, rng.uniform(0, 2 * np.pi)]))
        R = R0 @ Rt @ Rz
        t = ray * d - R @ board.center
        vis = cam.visible(obj @ R.T + t, margin=4.0)
        if vis.mean() >= min_visible:
            return R, t
    raise RuntimeError("姿勢のサンプリングに失敗")


def ideal_corners(cam: FisheyeCamera, board: Board, R, t):
    """真値パラメータでの内側コーナーの画素座標と入射角。"""
    P = board.object_points @ R.T + t
    return cam.project(P), np.arctan2(np.hypot(P[:, 0], P[:, 1]), P[:, 2])


class Renderer:
    """投影モデルから直接チェッカーボード画像を描く(画素ごとに視線を出して平面と交差させる)。"""

    def __init__(self, cam: FisheyeCamera, supersample: int = 2):
        self.cam = cam
        self.rays = []
        ss = supersample
        v, u = np.mgrid[0:cam.height, 0:cam.width].astype(np.float64)
        for j in range(ss):
            for i in range(ss):
                uv = np.stack([u + (i + 0.5) / ss - 0.5, v + (j + 0.5) / ss - 0.5], -1)
                rays, theta = cam.unproject(uv)
                self.rays.append((rays.astype(np.float32), theta <= cam.theta_max))

    def render(self, board: Board, R, t, blur_sigma=1.0, noise_sigma=2.0, rng=None,
               white=215.0, black=35.0, background=110.0):
        """8bit グレースケール画像を返す。blur_sigma [px], noise_sigma [階調]。"""
        R = np.asarray(R, np.float32)
        t = np.asarray(t, np.float32)
        n = R[:, 2]
        sq = board.square
        acc = np.zeros((self.cam.height, self.cam.width), np.float32)
        for rays, inside in self.rays:
            denom = rays @ n
            with np.errstate(divide="ignore", invalid="ignore"):
                s = float(n @ t) / denom
            hit = inside & (s > 0) & np.isfinite(s)
            Pb = (rays * np.where(hit, s, 0)[..., None] - t) @ R  # R^T (P - t)
            ix = np.floor(Pb[..., 0] / sq).astype(np.int32) + 1
            iy = np.floor(Pb[..., 1] / sq).astype(np.int32) + 1
            in_pat = hit & (ix >= 0) & (ix <= board.cols) & (iy >= 0) & (iy <= board.rows)
            in_margin = hit & (ix >= -1) & (ix <= board.cols + 1) & (iy >= -1) & (iy <= board.rows + 1)
            img = np.where(inside, background, 0.0).astype(np.float32)
            img[in_margin] = white
            img[in_pat & ((ix + iy) % 2 == 0)] = black
            acc += img
        img = acc / len(self.rays)
        if blur_sigma > 0:
            img = cv2.GaussianBlur(img, (0, 0), blur_sigma)
        if noise_sigma > 0:
            rng = rng or np.random.default_rng()
            img = img + rng.normal(0, noise_sigma, img.shape).astype(np.float32)
        return np.clip(np.round(img), 0, 255).astype(np.uint8)


def perturb_camera(nominal: FisheyeCamera, rng, f_rel=0.02, aspect_rel=0.002, c_px=8.0,
                   k_sigma=(0.01, 0.003, 0.0, 0.0)):
    """公称カメラに個体差を与える(ばらつきの大きさは仮定値。実機の分布は未検証)。"""
    f = nominal.fx * (1 + rng.normal(0, f_rel))
    return FisheyeCamera(
        nominal.width, nominal.height,
        f, f * (1 + rng.normal(0, aspect_rel)),
        nominal.cx + rng.normal(0, c_px), nominal.cy + rng.normal(0, c_px),
        np.asarray(nominal.k) + rng.normal(0, 1, 4) * np.array(k_sigma), nominal.fov_deg)
