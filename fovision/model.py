"""魚眼カメラの投影モデル(Kannala-Brandt / 等距離射影ベース)。

定義(カメラ座標系: X 右, Y 下, Z 光軸前方。画素座標は画素中心が整数):
    rho     = sqrt(X^2 + Y^2)
    theta   = atan2(rho, Z)              # 光軸からの入射角。90° を超えてもよい
    theta_d = theta * (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8)
    u = fx * theta_d * X / rho + cx
    v = fy * theta_d * Y / rho + cy

k1..k4 = 0 のとき純粋な等距離射影 r = f * theta。
Z > 0 の範囲では cv2.fisheye の (K, D)(skew=0)と同じ定義。
cv2.fisheye との違いは theta を atan2 で求める点で、これにより Z <= 0(入射角 90° 以上)も扱える。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MODEL_NAME = "kannala_brandt_equidistant_atan2"


def theta_d(theta, k):
    t2 = theta * theta
    return theta * (1.0 + t2 * (k[0] + t2 * (k[1] + t2 * (k[2] + t2 * k[3]))))


def project(intr, P):
    """intr = [fx, fy, cx, cy, k1, k2, k3, k4], P (...,3) カメラ座標 -> (...,2) 画素座標。"""
    X, Y, Z = P[..., 0], P[..., 1], P[..., 2]
    rho = np.hypot(X, Y)
    td = theta_d(np.arctan2(rho, Z), intr[4:8])
    s = td / np.maximum(rho, 1e-12)
    return np.stack([intr[0] * s * X + intr[2], intr[1] * s * Y + intr[3]], axis=-1)


def unproject(intr, uv, theta_max=np.pi):
    """画素座標 (...,2) -> (単位視線ベクトル (...,3), 入射角 theta (...))。

    theta_d(theta) が [0, theta_max] で単調増加であることを前提に、数表の補間で逆関数を求める。
    """
    mx = (uv[..., 0] - intr[2]) / intr[0]
    my = (uv[..., 1] - intr[3]) / intr[1]
    td = np.hypot(mx, my)
    th_tab = np.linspace(0.0, theta_max, 4097)
    td_tab = theta_d(th_tab, intr[4:8])
    theta = np.interp(td, td_tab, th_tab)
    # 補間後にニュートン法で仕上げる
    k = intr[4:8]
    for _ in range(3):
        t2 = theta * theta
        d = 1.0 + t2 * (3 * k[0] + t2 * (5 * k[1] + t2 * (7 * k[2] + t2 * 9 * k[3])))
        theta = np.clip(theta - (theta_d(theta, k) - td) / d, 0.0, theta_max)
    s = np.sin(theta) / np.maximum(td, 1e-12)
    rays = np.stack([s * mx, s * my, np.cos(theta)], axis=-1)
    return rays, theta


@dataclass
class FisheyeCamera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    k: np.ndarray = field(default_factory=lambda: np.zeros(4))
    fov_deg: float = 202.0  # レンズの全画角。これを超える入射角は像が無いものとして扱う

    @property
    def intr(self) -> np.ndarray:
        return np.array([self.fx, self.fy, self.cx, self.cy, *np.asarray(self.k, float)])

    @property
    def theta_max(self) -> float:
        return np.deg2rad(self.fov_deg / 2.0)

    @classmethod
    def from_intr(cls, width, height, intr, fov_deg=202.0):
        return cls(width, height, float(intr[0]), float(intr[1]), float(intr[2]), float(intr[3]),
                   np.array(intr[4:8], float), fov_deg)

    def project(self, P):
        return project(self.intr, np.asarray(P, float))

    def unproject(self, uv):
        return unproject(self.intr, np.asarray(uv, float), self.theta_max + 0.05)

    def visible(self, P, margin=0.0):
        """カメラ座標の点が画像内かつ画角内に写るか。"""
        P = np.asarray(P, float)
        theta = np.arctan2(np.hypot(P[..., 0], P[..., 1]), P[..., 2])
        uv = self.project(P)
        return ((theta <= self.theta_max)
                & (uv[..., 0] >= margin) & (uv[..., 0] <= self.width - 1 - margin)
                & (uv[..., 1] >= margin) & (uv[..., 1] <= self.height - 1 - margin))


def nominal_camera(width=1920, height=1536, fov_deg=202.0, circle_diameter_px=None):
    """公称値のカメラ(等距離射影)。既定はイメージサークルが水平幅いっぱいの場合。"""
    d = float(width if circle_diameter_px is None else circle_diameter_px)
    f = (d / 2.0) / np.deg2rad(fov_deg / 2.0)
    return FisheyeCamera(width, height, f, f, (width - 1) / 2.0, (height - 1) / 2.0, np.zeros(4), fov_deg)


def rodrigues(rvec):
    """回転ベクトル (...,3) -> 回転行列 (...,3,3)。"""
    rvec = np.asarray(rvec, float)
    th = np.linalg.norm(rvec, axis=-1)[..., None, None]
    small = th < 1e-9
    ths = np.where(small, 1.0, th)
    a = np.where(small, 1.0 - th * th / 6.0, np.sin(ths) / ths)
    b = np.where(small, 0.5 - th * th / 24.0, (1.0 - np.cos(ths)) / (ths * ths))
    x, y, z = rvec[..., 0], rvec[..., 1], rvec[..., 2]
    o = np.zeros_like(x)
    K = np.stack([np.stack([o, -z, y], -1), np.stack([z, o, -x], -1), np.stack([-y, x, o], -1)], -2)
    return np.eye(3) + a * K + b * (K @ K)
