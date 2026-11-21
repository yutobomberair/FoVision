"""歪み補正(出力投影を選んで再配置する)。

出力画像の各画素 -> 視線ベクトル -> 魚眼モデルで元画像の画素、の対応表を作って cv2.remap する。
出力投影:
  equirect       正距円筒。横が経度、縦が緯度。180° 超もそのまま入る
  cylindrical    円筒。横が経度、縦は透視投影。縦線がまっすぐになる
  stereographic  ステレオ投影。180° 超も入り、局所的な形が保たれる
  perspective    透視投影。直線がまっすぐになるが 180° 未満しか入らない
"""
from __future__ import annotations

import cv2
import numpy as np

from .model import FisheyeCamera

PROJECTIONS = ("equirect", "cylindrical", "stereographic", "perspective")


def build_map(cam: FisheyeCamera, proj: str = "equirect", width: int = 1600,
              hfov_deg: float = 200.0, vfov_deg: float = 150.0):
    """(map_x, map_y, 出力サイズ) を返す。元画像に像が無い画素は -1。

    hfov_deg / vfov_deg は出力に収める水平・垂直の画角。高さは投影に合わせて自動で決まる。
    """
    hh, hv = np.deg2rad(hfov_deg) / 2, np.deg2rad(vfov_deg) / 2
    if proj == "equirect":
        height = int(round(width * hv / hh))
        lon = (np.arange(width) + 0.5) / width * 2 * hh - hh
        lat = (np.arange(height) + 0.5) / height * 2 * hv - hv
        lon, lat = np.meshgrid(lon, lat)
        rays = np.stack([np.cos(lat) * np.sin(lon), np.sin(lat), np.cos(lat) * np.cos(lon)], -1)
    elif proj == "cylindrical":
        hv = min(hv, np.deg2rad(60))  # 縦は透視投影なので広げすぎると伸びる
        height = int(round(width * np.tan(hv) / hh))
        lon = (np.arange(width) + 0.5) / width * 2 * hh - hh
        y = ((np.arange(height) + 0.5) / height * 2 - 1) * np.tan(hv)
        lon, y = np.meshgrid(lon, y)
        rays = np.stack([np.sin(lon), y, np.cos(lon)], -1)
    elif proj in ("stereographic", "perspective"):
        if proj == "perspective":
            hh, hv = min(hh, np.deg2rad(80)), min(hv, np.deg2rad(80))
            rx, ry = np.tan(hh), np.tan(hv)
        else:
            rx, ry = 2 * np.tan(hh / 2), 2 * np.tan(hv / 2)
        height = int(round(width * ry / rx))
        x = ((np.arange(width) + 0.5) / width * 2 - 1) * rx
        y = ((np.arange(height) + 0.5) / height * 2 - 1) * ry
        x, y = np.meshgrid(x, y)
        r = np.hypot(x, y)
        theta = np.arctan(r) if proj == "perspective" else 2 * np.arctan(r / 2)
        s = np.sin(theta) / np.maximum(r, 1e-12)
        rays = np.stack([s * x, s * y, np.cos(theta)], -1)
    else:
        raise ValueError(f"未対応の出力投影: {proj}")
    uv = cam.project(rays)
    theta = np.arctan2(np.hypot(rays[..., 0], rays[..., 1]), rays[..., 2])
    bad = ((theta > cam.theta_max) | (uv[..., 0] < 0) | (uv[..., 0] > cam.width - 1)
           | (uv[..., 1] < 0) | (uv[..., 1] > cam.height - 1))
    uv[bad] = -1
    return uv[..., 0].astype(np.float32), uv[..., 1].astype(np.float32), (width, height)


def undistort(img: np.ndarray, cam: FisheyeCamera, proj: str = "equirect", **kw) -> np.ndarray:
    mx, my, _ = build_map(cam, proj, **kw)
    return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
