"""画像の入力元。どれも (名前, グレースケール画像) を順に返すので、検出・推定側は入力元を意識しない。"""
from __future__ import annotations

import glob
import time
from typing import Iterator

import cv2
import numpy as np

from .board import Board
from .model import FisheyeCamera
from .sim import Renderer, sample_pose

Frame = tuple[str, np.ndarray]


class SimSource:
    """シミュレーション。真値のカメラとボード姿勢から合成画像を作る。"""

    def __init__(self, cam: FisheyeCamera, board: Board, n_views: int, seed: int = 0,
                 blur_sigma: float = 1.0, noise_sigma: float = 2.0, supersample: int = 2):
        self.cam, self.board, self.n_views = cam, board, n_views
        self.blur_sigma, self.noise_sigma = blur_sigma, noise_sigma
        self.rng = np.random.default_rng(seed)
        self.renderer = Renderer(cam, supersample)
        self.poses = []  # 真値の (R, t)。評価用で、推定には使わない

    def __iter__(self) -> Iterator[Frame]:
        for i in range(self.n_views):
            R, t = sample_pose(self.cam, self.board, self.rng)
            self.poses.append((R, t))
            yield f"sim_{i:03d}", self.renderer.render(self.board, R, t, self.blur_sigma,
                                                       self.noise_sigma, self.rng)


class FileSource:
    """保存済み画像(glob パターン)。"""

    def __init__(self, pattern: str):
        self.paths = sorted(glob.glob(pattern))

    def __iter__(self) -> Iterator[Frame]:
        for p in self.paths:
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                yield p, img


class CameraSource:
    """実カメラ(cv2.VideoCapture)。Jetson では GStreamer パイプライン文字列を渡す想定。

    未検証: 実機が手元に無いため動作確認していない。会社で確認する。
    """

    def __init__(self, device: int | str = 0, n_frames: int = 30, api: int = cv2.CAP_ANY,
                 interval_s: float = 0.0):
        self.device, self.n_frames, self.api, self.interval_s = device, n_frames, api, interval_s

    def __iter__(self) -> Iterator[Frame]:
        cap = cv2.VideoCapture(self.device, self.api)
        try:
            for i in range(self.n_frames):
                # ボードを動かす時間を空ける。待つ間も読み捨てて、古いフレームが溜まらないようにする
                t_end = time.monotonic() + (self.interval_s if i else 0.0)
                while time.monotonic() < t_end:
                    cap.grab()
                ok, img = cap.read()
                if not ok:
                    break
                if img.ndim == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                yield f"cam_{i:03d}", img
        finally:
            cap.release()
