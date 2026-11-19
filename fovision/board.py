"""チェッカーボードの定義と検出。"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Board:
    cols: int = 9          # 内側コーナー数(横)
    rows: int = 6          # 内側コーナー数(縦)
    square: float = 0.03   # マス一辺 [m]

    @property
    def object_points(self) -> np.ndarray:
        """内側コーナーのボード座標 (rows*cols, 3)。原点は最初の内側コーナー、Z=0。行優先。"""
        gx, gy = np.meshgrid(np.arange(self.cols), np.arange(self.rows))
        return np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], -1) * self.square

    @property
    def center(self) -> np.ndarray:
        return np.array([(self.cols - 1) / 2.0, (self.rows - 1) / 2.0, 0.0]) * self.square


def detect_corners(gray: np.ndarray, board: Board):
    """チェッカーボードの内側コーナーを検出する。成功時 (N,2) float64、失敗時 None。

    並び順は object_points と同じ行優先。ボードの 180° 回転の取り違えは起こりうるが、
    内側コーナーの格子は 180° 回転対称なので、姿勢が変わるだけでキャリブレーションには影響しない。
    """
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    ok, corners = cv2.findChessboardCornersSB(gray, (board.cols, board.rows), flags=flags)
    if not ok or corners is None or len(corners) != board.cols * board.rows:
        return None
    return corners.reshape(-1, 2).astype(np.float64)
