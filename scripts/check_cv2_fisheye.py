"""cv2.fisheye / cv2.omnidir が入射角 90° 超を扱えるかの確認(真値既知の往復テスト)。

実行: python scripts/check_cv2_fisheye.py
"""
import os
import re
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fovision.board import Board
from fovision.calibrate import calibrate
from fovision.evaluate import compare
from fovision.model import FisheyeCamera, nominal_camera
from fovision.sim import ideal_corners, sample_pose

cam = nominal_camera()
K = np.array([[cam.fx, 0, cam.cx], [0, cam.fy, cam.cy], [0, 0, 1.0]])
D = np.zeros(4)
print(f"OpenCV {cv2.__version__} / 真値: 等距離射影 f={cam.fx:.3f} cx={cam.cx} cy={cam.cy}")

print("\n[1] 投影 cv2.fisheye.projectPoints と自前モデルの差(方位 0°、Y=0 の点)")
for deg in [30, 60, 85, 89, 91, 95, 100]:
    th = np.deg2rad(deg)
    P = np.array([[np.sin(th), 0.0, np.cos(th)]])
    mine = cam.project(P)[0]
    cvp = cv2.fisheye.projectPoints(P.reshape(1, 1, 3), np.zeros(3), np.zeros(3), K, D)[0].ravel()
    print(f"  theta={deg:3d}°  自前 u={mine[0]:8.2f}  cv2 u={cvp[0]:8.2f}  差={cvp[0] - mine[0]:9.2f} px")

print("\n[2] 逆投影 cv2.fisheye.undistortPoints(正規化座標 x/z に戻るか)")
for deg in [30, 60, 85, 89]:
    th = np.deg2rad(deg)
    uv = cam.project(np.array([[np.sin(th), 0.0, np.cos(th)]])).reshape(1, 1, 2)
    und = cv2.fisheye.undistortPoints(uv, K, D).ravel()
    print(f"  theta={deg:3d}°  期待 x/z={np.tan(th):9.3f}  cv2={und[0]:9.3f}")
print("  (90° 以上は x/z が定義できないため原理的に表現不可)")


def make_views(n, theta_limit_deg, seed):
    rng = np.random.default_rng(seed)
    board = Board()
    objs, imgs, tmax = [], [], 0.0
    while len(objs) < n:
        R, t = sample_pose(cam, board, rng)
        uv, th = ideal_corners(cam, board, R, t)
        if np.rad2deg(th.max()) > theta_limit_deg:
            continue
        objs.append(board.object_points)
        imgs.append(uv)
        tmax = max(tmax, np.rad2deg(th.max()))
    return objs, imgs, tmax


print("\n[3] キャリブレーション(ノイズ無しの理想コーナー、25 ビュー)")
for label, lim in [("全点 80° 以内", 80), ("全点 89° 以内", 89), ("90° 超を含む(101° まで)", 101)]:
    objs, imgs, tmax = make_views(25, lim, seed=1)
    print(f"  --- {label}(最大入射角 {tmax:.1f}°)")
    r = calibrate(objs, imgs, cam.width, cam.height, f_init=cam.fx * 1.1)
    c = compare(cam, r.camera)
    print(f"    自前      : rms={r.rms:.2e} px  場のずれ max={c['field_max_px']:.2e} px  d_fx={c['d_fx']:+.2e}")
    o32 = [o.reshape(-1, 1, 3).astype(np.float32) for o in objs]
    i32 = [p.reshape(-1, 1, 2).astype(np.float32) for p in imgs]
    base = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
    K0 = np.array([[cam.fx * 1.1, 0, cam.cx], [0, cam.fx * 1.1, cam.cy], [0, 0, 1.0]])
    for name, flags, Kin in [("cv2.fisheye(初期値なし)", base, None),
                             ("cv2.fisheye(初期値あり)", base | cv2.fisheye.CALIB_USE_INTRINSIC_GUESS, K0)]:
        try:
            crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-9)
            rms, Kc, Dc, _, _ = cv2.fisheye.calibrate(
                o32, i32, (cam.width, cam.height), None if Kin is None else Kin.copy(),
                None if Kin is None else np.zeros(4), flags=flags, criteria=crit)
            est = FisheyeCamera(cam.width, cam.height, Kc[0, 0], Kc[1, 1], Kc[0, 2], Kc[1, 2],
                                Dc.ravel(), cam.fov_deg)
            c = compare(cam, est)
            print(f"    {name}: rms={rms:.2e} px  場のずれ max={c['field_max_px']:.2e} px  d_fx={c['d_fx']:+.2e}")
        except cv2.error as e:
            msg = " ".join(x.strip() for x in str(e).strip().splitlines())
            m = re.search(r"(Too few image points[^(]*|Cannot compute the initial pose[^:]*)", msg)
            print(f"    {name}: 失敗 ({m.group(1).strip() if m else msg[-100:]})")
    if hasattr(cv2, "omnidir"):
        try:
            crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-9)
            o = [x.astype(np.float64) for x in o32]
            i = [x.astype(np.float64) for x in i32]
            out = cv2.omnidir.calibrate(o, i, (cam.width, cam.height), None, None, None,
                                        cv2.omnidir.CALIB_FIX_SKEW, crit)
            rms, Ko, xi, Do, idx = out[0], out[1], out[2], out[3], out[-1]
            # omnidir はモデルが異なる(単位球+xi)ので、係数ではなく再投影誤差と使えたビュー数だけ見る
            print(f"    cv2.omnidir: rms={rms:.2e} px  xi={float(np.ravel(xi)[0]):.3f}  "
                  f"使えたビュー {np.size(idx)}/{len(o)}")
        except cv2.error as e:
            print(f"    cv2.omnidir: 失敗 ({str(e).strip().splitlines()[-1][:110]})")
