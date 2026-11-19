"""テーマ 1 ベースライン(チェッカーボード)のシミュレーション評価。

実行: python scripts/run_baseline.py [--out results/sim] [--quick]
結果はすべてシミュレーション由来(source="simulation")。
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fovision.board import Board, detect_corners
from fovision.calibrate import calibrate
from fovision.evaluate import compare, save_params
from fovision.model import MODEL_NAME, FisheyeCamera, nominal_camera
from fovision.sim import ideal_corners, perturb_camera, sample_pose
from fovision.sources import SimSource

BOARD = Board()
F_INIT_SCALE = 1.08  # 初期値は真値から 8% ずらして与える(公称値と個体の差を想定)


def cv2_fisheye(objs, imgs, cam_init: FisheyeCamera, theta_list, limit_deg=88.0):
    """比較用: cv2.fisheye.calibrate。90° 以上の点を扱えないので、該当ビューは捨てる。"""
    keep = [i for i, th in enumerate(theta_list) if np.rad2deg(th.max()) < limit_deg]
    o = [objs[i].reshape(-1, 1, 3).astype(np.float32) for i in keep]
    p = [imgs[i].reshape(-1, 1, 2).astype(np.float32) for i in keep]
    K = np.array([[cam_init.fx, 0, cam_init.cx], [0, cam_init.fy, cam_init.cy], [0, 0, 1.0]])
    flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
             | cv2.fisheye.CALIB_USE_INTRINSIC_GUESS)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-9)
    rms, Kc, Dc, _, _ = cv2.fisheye.calibrate(o, p, (cam_init.width, cam_init.height), K, np.zeros(4),
                                              flags=flags, criteria=crit)
    return FisheyeCamera(cam_init.width, cam_init.height, Kc[0, 0], Kc[1, 1], Kc[0, 2], Kc[1, 2],
                         Dc.ravel(), cam_init.fov_deg), rms, len(keep)


def point_views(cam, n_views, sigma, rng):
    objs, imgs, thetas = [], [], []
    for _ in range(n_views):
        R, t = sample_pose(cam, BOARD, rng)
        uv, th = ideal_corners(cam, BOARD, R, t)
        objs.append(BOARD.object_points)
        imgs.append(uv + rng.normal(0, sigma, uv.shape))
        thetas.append(th)
    return objs, imgs, thetas


def stat(rows, key):
    v = np.array([r[key] for r in rows], float)
    return {"median": float(np.median(v)), "max": float(v.max())}


def exp_points(cam, sigmas, n_views_list, seeds):
    """A: 理想コーナー+ガウスノイズ。推定器そのものの精度を見る。"""
    out = []
    f0 = cam.fx * F_INIT_SCALE
    for n_views in n_views_list:
        for sigma in sigmas:
            res = {"own_k4": [], "own_k2": [], "cv2_fisheye_lt88deg": []}
            for seed in seeds:
                objs, imgs, thetas = point_views(cam, n_views, sigma, np.random.default_rng(seed))
                for name, nk in [("own_k4", 4), ("own_k2", 2)]:
                    r = calibrate(objs, imgs, cam.width, cam.height, f0, cam.fov_deg, n_k=nk)
                    res[name].append({**compare(cam, r.camera), "rms": r.rms, "time_s": r.elapsed_s,
                                      "views": len(r.view_rms)})
                try:
                    t0 = time.perf_counter()
                    init = FisheyeCamera(cam.width, cam.height, f0, f0, cam.cx, cam.cy, np.zeros(4), cam.fov_deg)
                    est, rms, nv = cv2_fisheye(objs, imgs, init, thetas)
                    res["cv2_fisheye_lt88deg"].append({**compare(cam, est), "rms": rms, "views": nv,
                                                       "time_s": time.perf_counter() - t0})
                except cv2.error:
                    pass
            for name, rows in res.items():
                if not rows:
                    out.append({"n_views": n_views, "sigma_px": sigma, "method": name, "runs": 0})
                    continue
                out.append({
                    "n_views": n_views, "sigma_px": sigma, "method": name, "runs": len(rows),
                    "views_used_median": float(np.median([r["views"] for r in rows])),
                    "rms_px": stat(rows, "rms"), "abs_d_fx": stat([{"x": abs(r["d_fx"])} for r in rows], "x"),
                    "abs_d_cx": stat([{"x": abs(r["d_cx"])} for r in rows], "x"),
                    "abs_d_cy": stat([{"x": abs(r["d_cy"])} for r in rows], "x"),
                    "field_rms_px": stat(rows, "field_rms_px"), "field_max_px": stat(rows, "field_max_px"),
                    "field_max_px_0_60deg": stat(rows, "field_max_px_0_60deg"),
                    "field_max_px_60_90deg": stat(rows, "field_max_px_60_90deg"),
                    "field_max_px_90_101deg": stat(rows, "field_max_px_90_101deg"),
                    "time_s": stat(rows, "time_s"),
                })
                o = out[-1]
                print(f"  views={n_views:2d} sigma={sigma:.1f} {name:20s} runs={o['runs']:2d} "
                      f"rms={o['rms_px']['median']:.3f} |dfx|={o['abs_d_fx']['median']:.3f} "
                      f"field max: 0-60°={o['field_max_px_0_60deg']['median']:.3f} "
                      f"60-90°={o['field_max_px_60_90deg']['median']:.3f} "
                      f"90-101°={o['field_max_px_90_101deg']['median']:.3f} "
                      f"(worst {o['field_max_px']['max']:.2f}) t={o['time_s']['median']:.2f}s", flush=True)
    return out


def run_images(cam, n_views, seed, blur, noise, overlay_path=None):
    """画像を合成 -> コーナー検出 -> 推定。検出精度・検出率・推定精度・処理時間を返す。"""
    src = SimSource(cam, BOARD, n_views, seed, blur, noise)
    objs, imgs, errs, t_det, n_gt90, overlay = [], [], [], [], 0, None
    for i, (name, img) in enumerate(src):
        t0 = time.perf_counter()
        c = detect_corners(img, BOARD)
        t_det.append(time.perf_counter() - t0)
        uv, th = ideal_corners(cam, BOARD, *src.poses[i])
        if c is None:
            continue
        # 精度評価のためだけに真値と突き合わせる(並びは正順か 180° 回転のどちらか)
        e = min(np.linalg.norm(c - uv, axis=1), np.linalg.norm(c[::-1] - uv, axis=1), key=lambda x: x.max())
        errs.append(e)
        objs.append(BOARD.object_points)
        imgs.append(c)
        n_gt90 += int(np.rad2deg(th.max()) > 90)
        if overlay_path and (overlay is None or np.rad2deg(th.max()) > 90) and i < 12:
            overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.drawChessboardCorners(overlay, (BOARD.cols, BOARD.rows), c.astype(np.float32).reshape(-1, 1, 2), True)
    if overlay_path and overlay is not None:
        cv2.imwrite(overlay_path, overlay)
    # 初期値は常に公称値(真値は使わない)
    f_init = nominal_camera(cam.width, cam.height, cam.fov_deg).fx
    r = calibrate(objs, imgs, cam.width, cam.height, f_init, cam.fov_deg, n_k=4)
    errs = np.concatenate(errs)
    return r, {
        "blur_sigma_px": blur, "noise_sigma_level": noise, "n_views": n_views,
        "detected_views": len(objs), "views_with_points_over_90deg": n_gt90,
        "corner_err_rms_px": float(np.sqrt(np.mean(errs ** 2))), "corner_err_max_px": float(errs.max()),
        "rms_px": r.rms, "detect_time_s_per_image": float(np.mean(t_det)), "calib_time_s": r.elapsed_s,
        "iterations": r.iterations, **compare(cam, r.camera),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/sim")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "params"), exist_ok=True)
    nominal = nominal_camera()
    summary = {"source": "simulation", "model": MODEL_NAME, "opencv": cv2.__version__,
               "nominal": {"fx": nominal.fx, "cx": nominal.cx, "cy": nominal.cy, "fov_deg": nominal.fov_deg,
                           "image_size": [nominal.width, nominal.height]},
               "board": {"cols": BOARD.cols, "rows": BOARD.rows, "square_m": BOARD.square}}

    print("[A] 理想コーナー+ノイズ(真値: 公称の等距離射影)")
    seeds = range(3 if a.quick else 10)
    summary["A_points"] = exp_points(nominal, [0.1, 0.3, 1.0], [25], seeds)
    summary["A_points"] += exp_points(nominal, [0.3], [10, 50], seeds)

    print("[B] 合成画像 -> 検出 -> 推定(真値: 公称の等距離射影、25 ビュー)")
    summary["B_images"] = []
    for blur, noise in [(1.0, 2.0), (2.0, 5.0), (3.0, 10.0)]:
        _, row = run_images(nominal, 25, 100, blur, noise,
                            os.path.join(a.out, "sample_detection.png") if blur == 1.0 else None)
        summary["B_images"].append(row)
        print(f"  blur={blur} noise={noise}: 検出 {row['detected_views']}/{row['n_views']} "
              f"(90°超を含む {row['views_with_points_over_90deg']}) コーナー誤差 rms={row['corner_err_rms_px']:.3f} "
              f"max={row['corner_err_max_px']:.2f} | 再投影 rms={row['rms_px']:.3f} d_fx={row['d_fx']:+.3f} "
              f"d_cx={row['d_cx']:+.3f} d_cy={row['d_cy']:+.3f} field rms={row['field_rms_px']:.3f} "
              f"max={row['field_max_px']:.3f} | 検出 {row['detect_time_s_per_image']:.2f}s/枚 "
              f"推定 {row['calib_time_s']:.2f}s", flush=True)

    print("[C] 個体差(公称からパラメータをばらつかせた仮想個体、初期値は公称値)")
    summary["C_units"] = []
    rng = np.random.default_rng(2026)
    for u in range(2 if a.quick else 5):
        module_id = f"SIM-{u + 1:04d}"
        true = perturb_camera(nominal, rng)
        r, row = run_images(true, 25, 200 + u, 1.0, 2.0)
        row["module_id"] = module_id
        row["true"] = {"fx": true.fx, "fy": true.fy, "cx": true.cx, "cy": true.cy, "k": true.k.tolist()}
        row["est"] = {"fx": r.camera.fx, "fy": r.camera.fy, "cx": r.camera.cx, "cy": r.camera.cy,
                      "k": r.camera.k.tolist()}
        summary["C_units"].append(row)
        save_params(os.path.join(a.out, "params", f"{module_id}.json"), r.camera, module_id, "simulation",
                    {"method": "checkerboard_baseline", "rms_px": r.rms, "n_views": row["detected_views"],
                     "n_points": r.n_points, "intr_std": r.intr_std.tolist()})
        print(f"  {module_id}: 真値 fx={true.fx:.2f} fy={true.fy:.2f} cx={true.cx:.2f} cy={true.cy:.2f} "
              f"k1={true.k[0]:+.4f} k2={true.k[1]:+.4f} | 差 d_fx={row['d_fx']:+.3f} d_fy={row['d_fy']:+.3f} "
              f"d_cx={row['d_cx']:+.3f} d_cy={row['d_cy']:+.3f} | 検出 {row['detected_views']}/25 "
              f"rms={row['rms_px']:.3f} field rms={row['field_rms_px']:.3f} max={row['field_max_px']:.3f}",
              flush=True)

    with open(os.path.join(a.out, "baseline_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("saved:", os.path.join(a.out, "baseline_summary.json"))


if __name__ == "__main__":
    main()
