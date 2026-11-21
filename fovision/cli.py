"""キャリブレーションツール(コマンドライン)。

    python -m fovision calibrate --images "shots/*.png" --module-id M0001 --source real --out out/
    python -m fovision undistort --params out/M0001.json --image shots/a.png --out a_eq.png
    python -m fovision simulate  --out sim/SIM-0001 --module-id SIM-0001     # 実機なしで試す用
    python -m fovision evaluate  --params out/SIM-0001.json --truth sim/SIM-0001/truth.json

各コマンドの引数は `python -m fovision <コマンド> -h`。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

from .board import Board, detect_corners
from .calibrate import calibrate_auto
from .evaluate import compare, load_params, save_params
from .model import FisheyeCamera, theta_d
from .sim import perturb_camera
from .sources import CameraSource, FileSource, SimSource
from .undistort import PROJECTIONS, undistort

GRID = (8, 6)  # 画像を覆えているかを見る格子(横 x 縦)


def _board(a) -> Board:
    cols, rows = (int(x) for x in a.board.lower().split("x"))
    return Board(cols, rows, a.square)


def _coverage(cam: FisheyeCamera, pts: np.ndarray) -> dict:
    """検出コーナーが画像と入射角をどれだけ覆っているか。"""
    gx = np.clip((pts[:, 0] / cam.width * GRID[0]).astype(int), 0, GRID[0] - 1)
    gy = np.clip((pts[:, 1] / cam.height * GRID[1]).astype(int), 0, GRID[1] - 1)
    cells = np.zeros(GRID[::-1], int)
    np.add.at(cells, (gy, gx), 1)
    # 像が届く格子だけを分母にする(イメージサークルの外は数えない)
    cx = (np.arange(GRID[0]) + 0.5) / GRID[0] * cam.width
    cy = (np.arange(GRID[1]) + 0.5) / GRID[1] * cam.height
    cxx, cyy = np.meshgrid(cx, cy)
    lit = cam.unproject(np.stack([cxx, cyy], -1))[1] <= cam.theta_max
    theta = np.rad2deg(cam.unproject(pts)[1])
    bands = {f"{lo}-{hi}deg": int(((theta >= lo) & (theta < hi)).sum())
             for lo, hi in [(0, 30), (30, 60), (60, 90), (90, 181)]}
    return {"grid": list(GRID), "cells_with_points": int(((cells > 0) & lit).sum()),
            "cells_total": int(lit.sum()), "points_per_cell": cells.tolist(),
            "points_by_incidence_angle": bands, "max_incidence_angle_deg": float(theta.max())}


def _draw_coverage(path, cam, views, view_rms, bad_thresh):
    img = np.zeros((cam.height, cam.width, 3), np.uint8)
    for i in range(1, GRID[0]):
        cv2.line(img, (i * cam.width // GRID[0], 0), (i * cam.width // GRID[0], cam.height), (50, 50, 50), 1)
    for j in range(1, GRID[1]):
        cv2.line(img, (0, j * cam.height // GRID[1]), (cam.width, j * cam.height // GRID[1]), (50, 50, 50), 1)
    for pts, rms in zip(views, view_rms):
        col = (0, 200, 0) if rms <= bad_thresh else (0, 0, 255)
        for p in pts:
            cv2.circle(img, (int(round(p[0])), int(round(p[1]))), 4, col, -1)
    cv2.circle(img, (int(round(cam.cx)), int(round(cam.cy))), 12, (0, 255, 255), 2)
    cv2.imwrite(path, img)


def cmd_calibrate(a):
    board = _board(a)
    if a.camera is not None:
        dev = int(a.camera) if a.camera.isdigit() else a.camera
        src = CameraSource(dev, a.frames, interval_s=a.interval)
    else:
        src = FileSource(a.images)
        if not src.paths:
            sys.exit(f"画像が見つからない: {a.images}")
    os.makedirs(a.out, exist_ok=True)

    names, corners, missed, size, t_det = [], [], [], None, 0.0
    for name, img in src:
        if size is None:
            size = (img.shape[1], img.shape[0])
        elif size != (img.shape[1], img.shape[0]):
            sys.exit(f"画像サイズが混在している: {name}")
        if a.camera is not None and a.save_frames:
            cv2.imwrite(os.path.join(a.out, f"{a.module_id}_{name}.png"), img)
        t0 = time.perf_counter()
        c = detect_corners(img, board)
        t_det += time.perf_counter() - t0
        print(f"  {os.path.basename(str(name))}: {'検出' if c is not None else '検出できず'}", flush=True)
        if c is None:
            missed.append(str(name))
        else:
            names.append(str(name))
            corners.append(c)
    if len(corners) < 5:
        sys.exit(f"ボードを検出できた画像が {len(corners)} 枚しかない(最低 5 枚、推奨 20 枚以上)")

    obj = [board.object_points] * len(corners)
    kw = dict(fov_deg=a.fov, n_k=a.nk)
    r, f0 = calibrate_auto(obj, corners, size[0], size[1], a.f_init, **kw)
    # 再投影誤差が飛び抜けて大きいビュー(誤検出など)を外して一度だけやり直す
    thresh = max(3.0 * float(np.median(r.view_rms)), 1.0)
    rejected = [names[v] for v, e in zip(r.views, r.view_rms) if e > thresh]
    rejected += [n for i, n in enumerate(names) if i not in r.views]
    if rejected and len(names) - len(rejected) >= 5:
        keep = [i for i, n in enumerate(names) if n not in rejected]
        names, corners = [names[i] for i in keep], [corners[i] for i in keep]
        r, _ = calibrate_auto([board.object_points] * len(corners), corners, size[0], size[1],
                              float(r.camera.fx), **kw)
    cam = r.camera
    used = [corners[v] for v in r.views]
    cov = _coverage(cam, np.concatenate(used))

    warnings = []
    if len(used) < 15:
        warnings.append(f"使えた画像が {len(used)} 枚と少ない(推奨 20 枚以上)")
    if cov["cells_with_points"] < 0.7 * cov["cells_total"]:
        warnings.append(f"画像の {cov['cells_with_points']}/{cov['cells_total']} 区画にしかコーナーが無い。"
                        "空いている区画にボードを写した画像を足す")
    edge = np.rad2deg(min(cam.theta_max, float(cam.unproject(np.array([0.0, 0.0]))[1])))
    if cov["max_incidence_angle_deg"] < 0.85 * edge:
        warnings.append(f"コーナーの最大入射角が {cov['max_incidence_angle_deg']:.0f}° で、画像に写る範囲"
                        f"(約 {edge:.0f}°)の外周が拘束されていない")
    th = np.linspace(0, cam.theta_max, 400)
    if np.any(np.diff(theta_d(th, cam.k)) <= 0):
        warnings.append("推定した歪み係数では画角内で像高が単調増加にならない。--nk を減らして再実行する")
    if r.rms > 1.0:
        warnings.append(f"再投影誤差 RMS が {r.rms:.2f} px と大きい")

    base = os.path.join(a.out, a.module_id)
    save_params(base + ".json", cam, a.module_id, a.source,
                {"method": "checkerboard", "rms_px": r.rms, "n_views": len(used), "n_points": r.n_points,
                 "intr_std": r.intr_std.tolist()})
    report = {
        "module_id": a.module_id, "source": a.source, "method": "checkerboard",
        "board": {"cols": board.cols, "rows": board.rows, "square_m": board.square},
        "images_total": len(names) + len(missed) + len(rejected), "images_board_not_detected": missed,
        "images_rejected": rejected, "images_used": [names[v] for v in r.views],
        "rms_px": r.rms, "view_rms_px": {names[v]: float(e) for v, e in zip(r.views, r.view_rms)},
        "f_init_px": float(f0), "coverage": cov, "warnings": warnings,
        "detect_time_s_per_image": t_det / max(len(names) + len(missed) + len(rejected), 1),
        "calib_time_s": r.elapsed_s, "opencv": cv2.__version__, "numpy": np.__version__,
    }
    with open(base + "_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    _draw_coverage(base + "_coverage.png", cam, used, r.view_rms, thresh)

    sd = r.intr_std
    print(f"\n個体 {a.module_id}({a.source}): {len(used)} 枚を使用、検出できず {len(missed)} 枚、除外 {len(rejected)} 枚")
    print(f"  fx = {cam.fx:9.3f} ± {sd[0]:.3f} px    fy = {cam.fy:9.3f} ± {sd[1]:.3f} px")
    print(f"  cx = {cam.cx:9.3f} ± {sd[2]:.3f} px    cy = {cam.cy:9.3f} ± {sd[3]:.3f} px")
    print("  k  = " + ", ".join(f"{x:+.5f}" for x in cam.k))
    print(f"  再投影誤差 RMS = {r.rms:.3f} px    コーナーのある区画 {cov['cells_with_points']}/{cov['cells_total']}"
          f"    最大入射角 {cov['max_incidence_angle_deg']:.0f}°")
    for w in warnings:
        print("  [注意] " + w)
    print(f"  保存: {base}.json, {base}_report.json, {base}_coverage.png")


def cmd_undistort(a):
    cam, _ = load_params(a.params)
    img = cv2.imread(a.image, cv2.IMREAD_UNCHANGED)
    if img is None:
        sys.exit(f"画像を読めない: {a.image}")
    if (img.shape[1], img.shape[0]) != (cam.width, cam.height):
        sys.exit(f"画像サイズ {img.shape[1]}x{img.shape[0]} がパラメータの {cam.width}x{cam.height} と違う")
    out = undistort(img, cam, a.proj, width=a.width, hfov_deg=a.hfov, vfov_deg=a.vfov)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    cv2.imwrite(a.out, out)
    print(f"保存: {a.out}({a.proj}, {out.shape[1]}x{out.shape[0]})")


def cmd_simulate(a):
    board = _board(a)
    w, h = (int(x) for x in a.size.lower().split("x"))
    f = a.f if a.f else (w / 2.0) / np.deg2rad(a.fov / 2.0)
    cam = FisheyeCamera(w, h, f, f, (w - 1) / 2.0, (h - 1) / 2.0, np.zeros(4), a.fov)
    if a.perturb:
        cam = perturb_camera(cam, np.random.default_rng(a.seed))
    os.makedirs(a.out, exist_ok=True)
    for name, img in SimSource(cam, board, a.n, a.seed, a.blur, a.noise):
        cv2.imwrite(os.path.join(a.out, name + ".png"), img)
    save_params(os.path.join(a.out, "truth.json"), cam, a.module_id, "simulation", {"is_truth": True})
    print(f"保存: {a.out}/sim_*.png({a.n} 枚)と truth.json")


def cmd_evaluate(a):
    est, d = load_params(a.params)
    true, t = load_params(a.truth)
    if not t.get("is_truth"):
        sys.exit("--truth には simulate が出力した truth.json を指定する")
    c = compare(true, est)
    print(f"個体 {d['module_id']}: 真値との差(シミュレーションのみ)")
    print(f"  d_fx = {c['d_fx']:+.3f}  d_fy = {c['d_fy']:+.3f}  d_cx = {c['d_cx']:+.3f}  d_cy = {c['d_cy']:+.3f} px")
    print(f"  場のずれ RMS = {c['field_rms_px']:.3f} px、最大 = {c['field_max_px']:.3f} px")
    for k, v in c.items():
        if k.startswith("field_max_px_") and np.isfinite(v):
            print(f"    入射角 {k[13:-3].replace('_', '〜')}°: 最大 {v:.3f} px")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fovision", description="魚眼カメラモジュールのキャリブレーションツール")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def board_args(p):
        p.add_argument("--board", default="9x6", help="内側コーナー数 横x縦(既定 9x6)")
        p.add_argument("--square", type=float, default=0.03, help="マス一辺 [m](既定 0.03)")

    p = sub.add_parser("calibrate", help="チェッカーボード画像からカメラパラメータを推定する")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--images", help='画像の glob パターン(例 "shots/*.png")')
    g.add_argument("--camera", help="カメラ番号または GStreamer パイプライン(実機では未検証)")
    p.add_argument("--module-id", required=True, help="モジュール個体の識別子")
    p.add_argument("--source", required=True, choices=["real", "simulation"], help="画像の由来")
    p.add_argument("--out", default="results/calib", help="出力先ディレクトリ")
    board_args(p)
    p.add_argument("--fov", type=float, default=202.0, help="レンズの全画角 [deg]")
    p.add_argument("--nk", type=int, default=4, choices=range(5), help="推定する歪み係数の個数")
    p.add_argument("--f-init", type=float, help="焦点距離の初期値 [px]。省略時は候補を試して自動で選ぶ")
    p.add_argument("--frames", type=int, default=30, help="--camera のとき取り込む枚数")
    p.add_argument("--interval", type=float, default=2.0, help="--camera のとき取り込み間隔 [秒]")
    p.add_argument("--save-frames", action="store_true", help="--camera のとき取り込んだ画像も保存する")
    p.set_defaults(fn=cmd_calibrate)

    p = sub.add_parser("undistort", help="推定したパラメータで画像の歪みを補正する")
    p.add_argument("--params", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--proj", default="equirect", choices=PROJECTIONS, help="出力投影")
    p.add_argument("--width", type=int, default=1600, help="出力画像の幅 [px]")
    p.add_argument("--hfov", type=float, default=200.0, help="出力に収める水平画角 [deg]")
    p.add_argument("--vfov", type=float, default=150.0, help="出力に収める垂直画角 [deg]")
    p.set_defaults(fn=cmd_undistort)

    p = sub.add_parser("simulate", help="合成のチェッカーボード画像一式を作る(実機なしで試す用)")
    p.add_argument("--out", required=True)
    p.add_argument("--module-id", default="SIM-0001")
    p.add_argument("--n", type=int, default=25, help="枚数")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--size", default="1920x1536")
    p.add_argument("--fov", type=float, default=202.0)
    p.add_argument("--f", type=float, help="焦点距離 [px]。省略時はイメージサークルが画像幅いっぱいになる値")
    p.add_argument("--perturb", action="store_true", help="個体差を与える")
    p.add_argument("--blur", type=float, default=1.0)
    p.add_argument("--noise", type=float, default=2.0)
    board_args(p)
    p.set_defaults(fn=cmd_simulate)

    p = sub.add_parser("evaluate", help="推定結果をシミュレーションの真値と比べる")
    p.add_argument("--params", required=True)
    p.add_argument("--truth", required=True)
    p.set_defaults(fn=cmd_evaluate)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
