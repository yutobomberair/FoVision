"""ボードなし自己校正(自然特徴点+バンドル調整)の点レベルのシミュレーション評価。

実行: python scripts/run_selfcal.py [--out results/sim] [--quick]

画像は作らない。部屋の壁に散らばった点を動くカメラで観測したときの画素座標に、特徴点追跡の
誤差に見立てたガウスノイズを加えて入力にする。特徴点の検出・追跡・誤対応は含まない。
比較相手はテーマ 1 のベースライン(同じノイズを与えたチェッカーボード、25 ビュー)。
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fovision.board import Board
from fovision.calibrate import calibrate
from fovision.evaluate import compare
from fovision.model import MODEL_NAME, nominal_camera, rodrigues
from fovision.selfcal import self_calibrate
from fovision.sim import ideal_corners, perturb_camera, sample_pose

ROOM = np.array([2.0, 1.5, 2.0])  # 部屋の半サイズ [m](カメラはほぼ中央)

MOTIONS = {
    # 名前: (並進の範囲 [m], 回転の範囲 [deg], 説明)
    "handheld": (0.30, 40.0, "手持ちで自由に動かす(並進 ±0.3 m、回転 ±40°)"),
    "small_translation": (0.03, 40.0, "ほぼその場で向きだけ変える(並進 ±0.03 m、回転 ±40°)"),
    "forward": (None, 2.0, "光軸方向にまっすぐ 1 m 進む(回転 ±2°)"),
    "forward_yaw": (None, 25.0, "まっすぐ 1 m 進みながら向きを振る(回転 ±25°)"),
}


def room_points(n, rng):
    p = rng.uniform(-1, 1, (n, 3))
    ax = rng.integers(0, 3, n)
    p[np.arange(n), ax] = rng.choice([-1.0, 1.0], n)
    return p * ROOM


def trajectory(kind, n_frames, rng):
    trans, rot, _ = MOTIONS[kind]
    Rs, ts = [], []
    for i in range(n_frames):
        if trans is None:
            C = np.array([0.0, 0.0, -0.5 + i / (n_frames - 1)]) + rng.normal(0, 0.005, 3)
        else:
            C = rng.uniform(-trans, trans, 3)
        ax = rng.normal(size=3)
        R = rodrigues(ax / np.linalg.norm(ax) * np.deg2rad(rot) * rng.uniform())
        if i == 0 and trans is not None:
            C, R = np.zeros(3), np.eye(3)
        Rs.append(R)
        ts.append(-R @ C)
    return np.array(Rs), np.array(ts)


def observe(cam, kind, n_frames, n_points, sigma, rng):
    X = room_points(n_points, rng)
    Rs, ts = trajectory(kind, n_frames, rng)
    fi, pi, uv = [], [], []
    for f in range(n_frames):
        P = X @ Rs[f].T + ts[f]
        vis = np.flatnonzero(cam.visible(P, margin=2.0))
        fi.append(np.full(len(vis), f))
        pi.append(vis)
        uv.append(cam.project(P[vis]) + rng.normal(0, sigma, (len(vis), 2)))
    return np.concatenate(fi), np.concatenate(pi), np.concatenate(uv)


def baseline(cam, f_init, sigma, rng, n_views=25):
    board = Board()
    objs, imgs = [], []
    for _ in range(n_views):
        uv, _ = ideal_corners(cam, board, *sample_pose(cam, board, rng))
        objs.append(board.object_points)
        imgs.append(uv + rng.normal(0, sigma, uv.shape))
    return calibrate(objs, imgs, cam.width, cam.height, f_init, cam.fov_deg)


KEYS = ["field_max_px_0_60deg", "field_max_px_60_90deg", "field_max_px_90_101deg", "field_max_px"]


def summarize(label, rows, fails, extra=""):
    if not rows:
        print(f"  {label:34s} 全試行で失敗")
        return {"label": label, "runs": 0, "failures": fails}
    med = {k: float(np.median([abs(r[k]) if k.startswith("d_") else r[k] for r in rows]))
           for k in KEYS + ["d_fx", "d_cx", "d_cy", "rms", "time_s"]}
    worst = float(max(r["field_max_px"] for r in rows))
    print(f"  {label:34s} 成功 {len(rows)}/{len(rows) + fails} rms={med['rms']:.2f} "
          f"|dfx|={med['d_fx']:.2f} |dcx|={med['d_cx']:.2f} |dcy|={med['d_cy']:.2f}  場のずれ最大: "
          f"0-60°={med['field_max_px_0_60deg']:.2f} 60-90°={med['field_max_px_60_90deg']:.2f} "
          f"90-101°={med['field_max_px_90_101deg']:.2f} (最悪 {worst:.2f}) t={med['time_s']:.1f}s {extra}",
          flush=True)
    return {"label": label, "runs": len(rows), "failures": fails, "median": med, "worst_field_max_px": worst}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/sim")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    nominal = nominal_camera()
    seeds = range(3 if a.quick else 8)
    out = {"source": "simulation", "model": MODEL_NAME, "level": "points_only", "results": []}

    def run(label, sigma, kind=None, n_frames=20, n_points=200):
        rows, fails = [], 0
        for s in seeds:
            rng = np.random.default_rng(5000 + s)
            true = perturb_camera(nominal, rng)  # 個体差のある真値。初期値は公称値
            try:
                if kind is None:
                    r = baseline(true, nominal.fx, sigma, rng)
                else:
                    fi, pi, uv = observe(true, kind, n_frames, n_points, sigma, rng)
                    r = self_calibrate(fi, pi, uv, n_frames, n_points, true.width, true.height,
                                       nominal.fx, true.fov_deg)
                rows.append({**compare(true, r.camera), "rms": r.rms, "time_s": r.elapsed_s})
            except (ValueError, np.linalg.LinAlgError):
                fails += 1
        res = summarize(label, rows, fails)
        res.update({"sigma_px": sigma, "motion": kind, "n_frames": n_frames, "n_points": n_points})
        out["results"].append(res)

    for sigma in [0.3, 1.0]:
        print(f"[ノイズ σ = {sigma} px]")
        run("ベースライン(ボード 25 ビュー)", sigma)
        for kind in MOTIONS:
            run(f"自己校正 {kind}", sigma, kind)
    print("[フレーム数・点数の影響(handheld、σ = 1.0 px)]")
    for nf, npt in [(8, 200), (40, 200), (20, 60), (20, 500)]:
        run(f"自己校正 {nf} フレーム {npt} 点", 1.0, "handheld", nf, npt)

    out["motions"] = {k: v[2] for k, v in MOTIONS.items()}
    with open(os.path.join(a.out, "selfcal_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("saved:", os.path.join(a.out, "selfcal_summary.json"))


if __name__ == "__main__":
    main()
