"""推定結果の評価(真値との差)とパラメータの保存。"""
from __future__ import annotations

import datetime
import json

import numpy as np

from .model import MODEL_NAME, FisheyeCamera


def compare(true: FisheyeCamera, est: FisheyeCamera, step: int = 8) -> dict:
    """真値と推定値の差。

    field_*: 画像内(画角内)の各画素について、真値モデルで視線に戻し、推定モデルで投影し直した
    ときの画素のずれ。係数同士は相関して打ち消し合うので、モデルとしての差はこちらで見る。
    """
    v, u = np.mgrid[0:true.height:step, 0:true.width:step].astype(float)
    uv = np.stack([u.ravel(), v.ravel()], -1)
    rays, theta = true.unproject(uv)
    ok = theta <= true.theta_max
    d = np.linalg.norm(est.project(rays[ok]) - uv[ok], axis=-1)
    th = np.rad2deg(theta[ok])
    out = {
        "d_fx": est.fx - true.fx, "d_fy": est.fy - true.fy,
        "d_cx": est.cx - true.cx, "d_cy": est.cy - true.cy,
        "d_k": (np.asarray(est.k) - np.asarray(true.k)).tolist(),
        "field_rms_px": float(np.sqrt(np.mean(d ** 2))),
        "field_max_px": float(d.max()),
    }
    for lo, hi in [(0, 60), (60, 90), (90, 101)]:
        m = (th >= lo) & (th <= hi)
        out[f"field_max_px_{lo}_{hi}deg"] = float(d[m].max()) if m.any() else float("nan")
    return out


def save_params(path, cam: FisheyeCamera, module_id: str, source: str, extra: dict | None = None):
    """カメラパラメータを JSON で保存する。source は "simulation" または "real"。"""
    assert source in ("simulation", "real")
    data = {
        "format_version": 1,
        "module_id": module_id,
        "source": source,
        "model": MODEL_NAME,
        "image_size": [cam.width, cam.height],
        "fov_deg": cam.fov_deg,
        "fx": cam.fx, "fy": cam.fy, "cx": cam.cx, "cy": cam.cy,
        "k": [float(x) for x in cam.k],
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    data.update(extra or {})
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_params(path) -> tuple[FisheyeCamera, dict]:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    if d["model"] != MODEL_NAME:
        raise ValueError(f"未対応の投影モデル: {d['model']}")
    cam = FisheyeCamera(d["image_size"][0], d["image_size"][1], d["fx"], d["fy"], d["cx"], d["cy"],
                        np.array(d["k"], float), d["fov_deg"])
    return cam, d
