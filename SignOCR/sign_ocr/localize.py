"""Sign localization: Depth-Anything-3 metric depth -> camera-relative 3D -> (optional) GPS.

    best frame of a track
      -> DA3METRIC-LARGE monocular metric depth (meters)
      -> median depth inside the sign box
      -> pinhole back-projection with camera intrinsics
         => forward Z, lateral X (right +), height above camera (up +)
      -> if a GPS log is supplied: vehicle lat/lon/heading at that timestamp
         + rotated offset => sign lat/lon

Depth Anything 3 is loaded from source (third_party/Depth-Anything-3/src) rather than
pip-installed: its package metadata pins numpy<2 and pulls xformers, which would
replace the cu128 torch build. Only the inference path is used.

Metric scaling (DA3 README): metric_depth = focal_px * net_output / 300, where focal_px
is the focal length at the resolution the network actually saw.

Camera intrinsics: the dashcam is uncalibrated. By default the focal length is
estimated once per video with DA3-BASE (which predicts intrinsics); `--focal-px`
overrides it and `--hfov` is the fallback if estimation fails.
"""

from __future__ import annotations

import csv
import importlib.util
import math
import os
import sys
import types
from typing import List, Optional

import cv2
import numpy as np

_DA3_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "third_party", "Depth-Anything-3", "src")


class _ExportOnlyStub(types.ModuleType):
    """Placeholder for packages DA3 imports eagerly but only uses for 3D export
    (COLMAP / GLB / PLY / pose alignment), none of which this pipeline calls."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        def _unavailable(*a, **k):
            raise RuntimeError(f"{self.__name__}.{name} is stubbed (DA3 export-only dependency)")
        return _unavailable


def _import_da3():
    if _DA3_SRC not in sys.path:
        sys.path.insert(0, _DA3_SRC)
    for mod in ("pycolmap", "trimesh", "plyfile", "evo", "evo.core", "evo.core.trajectory"):
        if mod not in sys.modules and importlib.util.find_spec(mod.split(".")[0]) is None:
            sys.modules[mod] = _ExportOnlyStub(mod)
    from depth_anything_3.api import DepthAnything3
    return DepthAnything3


def focal_from_hfov(width: int, hfov_deg: float) -> float:
    return (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)


def estimate_focal_da3(frames_bgr: List[np.ndarray], device="cuda",
                       model_id="depth-anything/DA3-BASE", process_res=504) -> float:
    """Median predicted fx over a few frames, rescaled to full frame width."""
    import torch
    DepthAnything3 = _import_da3()
    model = DepthAnything3.from_pretrained(model_id).to(device).eval()
    fxs = []
    with torch.no_grad():
        for fr in frames_bgr:  # one view at a time: each frame is an independent mono shot
            pred = model.inference([cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)], process_res=process_res)
            proc_w = pred.depth.shape[-1]
            fxs.append(float(pred.intrinsics[0][0, 0]) * fr.shape[1] / proc_w)
    del model
    torch.cuda.empty_cache()
    return float(np.median(fxs))


class MetricDepth:
    """DA3METRIC-LARGE wrapper returning a full-resolution metric depth map (meters)."""

    def __init__(self, focal_px: float, device="cuda",
                 model_id="depth-anything/DA3METRIC-LARGE", process_res=756):
        import torch
        DepthAnything3 = _import_da3()
        self.torch = torch
        self.model = DepthAnything3.from_pretrained(model_id).to(device).eval()
        self.focal_px = focal_px
        self.process_res = process_res

    def __call__(self, frame_bgr) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        with self.torch.no_grad():
            pred = self.model.inference([cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)],
                                        process_res=self.process_res)
        raw = pred.depth[0].astype(np.float32)
        ph, pw = raw.shape
        focal_proc = self.focal_px * 0.5 * (pw / w + ph / h)
        metric = raw if getattr(pred, "is_metric", 0) else raw * focal_proc / 300.0
        return cv2.resize(metric, (w, h), interpolation=cv2.INTER_LINEAR)


def sign_position(depth_m: np.ndarray, bbox, focal_px: float) -> dict:
    """Camera-relative position of a sign from its box and a metric depth map."""
    h, w = depth_m.shape
    x1, y1, x2, y2 = bbox
    # central 60% of the box: avoids sky / background leaking in at the panel edges
    cx1, cx2 = x1 + 0.2 * (x2 - x1), x2 - 0.2 * (x2 - x1)
    cy1, cy2 = y1 + 0.2 * (y2 - y1), y2 - 0.2 * (y2 - y1)
    patch = depth_m[max(0, int(cy1)):min(h, int(math.ceil(cy2))),
                    max(0, int(cx1)):min(w, int(math.ceil(cx2)))]
    patch = patch[np.isfinite(patch) & (patch > 0)]
    if patch.size == 0:
        return {}
    z = float(np.median(patch))
    u, v = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    x = (u - w / 2.0) * z / focal_px
    y = (v - h / 2.0) * z / focal_px
    return dict(forward_m=round(z, 2), lateral_m=round(x, 2), height_m=round(-y, 2),
                range_m=round(math.hypot(x, z), 2),
                bearing_deg=round(math.degrees(math.atan2(x, z)), 1),
                depth_spread_m=round(float(np.percentile(patch, 75) - np.percentile(patch, 25)), 2))


def depth_preview(depth_m: np.ndarray, bbox=None, max_m: float = 80.0) -> np.ndarray:
    """Colourised depth map (near = bright) for the HTML report."""
    d = np.clip(depth_m / max_m, 0, 1)
    vis = cv2.applyColorMap((255 * (1 - d)).astype(np.uint8), cv2.COLORMAP_INFERNO)
    if bbox is not None:
        x1, y1, x2, y2 = (int(v) for v in bbox)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), max(2, vis.shape[1] // 400))
    return vis


# ---- GPS (future: dashcam GPS log) ----------------------------------------

class GpsTrack:
    """Vehicle GPS log: CSV with columns t (s from video start), lat, lon[, heading].

    heading is degrees clockwise from north; if absent it's derived from motion.
    """

    def __init__(self, path: str):
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.t = np.array([float(r["t"]) for r in rows])
        self.lat = np.array([float(r["lat"]) for r in rows])
        self.lon = np.array([float(r["lon"]) for r in rows])
        self.heading = (np.array([float(r["heading"]) for r in rows])
                        if rows and "heading" in rows[0] else None)

    def pose(self, t: float):
        lat = float(np.interp(t, self.t, self.lat))
        lon = float(np.interp(t, self.t, self.lon))
        if self.heading is not None:
            return lat, lon, float(np.interp(t, self.t, self.heading))
        la0, lo0 = np.interp(t - 1, self.t, self.lat), np.interp(t - 1, self.t, self.lon)
        la1, lo1 = np.interp(t + 1, self.t, self.lat), np.interp(t + 1, self.t, self.lon)
        de = (lo1 - lo0) * 111320 * math.cos(math.radians(lat))
        dn = (la1 - la0) * 111320
        return lat, lon, math.degrees(math.atan2(de, dn)) % 360

    def sign_latlon(self, t: float, forward_m: float, lateral_m: float):
        lat, lon, hdg = self.pose(t)
        h = math.radians(hdg)
        east = forward_m * math.sin(h) + lateral_m * math.cos(h)
        north = forward_m * math.cos(h) - lateral_m * math.sin(h)
        return (round(lat + north / 111320, 7),
                round(lon + east / (111320 * math.cos(math.radians(lat))), 7))


def load_gps(path: Optional[str]) -> Optional[GpsTrack]:
    return GpsTrack(path) if path else None
