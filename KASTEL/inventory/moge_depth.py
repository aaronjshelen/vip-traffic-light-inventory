"""MoGe-2 monocular metric geometry for traffic-light localization.

MoGe-2 predicts, from ONE image with unknown camera, a metric depth map and the camera
intrinsics (field of view). The repo is cloned to third_party/MoGe and imported from source;
the only extra package is `utils3d-moge` (installed --no-deps into KASTEL/.venv).

The dashcam is uncalibrated, so the horizontal FOV is estimated ONCE per video (median over a
few frames) and then fixed for every inference, keeping all frames on one consistent camera.
"""

from __future__ import annotations

import math
import os
import sys

import cv2
import numpy as np

_MOGE_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "third_party", "MoGe")


class MoGe2:
    def __init__(self, model_id="Ruicheng/moge-2-vitl-normal", device="cuda"):
        import torch
        if _MOGE_SRC not in sys.path:
            sys.path.insert(0, _MOGE_SRC)
        from moge.model.v2 import MoGeModel
        self.torch = torch
        self.device = device
        self.model = MoGeModel.from_pretrained(model_id).to(device).eval()
        self.fov_x = None   # degrees; None = let MoGe estimate per frame

    def _infer(self, frame_bgr, fov_x=None):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = self.torch.tensor(rgb / 255.0, dtype=self.torch.float32, device=self.device).permute(2, 0, 1)
        with self.torch.no_grad():
            # apply_mask=False: keep depth on sky-adjacent pixels — lights are often seen against the sky
            return self.model.infer(img, fov_x=fov_x, apply_mask=False)

    def calibrate_fov(self, frames_bgr) -> float:
        fovs = []
        for fr in frames_bgr:
            k = self._infer(fr)["intrinsics"].cpu().numpy()
            fovs.append(math.degrees(2 * math.atan(0.5 / k[0, 0])))   # normalized fx (by width)
        self.fov_x = float(np.median(fovs))
        return self.fov_x

    def depth(self, frame_bgr):
        """Metric depth map (H, W) in metres and pixel intrinsics (fx, fy, cx, cy)."""
        out = self._infer(frame_bgr, self.fov_x)
        h, w = frame_bgr.shape[:2]
        k = out["intrinsics"].cpu().numpy()
        intr = (k[0, 0] * w, k[1, 1] * h, k[0, 2] * w, k[1, 2] * h)
        return out["depth"].float().cpu().numpy(), intr


def box_position(depth_m, intr, bbox, near_pct=30.0):
    """Camera-frame position of an object box: forward Z, right X, up (m).

    A low percentile of depth inside the box picks the light housing over sky / background
    showing through the box corners.
    """
    fx, fy, cx, cy = intr
    h, w = depth_m.shape
    x1, y1, x2, y2 = bbox
    patch = depth_m[max(0, int(y1)):min(h, int(math.ceil(y2))), max(0, int(x1)):min(w, int(math.ceil(x2)))]
    patch = patch[np.isfinite(patch) & (patch > 0) & (patch < 300)]
    if patch.size == 0:
        return None
    z = float(np.percentile(patch, near_pct))
    u, v = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return dict(forward_m=z, right_m=(u - cx) * z / fx, up_m=-(v - cy) * z / fy)


def depth_vis(depth_m, max_m=80.0):
    d = np.nan_to_num(np.clip(depth_m / max_m, 0, 1), nan=1.0)
    return cv2.applyColorMap((255 * (1 - d)).astype(np.uint8), cv2.COLORMAP_INFERNO)
