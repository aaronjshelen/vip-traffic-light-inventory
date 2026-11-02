"""Directional-sign PANEL detector (open-vocabulary YOLO-World).

Detects sign *panels* by text prompt, so ByteTrack has one stable object per
physical sign (better for tracking + future localization than many text boxes).

Surface:
    det = SignPanelDetector(prompts=[...], imgsz=1280, conf=0.05, device=0)
    dets = det.detect(frame_bgr)            # list[Panel]
    dets = det.track(frame_bgr)             # list[Panel] with track_id set

`Panel` mirrors KASTEL's Detection shape so the code reads familiarly, but this
is a SEPARATE class in a SEPARATE package — KASTEL is untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2

# Default open-vocabulary prompts for U.S. text-bearing directional signage.
DEFAULT_PROMPTS = [
    "green highway guide sign",
    "highway exit sign",
    "street name sign",
    "road direction sign",
    "destination sign",
]


@dataclass
class Panel:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_name: str
    track_id: Optional[int] = None


class SignPanelDetector:
    def __init__(
        self,
        prompts: Optional[List[str]] = None,
        weights: str = "yolov8x-worldv2.pt",
        imgsz: int = 1280,
        conf: float = 0.05,
        iou: float = 0.5,
        device=None,
    ):
        from ultralytics import YOLO
        import torch

        if device is None:
            device = 0 if torch.cuda.is_available() else "cpu"
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.prompts = prompts or DEFAULT_PROMPTS

        self.model = YOLO(weights)
        self.model.set_classes(self.prompts)

    def _to_panels(self, result) -> List[Panel]:
        panels: List[Panel] = []
        names = result.names
        for box in result.boxes:
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            cid = int(box.cls[0])
            tid = int(box.id[0]) if box.id is not None else None
            panels.append(Panel(
                x1=x1, y1=y1, x2=x2, y2=y2,
                confidence=float(box.conf[0]),
                class_name=names.get(cid, str(cid)),
                track_id=tid,
            ))
        return panels

    def detect(self, frame_bgr) -> List[Panel]:
        r = self.model.predict(frame_bgr, imgsz=self.imgsz, conf=self.conf,
                               iou=self.iou, device=self.device, verbose=False)
        return self._to_panels(r[0])

    def track(self, frame_bgr) -> List[Panel]:
        r = self.model.track(frame_bgr, imgsz=self.imgsz, conf=self.conf,
                             iou=self.iou, device=self.device,
                             tracker="bytetrack.yaml", persist=True, verbose=False)
        return self._to_panels(r[0])


# ---- crop quality helpers -------------------------------------------------

def crop_from(frame, box, pad: int = 4):
    h, w = frame.shape[:2]
    x1 = max(0, int(box[0]) - pad)
    y1 = max(0, int(box[1]) - pad)
    x2 = min(w, int(box[2]) + pad)
    y2 = min(h, int(box[3]) + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def sharpness(crop) -> float:
    """Variance of the Laplacian — higher = sharper (less motion blur)."""
    if crop is None or crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def quality_score(crop, confidence: float) -> float:
    """Rank crops for OCR: bigger + sharper + more confident is better."""
    if crop is None or crop.size == 0:
        return 0.0
    h, w = crop.shape[:2]
    area = w * h
    return area * sharpness(crop) * max(confidence, 1e-3)
