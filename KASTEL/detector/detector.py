"""Swappable traffic-light detector wrapper.

The rest of the baseline pipeline depends ONLY on the small surface defined here:

    - `Detection`         : one bounding box + score + class for a single frame
    - `TrafficLightDetector.detect(frame_bgr) -> list[Detection]`
    - `TrafficLightDetector.names`  : {class_id: class_name}

To swap the detector later (a fine-tuned model, a different architecture, a
COCO model filtered to the traffic-light class, etc.) you only need to provide
another object exposing the same `.detect()` / `.names` surface. Nothing in
`run_baseline.py` reaches into ultralytics directly.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional, Union


@dataclass
class Detection:
    """A single detected signal head in a single frame (pixel coordinates).

    `track_id` is populated only when the frame was processed with
    `TrafficLightDetector.track()`; it is None for plain `detect()`.
    """
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    def as_dict(self) -> dict:
        return asdict(self)


class TrafficLightDetector:
    """Thin wrapper around an Ultralytics YOLO model.

    Defaults target the KASTEL MobilityLab DTLD-trained YOLOv8 weights, whose 20
    classes jointly encode state + indication (e.g. ``circle_red``,
    ``arrow_left_green``). We keep that raw class string here; splitting it into
    separate state / indication labels is deliberately left to a later stage.
    """

    def __init__(
        self,
        weights_path: str,
        imgsz: int = 1280,
        conf: float = 0.25,
        iou: float = 0.7,
        device: Optional[Union[int, str]] = None,
    ):
        # Imported lazily so that importing this module stays cheap/testable.
        from ultralytics import YOLO
        import torch

        if device is None:
            device = 0 if torch.cuda.is_available() else "cpu"

        self.weights_path = weights_path
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.device = device

        self.model = YOLO(weights_path)
        # {id: name}, taken straight from the checkpoint so it stays correct
        # even if we swap in a different model.
        self.names = dict(self.model.names)

    def detect(self, frame_bgr, conf: Optional[float] = None) -> List[Detection]:
        """Run detection on one BGR frame (numpy array as returned by OpenCV)."""
        results = self.model.predict(
            frame_bgr,
            imgsz=self.imgsz,
            conf=self.conf if conf is None else conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )
        return self._boxes_to_detections(results[0])

    def track(
        self,
        frame_bgr,
        conf: Optional[float] = None,
        tracker: str = "bytetrack.yaml",
    ) -> List[Detection]:
        """Detect + associate across frames (tracking-by-detection).

        Must be called on CONSECUTIVE frames: the tracker keeps state between
        calls via ``persist=True``. Returns the same `Detection` objects as
        `detect()` but with `track_id` populated for confirmed tracks (it may
        still be None for freshly-seen / unconfirmed boxes).
        """
        results = self.model.track(
            frame_bgr,
            imgsz=self.imgsz,
            conf=self.conf if conf is None else conf,
            iou=self.iou,
            device=self.device,
            tracker=tracker,
            persist=True,
            verbose=False,
        )
        return self._boxes_to_detections(results[0])

    @staticmethod
    def _boxes_to_detections(result) -> List[Detection]:
        detections: List[Detection] = []
        names = result.names
        for box in result.boxes:
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            class_id = int(box.cls[0])
            track_id = int(box.id[0]) if box.id is not None else None
            detections.append(
                Detection(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=float(box.conf[0]),
                    class_id=class_id,
                    class_name=names.get(class_id, str(class_id)),
                    track_id=track_id,
                )
            )
        return detections
