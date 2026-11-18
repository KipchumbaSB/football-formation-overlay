"""
Stage 1 — Player (and ball) detection.

Reference approach per realtime-formation-overlay-pipeline.md: YOLOv8/v11
fine-tuned on a soccer dataset (e.g. SoccerNet). That fine-tuned checkpoint
is a training exercise of its own and isn't bundled here.

This module wraps a generic Ultralytics YOLO model (COCO classes) as the
Phase 1 stand-in: class 0 ("person") covers players/GK/referee, and class
32 ("sports ball") is a usable fallback for Stage 5's ball detection. This
is explicitly a placeholder — production accuracy (especially small/distant
players, ball under player feet) needs the soccer-specific fine-tune. The
interface (Detection, PlayerDetector.detect) is what the rest of the
pipeline depends on, so swapping in a fine-tuned checkpoint later is a
one-line change (just point `model_path` at the new weights).
"""
from dataclasses import dataclass
from typing import List

import numpy as np

COCO_PERSON_CLASS = 0
COCO_BALL_CLASS = 32


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int

    @property
    def foot_point(self):
        """Bottom-center of the bounding box — the usual proxy for where a
        player's feet touch the pitch, and the point Stage 4's homography
        projects onto real pitch coordinates."""
        return ((self.x1 + self.x2) / 2.0, self.y2)

    @property
    def center(self):
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


class PlayerDetector:
    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.35):
        self._model_path = model_path
        self._conf_threshold = conf_threshold
        self._model = None  # lazy-loaded; keeps this module importable
                             # without ultralytics installed, e.g. for unit tests

    def _ensure_loaded(self):
        if self._model is None:
            from ultralytics import YOLO  # local import: heavy, optional dep
            self._model = YOLO(self._model_path)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run detection on a single BGR frame. Returns player detections
        (class 0) only; use detect_ball() for the ball."""
        self._ensure_loaded()
        results = self._model.predict(frame, conf=self._conf_threshold, verbose=False)[0]
        out = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id != COCO_PERSON_CLASS:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            out.append(Detection(x1, y1, x2, y2, float(box.conf[0]), cls_id))
        return out

    def detect_ball(self, frame: np.ndarray) -> Detection | None:
        self._ensure_loaded()
        results = self._model.predict(frame, conf=self._conf_threshold, verbose=False)[0]
        best = None
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id != COCO_BALL_CLASS:
                continue
            conf = float(box.conf[0])
            if best is None or conf > best.confidence:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                best = Detection(x1, y1, x2, y2, conf, cls_id)
        return best
