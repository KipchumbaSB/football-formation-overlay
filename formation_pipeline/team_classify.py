"""
Stage 3 — Team / role classification.

Clusters each tracked player's shirt-region pixel color into two teams via
k-means (k=2), per the doc. Goalkeepers and the referee are typically
color outliers from both outfield clusters (GK kits and ref kits are
usually visually distinct by design) — a production version separates them
out as a third/fourth cluster or with simple hue-distance rules; this
module keeps it to the two outfield-team split, which is what Stage 6
(formation engine) actually needs.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

import cv2
import numpy as np

from detection import Detection
from tracking import Track


class Team(Enum):
    A = "A"
    B = "B"
    UNKNOWN = "unknown"


def _shirt_color(frame: np.ndarray, det: Detection) -> np.ndarray:
    """Sample the upper third of the bounding box (torso/shirt), away from
    shorts and grass, and return its mean BGR color."""
    x1, y1, x2, y2 = map(int, (det.x1, det.y1, det.x2, det.y2))
    h = y2 - y1
    shirt_y1 = y1 + int(h * 0.15)
    shirt_y2 = y1 + int(h * 0.45)
    shirt_y1, shirt_y2 = max(0, shirt_y1), max(shirt_y1 + 1, shirt_y2)
    x1, x2 = max(0, x1), max(x1 + 1, x2)
    crop = frame[shirt_y1:shirt_y2, x1:x2]
    if crop.size == 0:
        return np.array([0.0, 0.0, 0.0])
    return crop.reshape(-1, 3).mean(axis=0)


def classify_teams(frame: np.ndarray, tracks: List[Track]) -> Dict[int, Team]:
    """Returns {track_id: Team} for the given frame's tracks."""
    if len(tracks) < 2:
        return {t.track_id: Team.UNKNOWN for t in tracks}

    colors = np.array([_shirt_color(frame, t.detection) for t in tracks], dtype=np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, _ = cv2.kmeans(colors, 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    labels = labels.flatten()

    team_map = {0: Team.A, 1: Team.B}
    return {t.track_id: team_map[labels[i]] for i, t in enumerate(tracks)}
