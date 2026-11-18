"""
Stage 4 — Pitch calibration (image pixels -> real pitch coordinates).

This is the step that makes "depth" (distance from a team's own goal line)
a measurable quantity at all, which is what Stage 6's formation engine
clusters on. Per the doc, a production system detects pitch-line keypoints
automatically per camera shot; that detector is out of scope here. Instead
this module solves a homography from manually supplied point
correspondences (see config.CalibrationConfig) — same math, just without
the automatic keypoint-finding step in front of it.
"""
from typing import Optional, Tuple

import cv2
import numpy as np

from config import CalibrationConfig


class PitchCalibrator:
    def __init__(self, cfg: CalibrationConfig):
        if len(cfg.image_points) < 4 or len(cfg.image_points) != len(cfg.pitch_points_m):
            raise ValueError(
                "Calibration needs >=4 matched (image_point, pitch_point_m) pairs. "
                "See config.CalibrationConfig docstring."
            )
        src = np.array(cfg.image_points, dtype=np.float32)
        dst = np.array(cfg.pitch_points_m, dtype=np.float32)
        self._H, _ = cv2.findHomography(src, dst, method=cv2.RANSAC)
        self._H_inv = np.linalg.inv(self._H)

    def image_to_pitch(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """Project an image-space point (e.g. a player's foot_point) onto
        real pitch coordinates in meters, origin at one corner flag."""
        p = np.array([point[0], point[1], 1.0])
        q = self._H @ p
        q /= q[2]
        return float(q[0]), float(q[1])

    def pitch_to_image(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """Inverse projection — used by render.py to place overlay graphics
        back at the correct pixel location."""
        p = np.array([point[0], point[1], 1.0])
        q = self._H_inv @ p
        q /= q[2]
        return float(q[0]), float(q[1])


def example_config_for_frame() -> CalibrationConfig:
    """
    Illustrates the shape of a calibration config — NOT real values.
    Replace image_points with pixel coordinates read off an actual frame
    (e.g. via a quick annotation tool / clicking on the video), matched in
    the same order to known real-world pitch coordinates (meters, origin
    at the corner flag nearest the top-left of a standard broadcast angle).
    """
    return CalibrationConfig(
        image_points=[
            (660.0, 115.0),   # e.g. halfway line x touchline (top)
            (1340.0, 115.0),  # e.g. halfway line x touchline (bottom edge of center circle)
            (660.0, 780.0),
            (1340.0, 780.0),
        ],
        pitch_points_m=[
            (52.5, 0.0),
            (52.5, 68.0),
            (52.5 - 9.15, 0.0),
            (52.5 - 9.15, 68.0),
        ],
    )
