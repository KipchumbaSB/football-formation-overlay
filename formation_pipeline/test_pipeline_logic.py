"""
Smoke tests for the non-CV stages (tracking, calibration, team
classification, formation engine) using synthetic data — no video file,
no model weights, no network access required. Run with:

    python test_pipeline_logic.py

This validates the math/logic independently of detection quality, which
is exactly the kind of thing to check in isolation before ever pointing
the full pipeline at real footage.
"""
import numpy as np

from calibration import PitchCalibrator
from config import CalibrationConfig
from detection import Detection
from formation import LabelSmoother, detect_formation
from tracking import SimpleTracker


def test_tracker_keeps_id_across_frames():
    tracker = SimpleTracker(iou_threshold=0.3, max_missed_frames=5)
    d1 = Detection(100, 100, 140, 220, 0.9, 0)
    tracks = tracker.update([d1])
    assert len(tracks) == 1
    first_id = tracks[0].track_id

    d2 = Detection(104, 101, 144, 221, 0.9, 0)  # small shift, same player
    tracks = tracker.update([d2])
    assert len(tracks) == 1
    assert tracks[0].track_id == first_id, "tracker should keep the same ID across a small shift"
    print("OK: tracker ID persistence")


def test_tracker_survives_occlusion():
    tracker = SimpleTracker(iou_threshold=0.3, max_missed_frames=3)
    d1 = Detection(100, 100, 140, 220, 0.9, 0)
    tracks = tracker.update([d1])
    first_id = tracks[0].track_id

    tracker.update([])  # occluded for 1 frame
    tracks = tracker.update([Detection(102, 100, 142, 220, 0.9, 0)])
    assert tracks[0].track_id == first_id, "track should survive a brief occlusion"
    print("OK: tracker occlusion tolerance")


def test_calibration_round_trip():
    cfg = CalibrationConfig(
        image_points=[(0, 0), (100, 0), (0, 100), (100, 100)],
        pitch_points_m=[(0, 0), (50, 0), (0, 30), (50, 30)],
    )
    calib = PitchCalibrator(cfg)
    pitch_pt = calib.image_to_pitch((50, 50))
    assert abs(pitch_pt[0] - 25.0) < 1e-3
    assert abs(pitch_pt[1] - 15.0) < 1e-3

    image_pt = calib.pitch_to_image(pitch_pt)
    assert abs(image_pt[0] - 50.0) < 1e-3
    assert abs(image_pt[1] - 50.0) < 1e-3
    print("OK: calibration homography round-trip")


def test_formation_detects_back_four_three_three():
    # 4 defenders ~5-10m deep, 3 midfielders ~35-42m, 3 attackers ~65-70m
    depths = [5, 6, 8, 10, 35, 38, 42, 65, 68, 70]
    result = detect_formation(depths, min_gap_m=7.0)
    assert result.formation == "4-3-3", f"expected 4-3-3, got {result.formation}"
    assert result.confidence > 0
    print(f"OK: formation engine -> {result.formation} (confidence {result.confidence:.0f}%)")


def test_label_smoother_majority_votes():
    sm = LabelSmoother(window=5)
    labels = ["4-3-3", "4-3-3", "4-2-3-1", "4-3-3", "4-3-3"]
    out = None
    for lbl in labels:
        out = sm.push(lbl)
    assert out == "4-3-3", f"expected majority label 4-3-3, got {out}"
    print("OK: label smoother majority vote")


if __name__ == "__main__":
    test_tracker_keeps_id_across_frames()
    test_tracker_survives_occlusion()
    test_calibration_round_trip()
    test_formation_detects_back_four_three_three()
    test_label_smoother_majority_votes()
    print("\nAll logic tests passed.")
