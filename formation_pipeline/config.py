"""
Stage 0 — Configuration for the Phase 1 (local video file) pipeline.

Mirrors the dependency chain in realtime-formation-overlay-pipeline.md:
detection -> tracking -> team classification -> calibration -> possession
-> formation engine -> render.
"""
from dataclasses import dataclass, field
from typing import List, Tuple


# Standard FIFA pitch dimensions in meters. Used as the real-world target
# for the homography solved in calibration.py.
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0


@dataclass
class CalibrationConfig:
    """
    Phase 1 placeholder for Stage 4 (pitch calibration).

    A production system detects pitch-line keypoints automatically per
    camera shot (see realtime-formation-overlay-pipeline.md, Stage 4).
    That model isn't included here. Instead, this config takes manually
    supplied point correspondences for one representative frame per shot:
    four or more (image_x, image_y) points matched to known
    (pitch_x_m, pitch_y_m) positions (e.g. center circle, penalty box
    corners). This is the fastest way to get a correct, testable
    homography while the automatic keypoint detector is still a TODO.
    """
    image_points: List[Tuple[float, float]] = field(default_factory=list)
    pitch_points_m: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class FormationConfig:
    # Minimum normalized-depth gap to count as a real line break between
    # clusters of players, rather than noise. Same tunable as the browser
    # prototype (formation_display_demo.html).
    min_gap_m: float = 7.0          # in meters, since Stage 6 now runs on
                                     # real pitch coordinates (Stage 4 output)
                                     # instead of the 0-1 normalized space
                                     # used in the browser prototype.
    smoothing_window_frames: int = 15   # centered window (Phase 1 is non-causal)
    label_vote_window: int = 9          # majority-vote smoothing on the label itself


@dataclass
class TrackingConfig:
    iou_match_threshold: float = 0.3
    max_missed_frames: int = 10     # how long a track survives an occlusion
                                     # before being dropped


@dataclass
class PipelineConfig:
    input_video_path: str = ""
    output_video_path: str = ""
    detection_every_n_frames: int = 1   # Phase 1 can afford full-rate detection;
                                         # Phase 2 will subsample (see doc, Stage 1 note)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    formation: FormationConfig = field(default_factory=FormationConfig)
