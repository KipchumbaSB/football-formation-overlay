"""
Phase 1 orchestrator — local video file, two-pass, non-causal.

Pass A (analysis): decode the whole file once; run Stages 1-5 on every
frame; store each track's projected pitch position and possession state,
indexed by frame. Nothing is rendered yet.

Pass B (render): re-decode the file; for each frame, build a *centered*
smoothing window over Pass A's stored data (using frames both before and
after the current one — only possible because the whole file is already
on disk, per the Phase 1 vs Phase 2 trade-off in
realtime-formation-overlay-pipeline.md), run Stage 6 on the smoothed
positions, smooth the resulting label, then render (Stage 7) and write
the output file.

This two-pass structure is the concrete implementation of the doc's
"Phase 1 can use past AND future frames" design decision — Phase 2 will
collapse this into a single causal, trailing-only pass.
"""
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2

from calibration import PitchCalibrator
from config import PipelineConfig
from detection import PlayerDetector
from formation import LabelSmoother, detect_formation
from possession import PossessionState, attribute_possession
from team_classify import Team, classify_teams
from tracking import SimpleTracker, Track


class FrameRecord:
    __slots__ = ("tracks", "team_of_track", "pitch_points", "possession")

    def __init__(self):
        self.tracks: List[Track] = []
        self.team_of_track: Dict[int, Team] = {}
        self.pitch_points: Dict[int, Tuple[float, float]] = {}
        self.possession: PossessionState = PossessionState(None, None)


class Phase1Pipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.detector = PlayerDetector()
        self.calibrator = PitchCalibrator(cfg.calibration)

    # ---------- Pass A ----------
    def _analyze(self) -> List[FrameRecord]:
        cap = cv2.VideoCapture(self.cfg.input_video_path)
        tracker = SimpleTracker(
            iou_threshold=self.cfg.tracking.iou_match_threshold,
            max_missed_frames=self.cfg.tracking.max_missed_frames,
        )
        records: List[FrameRecord] = []
        frame_idx = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rec = FrameRecord()
            detections = self.detector.detect(frame)
            tracks = tracker.update(detections)
            rec.tracks = tracks

            rec.team_of_track = classify_teams(frame, tracks)
            rec.pitch_points = {
                t.track_id: self.calibrator.image_to_pitch(t.detection.foot_point)
                for t in tracks
            }

            ball_det = self.detector.detect_ball(frame)
            ball_pitch = (
                self.calibrator.image_to_pitch(ball_det.foot_point) if ball_det else None
            )
            rec.possession = attribute_possession(
                ball_pitch, rec.pitch_points, rec.team_of_track
            )

            records.append(rec)
            frame_idx += 1

        cap.release()
        return records

    # ---------- Pass B ----------
    def _smoothed_depth(
        self, records: List[FrameRecord], frame_idx: int, track_id: int, team_axis_fn
    ) -> Optional[float]:
        w = self.cfg.formation.smoothing_window_frames
        lo, hi = max(0, frame_idx - w), min(len(records), frame_idx + w + 1)
        vals = []
        for f in range(lo, hi):
            p = records[f].pitch_points.get(track_id)
            if p is not None:
                vals.append(team_axis_fn(p))
        return sum(vals) / len(vals) if vals else None

    def run(self) -> None:
        records = self._analyze()
        if not records:
            raise RuntimeError("No frames decoded from input video.")

        # Pitch depth axis: x-coordinate in the calibration's pitch space
        # (see calibration.example_config_for_frame for the convention used).
        depth_axis = lambda p: p[0]

        label_smoother = LabelSmoother(window=self.cfg.formation.label_vote_window)

        cap = cv2.VideoCapture(self.cfg.input_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(self.cfg.output_video_path, fourcc, fps, (width, height))

        from render import draw_overlay  # local import to avoid a cycle with tracking.Track typing

        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rec = records[frame_idx]
            out_team = rec.possession.out_of_possession

            raw_label = ""
            if out_team is not None:
                target_ids = [
                    tid for tid, team in rec.team_of_track.items() if team == out_team
                ]
                smoothed_depths = []
                for tid in target_ids:
                    d = self._smoothed_depth(records, frame_idx, tid, depth_axis)
                    if d is not None:
                        smoothed_depths.append(d)

                if len(smoothed_depths) >= 3:  # need enough players for a meaningful shape
                    result = detect_formation(
                        smoothed_depths, min_gap_m=self.cfg.formation.min_gap_m
                    )
                    raw_label = result.formation

            smoothed_label = label_smoother.push(raw_label) if raw_label else ""

            out_frame = draw_overlay(
                frame, rec.tracks, rec.team_of_track, out_team, smoothed_label
            )
            writer.write(out_frame)
            frame_idx += 1

        cap.release()
        writer.release()
