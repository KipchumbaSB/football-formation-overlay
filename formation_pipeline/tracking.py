"""
Stage 2 — Multi-object tracking.

Reference approach per the doc: ByteTrack / BoT-SORT. This module
implements a simpler IoU + Hungarian-assignment tracker that covers the
same core job (persistent IDs across frames) without pulling in a full
tracking library, so the pipeline has no hard external dependency beyond
numpy/scipy for this stage. Swap in ByteTrack/BoT-SORT later without
touching anything downstream — everything after this stage only consumes
`Track` objects.

Phase 1 non-causal note: `max_missed_frames` lets a track survive a short
occlusion. A full implementation would additionally re-link a track that
reappears after exceeding that window using appearance (jersey color/
embedding) similarity plus trajectory extrapolation — worth building once
Phase 1's basic accuracy bar (doc, section 3, exit criteria) is otherwise met.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from detection import Detection


def iou(a: Detection, b: Detection) -> float:
    xx1, yy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    xx2, yy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
    area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
    area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    track_id: int
    detection: Detection
    missed_frames: int = 0
    history: List[Detection] = field(default_factory=list)


class SimpleTracker:
    def __init__(self, iou_threshold: float = 0.3, max_missed_frames: int = 10):
        self._iou_threshold = iou_threshold
        self._max_missed_frames = max_missed_frames
        self._next_id = 0
        self._tracks: Dict[int, Track] = {}

    def update(self, detections: List[Detection]) -> List[Track]:
        track_ids = list(self._tracks.keys())
        if track_ids and detections:
            cost = np.zeros((len(track_ids), len(detections)))
            for i, tid in enumerate(track_ids):
                for j, det in enumerate(detections):
                    cost[i, j] = 1.0 - iou(self._tracks[tid].detection, det)
            row_idx, col_idx = linear_sum_assignment(cost)

            matched_tracks, matched_dets = set(), set()
            for r, c in zip(row_idx, col_idx):
                if cost[r, c] <= (1.0 - self._iou_threshold):
                    tid = track_ids[r]
                    self._tracks[tid].detection = detections[c]
                    self._tracks[tid].history.append(detections[c])
                    self._tracks[tid].missed_frames = 0
                    matched_tracks.add(tid)
                    matched_dets.add(c)

            for i, tid in enumerate(track_ids):
                if tid not in matched_tracks:
                    self._tracks[tid].missed_frames += 1

            for j, det in enumerate(detections):
                if j not in matched_dets:
                    self._spawn(det)
        elif detections:
            for det in detections:
                self._spawn(det)
        else:
            for t in self._tracks.values():
                t.missed_frames += 1

        self._tracks = {
            tid: t for tid, t in self._tracks.items()
            if t.missed_frames <= self._max_missed_frames
        }
        return list(self._tracks.values())

    def _spawn(self, det: Detection):
        t = Track(track_id=self._next_id, detection=det, history=[det])
        self._tracks[self._next_id] = t
        self._next_id += 1
