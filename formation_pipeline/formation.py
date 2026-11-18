"""
Stage 6 — Formation / shape clustering engine.

Direct port of the gap-based depth-clustering algorithm from
formation_display_demo.html (the earlier browser prototype), adapted to
run on real pitch coordinates (meters, from Stage 4's calibration) instead
of normalized 0-1 frame space. Same core idea: sort outfield players by
depth (distance from own goal line), find the largest gaps, and read off
cluster sizes as the formation string.

Only ever called on the out-of-possession team's 10 outfield players
(Stage 5 decides which team that is, each frame/window).
"""
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class FormationResult:
    formation: str            # e.g. "4-3-3"
    line_counts: List[int]
    line_mean_depth: List[float]
    confidence: float


def detect_formation(depths_m: List[float], min_gap_m: float = 7.0) -> FormationResult:
    """
    depths_m: distance-from-own-goal-line (meters) for exactly the 10
    outfield players of one team. Order doesn't matter; this sorts them.
    """
    xs = sorted(depths_m)
    n = len(xs)
    if n == 0:
        return FormationResult("", [], [], 0.0)

    gaps = [(i, xs[i] - xs[i - 1]) for i in range(1, n)]
    top_gaps = sorted(gaps, key=lambda g: g[1], reverse=True)[:3]
    split_points = sorted(idx for idx, size in top_gaps if size >= min_gap_m)

    clusters: List[List[float]] = []
    start = 0
    for sp in split_points:
        clusters.append(xs[start:sp])
        start = sp
    clusters.append(xs[start:])

    line_counts = [len(c) for c in clusters]
    line_mean_depth = [sum(c) / len(c) for c in clusters]
    formation = "-".join(str(c) for c in line_counts)

    kept_sizes = [size for idx, size in top_gaps if size >= min_gap_m]
    smallest_kept = min(kept_sizes) if kept_sizes else 0.0
    confidence = min(100.0, (smallest_kept / min_gap_m) * 55.0 + 30.0) if kept_sizes else 0.0

    return FormationResult(formation, line_counts, line_mean_depth, confidence)


class LabelSmoother:
    """
    Majority-vote smoothing over the last N formation labels, so the
    on-screen label doesn't flicker between adjacent readings
    (e.g. '4-3-3' <-> '4-2-3-1') during a genuinely ambiguous shape.
    Phase 1 note: because this is non-causal, a stronger version could
    vote using a window centered on the current frame instead of a
    trailing one — this trailing version is kept deliberately simple so
    the exact same class can be reused unmodified in Phase 2 (streaming),
    where only a trailing window is ever possible.
    """

    def __init__(self, window: int = 9):
        self._window = window
        self._history: List[str] = []

    def push(self, label: str) -> str:
        self._history.append(label)
        if len(self._history) > self._window:
            self._history.pop(0)
        counts = {}
        for lbl in self._history:
            counts[lbl] = counts.get(lbl, 0) + 1
        return max(counts.items(), key=lambda kv: kv[1])[0]
