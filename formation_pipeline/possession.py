"""
Stage 5 — Ball detection + possession attribution.

This is the gate the whole overlay depends on: it decides which team is
*out* of possession, i.e. which team's 10 outfield players should be fed
into Stage 6's formation engine at all. Per the doc, the production
version can be a trained event model; this module implements the
proximity/velocity heuristic named as the lighter-weight fallback:
whichever team has a player closest to the ball (within a distance
threshold, in real pitch meters via Stage 4's calibration) is considered
in possession.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from team_classify import Team
from tracking import Track

POSSESSION_DISTANCE_THRESHOLD_M = 3.0


@dataclass
class PossessionState:
    in_possession: Optional[Team]       # None if ball not found / no team close enough
    out_of_possession: Optional[Team]   # the team the formation engine should run on


def attribute_possession(
    ball_pitch_point: Optional[Tuple[float, float]],
    track_pitch_points: Dict[int, Tuple[float, float]],
    team_of_track: Dict[int, Team],
) -> PossessionState:
    if ball_pitch_point is None:
        return PossessionState(in_possession=None, out_of_possession=None)

    bx, by = ball_pitch_point
    closest_team: Optional[Team] = None
    closest_dist = float("inf")

    for track_id, (px, py) in track_pitch_points.items():
        team = team_of_track.get(track_id, Team.UNKNOWN)
        if team == Team.UNKNOWN:
            continue
        dist = ((px - bx) ** 2 + (py - by) ** 2) ** 0.5
        if dist < closest_dist:
            closest_dist = dist
            closest_team = team

    if closest_team is None or closest_dist > POSSESSION_DISTANCE_THRESHOLD_M:
        return PossessionState(in_possession=None, out_of_possession=None)

    out_team = Team.B if closest_team == Team.A else Team.A
    return PossessionState(in_possession=closest_team, out_of_possession=out_team)
