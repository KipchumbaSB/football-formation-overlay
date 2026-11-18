"""
Stage 7 — Render the overlay back onto the frame.

Draws the same two elements identified in the source clip: a ground
ellipse at each tracked player's projected foot position, and a red arrow
above each player on the out-of-possession team (the team the formation
engine just ran on), plus the formation label as on-screen text.
"""
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from team_classify import Team
from tracking import Track

TEAM_COLORS = {
    Team.A: (209, 122, 63),   # BGR, matches the teal used in the earlier prototype
    Team.B: (69, 122, 255),   # BGR, matches the orange used in the earlier prototype
}
ARROW_COLOR = (0, 0, 220)  # red


def draw_overlay(
    frame: np.ndarray,
    tracks: List[Track],
    team_of_track: Dict[int, Team],
    out_of_possession_team: Optional[Team],
    formation_label: str,
) -> np.ndarray:
    out = frame.copy()

    for t in tracks:
        fx, fy = t.detection.foot_point
        team = team_of_track.get(t.track_id, Team.UNKNOWN)
        color = TEAM_COLORS.get(team, (180, 180, 180))

        # ground ellipse at the projected foot position
        cv2.ellipse(
            out, (int(fx), int(fy)), (22, 10), 0, 0, 360, color, 2, cv2.LINE_AA
        )

        # red arrow tag: only on the out-of-possession team's players,
        # i.e. exactly the players Stage 6 just computed a shape for
        if out_of_possession_team is not None and team == out_of_possession_team:
            tip = (int(fx), int(t.detection.y1) - 28)
            base_l = (int(fx) - 7, int(t.detection.y1) - 42)
            base_r = (int(fx) + 7, int(t.detection.y1) - 42)
            cv2.fillPoly(out, [np.array([tip, base_l, base_r])], ARROW_COLOR)

    if formation_label:
        cv2.putText(
            out, formation_label, (24, 48),
            cv2.FONT_HERSHEY_DUPLEX, 1.1, (3, 183, 255), 2, cv2.LINE_AA
        )

    return out
