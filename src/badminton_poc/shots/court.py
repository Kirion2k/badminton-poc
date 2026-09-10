"""Court-relative depth from the calibrated court polygon, without a homography.

The rules in :mod:`badminton_poc.shots.baseline` need to know whether a player is at the net or
at the back, and whether a shuttle landed short or deep. Image ``y`` alone cannot say that: the
same y is the far player's midcourt and the near player's service line.

What we have is the hand-calibrated ``CourtROI``: a polygon whose top and bottom edges sit on the
two baselines, and a net line where the net meets the floor. That is enough for a **depth along
the court**, measured from each player's own net, which is all the rules ask for.

What it is not is a homography. Depth here is linear in image y, and perspective is not, so a
figure of 0.5 is "about halfway back" rather than a measured distance. It is used only for
coarse comparisons against thresholds, never reported as a court coordinate, and the BST path
does not touch it at all.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..config import CourtROI


@dataclass(frozen=True)
class CourtDepth:
    """Depth along the court from each player's own net, in units of that player's half."""

    net_left: tuple[float, float]
    net_right: tuple[float, float]
    near_baseline_y: float
    far_baseline_y: float

    @classmethod
    def from_roi(cls, court: CourtROI) -> "CourtDepth | None":
        """None when the ROI has no net line, which makes depth undefined rather than guessed."""
        if court.net_line is None:
            return None
        left, right = court.net_line
        ys = [y for _, y in court.polygon]
        return cls(
            net_left=(float(left[0]), float(left[1])),
            net_right=(float(right[0]), float(right[1])),
            near_baseline_y=max(ys),
            far_baseline_y=min(ys),
        )

    def net_y_at(self, x: float) -> float:
        """Where the net meets the floor at this horizontal position."""
        x0, y0 = self.net_left
        x1, y1 = self.net_right
        if abs(x1 - x0) < 1e-9:
            return y0
        t = (x - x0) / (x1 - x0)
        return y0 + t * (y1 - y0)

    def depth(self, x: float, y: float, side: str) -> float | None:
        """0 at ``side``'s net, 1 at ``side``'s baseline. Negative means past the net.

        Both coordinates are normalized to the frame. Values beyond 1 are possible and kept:
        a shuttle high in the air over the far court reads as deep, which is the truth about
        the image even though it is not the truth about the floor.
        """
        net = self.net_y_at(x)
        if side == "near":
            span = self.near_baseline_y - net
            return None if span <= 0 else (y - net) / span
        if side == "far":
            span = net - self.far_baseline_y
            return None if span <= 0 else (net - y) / span
        return None

    def crossed_to(self, x: float, y: float) -> str:
        """Which half of the court an image point falls in."""
        return "near" if y > self.net_y_at(x) else "far"
