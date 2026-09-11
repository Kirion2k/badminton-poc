"""Associating a detected shuttle hit with the player who most plausibly played it.

The output of this module is deliberately verbose. For every hit we record the shuttle sample
used, the anchor points considered for each player, the distances, and which rule decided the
outcome, because the only way to improve this heuristic is to look at the cases it gets wrong.

The reported ``confidence`` is a *separation score*, not a probability. It says how much more
plausible the winner was than the runner-up on this frame, on a 0-1 scale. It is not calibrated
against labelled data and must not be read as "90% likely to be correct".
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .config import AnalysisConfig, CourtROI
from .parsing import BallSample, BallTrack, ShuttleEvent
from .players import wrist_points

UNASSIGNED = None

NOT_FILLABLE_BY_ALTERNATION = frozenset({"out_of_racket_reach", "closest_player_too_far"})
"""Outcomes that mean the contact did not happen, rather than that we could not decide.

`enforce_alternation` fills unassigned hits from their neighbours. That is right for a hit whose
two candidates were too close to separate, and wrong for one where the shuttle was nowhere near
either player: filling those puts the spurious contacts straight back."""


@dataclass
class Anchor:
    name: str
    point: tuple[float, float]
    distance: float

    def to_dict(self) -> dict[str, Any]:
        return {"anchor": self.name, "point": [round(self.point[0], 1), round(self.point[1], 1)],
                "distance_px": round(self.distance, 1)}


@dataclass
class HitAssignment:
    frame: int
    hitter: str | None
    confidence: float | None
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hitter": self.hitter,
            "assignment_separation": self.confidence,
            "assignment_reason": self.reason,
            "assignment_evidence": self.evidence,
        }


def player_anchors(player: dict[str, Any] | None, shuttle: tuple[float, float]) -> list[Anchor]:
    """Candidate contact points for a player, nearest first.

    Wrists first because contact happens at the racket hand; the racket head extends roughly
    another 60-70 cm beyond the wrist, so wrist distance is an underestimate of reach but a good
    ordering signal. Box top-centre is a fallback for overhead shots where the wrist keypoint is
    often missing or wrong, and box centre is the last resort when pose fails entirely.
    """
    if not player:
        return []
    anchors: list[Anchor] = []
    keypoints = player.get("keypoints") or []
    for index, (wx, wy) in enumerate(wrist_points(keypoints)):
        anchors.append(Anchor(f"wrist_{index}", (wx, wy), math.dist((wx, wy), shuttle)))
    bbox = player.get("bbox")
    if bbox:
        x1, y1, x2, y2 = (float(v) for v in bbox)
        top_centre = ((x1 + x2) / 2.0, y1)
        centre = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        anchors.append(Anchor("box_top_centre", top_centre, math.dist(top_centre, shuttle)))
        anchors.append(Anchor("box_centre", centre, math.dist(centre, shuttle)))
    anchors.sort(key=lambda a: a.distance)
    return anchors


def _best(anchors: list[Anchor]) -> Anchor | None:
    return anchors[0] if anchors else None


def body_height(player: dict[str, Any] | None) -> float | None:
    """The player's box height, which stands in for their distance from the camera.

    People are roughly the same height, so how tall someone is in frame says how many pixels a
    metre is worth where they are standing. It is the cheapest depth estimate available without a
    court homography, and it is the unit the swing features already use.
    """
    if not player:
        return None
    bbox = player.get("bbox")
    if not bbox:
        return None
    height = float(bbox[3]) - float(bbox[1])
    return height if height > 1.0 else None


def racket_reach(
    player_row: dict[str, Any] | None, shuttle: tuple[float, float]
) -> float | None:
    """How far the shuttle is from the nearer player's reach, in that player's body heights.

    The same measure the assignment gate uses, exposed so the flight segmentation can ask "was
    there a racket here" without knowing anything about players. Returns None when there is
    nobody to measure against, which must not be read as "far away".
    """
    if not player_row:
        return None
    best: float | None = None
    for slot in ("near", "far"):
        player = player_row.get(slot)
        anchors = player_anchors(player, shuttle)
        height = body_height(player)
        if not anchors or not height:
            continue
        reach = anchors[0].distance / height
        if best is None or reach < best:
            best = reach
    return best


def assign_hit(
    event: ShuttleEvent,
    ball: BallTrack,
    players_by_frame: dict[int, dict[str, Any]],
    court: CourtROI,
    cfg: AnalysisConfig,
    *,
    width: int,
    height: int,
) -> HitAssignment:
    """Decide which player played a hit. Pure function; unit-testable without any model."""
    frame = event.frame
    evidence: dict[str, Any] = {"frame": frame}

    sample = ball.nearest_visible(frame, cfg.hit_search_radius_frames)
    if sample is None and event.x is not None and event.y is not None:
        # The event carries its own coordinate even when the trajectory row is invisible.
        sample = BallSample(frame=frame, x=event.x, y=event.y, visible=False)
        evidence["shuttle_source"] = "event_coordinate"
    elif sample is not None:
        evidence["shuttle_source"] = "trajectory"
        evidence["shuttle_frame_offset"] = sample.frame - frame
    if sample is None:
        return HitAssignment(frame, UNASSIGNED, None, "no_shuttle_position", evidence)
    shuttle = sample.position()
    evidence["shuttle"] = [round(shuttle[0], 1), round(shuttle[1], 1)]

    player_row = players_by_frame.get(frame)
    if player_row is None:
        return HitAssignment(frame, UNASSIGNED, None, "no_player_row_for_frame", evidence)

    # A hit frame that happens to fall in a one- or two-frame dropout of one player leaves the
    # only remaining candidate to win by default, and which frame a detector lands on is close
    # to chance: three of six wrong assignments on the development clip were hits where the far
    # player was simply missing, with the same player present a few frames either side. Nobody
    # crosses a court in 60ms, so borrowing the nearest frame that has both is a better estimate
    # of where they stood than deciding from one of them.
    if not (player_row.get("near") and player_row.get("far")):
        for offset in range(1, max(0, cfg.hit_search_radius_frames) + 1):
            for candidate_frame in (frame - offset, frame + offset):
                candidate = players_by_frame.get(candidate_frame)
                if candidate and candidate.get("near") and candidate.get("far"):
                    player_row = candidate
                    evidence["player_row_frame_offset"] = candidate_frame - frame
                    break
            if "player_row_frame_offset" in evidence:
                break

    near_anchors = player_anchors(player_row.get("near"), shuttle)
    far_anchors = player_anchors(player_row.get("far"), shuttle)
    near_best, far_best = _best(near_anchors), _best(far_anchors)
    evidence["near"] = near_best.to_dict() if near_best else None
    evidence["far"] = far_best.to_dict() if far_best else None
    # Deliberately NOT recording which side of the net the shuttle is on. `court.side_of`
    # compares against the line where the net meets the *floor*, which is only meaningful for a
    # point on the ground. At the moment of a hit the shuttle is in the air, so it sits high in
    # frame regardless of which half it is over: on our test clip that test called 33 of 50
    # assignments wrong, including hits played from directly under the camera. Deciding a
    # shuttle's court half from a monocular view needs homography plus a height estimate, which
    # is a later milestone.

    near_body = body_height(player_row.get("near"))
    far_body = body_height(player_row.get("far"))
    evidence["body_height_px"] = {
        "near": round(near_body, 1) if near_body else None,
        "far": round(far_body, 1) if far_body else None,
    }

    if near_best is None and far_best is None:
        return HitAssignment(frame, UNASSIGNED, None, "no_players_detected", evidence)
    if near_best is None:
        return _distance_gated("far", far_best, None, far_body, cfg, width, height, evidence,
                               "only_far_detected")
    if far_best is None:
        return _distance_gated("near", near_best, None, near_body, cfg, width, height, evidence,
                               "only_near_detected")

    winner, loser = ("near", "far") if near_best.distance <= far_best.distance else ("far", "near")
    winner_anchor = near_best if winner == "near" else far_best
    loser_anchor = far_best if winner == "near" else near_best
    winner_body = near_body if winner == "near" else far_body
    return _distance_gated(
        winner, winner_anchor, loser_anchor, winner_body, cfg, width, height, evidence,
        "closest_anchor"
    )


def _distance_gated(
    hitter: str,
    winner: Anchor,
    loser: Anchor | None,
    winner_body_height: float | None,
    cfg: AnalysisConfig,
    width: int,
    height: int,
    evidence: dict[str, Any],
    reason: str,
) -> HitAssignment:
    diagonal = math.hypot(width, height)
    limit = cfg.max_assignment_distance_frac * diagonal
    evidence["max_distance_px"] = round(limit, 1)
    frame = int(evidence["frame"])

    if winner.distance > limit:
        evidence["rejected_distance_px"] = round(winner.distance, 1)
        return HitAssignment(frame, UNASSIGNED, None, "closest_player_too_far", evidence)

    # Out of reach for a racket. Measured in the winner's own body heights, because the far
    # player is half the near player's size in frame and the same real distance is half as many
    # pixels at the far end. Without a body height to scale by we cannot judge reach at all, so
    # the contact passes rather than being rejected on a guess.
    if winner_body_height:
        reach = winner.distance / winner_body_height
        evidence["distance_body_heights"] = round(reach, 3)
        if reach > cfg.max_contact_distance_body_heights:
            evidence["max_body_heights"] = cfg.max_contact_distance_body_heights
            return HitAssignment(frame, UNASSIGNED, None, "out_of_racket_reach", evidence)

    if loser is None:
        # Only one side had a detection: plausible, but we cannot compare, so separation is 0.
        return HitAssignment(frame, hitter, 0.0, reason, evidence)

    gap = loser.distance - winner.distance
    evidence["distance_gap_px"] = round(gap, 1)
    if gap < cfg.assignment_margin_px:
        evidence["margin_px"] = cfg.assignment_margin_px
        return HitAssignment(frame, UNASSIGNED, None, "ambiguous_both_players_close", evidence)

    separation = gap / max(loser.distance, 1.0)
    return HitAssignment(frame, hitter, round(min(1.0, max(0.0, separation)), 4), reason, evidence)


def enforce_alternation(assignments: list[HitAssignment], max_gap_frames: int) -> list[HitAssignment]:
    """Fill in unassigned hits from context.

    In singles, consecutive hits within one rally alternate sides. When a hit could not be
    assigned but both of its immediate neighbours were assigned to the same player, the missing
    one is almost certainly the other player. This is applied as a clearly labelled second pass
    so the raw geometric decision stays visible in the output.
    """
    result = list(assignments)
    for i, current in enumerate(result):
        if current.hitter is not None:
            continue
        if current.reason in NOT_FILLABLE_BY_ALTERNATION:
            # "Nobody could reach it" is a finding, not a gap to be filled. Alternation is for
            # contacts we could not call, and treating the two alike lets this pass hand a
            # hitter back to every contact the reach gate just rejected: on clip 5 it restored
            # four of the seven spurious contacts that gate had removed.
            continue
        previous = next((result[j] for j in range(i - 1, -1, -1) if result[j].hitter), None)
        following = next((result[j] for j in range(i + 1, len(result)) if result[j].hitter), None)
        if previous is None or following is None:
            continue
        if previous.hitter != following.hitter:
            continue
        if abs(current.frame - previous.frame) > max_gap_frames:
            continue
        if abs(following.frame - current.frame) > max_gap_frames:
            continue
        inferred = "far" if previous.hitter == "near" else "near"
        result[i] = HitAssignment(
            frame=current.frame,
            hitter=inferred,
            confidence=None,
            reason="inferred_by_alternation",
            evidence={
                **current.evidence,
                "original_reason": current.reason,
                "neighbours": [previous.frame, following.frame],
                "neighbour_hitter": previous.hitter,
            },
        )
    return result
