"""A deliberately simple, readable shot classifier built from Milestone 1 data alone.

Its job is not to be good. It is to say what a few obvious rules are worth, so that any claim
about the pretrained model is a claim about the *difference*. Without it, an accuracy of 0.55
means nothing: it could be excellent or it could be worse than asking whether the shuttle went
up or down.

Every rule below is one line of badminton, in this order, first match wins:

    the first contact of a rally is a serve
    the shuttle leaves fast and steeply downward, from above the hitter        -> smash
    the shuttle leaves upward and ends deep                                    -> lift or clear
    the shuttle stays in the forecourt and travels slowly                      -> net
    the shuttle leaves roughly flat and fast                                   -> drive
    the shuttle leaves downward from the rear court and lands short            -> drop
    nothing matched                                                            -> other

All the geometry is normalized: positions by frame size, speeds in frame-heights per second, and
the vertical sign is flipped so positive means *upward* in the world rather than downward in the
image. No pixel or pixel-per-frame quantity escapes this module, so the rules mean the same thing
on a clip of a different size or frame rate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .court import CourtDepth
from .features import ShotFeatures
from .taxonomy import BROAD_CLASSES


@dataclass(frozen=True)
class BaselineConfig:
    outgoing_seconds: float = 0.25
    """How long after the contact to measure the departing shuttle."""
    incoming_seconds: float = 0.20
    smash_speed: float = 1.2
    """Frame-heights per second. A smash crosses the court in a fraction of a second."""
    smash_descent: float = 0.45
    """Fraction of the departing motion that must be downward for a smash."""
    fast_speed: float = 0.9
    flat_band: float = 0.25
    """Below this absolute vertical fraction the flight counts as flat."""
    rising: float = 0.25
    """Above this upward fraction the shuttle is climbing."""
    forecourt: float = 0.28
    """Depth from the net, below which a player counts as being in the forecourt."""
    deep: float = 0.40
    """Depth from the net, above which a player counts as being at the back.

    These two are placed against the observed spread of foot depth on the development clip,
    which runs from about 0.19 to 0.48, not against any shot label. The scale is compressed
    because the court polygon's near edge closes on the bottom of the frame rather than on the
    real baseline, which runs out of shot."""
    slow_speed: float = 0.55
    min_observations: int = 3


@dataclass(frozen=True)
class BaselineFeatures:
    """The handful of numbers the rules read. Kept separate so they can be inspected."""
    outgoing_speed: float | None
    outgoing_rise: float | None
    """Upward fraction of the departing motion: +1 straight up, -1 straight down."""
    contact_height_above_hitter: float | None
    """Contact height relative to the hitter's body, in body heights. Positive is overhead."""
    hitter_depth: float | None
    """0 at the hitter's net, 1 at the hitter's own baseline. None without a calibrated net."""
    destination_depth: float | None
    """How deep the *receiver* was standing when they met this shot, from their own net.

    Deliberately the receiver's feet rather than the shuttle. A court depth is derived from the
    line where the net meets the floor, so it only means anything for a point on the floor. This
    project already learned that once: judging which side of the net an airborne shuttle was over
    called 33 of 50 hit assignments wrong. Feet are on the floor; a shuttle in flight is not."""
    observations: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "outgoing_speed_frame_heights_per_s": _round(self.outgoing_speed),
            "outgoing_rise_fraction": _round(self.outgoing_rise),
            "contact_height_above_hitter_body_heights": _round(self.contact_height_above_hitter),
            "hitter_depth_from_net": _round(self.hitter_depth),
            "destination_depth_from_net": _round(self.destination_depth),
            "observations": self.observations,
            "units": (
                "speeds in frame-heights per second; rise is the upward fraction of motion with "
                "image y inverted so positive is up; depths are 0 at the net and 1 at a baseline"
            ),
        }


def _round(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(float(value), digits)


def extract(
    features: ShotFeatures,
    cfg: BaselineConfig | None = None,
    depth: CourtDepth | None = None,
) -> BaselineFeatures:
    cfg = cfg or BaselineConfig()
    times = features.times_s
    shuttle = features.shuttle
    mask = features.shuttle_mask
    side = features.sample.hitter or "near"
    other = "far" if side == "near" else "near"

    after = mask & (times >= 0) & (times <= cfg.outgoing_seconds)
    n_after = int(after.sum())
    speed = rise = None
    if n_after >= cfg.min_observations:
        pts = shuttle[after]
        ts = times[after]
        span = float(ts[-1] - ts[0])
        if span > 0:
            dx = float(pts[-1, 0] - pts[0, 0])
            dy = float(pts[-1, 1] - pts[0, 1])
            distance = float(np.hypot(dx, dy))
            speed = distance / span
            if distance > 1e-6:
                # Image y grows downward, so a negative dy is the shuttle climbing.
                rise = -dy / distance

    hitter = features.hitter_index
    at_contact = int(np.argmin(np.abs(times)))
    box = features.bbox[at_contact, hitter]
    body_px = float(box[3] - box[1])

    contact_height = None
    if body_px > 0 and features.shuttle_mask[at_contact]:
        # Shuttle height above the hitter's feet, in body heights. Above 1 is over the head.
        feet_y = float(box[3]) / features.sample.height
        shuttle_y = float(features.shuttle[at_contact, 1])
        contact_height = (feet_y - shuttle_y) / (body_px / features.sample.height)

    hitter_depth = None
    if depth is not None and features.position_mask[at_contact, hitter]:
        px, py = features.position[at_contact, hitter]
        hitter_depth = depth.depth(float(px), float(py), side)

    destination_depth = None
    sample = features.sample
    if depth is not None and sample.next_hit_frame is not None:
        arrival = (sample.next_hit_frame - sample.hit_frame) / sample.fps
        at_arrival = int(np.argmin(np.abs(times - arrival)))
        receiver = 1 - hitter
        if features.position_mask[at_arrival, receiver]:
            rx, ry = features.position[at_arrival, receiver]
            destination_depth = depth.depth(float(rx), float(ry), other)

    return BaselineFeatures(
        outgoing_speed=speed,
        outgoing_rise=rise,
        contact_height_above_hitter=contact_height,
        hitter_depth=hitter_depth,
        destination_depth=destination_depth,
        observations=n_after,
    )


def classify(
    features: ShotFeatures,
    cfg: BaselineConfig | None = None,
    depth: CourtDepth | None = None,
) -> tuple[str, dict[str, Any]]:
    """Broad class plus the reason, so a wrong answer can be argued with."""
    cfg = cfg or BaselineConfig()
    f = extract(features, cfg, depth)

    if features.sample.is_first_in_rally:
        return "serve", {"rule": "first contact of the rally", "features": f.to_dict()}

    if f.outgoing_speed is None or f.outgoing_rise is None:
        return "other", {"rule": "not enough shuttle track after the contact to say",
                         "features": f.to_dict()}

    fast = f.outgoing_speed >= cfg.smash_speed
    steeply_down = f.outgoing_rise <= -cfg.smash_descent
    overhead = f.contact_height_above_hitter is not None and f.contact_height_above_hitter > 0.9

    if fast and steeply_down and overhead:
        return "smash", {"rule": "fast, steeply downward, struck above the hitter",
                         "features": f.to_dict()}
    if f.outgoing_rise >= cfg.rising:
        deep = f.destination_depth is not None and f.destination_depth >= cfg.deep
        near_the_net = f.hitter_depth is not None and f.hitter_depth <= cfg.forecourt
        if near_the_net and deep:
            return "lift", {"rule": "struck upward from the forecourt and lands deep",
                            "features": f.to_dict()}
        if deep:
            return "clear", {"rule": "struck upward and lands deep",
                             "features": f.to_dict()}
        return "lift", {"rule": "struck upward", "features": f.to_dict()}
    if (f.hitter_depth is not None and f.hitter_depth <= cfg.forecourt
            and f.destination_depth is not None and f.destination_depth <= cfg.forecourt
            and f.outgoing_speed < cfg.slow_speed):
        return "net", {"rule": "played in the forecourt, stays in the forecourt, slow",
                       "features": f.to_dict()}
    if abs(f.outgoing_rise) <= cfg.flat_band and f.outgoing_speed >= cfg.fast_speed:
        return "drive", {"rule": "flat and fast", "features": f.to_dict()}
    if f.outgoing_rise < 0 and f.destination_depth is not None \
            and f.destination_depth <= cfg.forecourt:
        return "drop", {"rule": "struck downward and lands short",
                        "features": f.to_dict()}
    return "other", {"rule": "no rule matched", "features": f.to_dict()}


def predict(
    features_list, cfg: BaselineConfig | None = None, depth: CourtDepth | None = None
) -> list[dict[str, Any]]:
    out = []
    for features in features_list:
        broad, why = classify(features, cfg, depth)
        scores = {c: 0.0 for c in BROAD_CLASSES}
        # The rules are hard decisions, so the "score" is only a flag that one fired at all.
        scores[broad] = 1.0
        out.append({
            "clip": features.sample.clip,
            "shot_id": features.sample.shot_id,
            "hit_frame": features.sample.hit_frame,
            "hitter": features.sample.hitter,
            "source_model": "rules",
            "broad_prediction": broad,
            "broad_score": 1.0,
            "broad_scores": scores,
            "rule": why["rule"],
            "features": why["features"],
        })
    return out
