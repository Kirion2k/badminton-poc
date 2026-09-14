"""Rally segmentation from shuttle events.

Kept deliberately simple and rule-shaped for the first POC: a rally is a run of hits that are
close together in time, terminated by a landing, by the shuttle disappearing for a while, or by
a long silence before the next hit.

Upstream's events CLI filters its detections down to ``hit`` and ``landing`` before writing the
file, so ``out_of_frame`` never arrives from TrackNet. We recover an equivalent signal ourselves
from runs of invisible shuttle frames and tag those events ``source="shuttle_absence"`` so they
are never confused with an upstream detection.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .assoc import HitAssignment
from .parsing import BallTrack, ShuttleEvent

TERMINAL_TYPES = frozenset({"landing", "out_of_frame", "shuttle_absent"})


def detect_play_spans(
    ball: "BallTrack",
    fps: float,
    *,
    min_speed_px: float = 4.0,
    smooth_seconds: float = 0.25,
    bridge_seconds: float = 1.5,
    min_span_seconds: float = 1.0,
) -> list[tuple[int, int]]:
    """Frame spans where a rally is actually in progress, from shuttle motion alone.

    Rally segmentation was driven entirely by hits, which is circular: a hit candidate fired
    during dead time keeps a rally alive past its end or invents one outright. On the second
    test clip 22 of 33 false-positive hits fell outside real play, and an entire rally of
    eleven hits was invented between points.

    Shuttle *visibility* says nothing -- TrackNetV3 emits a shuttle on every frame whether or
    not one is in play, and between points it pins to a fixed position (y=219 throughout every
    dead period on that clip). Shuttle *motion* separates cleanly: a median of 4.85 to 11.66
    px/frame within each real rally against 0.00 in five of six dead stretches.

    Three shaping rules, each for an observed failure:

    - a rolling median rather than a raw frame-to-frame speed, because one noisy sample is not
      a rally;
    - short gaps are bridged, because a high clear leaves the top of the frame and TrackNet
      loses it for the better part of a second while play continues (frames 1750-1797 on the
      test clip, exiting at y=6 and returning at y=12);
    - spans shorter than ``min_span_seconds`` are discarded, because somebody picking the
      shuttle up off the floor is not a rally.
    """
    frames = ball.frames
    if not frames:
        return []
    half = max(1, int(round(smooth_seconds * fps / 2)))
    bridge = int(round(bridge_seconds * fps))
    min_span = int(round(min_span_seconds * fps))

    speed: dict[int, float] = {}
    previous: tuple[int, float, float] | None = None
    for frame in frames:
        sample = ball.get(frame)
        if sample is None or not sample.visible:
            continue
        if previous is not None and frame - previous[0] <= 2:
            step = math.hypot(sample.x - previous[1], sample.y - previous[2])
            speed[frame] = step / (frame - previous[0])
        previous = (frame, sample.x, sample.y)

    lo, hi = frames[0], frames[-1]
    active: list[int] = []
    for frame in range(lo, hi + 1):
        window = [speed[f] for f in range(frame - half, frame + half + 1) if f in speed]
        if not window:
            continue
        window.sort()
        if window[len(window) // 2] >= min_speed_px:
            active.append(frame)
    if not active:
        return []

    spans: list[list[int]] = [[active[0], active[0]]]
    for frame in active[1:]:
        if frame - spans[-1][1] <= bridge:
            spans[-1][1] = frame
        else:
            spans.append([frame, frame])
    return [(s, e) for s, e in spans if e - s + 1 >= min_span]


# The range each measurement takes across every hand-labelled rally in this project: clip 2's six
# and clip 3's four, ten in all, on two recordings. These are observed extremes, not thresholds.
# A rally outside them is unlike everything that has been labelled, which is a reason to look at
# it and not a reason to believe anything -- ten rallies is a small sample and a real rally that
# is short, flat or central could sit outside on any of these axes.
LABELLED_RALLY_RANGES: dict[str, tuple[float, float]] = {
    "shuttle_lateral_range_frac": (0.188, 0.568),
    "slower_slot_movement_bh_s": (0.318, 0.742),
    "median_shuttle_speed_px": (5.30, 11.88),
    "usable_shuttle_fraction": (0.81, 1.00),
}

MIN_EVIDENCE_FRAMES = 30
"""Below this the medians describe noise. Half a second at 60fps."""

MIN_USABLE_SHUTTLE = 0.15
"""Below this the shuttle measurements describe the tracker, not the shuttle."""


def _usable_samples(ball, start, end, fixed_points):
    """Shuttle samples in the span that are a shuttle rather than the tracker sitting still.

    TrackNetV3 emits one shuttle per frame whether or not one is in play and settles on a fixed
    fixture between points, so a parked sample is not evidence of a slow rally -- counting it
    would make an interval look like a rally with a stationary shuttle.
    """
    tolerance = 12.0
    out = []
    for frame in range(start, end + 1):
        sample = ball.get(frame)
        if sample is None or not sample.visible:
            continue
        if any(abs(sample.x - float(p["x"])) < tolerance and abs(sample.y - float(p["y"])) < tolerance
               for p in fixed_points):
            continue
        out.append((frame, sample.x, sample.y))
    return out


def _slot_movement(player_rows, start, end, slot, fps):
    """Median frame-to-frame travel of one slot, in its own body heights per second.

    Dividing by the tracked person's own height takes frame rate, resolution and distance from
    the camera out of it, so the number means the same on any clip.
    """
    steps: list[float] = []
    previous: tuple[int, float, float, float] | None = None
    for row in player_rows:
        frame = int(row["frame"])
        if not (start <= frame <= end):
            continue
        player = row.get(slot)
        if not player:
            previous = None
            continue
        box = player["bbox"]
        cx, cy, height = (box[0] + box[2]) / 2.0, box[3], box[3] - box[1]
        if previous is not None and frame - previous[0] == 1 and height > 0:
            steps.append(math.hypot(cx - previous[1], cy - previous[2]) / height * fps)
        previous = (frame, cx, cy, height)
    if not steps:
        return None
    steps.sort()
    return steps[len(steps) // 2]


def play_evidence(
    ball: "BallTrack",
    player_rows: list[dict[str, Any]],
    start_frame: int,
    end_frame: int,
    *,
    fps: float,
    width: int,
    fixed_points: "list[dict[str, Any]] | tuple" = (),
) -> dict[str, Any]:
    """What a detected rally looks like, measured, beside the rally it claims to be.

    Clip 3's detector invented a rally at frames 2017-2122 from two people hand-throwing shuttles
    during an interval, and the obvious cure is a gate that suppresses hits when no rally is in
    progress. Seven candidate signals were measured for it and every one of them overlaps on clip
    2, the only clip with hand labels:

      median shuttle speed   5.46 px/frame in the invented rally against 4.85-6.29 in real ones
      player motion          0.235 body heights/s against a real rally's 0.322 -- a 1.4x margin
      shuttle p90 speed      11.4 against 15.8-22.8, but shuttle collection scores 29.8
      wrist speed            4.13 against a real rally's 2.37, the wrong way round
      per-rally plausibility flags two real rallies, and the invented one scores best on sway
      shuttle lateral range  separates clip 3, but a clip 2 dead gap beats four real rallies
      slot identity switches two ids, where clip 2's real rallies reach three

    So this suppresses nothing and returns no verdict. It reports four measurements and names the
    ones that fall outside ``LABELLED_RALLY_RANGES``, which is where every hand-labelled rally in
    the project sits. On the invented rally two of the four fall outside; on all ten labelled
    rallies none does. That is an observation about a sample of ten, offered so somebody looks at
    the rally, and it is deliberately not wired to anything that drops a hit.
    """
    frames = end_frame - start_frame + 1
    note_tail = (" This does not decide anything: no signal measured so far separates a rally "
                 "from an interval on both labelled clips, so the numbers are reported and the "
                 "rally is kept.")
    if frames < MIN_EVIDENCE_FRAMES:
        return {"frames": frames, "outside_labelled_range": [],
                "note": f"span is too short to measure over ({frames} frames)." + note_tail}

    samples = _usable_samples(ball, start_frame, end_frame, list(fixed_points))
    usable = len(samples) / frames
    if usable < MIN_USABLE_SHUTTLE:
        return {"frames": frames, "usable_shuttle_fraction": round(usable, 3),
                "outside_labelled_range": [],
                "note": ("too little usable shuttle track to measure over; what is here "
                         "describes the tracker rather than a shuttle." + note_tail)}

    xs = [x for _, x, _ in samples]
    steps = [math.hypot(b[1] - a[1], b[2] - a[2])
             for a, b in zip(samples, samples[1:]) if b[0] - a[0] == 1]
    steps.sort()
    near = _slot_movement(player_rows, start_frame, end_frame, "near", fps)
    far = _slot_movement(player_rows, start_frame, end_frame, "far", fps)
    both = [v for v in (near, far) if v is not None]

    measured: dict[str, Any] = {
        "frames": frames,
        # the slower slot, not the average: a player standing still is not excused by their
        # opponent running about, and in every wrong-person case so far one slot was the problem
        "slower_slot_movement_bh_s": round(min(both), 3) if len(both) == 2 else None,
        "shuttle_lateral_range_frac": round((max(xs) - min(xs)) / max(1, width), 3),
        "median_shuttle_speed_px": round(steps[len(steps) // 2], 2) if steps else None,
        "usable_shuttle_fraction": round(usable, 3),
    }
    outside = []
    for name, (low, high) in LABELLED_RALLY_RANGES.items():
        value = measured.get(name)
        if value is None or low <= value <= high:
            continue
        outside.append({"measurement": name, "value": value,
                        "labelled_range": [low, high],
                        "direction": "below" if value < low else "above"})
    measured["outside_labelled_range"] = outside
    measured["note"] = (
        "Measurements over this detected rally, against the range every hand-labelled rally in "
        "the project falls in (ten rallies, two clips)." + note_tail
    )
    return measured


def reject_hits_outside_play(
    events: list[ShuttleEvent],
    play_spans: list[tuple[int, int]],
    *,
    fps: float,
) -> tuple[list[ShuttleEvent], list[dict[str, Any]]]:
    """Drop hit candidates that fall outside every detected play span.

    Only hits. A landing or a shuttle-absence marker is what *ends* a rally and legitimately
    sits at or past its edge, so gating those on play would remove the terminal event the
    segmentation depends on.

    With no play spans at all the gate is off rather than total: refusing every hit because the
    shuttle signal was unusable would be a worse answer than saying nothing.
    """
    if not play_spans:
        return list(events), []
    kept: list[ShuttleEvent] = []
    dropped: list[dict[str, Any]] = []
    for event in events:
        if event.event_type != "hit" or any(s <= event.frame <= e for s, e in play_spans):
            kept.append(event)
        else:
            dropped.append({
                "frame": event.frame,
                "time_s": round(event.frame / fps, 3),
                "confidence": event.confidence,
                "rule": event.rule,
                "reason": "outside detected play",
            })
    return kept, dropped


def trim_trailing_weak_hits(
    rally: dict[str, Any], min_speed: float, *, fps: float
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Walk back from a rally's last hit while the swing evidence is weak.

    A swing gate applied to every hit removes real contacts mid-rally, opening gaps longer than
    the rally break and cutting the rally in two: on the development clip that took rally
    accuracy from 4 of 6 to 1 of 6 while precision improved. Applied only to the tail it cannot
    split anything, because it never removes a hit with surviving hits after it.

    The rally always keeps at least one hit. If every candidate looks weak, something still
    happened there and reporting nothing would be worse than reporting a rally with a doubtful
    end.
    """
    if min_speed <= 0:
        return rally, []
    hits = list(rally.get("hits") or [])
    dropped: list[dict[str, Any]] = []
    while len(hits) > 1:
        peak = hits[-1].get("swing_peak_wrist_speed")
        if peak is None or peak >= min_speed:
            break
        dropped.append({**hits[-1], "reason": "weak swing at the end of a rally",
                        "threshold": min_speed})
        hits.pop()
    if not dropped:
        return rally, []
    dropped.reverse()   # report in frame order, not the order they were peeled off
    out = dict(rally)
    out["hits"] = hits
    out["end_frame"] = hits[-1]["frame"]
    out["end_s"] = round(hits[-1]["frame"] / fps, 3)
    out["duration_s"] = round(max(0.0, (out["end_frame"] - out["start_frame"]) / fps), 3)
    out["hit_count"] = len(hits)
    out["near_hits"] = sum(1 for h in hits if h.get("hitter") == "near")
    out["far_hits"] = sum(1 for h in hits if h.get("hitter") == "far")
    out["unassigned_hits"] = sum(1 for h in hits if h.get("hitter") is None)
    out["termination_reason"] = "trimmed_trailing_weak_hits"
    return out, dropped


def drop_weak_swings(
    events: list[ShuttleEvent],
    swing_by_frame: dict[int, dict[str, Any]],
    hitters: dict[int, str | None],
    min_speed: float,
    *,
    fps: float,
) -> tuple[list[ShuttleEvent], list[dict[str, Any]]]:
    """Drop hit candidates with no racket swing behind them, recording each one.

    The last lever available on rally boundaries. Ends run long because false hits during dead
    time keep a rally alive past its real end, and the swing signal is the only measured
    discriminator that does not come from the event detector itself.

    Only hits are gated: a landing is what ends a rally and has no swing associated with it.
    A candidate whose pose could not be measured is kept -- absence of evidence is not evidence
    of absence, and dropping those would silently penalise occluded frames.
    """
    if min_speed <= 0:
        return list(events), []
    kept: list[ShuttleEvent] = []
    dropped: list[dict[str, Any]] = []
    for event in events:
        if event.event_type != "hit":
            kept.append(event)
            continue
        hitter = hitters.get(event.frame)
        features = (swing_by_frame.get(event.frame) or {}).get(hitter) or {}
        peak = features.get("peak_wrist_speed")
        if peak is None or peak >= min_speed:
            kept.append(event)
        else:
            dropped.append({
                "frame": event.frame,
                "time_s": round(event.frame / fps, 3),
                "hitter": hitter,
                "peak_wrist_speed": peak,
                "threshold": min_speed,
                "reason": "no racket swing behind the candidate",
            })
    return kept, dropped


def drop_low_confidence(
    events: list[ShuttleEvent],
    min_confidence: float,
    event_type: str = "hit",
) -> tuple[list[ShuttleEvent], list[dict[str, Any]]]:
    """Discard candidates of one type below a confidence floor, recording each one.

    Applied before consolidation, not after: suppression keeps the strongest candidate in a
    neighbourhood, so a weak candidate with nothing stronger near it survives on its own and
    becomes a hit. Filtering afterwards would leave exactly those in place.

    The threshold is a judgement about upstream's rules, not a measurement. On the second test
    clip 613 of 777 hit candidates came from ``acceleration_peak`` at a flat confidence of
    0.75, and every dropped candidate is written to analysis.json so the choice can be argued
    with rather than taken on trust.
    """
    if min_confidence <= 0:
        return list(events), []
    kept: list[ShuttleEvent] = []
    dropped: list[dict[str, Any]] = []
    for event in events:
        if event.event_type == event_type and (event.confidence or 0.0) < min_confidence:
            dropped.append({
                "frame": event.frame,
                "event_type": event.event_type,
                "confidence": event.confidence,
                "rule": event.rule,
                "threshold": min_confidence,
            })
        else:
            kept.append(event)
    return kept, dropped


def consolidate_events(
    events: list[ShuttleEvent],
    fps: float,
    min_interval_seconds: float,
    event_type: str = "hit",
) -> tuple[list[ShuttleEvent], list[dict[str, Any]]]:
    """Collapse bursts of candidates of one type that describe a single physical event.

    Upstream's rule-based detector fires several times around one physical event. On our test
    clip it produced hits 1 to 8 frames apart -- up to four per second, far above the rate a
    singles rally can physically contain -- and 250 landings, 215 of which were spaced exactly
    four frames apart.

    The cause is the variable-frame-rate source. Its frames were captured 2/60 and 3/60 of a
    second apart in an alternating pattern; placed on a uniform grid, the real time between
    consecutive frames alternates 33 ms / 50 ms, so measured speed oscillates by about 20% with
    a period of two frames. That repeatedly trips the speed-step, vy-reversal and
    acceleration-peak rules, which all reason about frame-to-frame velocity.

    We keep the highest-confidence candidate in each burst. Even the fastest exchange (a net kill
    answered immediately) leaves more than a third of a second between contacts, so merging below
    that threshold cannot discard a real, separate event.

    Returns the consolidated events and a record of every merge, so the raw detector output
    remains auditable.
    """
    min_gap = max(1, int(round(min_interval_seconds * fps)))
    hits = sorted([e for e in events if e.event_type == event_type], key=lambda e: e.frame)
    others = [e for e in events if e.event_type != event_type]

    # Non-maximum suppression: take the most confident candidate, suppress everything within
    # min_gap of it, repeat. Two earlier shapes of this were both wrong, in opposite ways.
    #
    # Growing a burst while each candidate is within min_gap of the *previous* one makes
    # merging transitive, so a dense stream chains without limit: on the second test clip one
    # burst swallowed 84 candidates across 8.15 seconds and called them a single hit.
    #
    # Cutting the stream into fixed windows instead fixes the chaining but not the guarantee
    # that matters. Each window keeps its own winner, so a winner at the end of one window and
    # a winner at the start of the next survive a few frames apart -- 5% of consolidated hits
    # ended up inside 0.12s of each other, well under the 0.3s floor.
    #
    # Suppression is defined by distance from what was kept, so no two survivors can be closer
    # than min_gap and no chain can form. That is exactly the property the parameter claims.
    order = sorted(hits, key=lambda e: (-(e.confidence or 0.0), e.frame))
    kept: list[ShuttleEvent] = []
    absorbed: dict[int, list[int]] = {}

    for event in order:
        near = [k for k in kept if abs(event.frame - k.frame) <= min_gap]
        if near:
            winner = min(near, key=lambda k: (abs(event.frame - k.frame), k.frame))
            absorbed.setdefault(winner.frame, []).append(event.frame)
        else:
            kept.append(event)
            absorbed.setdefault(event.frame, [])

    audit: list[dict[str, Any]] = []
    for event in sorted(kept, key=lambda e: e.frame):
        merged_frames = sorted(absorbed[event.frame] + [event.frame])
        if len(merged_frames) > 1:
            audit.append({
                "kept_frame": event.frame,
                "merged_frames": merged_frames,
                "kept_confidence": event.confidence,
                "event_type": event_type,
                "reason": f"{event_type} candidates within {min_gap} frames of a stronger one",
            })

    return sorted(kept + others, key=lambda e: e.frame), audit


@dataclass
class TimelineEvent:
    frame: int
    event_type: str
    source: str
    hitter: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, fps: float) -> dict[str, Any]:
        payload = {
            "frame": self.frame,
            "time_s": round(self.frame / fps, 3),
            "event_type": self.event_type,
            "source": self.source,
        }
        if self.event_type == "hit":
            payload["hitter"] = self.hitter
        payload.update(self.detail)
        return payload


def derive_absence_events(ball: BallTrack, fps: float, min_seconds: float) -> list[TimelineEvent]:
    """Turn long runs of invisible shuttle into rally-termination candidates."""
    min_frames = max(1, int(round(min_seconds * fps)))
    events: list[TimelineEvent] = []
    for start, end in ball.absence_spans(min_frames):
        events.append(
            TimelineEvent(
                frame=start,
                event_type="shuttle_absent",
                source="shuttle_absence",
                detail={
                    "span_frames": end - start + 1,
                    "span_seconds": round((end - start + 1) / fps, 3),
                    "note": "derived by this project from TrackNet visibility, not an upstream detection",
                },
            )
        )
    return events


def build_timeline(
    events: list[ShuttleEvent],
    assignments: dict[int, HitAssignment],
    ball: BallTrack,
    fps: float,
    shuttle_absent_seconds: float,
    swing_by_frame: dict[int, dict[str, Any]] | None = None,
) -> list[TimelineEvent]:
    timeline: list[TimelineEvent] = []
    for event in events:
        detail: dict[str, Any] = {
            "detector_confidence": event.confidence,
            "rule": event.rule,
        }
        if event.x is not None and event.y is not None:
            detail["position"] = [round(event.x, 1), round(event.y, 1)]
        hitter = None
        if event.event_type == "hit":
            assignment = assignments.get(event.frame)
            if assignment is not None:
                hitter = assignment.hitter
                detail.update(assignment.to_dict())
            swings = (swing_by_frame or {}).get(event.frame)
            if swings:
                # Evidence, not a filter. A hit with no swing behind it is worth marking; the
                # measurement costs more recall than it buys precision if used to discard.
                detail["swing_evidence"] = {
                    "near": swings.get("near"),
                    "far": swings.get("far"),
                    # The assigned player's own numbers, lifted for convenience; both
                    # players' full feature sets stay above so a wrong assignment can be
                    # re-examined without re-running the pose model.
                    "hitter": (swings.get(hitter) if hitter in ("near", "far") else None),
                }
                detail.pop("hitter", None)
        timeline.append(
            TimelineEvent(
                frame=event.frame,
                event_type=event.event_type,
                # The event's own provenance, not a constant. Under event_source "flight" these
                # come from this project's segmentation, and stamping them "tracknet" made
                # analysis.json claim an origin the events did not have.
                source=event.source,
                hitter=hitter,
                detail=detail,
            )
        )
    timeline.extend(derive_absence_events(ball, fps, shuttle_absent_seconds))
    timeline.sort(key=lambda e: (e.frame, 0 if e.event_type == "hit" else 1))
    return timeline


def segment_rallies(
    timeline: list[TimelineEvent],
    fps: float,
    *,
    gap_seconds: float,
    min_hits: int,
    end_at_last_hit: bool = False,
) -> list[dict[str, Any]]:
    """Group hits into rallies, then attach a terminating event to each.

    Rally boundaries are driven by *hits*, not by terminal events. An earlier version split on
    any landing, which collapsed under this clip's 250 spurious landing candidates: a rally was
    cut every time the shuttle's measured speed dipped. Hits are the more reliable signal, and a
    rally is fundamentally a run of hits close together in time.

    A terminal event is then attached if one falls between the rally's last hit and the next
    rally's first hit, which is where a real landing or shuttle-absence belongs.
    """
    gap_frames = max(1, int(round(gap_seconds * fps)))
    hits = [e for e in timeline if e.event_type == "hit"]
    terminals = [e for e in timeline if e.event_type in TERMINAL_TYPES]

    groups: list[list[TimelineEvent]] = []
    for hit in hits:
        if groups and hit.frame - groups[-1][-1].frame <= gap_frames:
            groups[-1].append(hit)
        else:
            groups.append([hit])

    groups = [g for g in groups if len(g) >= min_hits]

    rallies: list[dict[str, Any]] = []
    for index, group in enumerate(groups):
        last_hit = group[-1].frame
        next_start = groups[index + 1][0].frame if index + 1 < len(groups) else None
        terminator = next(
            (
                t for t in terminals
                if t.frame >= last_hit and (next_start is None or t.frame < next_start)
            ),
            None,
        )
        rallies.append(_finalise(group, terminator, fps, len(rallies) + 1,
                                 end_at_last_hit=end_at_last_hit))
    return rallies


def _finalise(
    events: list[TimelineEvent],
    terminator: TimelineEvent | None,
    fps: float,
    index: int,
    *,
    end_at_last_hit: bool = False,
) -> dict[str, Any]:
    hits = [e for e in events if e.event_type == "hit"]
    start = hits[0].frame if hits else events[0].frame
    # Ending on the terminating event puts the boundary wherever the landing detector next
    # ticks, and on these clips it ticks every 0.8s right through dead time. Ending at the last
    # hit is at worst one stroke early, which is an error that can be reasoned about.
    if end_at_last_hit and hits:
        end = hits[-1].frame
        reason = "last_hit"
    elif terminator is not None:
        end = terminator.frame
        reason = "terminal_event"
    else:
        end = events[-1].frame
        reason = "time_gap_or_end_of_clip"
    ordered = [
        {
            "index": i,
            "frame": hit.frame,
            "time_s": round(hit.frame / fps, 3),
            "hitter": hit.hitter,
            "detector_confidence": hit.detail.get("detector_confidence"),
            "swing_peak_wrist_speed": (
                ((hit.detail.get("swing_evidence") or {}).get("hitter") or {})
                .get("peak_wrist_speed")
            ),
            "assignment_separation": hit.detail.get("assignment_separation"),
            "assignment_reason": hit.detail.get("assignment_reason"),
        }
        for i, hit in enumerate(hits)
    ]
    return {
        "rally": index,
        "start_frame": start,
        "end_frame": end,
        "start_s": round(start / fps, 3),
        "end_s": round(end / fps, 3),
        "duration_s": round(max(0.0, (end - start) / fps), 3),
        "hit_count": len(hits),
        "near_hits": sum(1 for h in hits if h.hitter == "near"),
        "far_hits": sum(1 for h in hits if h.hitter == "far"),
        "unassigned_hits": sum(1 for h in hits if h.hitter is None),
        "terminating_event": None if terminator is None else terminator.event_type,
        "terminating_event_source": None if terminator is None else terminator.source,
        "termination_reason": reason,
        "hits": ordered,
    }


def rallies_from_flight(
    model,
    timeline: list[TimelineEvent],
    fps: float,
    *,
    require_hitter: bool = False,
    require_both_players: bool = False,
    min_hits: int = 2,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Turn flight-segment chains into rally records.

    Chains already know where they start and end, so unlike :func:`segment_rallies` this does no
    grouping: it attaches the hits and the terminating landing to a span the segmentation
    decided. The span survives ``require_hitter`` untouched. Whether a player could be put
    behind a contact says something about that contact, not about whether the shuttle flew.
    """
    hits_by_frame = {e.frame: e for e in timeline if e.event_type == "hit"}
    landings = {e.frame: e for e in timeline if e.event_type == "landing"}
    rallies: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for chain in model.rallies:
        hits = [hits_by_frame[f] for f in chain.hit_frames if f in hits_by_frame]
        if require_hitter:
            keep = []
            for hit in hits:
                if hit.hitter in ("near", "far"):
                    keep.append(hit)
                else:
                    dropped.append({
                        "frame": hit.frame,
                        "time_s": round(hit.frame / fps, 3),
                        "reason": "no player close enough to have played it",
                        "assignment_reason": hit.detail.get("assignment_reason"),
                    })
            hits = keep
        if len(hits) < min_hits:
            dropped.extend({
                "frame": hit.frame,
                "time_s": round(hit.frame / fps, 3),
                "reason": f"chain kept fewer than {min_hits} hits",
            } for hit in hits)
            continue
        if require_both_players and len({h.hitter for h in hits} & {"near", "far"}) < 2:
            dropped.extend({
                "frame": hit.frame,
                "time_s": round(hit.frame / fps, 3),
                "reason": "only one player made contact, so this is not an exchange",
                "hitter": hit.hitter,
            } for hit in hits)
            continue
        rally = _finalise(hits, landings.get(chain.end_frame), fps, len(rallies) + 1)
        # The chain, not the hits, decides the span: a rally is in progress from the first
        # flight we can see to the last, and the serve that opened it may be off the clip.
        rally["start_frame"] = chain.start_frame
        rally["end_frame"] = chain.end_frame
        rally["start_s"] = round(chain.start_frame / fps, 3)
        rally["end_s"] = round(chain.end_frame / fps, 3)
        rally["duration_s"] = round(max(0.0, (chain.end_frame - chain.start_frame) / fps), 3)
        rally["termination_reason"] = "last_flight_ended"
        rally["start_kind"] = chain.start_kind
        rally["flight_segments"] = list(chain.segment_indices)
        rally["shuttle_seen_at_rest"] = chain.rest_seen_at
        rallies.append(rally)
    return rallies, dropped
