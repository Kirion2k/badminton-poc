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

from dataclasses import dataclass, field
from typing import Any

from .assoc import HitAssignment
from .parsing import BallTrack, ShuttleEvent

TERMINAL_TYPES = frozenset({"landing", "out_of_frame", "shuttle_absent"})


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

    merged: list[ShuttleEvent] = []
    audit: list[dict[str, Any]] = []
    burst: list[ShuttleEvent] = []

    def flush() -> None:
        if not burst:
            return
        best = max(burst, key=lambda e: (e.confidence or 0.0, -abs(e.frame - burst[0].frame)))
        merged.append(best)
        if len(burst) > 1:
            audit.append({
                "kept_frame": best.frame,
                "merged_frames": [e.frame for e in burst],
                "kept_confidence": best.confidence,
                "event_type": event_type,
                "reason": f"{event_type} candidates within {min_gap} frames of each other",
            })

    for event in hits:
        if burst and event.frame - burst[-1].frame > min_gap:
            flush()
            burst = []
        burst.append(event)
    flush()

    return sorted(merged + others, key=lambda e: e.frame), audit


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
                detail.pop("hitter", None)
        timeline.append(
            TimelineEvent(
                frame=event.frame,
                event_type=event.event_type,
                source="tracknet",
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
        rallies.append(_finalise(group, terminator, fps, len(rallies) + 1))
    return rallies


def _finalise(
    events: list[TimelineEvent],
    terminator: TimelineEvent | None,
    fps: float,
    index: int,
) -> dict[str, Any]:
    hits = [e for e in events if e.event_type == "hit"]
    start = hits[0].frame if hits else events[0].frame
    end = terminator.frame if terminator is not None else events[-1].frame
    ordered = [
        {
            "index": i,
            "frame": hit.frame,
            "time_s": round(hit.frame / fps, 3),
            "hitter": hit.hitter,
            "detector_confidence": hit.detail.get("detector_confidence"),
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
        "termination_reason": "terminal_event" if terminator is not None else "time_gap_or_end_of_clip",
        "hits": ordered,
    }
