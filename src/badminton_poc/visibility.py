"""Why a player slot is empty: a diagnostic over ``players.jsonl``, reported and never acted on.

Two-player coverage says how often a slot was filled. On a low camera that number mixes two
different things -- a far player standing behind the near one, who no detector could see, and a
far player in plain view whom the tracker dropped -- and they need different answers. This names
the likely reason for every empty slot frame. It never fills a slot, never moves a box and never
changes coverage: a slot is ``observed`` exactly when ``players.jsonl`` has it.

States, first match wins:

``observed``             the slot has a detection this frame.
``candidate_not_selected``  a person on this slot's side of the net was detected and passed the
                         court filter but was not chosen: a selection miss, not a visibility one.
``occluded``             the slot's last observed box is at least ``OCCLUDED_COVER`` covered by
                         the other player's box this frame.
``at_frame_edge``        the slot's last observed box touched the image border: the player is
                         partly or wholly out of frame. On clip 6 this is mostly a near player
                         still in shot with their feet cut off, whom the pose model then drops.
``unknown``              none of the above; the reason is not in the data. On clip 6 this
                         includes players walking off court between points (detected, then
                         rejected by the court filter) and a far player in plain view whom the
                         selection lost.
``never_seen``           the slot has not been observed yet in this clip.
"""
from __future__ import annotations

from typing import Any, Iterable

SLOTS = ("near", "far")
STATES = (
    "observed", "candidate_not_selected", "occluded", "at_frame_edge", "unknown", "never_seen",
)

OCCLUDED_COVER = 0.5
"""Measured on clips 2, 3, 5 and 6, far slot, near player present:

    observed far box covered >= 0.5:        1.3%, 1.5%, 11.2%, 15.2% of frames
    missing far player's last box >= 0.5:  19.6%, 92.9%, 69.6%, 67.5%
    missing, cover 0.4-0.6:                 0.2%, 21.4%, 11.6%,  4.3%

So 0.5 separates the two populations on every clip, but not cleanly: boxes are rectangles, and
on the low cameras (5, 6) a far player is still detected behind 11-15% box cover. It reads as
"likely", and ``max_cover`` is reported beside every gap so it can be checked. Clip 3's middle
band is 6 of 28 frames."""

EDGE_FRAC = 0.01
"""A box within 1% of the frame size of a border is touching it."""

GAP_BUCKETS_S = (0.25, 1.0, 3.0)


def box_cover(box: list[float], by: list[float]) -> float:
    """Fraction of ``box``'s area that ``by`` covers."""
    ix = max(0.0, min(box[2], by[2]) - max(box[0], by[0]))
    iy = max(0.0, min(box[3], by[3]) - max(box[1], by[1]))
    area = (box[2] - box[0]) * (box[3] - box[1])
    return ix * iy / area if area > 0 else 0.0


def _touches_edge(box: list[float], width: float, height: float) -> bool:
    mx, my = EDGE_FRAC * width, EDGE_FRAC * height
    return box[0] <= mx or box[1] <= my or box[2] >= width - mx or box[3] >= height - my


def classify(player_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per frame, per slot: ``{"frame", "near": (state, cover), "far": (state, cover)}``.

    ``cover`` is how much of the slot's last observed box the other player covers now, or None
    when either box is unavailable.
    """
    last: dict[str, list[float] | None] = {s: None for s in SLOTS}
    out = []
    for row in player_rows:
        width, height = float(row.get("width") or 0), float(row.get("height") or 0)
        entry: dict[str, Any] = {"frame": int(row["frame"])}
        for slot, other in (("near", "far"), ("far", "near")):
            if row.get(slot):
                last[slot] = row[slot]["bbox"]
                entry[slot] = ("observed", None)
                continue
            box = last[slot]
            cover = box_cover(box, row[other]["bbox"]) if box and row.get(other) else None
            if box is None:
                state = "never_seen"
            elif any(c.get("side") == slot for c in row.get("not_selected") or []):
                state = "candidate_not_selected"
            elif cover is not None and cover >= OCCLUDED_COVER:
                state = "occluded"
            elif width and height and _touches_edge(box, width, height):
                state = "at_frame_edge"
            else:
                state = "unknown"
            entry[slot] = (state, None if cover is None else round(cover, 3))
        out.append(entry)
    return out


def _gaps(frames: list[dict[str, Any]], slot: str) -> list[dict[str, Any]]:
    """Runs of consecutive frames with this slot empty, labelled by their commonest state."""
    gaps, run = [], []
    for entry in frames + [None]:
        if entry is not None and entry[slot][0] != "observed":
            run.append(entry)
            continue
        if run:
            counts: dict[str, int] = {}
            for e in run:
                counts[e[slot][0]] = counts.get(e[slot][0], 0) + 1
            covers = [e[slot][1] for e in run if e[slot][1] is not None]
            gaps.append({
                "start_frame": run[0]["frame"], "end_frame": run[-1]["frame"],
                "frames": len(run), "state": max(counts, key=counts.get), "states": counts,
                "max_cover": max(covers) if covers else None,
            })
            run = []
    return gaps


def summarise(
    player_rows: list[dict[str, Any]], fps: float, frames: Iterable[int] | None = None,
    examples: int = 5,
) -> dict[str, Any]:
    """The report. ``frames`` restricts it (e.g. to detected rallies); gaps are cut at its edges."""
    classified = classify(player_rows)
    if frames is not None:
        wanted = set(frames)
        classified = [e for e in classified if e["frame"] in wanted]
    n = len(classified)
    out: dict[str, Any] = {"frames": n}
    if not n:
        return out
    both = sum(1 for e in classified if e["near"][0] == e["far"][0] == "observed")
    out["both_observed_fraction"] = round(both / n, 4)
    out["one_occluded_frames"] = sum(
        1 for e in classified if "occluded" in (e["near"][0], e["far"][0]))
    out["unexplained_missing_frames"] = sum(
        1 for e in classified if "unknown" in (e["near"][0], e["far"][0]))
    for slot in SLOTS:
        counts = {s: 0 for s in STATES}
        for e in classified:
            counts[e[slot][0]] += 1
        # Split where frame numbers jump, so a restriction to rallies does not join two gaps.
        gaps = []
        segment: list[dict[str, Any]] = []
        for e in classified:
            if segment and e["frame"] != segment[-1]["frame"] + 1:
                gaps += _gaps(segment, slot)
                segment = []
            segment.append(e)
        gaps += _gaps(segment, slot)
        buckets: dict[str, dict[str, int]] = {}
        for gap in gaps:
            seconds = gap["frames"] / fps
            label = next((f"<{b}s" for b in GAP_BUCKETS_S if seconds < b), f">={GAP_BUCKETS_S[-1]}s")
            buckets.setdefault(gap["state"], {}).setdefault(label, 0)
            buckets[gap["state"]][label] += 1
        out[slot] = {
            "observed_fraction": round(counts["observed"] / n, 4),
            "frames_by_state": counts,
            "gaps": len(gaps),
            "gap_durations_by_state": buckets,
            "longest_gaps": [
                {**g, "start_s": round(g["start_frame"] / fps, 2), "seconds": round(g["frames"] / fps, 2)}
                for g in sorted(gaps, key=lambda g: -g["frames"])[:examples]
            ],
        }
    out["note"] = (
        "Diagnostic only. Every state but 'observed' is a likely reason for an empty slot, read "
        "from the slot's last observed box; no position is predicted and coverage is unchanged."
    )
    return out
