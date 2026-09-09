"""Manual evaluation support.

The pipeline can count what it produced, but it cannot tell you whether those counts are right.
This module does two things:

1. Writes a ground-truth template pre-filled with what the pipeline detected, so labelling means
   correcting a list rather than building one from nothing.
2. Scores a completed ground-truth file against analysis.json to produce the six numbers the
   POC is actually judged on.

Nothing here invents an accuracy figure. Until ``ground_truth.json`` has ``labelled: true`` and
real entries, ``evaluate`` refuses to report precision, recall or accuracy.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

TEMPLATE_NOTE = (
    "Fill this in by watching annotated.mp4. Set labelled=true when done. "
    "Frame numbers refer to the working video (outputs/<clip>/work.mp4); metadata.json records "
    "the offset back to the source video."
)


def build_template(analysis: dict[str, Any], fps: float, active_play_hint: list[list[int]] | None = None) -> dict[str, Any]:
    """Pre-fill a ground-truth file from the detections, for a human to correct."""
    detected_hits = [
        {
            "frame": event["frame"],
            "time_s": event["time_s"],
            "detected_hitter": event.get("hitter"),
            "true_hitter": event.get("hitter"),
            "is_real_hit": True,
            "notes": "",
        }
        for event in analysis.get("events", [])
        if event.get("event_type") == "hit"
    ]
    detected_rallies = [
        {
            "rally": rally["rally"],
            "detected_start_frame": rally["start_frame"],
            "detected_end_frame": rally["end_frame"],
            "true_start_frame": rally["start_frame"],
            "true_end_frame": rally["end_frame"],
            "boundary_ok": None,
            "notes": "",
        }
        for rally in analysis.get("rallies", [])
    ]
    return {
        "_instructions": TEMPLATE_NOTE,
        "labelled": False,
        "fps": fps,
        "active_play_ranges": active_play_hint or [],
        "_active_play_help": (
            "Inclusive [start_frame, end_frame] ranges where a rally is actually in progress. "
            "The >=90% two-player tracking target is measured over these frames only, because "
            "coverage between rallies is not what the criterion is about."
        ),
        "missed_hits": [],
        "_missed_hits_help": "Frames of real hits the pipeline did not detect at all: [{\"frame\": 123}]",
        "hits": detected_hits,
        "_hits_help": (
            "One row per detected hit. Set is_real_hit=false for a false positive. "
            "Correct true_hitter to 'near' or 'far' where the assignment is wrong."
        ),
        "rallies": detected_rallies,
        "_rallies_help": "Set boundary_ok=true/false for each detected rally.",
    }


def write_template(path: str | Path, analysis: dict[str, Any], fps: float) -> Path:
    out = Path(path)
    if out.exists():
        return out  # never clobber labelling work
    out.parent.mkdir(parents=True, exist_ok=True)
    # Seed active_play_ranges from the detected rallies so labelling starts from a draft, but
    # they must still be corrected by hand: a rally the pipeline missed is not in this list.
    hint = [[r["start_frame"], r["end_frame"]] for r in analysis.get("rallies", [])]
    out.write_text(
        json.dumps(build_template(analysis, fps, hint), indent=2), encoding="utf-8"
    )
    return out


class NotLabelledError(RuntimeError):
    pass


def evaluate(analysis: dict[str, Any], ground_truth: dict[str, Any], players: list[dict[str, Any]]) -> dict[str, Any]:
    if not ground_truth.get("labelled"):
        raise NotLabelledError(
            "ground_truth.json still has labelled=false. Fill it in from annotated.mp4 first; "
            "reporting accuracy from unlabelled data would be fabricating results."
        )

    hits = [h for h in ground_truth.get("hits", []) if isinstance(h, dict)]
    if not hits:
        raise NotLabelledError("ground_truth.json has labelled=true but no hit rows to score.")

    true_positives = [h for h in hits if h.get("is_real_hit")]
    false_positives = [h for h in hits if not h.get("is_real_hit")]
    missed = ground_truth.get("missed_hits", [])

    precision = _ratio(len(true_positives), len(true_positives) + len(false_positives))
    recall = _ratio(len(true_positives), len(true_positives) + len(missed))

    scorable = [h for h in true_positives if h.get("true_hitter") in {"near", "far"}]
    correct = [h for h in scorable if h.get("detected_hitter") == h.get("true_hitter")]
    hitter_accuracy = _ratio(len(correct), len(scorable))

    rallies = [r for r in ground_truth.get("rallies", []) if isinstance(r, dict)]
    judged = [r for r in rallies if r.get("boundary_ok") is not None]
    rally_accuracy = _ratio(sum(1 for r in judged if r["boundary_ok"]), len(judged))

    active = _active_frames(ground_truth.get("active_play_ranges") or [])
    coverage = _coverage(players, active)

    return {
        "labelled": True,
        "two_player_tracking": coverage,
        "shuttle_trajectory": {
            "visible_fraction_all_frames": analysis["summary"]["shuttle_coverage"]["visible_fraction"],
            "note": "Coherence through rallies is a visual judgement; record it in notes.",
        },
        "hit_detection": {
            "true_positives": len(true_positives),
            "false_positives": len(false_positives),
            "missed": len(missed),
            "precision": precision,
            "recall": recall,
        },
        "hitter_assignment": {
            "scored_hits": len(scorable),
            "correct": len(correct),
            "accuracy": hitter_accuracy,
        },
        "rally_boundaries": {
            "judged": len(judged),
            "approximately_correct": sum(1 for r in judged if r["boundary_ok"]),
            "accuracy": rally_accuracy,
        },
        "targets": _targets(coverage.get("both_players_fraction"), hitter_accuracy, rally_accuracy),
    }


def _targets(coverage: float | None, hitter: float | None, rally: float | None) -> dict[str, Any]:
    def verdict(value: float | None, threshold: float) -> str:
        if value is None:
            return "not measurable (missing labels)"
        return "PASS" if value >= threshold else "FAIL"

    return {
        "two_player_coverage_active_play >= 0.90": verdict(coverage, 0.90),
        "hitter_assignment_accuracy >= 0.90": verdict(hitter, 0.90),
        "rally_boundary_accuracy >= 0.85": verdict(rally, 0.85),
    }


def _active_frames(ranges: list[list[int]]) -> set[int] | None:
    if not ranges:
        return None
    frames: set[int] = set()
    for entry in ranges:
        if len(entry) != 2:
            raise ValueError(f"active_play_ranges entry must be [start, end], got {entry!r}")
        start, end = int(entry[0]), int(entry[1])
        if end < start:
            raise ValueError(f"active_play_ranges entry {entry!r} ends before it starts")
        frames.update(range(start, end + 1))
    return frames


def _coverage(players: list[dict[str, Any]], active: set[int] | None) -> dict[str, Any]:
    if active is None:
        return {
            "both_players_fraction": None,
            "note": (
                "active_play_ranges is empty, so the acceptance-criteria coverage number cannot "
                "be computed. Coverage over all frames is in analysis.json, but it includes "
                "between-rally frames and is not what the >=90% target refers to."
            ),
        }
    rows = [r for r in players if int(r["frame"]) in active]
    both = sum(1 for r in rows if r.get("near") and r.get("far"))
    return {
        "active_play_frames": len(rows),
        "both_players_frames": both,
        "both_players_fraction": _ratio(both, len(rows)),
        "near_only": sum(1 for r in rows if r.get("near") and not r.get("far")),
        "far_only": sum(1 for r in rows if r.get("far") and not r.get("near")),
        "neither": sum(1 for r in rows if not r.get("near") and not r.get("far")),
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None
