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


def score_rally_boundaries(
    detected: list[dict[str, Any]],
    active_play_ranges: list[Any],
    *,
    fps: float,
    tolerance_seconds: float = 1.0,
) -> dict[str, Any]:
    """Score rally segmentation against the hand-labelled play ranges.

    A manual ``boundary_ok`` flag per detected rally cannot express the two failures that
    matter most. A rally the pipeline invented has no real rally to be flagged against, and a
    real rally split in two collects two flags for one error. On the second test clip the flags
    reported 12.5% while the actual picture was four of six rallies bounded correctly, one real
    rally split in two, and one rally invented out of nothing.

    A real rally counts as approximately correct when exactly one detected rally overlaps it
    and both of its edges land within ``tolerance_seconds``.
    """
    real: list[tuple[int, int]] = []
    for entry in active_play_ranges or []:
        if isinstance(entry, dict):
            start, end = entry.get("start"), entry.get("end")
        else:
            if len(entry) != 2:
                raise ValueError(
                    f"active_play_ranges entry must be [start, end], got {entry!r}")
            start, end = entry[0], entry[1]
        if end is None or start is None or int(end) < int(start):
            raise ValueError(f"active_play_ranges entry {entry!r} ends before it starts")
        real.append((int(start), int(end)))
    real.sort()

    tolerance = tolerance_seconds * fps
    detail: list[dict[str, Any]] = []
    matched_detected: set[int] = set()
    correct = split = missed = 0

    for start, end in real:
        overlapping = [
            d for d in detected
            if not (d["end_frame"] < start or d["start_frame"] > end)
        ]
        for d in overlapping:
            matched_detected.add(id(d))
        row: dict[str, Any] = {
            "true_start_frame": start,
            "true_end_frame": end,
            "detected_rallies": [d.get("rally") for d in overlapping],
        }
        if not overlapping:
            missed += 1
            row["verdict"] = "missed"
        elif len(overlapping) > 1:
            split += 1
            row["verdict"] = "split across several detected rallies"
        else:
            only = overlapping[0]
            start_error = (only["start_frame"] - start) / fps
            end_error = (only["end_frame"] - end) / fps
            row["start_error_s"] = round(start_error, 3)
            row["end_error_s"] = round(end_error, 3)
            if abs(start_error) * fps <= tolerance and abs(end_error) * fps <= tolerance:
                correct += 1
                row["verdict"] = "ok"
            else:
                row["verdict"] = "edges outside tolerance"
        detail.append(row)

    spurious = [d for d in detected if id(d) not in matched_detected]
    return {
        "real_rallies": len(real),
        "detected_rallies": len(detected),
        "approximately_correct": correct,
        "accuracy": (correct / len(real)) if real else None,
        "split": split,
        "missed": missed,
        "spurious": len(spurious),
        "spurious_rallies": [d.get("rally") for d in spurious],
        "tolerance_seconds": tolerance_seconds,
        "detail": detail,
        "note": (
            "Scored against active_play_ranges. A real rally is correct when exactly one "
            "detected rally overlaps it and both edges land within the tolerance."
        ),
    }


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

    # Labels describe the run that produced them. Change a threshold and the pipeline emits a
    # different set of hits while ground_truth.json still holds the old rows; scoring those
    # yields a precision figure for a configuration nobody labelled.
    detected_frames = {
        e["frame"] for e in analysis.get("events", []) if e.get("event_type") == "hit"
    }
    labelled_frames = {h["frame"] for h in hits if h.get("frame") is not None}
    unlabelled = detected_frames - labelled_frames
    stale = labelled_frames - detected_frames
    labels_match_run = not unlabelled and not stale

    precision = _ratio(len(true_positives), len(true_positives) + len(false_positives))
    recall = _ratio(len(true_positives), len(true_positives) + len(missed))

    scorable = [h for h in true_positives if h.get("true_hitter") in {"near", "far"}]
    correct = [h for h in scorable if h.get("detected_hitter") == h.get("true_hitter")]
    hitter_accuracy = _ratio(len(correct), len(scorable))

    rallies = [r for r in ground_truth.get("rallies", []) if isinstance(r, dict)]
    judged = [r for r in rallies if r.get("boundary_ok") is not None]
    ranges = ground_truth.get("active_play_ranges") or []
    boundaries = score_rally_boundaries(
        analysis.get("rallies", []), ranges, fps=float(ground_truth.get("fps") or 0.0) or 1.0
    )
    boundaries["manual_boundary_ok"] = {
        "judged": len(judged),
        "marked_ok": sum(1 for r in judged if r["boundary_ok"]),
        "note": "The labeller's own per-detected-rally flags, kept for cross-checking.",
    }
    rally_accuracy = boundaries["accuracy"]

    active = _active_frames(ranges)
    coverage = _coverage(players, active)

    return {
        "labelled": True,
        "two_player_tracking": coverage,
        "shuttle_trajectory": {
            "visible_fraction_all_frames": analysis["summary"]["shuttle_coverage"]["visible_fraction"],
            "note": "Coherence through rallies is a visual judgement; record it in notes.",
        },
        "hit_detection": _hit_detection(
            true_positives, false_positives, missed, precision, recall,
            labels_match_run, unlabelled, stale, len(detected_frames),
        ),
        "hitter_assignment": {
            "scored_hits": len(scorable),
            "correct": len(correct),
            "accuracy": hitter_accuracy,
        },
        "rally_boundaries": boundaries,
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


def _hit_detection(
    true_positives: list[Any],
    false_positives: list[Any],
    missed: list[Any],
    precision: float | None,
    recall: float | None,
    labels_match_run: bool,
    unlabelled: set[int],
    stale: set[int],
    detected: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "scorable": labels_match_run,
        "detected_hits": detected,
        "labelled_hits": len(true_positives) + len(false_positives),
        "unlabelled_hits": len(unlabelled),
        "labelled_but_not_detected": len(stale),
    }
    if not labels_match_run:
        out["note"] = (
            "Precision and recall are withheld: the labels do not describe this run. "
            f"{len(unlabelled)} detected hits have no label and {len(stale)} labelled hits are "
            "no longer produced, which happens when a threshold or filter changed after "
            "labelling. Re-label against the current output, or restore the configuration the "
            "labels were made against."
        )
        return out
    out.update({
        "true_positives": len(true_positives),
        "false_positives": len(false_positives),
        "missed": len(missed),
        "precision": precision,
        "recall": recall,
    })
    return out


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
