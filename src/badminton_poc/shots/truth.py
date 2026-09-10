"""Hand labels for shot type, attached to the Milestone 1 event timeline.

The file is seeded from the shot samples, which come from the hit timeline, and **never** from a
classifier's output. Seeding a label with a prediction and asking a human to correct it produces
agreement with the prediction, not truth, so the ``shot_type`` field starts empty and the scorer
refuses to run while any row is still unreviewed.

That refusal is the same guard Milestone 1 uses. It exists because this project has already been
caught scoring a detector against labelling that had been seeded from that detector's own output.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .taxonomy import BROAD_CLASSES, MAPPING_VERSION
from .windows import ShotSample

TRUTH_FORMAT = "badminton-poc/shot-truth-v1"

LABEL_GUIDE = {
    "serve": "the contact that starts a rally",
    "smash": "steep and fast downward, struck from above the hitter",
    "drop": "struck from the rear court, falling gently into the opponent's forecourt",
    "clear": "high and deep, landing near the opponent's back line",
    "lift": "struck upward from the forecourt, sending the shuttle high and deep",
    "drive": "flat and fast, travelling roughly parallel to the floor",
    "net": "played in the forecourt and staying in the forecourt, including a block off a smash",
    "other": (
        "a real contact that is none of the above, or that you cannot call from the video: "
        "pushes, net kills, mishits, and anything ambiguous"
    ),
}


class NotLabelledError(RuntimeError):
    pass


def empty_truth(clip: str, samples: Iterable[ShotSample], fps: float) -> dict[str, Any]:
    """A blank labelling file, one row per shot, with no shot type filled in."""
    rows = []
    for sample in samples:
        rows.append({
            "shot_id": sample.shot_id,
            "rally": sample.rally,
            "index_in_rally": sample.index_in_rally,
            "hit_frame": sample.hit_frame,
            "hit_time_s": sample.hit_time_s,
            "hitter": sample.hitter,
            "is_first_in_rally": sample.is_first_in_rally,
            "shot_type": None,
            "uncertain": False,
            "hitter_looks_wrong": False,
            "not_a_real_contact": False,
            "notes": "",
            "reviewed": False,
        })
    return {
        "_format": TRUTH_FORMAT,
        "_instructions": (
            "Watch the preview clip for each shot and set shot_type to what you saw. Use only "
            "the classes in _classes. Set uncertain=true rather than guessing. Set "
            "not_a_real_contact=true if the pipeline invented this hit, and hitter_looks_wrong=true "
            "if the wrong player is credited. Set reviewed=true on each row you have judged, and "
            "labelled=true at the top when the file is finished. Nothing here should describe a "
            "classifier; label what happened in the video."
        ),
        "_classes": list(BROAD_CLASSES),
        "_class_guide": dict(LABEL_GUIDE),
        "_taxonomy_version": MAPPING_VERSION,
        "_seeded_from": (
            "the Milestone 1 hit timeline only. shot_type is deliberately empty: seeding it from "
            "a classifier would measure agreement with that classifier rather than accuracy."
        ),
        "clip": clip,
        "fps": fps,
        "labelled": False,
        "shots": rows,
    }


def load(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("_format") != TRUTH_FORMAT:
        raise ValueError(f"{path} is not {TRUTH_FORMAT}: got {data.get('_format')!r}")
    return data


def save(path: str | Path, truth: dict[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(truth, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def refresh(existing: dict[str, Any], samples: Iterable[ShotSample], fps: float) -> dict[str, Any]:
    """Re-seed against a new run while keeping every judgement already made.

    Rows are carried across by hit frame, not by shot id, because a change upstream renumbers
    shots. A label whose hit frame no longer exists is kept in ``orphaned`` rather than dropped,
    so a hand judgement is never silently lost.
    """
    by_frame = {int(r["hit_frame"]): r for r in existing.get("shots", []) if r.get("reviewed")}
    fresh = empty_truth(existing.get("clip", ""), samples, fps)
    used = set()
    for row in fresh["shots"]:
        old = by_frame.get(row["hit_frame"])
        if old is not None:
            used.add(row["hit_frame"])
            for key in ("shot_type", "uncertain", "hitter_looks_wrong",
                        "not_a_real_contact", "notes", "reviewed"):
                if key in old:
                    row[key] = old[key]
    orphaned = [r for f, r in by_frame.items() if f not in used]
    fresh["labelled"] = existing.get("labelled", False)
    if orphaned:
        fresh["orphaned"] = orphaned
        fresh["labelled"] = False
        fresh["_orphan_warning"] = (
            f"{len(orphaned)} judged rows no longer match any detected hit and are parked in "
            "'orphaned'. Reconcile them before setting labelled=true again."
        )
    return fresh


def labelled_rows(truth: dict[str, Any]) -> list[dict[str, Any]]:
    """The rows a scorer may use, refusing anything unfinished or self-contradictory."""
    if not truth.get("labelled"):
        raise NotLabelledError(
            "shot ground truth has labelled=false. Finish it before reporting accuracy; "
            "reporting from an unfinished file would be fabricating results."
        )
    rows = truth.get("shots") or []
    unreviewed = [r for r in rows if not r.get("reviewed")]
    if unreviewed:
        raise NotLabelledError(
            f"{len(unreviewed)} shots are still unreviewed but the file claims to be labelled."
        )
    usable = []
    for row in rows:
        if row.get("not_a_real_contact"):
            continue
        label = row.get("shot_type")
        if label is None:
            raise NotLabelledError(
                f"shot {row.get('shot_id')} is reviewed but has no shot_type. Use 'other' if you "
                "could not call it, rather than leaving it blank."
            )
        if label not in BROAD_CLASSES:
            raise ValueError(
                f"shot {row.get('shot_id')} has shot_type {label!r}, which is not one of "
                f"{list(BROAD_CLASSES)}"
            )
        usable.append(row)
    return usable


def summarise(truth: dict[str, Any]) -> dict[str, Any]:
    rows = truth.get("shots") or []
    counts: dict[str, int] = {}
    for row in rows:
        if row.get("shot_type"):
            counts[row["shot_type"]] = counts.get(row["shot_type"], 0) + 1
    return {
        "shots": len(rows),
        "reviewed": sum(1 for r in rows if r.get("reviewed")),
        "labelled": bool(truth.get("labelled")),
        "uncertain": sum(1 for r in rows if r.get("uncertain")),
        "not_a_real_contact": sum(1 for r in rows if r.get("not_a_real_contact")),
        "hitter_looks_wrong": sum(1 for r in rows if r.get("hitter_looks_wrong")),
        "class_counts": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
    }
