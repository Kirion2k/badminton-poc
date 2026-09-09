"""Detector-independent ground truth, and scoring predictions against it.

The first labelling format stored one row per *detected candidate* with an ``is_real_hit``
flag. That makes the labels a property of the configuration that produced them: "the candidate
at frame 140 was not a hit" says nothing to a run that never emitted frame 140, and a real
contact the detector missed entirely exists only if someone remembers to add it by hand. On the
development clip that produced a precision figure for a configuration nobody had labelled, and
needed a frame-matching migration command every time a threshold moved.

This format stores only what is true of the video:

* ``contacts`` -- every racket contact that happened, with the player who made it;
* ``rallies`` -- the frame spans where a rally was actually in progress;
* ``tracking_failures`` -- optional notes about where the vision stack visibly went wrong.

Nothing about what any detector emitted. A false positive is not recorded, because it is not a
property of the video: it is re-derived for whatever configuration is being scored, as any
prediction that matches no contact. One labelling therefore scores every configuration, which
is the whole point of having labelled.

Frame numbers refer to the working video (``outputs/<clip>/work.mp4``); ``metadata.json``
records the offset back to the source.
"""
from __future__ import annotations

import statistics
from typing import Any, Iterable

TRUTH_FORMAT = "badminton-poc/truth-v2"

# A racket contact is an instant, but the detector places it a few frames either side of the
# real moment: the shuttle's direction change is what it sees, not the impact. A quarter of a
# second is comfortably inside the fastest interval between two real contacts (0.3s) so it
# cannot merge two of them, and wide enough to absorb that placement error.
DEFAULT_MATCH_TOLERANCE_S = 0.25

# A rally edge is a judgement call for a human too -- play stops over a few frames, not one --
# so a second either side counts as recovered.
DEFAULT_RALLY_TOLERANCE_S = 1.0


def empty_truth(clip: str, fps: float) -> dict[str, Any]:
    """A blank truth file for a clip nobody has labelled yet."""
    return {
        "_format": TRUTH_FORMAT,
        "_instructions": (
            "Watch annotated.mp4 and record what happened in the video, not what the pipeline "
            "said. contacts: one entry per real racket contact, with hitter 'near' or 'far'. "
            "rallies: one entry per rally actually played. Delete any contact you cannot see "
            "happen. Set labelled=true when finished. Nothing here should describe a detector."
        ),
        "clip": clip,
        "fps": fps,
        "frame_reference": "outputs/<clip>/work.mp4; metadata.json holds the offset to source",
        "labelled": False,
        "contacts": [],
        "_contacts_help": (
            "[{\"frame\": 1234, \"hitter\": \"near\"|\"far\"|null, \"notes\": \"\"}] -- every "
            "racket contact in the clip, whether or not anything detected it."
        ),
        "rallies": [],
        "_rallies_help": (
            "[{\"rally\": 1, \"start_frame\": 83, \"end_frame\": 236, \"notes\": \"\"}] -- "
            "inclusive spans where a rally was in progress."
        ),
        "tracking_failures": [],
        "_tracking_failures_help": (
            "Optional. [{\"start_frame\": 1918, \"end_frame\": 3291, \"kind\": \"...\", "
            "\"notes\": \"...\"}] -- stretches where the vision stack visibly followed the "
            "wrong thing. Not scored; kept so a known failure is not rediscovered twice."
        ),
    }


def migrate_legacy_ground_truth(legacy: dict[str, Any], *, clip: str) -> dict[str, Any]:
    """Convert a candidate-verdict ground_truth.json into truth-first form.

    Rows judged real become contacts, and so do hand-added missed hits. Rows judged false are
    dropped: they describe one configuration's output rather than the video, and are re-derived
    for any configuration by the matcher. Rally spans carry over unchanged -- they were already
    a property of the video.
    """
    if not legacy.get("labelled"):
        raise ValueError(
            "refusing to migrate: the source file is not labelled, so its rows are drafts "
            "rather than judgements"
        )
    fps = float(legacy.get("fps") or 0.0)
    out = empty_truth(clip, fps)

    contacts: list[dict[str, Any]] = []
    discarded = 0
    for row in legacy.get("hits", []):
        if row.get("is_real_hit"):
            contacts.append({
                "frame": int(row["frame"]),
                "hitter": row.get("true_hitter"),
                "notes": row.get("notes") or "",
            })
        else:
            discarded += 1
    for row in legacy.get("missed_hits", []):
        contacts.append({
            "frame": int(row["frame"]),
            "hitter": row.get("true_hitter"),
            "notes": (row.get("notes") or "").strip(),
        })
    contacts.sort(key=lambda c: c["frame"])

    rallies = []
    for index, entry in enumerate(legacy.get("active_play_ranges") or [], start=1):
        start, end = (entry["start"], entry["end"]) if isinstance(entry, dict) else entry
        rallies.append({"rally": index, "start_frame": int(start), "end_frame": int(end),
                        "notes": ""})

    out["contacts"] = contacts
    out["rallies"] = rallies
    out["labelled"] = True
    out["migrated_from"] = {
        "format": "candidate verdicts (ground_truth.json)",
        "contacts_kept": len(contacts),
        "discarded_candidate_verdicts": discarded,
        "note": (
            "Discarded rows recorded that a particular candidate was not a hit, which is true "
            "of a configuration rather than of the video. The matcher re-derives false "
            "positives for whatever configuration is scored."
        ),
    }
    return out


def match_contacts(
    predicted_frames: Iterable[int],
    contacts: list[dict[str, Any]],
    *,
    fps: float,
    tolerance_seconds: float = DEFAULT_MATCH_TOLERANCE_S,
) -> dict[str, Any]:
    """Pair predicted hit frames with labelled contacts, closest pair first.

    Closest-first rather than in time order, so a burst that survives consolidation cannot score
    twice against one contact: the nearer prediction takes it and the other becomes a false
    positive, which is what it is. Each contact is claimed at most once.
    """
    tolerance = tolerance_seconds * fps
    predicted = sorted(int(f) for f in predicted_frames)
    truth_frames = [int(c["frame"]) for c in contacts]

    pairs = sorted(
        (
            (abs(p - t), p, t)
            for p in predicted
            for t in truth_frames
            if abs(p - t) <= tolerance
        ),
        key=lambda item: (item[0], item[1], item[2]),
    )
    used_predictions: set[int] = set()
    used_truth: set[int] = set()
    matched: list[tuple[int, int]] = []
    for _, prediction, contact in pairs:
        if prediction in used_predictions or contact in used_truth:
            continue
        used_predictions.add(prediction)
        used_truth.add(contact)
        matched.append((prediction, contact))

    matched.sort()
    return {
        "matched": matched,
        "false_positives": [p for p in predicted if p not in used_predictions],
        "missed": [t for t in truth_frames if t not in used_truth],
        "tolerance_frames": round(tolerance, 1),
        "tolerance_seconds": tolerance_seconds,
    }


def score_hit_detection(
    predicted_frames: Iterable[int],
    contacts: list[dict[str, Any]],
    *,
    fps: float,
    tolerance_seconds: float = DEFAULT_MATCH_TOLERANCE_S,
) -> dict[str, Any]:
    """Precision, recall and F1 of hit detection against the labelled contacts."""
    match = match_contacts(predicted_frames, contacts, fps=fps,
                           tolerance_seconds=tolerance_seconds)
    tp, fp, fn = len(match["matched"]), len(match["false_positives"]), len(match["missed"])
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "predicted": tp + fp,
        "labelled_contacts": len(contacts),
        "match": match,
    }


def score_hitter_assignment(
    predicted_hitters: dict[int, str | None],
    contacts: list[dict[str, Any]],
    *,
    fps: float,
    tolerance_seconds: float = DEFAULT_MATCH_TOLERANCE_S,
) -> dict[str, Any]:
    """Hitter accuracy over matched pairs only.

    A false positive has no true hitter to be judged against and a missed contact was never
    assigned one, so neither belongs in the denominator. A prediction that declined to assign a
    hitter is counted separately: refusing to guess is not the same kind of event as guessing
    wrong, and the pipeline reports it deliberately.
    """
    by_frame = {int(c["frame"]): c.get("hitter") for c in contacts}
    match = match_contacts(predicted_hitters.keys(), contacts, fps=fps,
                           tolerance_seconds=tolerance_seconds)
    correct = incorrect = unassigned = scored = 0
    for prediction, contact in match["matched"]:
        predicted = predicted_hitters.get(prediction)
        actual = by_frame.get(contact)
        if predicted is None:
            # Counted, but kept out of the denominator: the pipeline reports "I could not tell"
            # deliberately, and folding that into an error rate hides the difference between a
            # wrong answer and a refused one. Both are visible in the output.
            unassigned += 1
            continue
        if actual not in ("near", "far"):
            continue
        scored += 1
        if predicted == actual:
            correct += 1
        else:
            incorrect += 1
    return {
        "matched_hits": len(match["matched"]),
        "scored": scored,
        "correct": correct,
        "incorrect": incorrect,
        "unassigned": unassigned,
        "accuracy": round(correct / scored, 4) if scored else None,
    }


def score_rallies(
    predicted: list[dict[str, Any]],
    labelled: list[dict[str, Any]],
    *,
    fps: float,
    tolerance_seconds: float = DEFAULT_RALLY_TOLERANCE_S,
) -> dict[str, Any]:
    """Score rally segmentation, reporting signed boundary errors.

    Signed rather than absolute because the failure on the development clip is one-directional:
    ends run long and never short, by a median of several seconds. A median of absolute values
    would report the same number for a segmentation that scattered either way, and the two need
    different fixes. Both are reported.

    A labelled rally counts as recovered when exactly one predicted rally overlaps it and both
    edges land within the tolerance. Splits, misses and inventions are counted separately
    because a single accuracy figure cannot express them: an invented rally has no labelled
    rally to be wrong about, and a split collects two errors for one mistake.
    """
    real = sorted(
        ((int(r["start_frame"]), int(r["end_frame"])) for r in labelled), key=lambda r: r[0]
    )
    tolerance = tolerance_seconds * fps
    detail: list[dict[str, Any]] = []
    claimed: set[int] = set()
    recovered = split = missed = 0
    start_errors: list[float] = []
    end_errors: list[float] = []

    for index, (start, end) in enumerate(real, start=1):
        overlapping = [
            p for p in predicted
            if not (int(p["end_frame"]) < start or int(p["start_frame"]) > end)
        ]
        for p in overlapping:
            claimed.add(id(p))
        row: dict[str, Any] = {
            "rally": index,
            "true_start_frame": start,
            "true_end_frame": end,
            "predicted_count": len(overlapping),
        }
        if not overlapping:
            missed += 1
            row["verdict"] = "missed"
        elif len(overlapping) > 1:
            split += 1
            row["verdict"] = "split across several predicted rallies"
        else:
            only = overlapping[0]
            start_error = (int(only["start_frame"]) - start) / fps
            end_error = (int(only["end_frame"]) - end) / fps
            start_errors.append(start_error)
            end_errors.append(end_error)
            row["start_error_s"] = round(start_error, 3)
            row["end_error_s"] = round(end_error, 3)
            if abs(start_error) * fps <= tolerance and abs(end_error) * fps <= tolerance:
                recovered += 1
                row["verdict"] = "recovered"
            else:
                row["verdict"] = "edges outside tolerance"
        detail.append(row)

    spurious = [p for p in predicted if id(p) not in claimed]
    median = lambda values: round(statistics.median(values), 3) if values else None
    return {
        "labelled_rallies": len(real),
        "predicted_rallies": len(predicted),
        "recovered": recovered,
        "accuracy": (recovered / len(real)) if real else None,
        "split": split,
        "missed": missed,
        "spurious": len(spurious),
        "median_start_error_s": median(start_errors),
        "median_end_error_s": median(end_errors),
        "median_abs_start_error_s": median([abs(v) for v in start_errors]),
        "median_abs_end_error_s": median([abs(v) for v in end_errors]),
        "tolerance_seconds": tolerance_seconds,
        "detail": detail,
    }


def score_player_tracking(
    player_rows: list[dict[str, Any]],
    labelled_rallies: list[dict[str, Any]],
    *,
    fps: float,
) -> dict[str, Any]:
    """Two-player coverage and identity stability over the labelled rally spans.

    Measured over rallies only: coverage between points is not what the acceptance criterion is
    about, and including it rewards or punishes the tracker for frames nobody cares about.

    Near and far are reported separately because they fail for different reasons -- the far
    player is half the size in frame and far more often occluded -- and a combined figure hides
    which one needs work.

    Identity breaks and swaps are reported because coverage cannot see them. A slot that quietly
    changes track id mid-rally has lost the player and re-acquired them while looking completely
    healthy, and on the development clip a slot held an official for 23 seconds at 100%
    coverage. A swap is at least detectable; following the wrong person is not, which is what
    the tracking_failures section of the truth file is for.
    """
    active: set[int] = set()
    for rally in labelled_rallies:
        active.update(range(int(rally["start_frame"]), int(rally["end_frame"]) + 1))
    rows = [r for r in player_rows if int(r["frame"]) in active]
    rows.sort(key=lambda r: int(r["frame"]))

    if not rows:
        return {
            "active_play_frames": 0,
            "both_players_fraction": None,
            "near_fraction": None,
            "far_fraction": None,
            "near_identity_breaks": 0,
            "far_identity_breaks": 0,
            "identity_swaps": 0,
            "note": "no labelled rally spans, so there is no active play to measure over",
        }

    near = sum(1 for r in rows if r.get("near"))
    far = sum(1 for r in rows if r.get("far"))
    both = sum(1 for r in rows if r.get("near") and r.get("far"))
    neither = sum(1 for r in rows if not r.get("near") and not r.get("far"))

    breaks = {"near": 0, "far": 0}
    last: dict[str, int | None] = {"near": None, "far": None}
    swaps = 0
    for row in rows:
        ids: dict[str, int | None] = {}
        for slot in ("near", "far"):
            player = row.get(slot)
            ids[slot] = player.get("track_id") if player else None
        # Compare both slots against the same snapshot: updating one before reading the other
        # makes an exchange invisible from the second slot's side.
        previous = dict(last)
        swapped = (
            ids["near"] is not None and ids["far"] is not None
            and ids["near"] == previous["far"] and ids["far"] == previous["near"]
            and previous["near"] != previous["far"]
        )
        if swapped:
            swaps += 1
        for slot in ("near", "far"):
            current = ids[slot]
            if current is None:
                continue
            if previous[slot] is not None and current != previous[slot]:
                breaks[slot] += 1
            last[slot] = current

    total = len(rows)
    return {
        "active_play_frames": total,
        "both_players_frames": both,
        "both_players_fraction": round(both / total, 4),
        "near_fraction": round(near / total, 4),
        "far_fraction": round(far / total, 4),
        "no_player_frames": neither,
        "near_identity_breaks": breaks["near"],
        "far_identity_breaks": breaks["far"],
        "identity_swaps": swaps,
        "note": (
            "Coverage says a slot was filled, not that it held the right person. A slot "
            "following a bystander reads as 100%; record those in tracking_failures."
        ),
    }


def evaluate_against_truth(
    analysis: dict[str, Any],
    truth: dict[str, Any],
    player_rows: list[dict[str, Any]],
    *,
    match_tolerance_seconds: float = DEFAULT_MATCH_TOLERANCE_S,
    rally_tolerance_seconds: float = DEFAULT_RALLY_TOLERANCE_S,
) -> dict[str, Any]:
    """The whole baseline report for one configuration against one labelling.

    Assembled in one place so a baseline is a single artefact that can be diffed against a later
    run, rather than four numbers gathered by hand and quoted from memory.
    """
    if not truth.get("labelled"):
        raise ValueError(
            "truth file is not labelled; refusing to report accuracy from an unfinished file"
        )
    fps = float(truth.get("fps") or analysis.get("config", {}).get("fps") or 0.0)
    if fps <= 0:
        raise ValueError("cannot score without a frame rate")

    hits = [e for e in analysis.get("events", []) if e.get("event_type") == "hit"]
    contacts = truth.get("contacts") or []
    rallies = truth.get("rallies") or []

    detection = score_hit_detection(
        (h["frame"] for h in hits), contacts, fps=fps,
        tolerance_seconds=match_tolerance_seconds)
    hitter = score_hitter_assignment(
        {h["frame"]: h.get("hitter") for h in hits}, contacts, fps=fps,
        tolerance_seconds=match_tolerance_seconds)
    segmentation = score_rallies(
        analysis.get("rallies", []), rallies, fps=fps,
        tolerance_seconds=rally_tolerance_seconds)
    tracking = score_player_tracking(player_rows, rallies, fps=fps)

    return {
        "_format": TRUTH_FORMAT,
        "clip": truth.get("clip"),
        "fps": fps,
        "labelled_contacts": len(contacts),
        "labelled_rallies": len(rallies),
        "hit_detection": {k: v for k, v in detection.items() if k != "match"},
        "hitter_assignment": hitter,
        "rally_segmentation": segmentation,
        "player_tracking": tracking,
        "tracking_failures_recorded": len(truth.get("tracking_failures") or []),
        "caveat": (
            "Scored against hand labels for this clip only. A configuration that has been "
            "tuned on this clip is not validated by it."
        ),
    }
