"""Merge player tracks with shuttle trajectory and events into analysis.json."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

from .assoc import assign_hit, enforce_alternation, racket_reach
from .config import AnalysisConfig, CourtROI, FlightConfig
from .parsing import BallTrack, load_ball_csv, load_events_json, load_jsonl
from .players import swing_features
from .flight import find_fixed_points, segment_flight, track_quality
from .rallies import (
    build_timeline,
    rallies_from_flight,
    consolidate_events,
    detect_play_spans,
    drop_weak_swings,
    trim_trailing_weak_hits,
    drop_low_confidence,
    play_evidence,
    reject_hits_outside_play,
    segment_rallies,
)


def analyze(
    players_jsonl: str | Path,
    ball_csv: str | Path,
    events_json: str | Path,
    output_json: str | Path,
    *,
    fps: float,
    court: CourtROI,
    cfg: AnalysisConfig,
    flight_cfg: FlightConfig | None = None,
    width: int = 1920,
    height: int,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if fps <= 0:
        raise ValueError("fps must be positive")
    if cfg.event_source == "flight" and flight_cfg is None:
        flight_cfg = FlightConfig()

    player_rows = load_jsonl(players_jsonl)
    players_by_frame = {int(row["frame"]): row for row in player_rows}
    ball_raw = load_ball_csv(ball_csv)
    ball, offcourt_frames = mask_offcourt_shuttle(ball_raw, court, width, height)
    upstream_events = load_events_json(events_json)
    upstream_events, offcourt_events = drop_offcourt_events(upstream_events, court, width, height)
    raw_counts: dict[str, int] = {}
    for event in upstream_events:
        raw_counts[event.event_type] = raw_counts.get(event.event_type, 0) + 1

    # Computed whichever detector is in use. It costs one pass over the shuttle track, it is
    # what tells the annotated video that a motionless mid-court shuttle is a tracker parked on
    # a light rather than a shuttle, and it is worth knowing on a clip whatever produced its
    # events.
    _flight_cfg = flight_cfg or FlightConfig()
    fixed_points = find_fixed_points(
        ball, fps, _flight_cfg,
        width / _flight_cfg.reference_width if _flight_cfg.reference_width > 0 else 1.0,
    )

    flight_model = None
    if cfg.event_source == "flight":
        # The whole upstream filtering chain below exists to thin 777 candidates into something
        # usable. A segmentation does not produce them, so none of it runs; upstream's file is
        # still parsed above and its counts still reported, as a comparison rather than an input.
        def racket_reach_at(frame: int) -> float | None:
            """How close the shuttle came to a racket on this frame, in body heights.

            Supplied to the segmentation so it can tell a contact that barely deflected the
            shuttle from a tracker dropout mid-flight, which look identical by turn angle.
            """
            sample = ball.nearest_visible(frame, radius=cfg.hit_search_radius_frames)
            if sample is None:
                return None
            return racket_reach(players_by_frame.get(frame), sample.position())

        flight_model = segment_flight(ball, fps, flight_cfg, width=width,
                                      racket_reach_at=racket_reach_at)
        raw_events = list(flight_model.events)
    elif cfg.event_source == "tracknet":
        raw_events = upstream_events
    else:
        raise ValueError(
            f"unknown event_source {cfg.event_source!r}; expected 'tracknet' or 'flight'"
        )

    play_spans = detect_play_spans(
        ball, fps,
        min_speed_px=cfg.play_gate_min_speed_px,
        smooth_seconds=cfg.play_gate_smooth_seconds,
        bridge_seconds=cfg.play_gate_bridge_seconds,
        min_span_seconds=cfg.play_gate_min_span_seconds,
    ) if cfg.play_gate_min_speed_px > 0 and flight_model is None else []
    if flight_model is None:
        raw_events, off_play = reject_hits_outside_play(raw_events, play_spans, fps=fps)
        raw_events, low_confidence = drop_low_confidence(
            raw_events, cfg.min_hit_confidence, "hit"
        )
        events, merge_audit = consolidate_events(
            raw_events, fps, cfg.min_hit_interval_seconds, "hit"
        )
        events, landing_audit = consolidate_events(
            events, fps, cfg.min_landing_interval_seconds, "landing"
        )
        merge_audit = merge_audit + landing_audit
    else:
        events, off_play, low_confidence, merge_audit = raw_events, [], [], []

    hits = [e for e in events if e.event_type == "hit"]
    swing_by_frame = {}
    for event in hits:
        swing_by_frame[event.frame] = {
            slot: swing_features(players_by_frame, event.frame, slot, fps=fps)
            for slot in ("near", "far")
        }
    raw_assignments = [
        assign_hit(event, ball, players_by_frame, court, cfg, width=width, height=height)
        for event in hits
    ]
    max_gap = max(1, int(round(cfg.rally_gap_seconds * fps)))
    assignments = enforce_alternation(raw_assignments, max_gap)
    by_frame = {a.frame: a for a in assignments}

    # Gated after assignment, because which player's arm to look at is only known once the hit
    # has been attributed to one.
    events, weak_swings = drop_weak_swings(
        events, swing_by_frame, {a.frame: a.hitter for a in assignments},
        cfg.min_swing_speed, fps=fps)
    if weak_swings:
        gated = {d["frame"] for d in weak_swings}
        by_frame = {f: a for f, a in by_frame.items() if f not in gated}

    timeline = build_timeline(events, by_frame, ball, fps, cfg.shuttle_absent_seconds,
                              swing_by_frame=swing_by_frame)
    unattributed: list[dict[str, Any]] = []
    orphaned: list[dict[str, Any]] = []
    if flight_model is not None:
        rallies, unattributed = rallies_from_flight(
            flight_model, timeline, fps,
            require_hitter=cfg.require_hitter,
            require_both_players=cfg.require_both_players,
            min_hits=cfg.min_hits_per_rally,
        )
        timeline, orphaned = prune_to_rallies(timeline, rallies)
    else:
        rallies = segment_rallies(
            timeline, fps, gap_seconds=cfg.rally_gap_seconds, min_hits=cfg.min_hits_per_rally,
            end_at_last_hit=cfg.rally_end_at_last_hit,
        )

    trailing_trimmed: list[dict[str, Any]] = []
    if cfg.trailing_swing_trim > 0:
        trimmed = []
        for rally in rallies:
            out, dropped = trim_trailing_weak_hits(rally, cfg.trailing_swing_trim, fps=fps)
            trimmed.append(out)
            trailing_trimmed.extend(dropped)
        rallies = trimmed

    # Say what each rally actually looks like, next to the rally itself. A rally invented out of
    # an interval reads exactly like a real one in this file otherwise: clip 3 reported one at
    # frames 2017-2122 built from two people hand-throwing shuttles. Nothing here drops it --
    # every signal measured for that job overlaps on the one clip with hand labels -- so the
    # measurements go beside it and the reader decides.
    for rally in rallies:
        rally["play_evidence"] = play_evidence(
            ball, player_rows, rally["start_frame"], rally["end_frame"],
            fps=fps, width=width, fixed_points=fixed_points,
        )

    # Everything above is a reading of the shuttle track, so say plainly how much of that track
    # was physically possible. Measured over rally frames, where it matters, and scaled by the
    # near player's size so the verdict transfers between clips and frame rates.
    rally_frames = {f for r in rallies for f in range(r["start_frame"], r["end_frame"] + 1)}
    near_heights = sorted(
        row["near"]["bbox"][3] - row["near"]["bbox"][1]
        for row in players_by_frame.values() if row.get("near") and row["near"].get("bbox")
    )
    shuttle_quality = track_quality(
        ball, fps,
        body_height_px=near_heights[len(near_heights) // 2] if near_heights else None,
        frames=rally_frames or None,
    )

    summary = _summarise(player_rows, ball, timeline, rallies, assignments, fps)
    summary["two_player_coverage"]["during_detected_rallies"] = _rally_coverage(
        player_rows, rallies
    )
    summary["tracked_player_plausibility"] = tracked_player_plausibility(
        player_rows, rallies, fps
    )
    summary["hits"]["raw_detector_candidates"] = raw_counts.get("hit", 0)
    summary["raw_detector_candidates"] = raw_counts
    summary["merged_bursts"] = len(merge_audit)
    summary["shuttle_coverage"]["offcourt_frames"] = offcourt_frames
    summary["shuttle_coverage"]["offcourt_fraction"] = _frac(offcourt_frames, len(ball_raw))
    summary["shuttle_coverage"]["offcourt_events_dropped"] = len(offcourt_events)
    summary["shuttle_coverage"]["offcourt_note"] = (
        "TrackNetV3 emits one shuttle per frame; with more than one court in view it locks onto "
        "a neighbouring court's shuttle. These samples fell outside the configured shuttle "
        "airspace polygon and were excluded from association. They remain in tracknet/ unchanged."
    )
    output = {
        "metadata": metadata or {},
        "config": {"analysis": vars(cfg), "court": court.to_dict(), "fps": fps,
                   "flight": vars(_flight_cfg) if cfg.event_source == "flight" else None},
        "summary": summary,
        "rallies": rallies,
        "events": [e.to_dict(fps) for e in timeline],
        "hit_consolidation": merge_audit,
        "low_confidence_hits_dropped": low_confidence,
        "play_spans": [
            {"start_frame": s, "end_frame": e, "start_s": round(s / fps, 3),
             "end_s": round(e / fps, 3), "duration_s": round((e - s + 1) / fps, 3)}
            for s, e in play_spans
        ],
        "hits_outside_play_dropped": off_play,
        "unattributed_hits_dropped": unattributed,
        "events_orphaned_by_rally_gates": orphaned,
        "weak_swing_hits_dropped": weak_swings,
        "trailing_hits_trimmed": trailing_trimmed,
        "offcourt_events_dropped": offcourt_events,
        "shuttle_track_quality": shuttle_quality,
        "shuttle_fixed_points": fixed_points,
        "flight": None if flight_model is None else flight_model.to_dict(fps),
        "upstream_events_not_used": (
            None if flight_model is None else {
                "counts": raw_counts,
                "note": (
                    "Parsed from upstream's events file and reported for comparison. Under "
                    "event_source 'flight' nothing here reached the timeline."
                ),
            }
        ),
    }
    out = Path(output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def prune_to_rallies(
    timeline: list[Any], rallies: list[dict[str, Any]]
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Drop hits and landings belonging to a chain the rally gates threw out.

    The events are built before the gates run, so without this a chain rejected as "not an
    exchange" still leaves its landing behind, and the annotated video flashes LANDING in the
    middle of dead time for a rally the pipeline decided was not one. Events that belong to no
    rally by design, such as shuttle absence, are left alone.
    """
    keep_hits = {int(h["frame"]) for rally in rallies for h in rally.get("hits", [])}
    spans = [(int(r["start_frame"]), int(r["end_frame"])) for r in rallies]
    kept, dropped = [], []
    for event in timeline:
        if event.event_type == "hit":
            survives = event.frame in keep_hits
        elif event.event_type == "landing":
            survives = any(start <= event.frame <= end for start, end in spans)
        else:
            survives = True
        if survives:
            kept.append(event)
        else:
            dropped.append({
                "frame": event.frame,
                "event_type": event.event_type,
                "reason": "the chain it belonged to was not kept as a rally",
            })
    return kept, dropped


def _rally_coverage(player_rows, rallies) -> dict[str, Any]:
    """Two-player coverage restricted to the frames we believe are rallies.

    This is a diagnostic, not the acceptance-criteria number: the spans come from the pipeline's
    own rally detection, so a rally the pipeline missed entirely cannot lower it. The real
    figure needs the hand-labelled active_play_ranges in ground_truth.json.
    """
    spans = [(r["start_frame"], r["end_frame"]) for r in rallies]
    if not spans:
        return {"both_players_fraction": None, "note": "no rallies detected"}
    frames = {f for start, end in spans for f in range(start, end + 1)}
    rows = [r for r in player_rows if int(r["frame"]) in frames]
    both = sum(1 for r in rows if r.get("near") and r.get("far"))
    return {
        "frames": len(rows),
        "both_players_frames": both,
        "both_players_fraction": _frac(both, len(rows)),
        "near_detected_fraction": _frac(sum(1 for r in rows if r.get("near")), len(rows)),
        "far_detected_fraction": _frac(sum(1 for r in rows if r.get("far")), len(rows)),
        "note": (
            "Diagnostic only: rally spans are pipeline output, not ground truth. The "
            "acceptance-criteria figure comes from hand-labelled active_play_ranges."
        ),
    }


def tracked_player_plausibility(player_rows, rallies, fps: float,
                                *, spans: str = "detected") -> dict[str, Any]:
    """Ask whether the thing in each slot behaves like a player, not just how often it was there.

    Coverage rises whether the pipeline is holding the right person or the wrong one, and twice
    now it has been the wrong one. With a permissive court polygon, yt-ccaa reported 94% coverage
    while its far slot sat on the umpire, whose foot point never left (0.906, 0.508); relaxing the
    detection threshold on the same clip later reported 90% during rallies while the slot held
    officials standing behind the back boards. Both read as a pass. Neither was one.

    Three things separate a player from a bystander, measured over rally frames only:

      confidence   how sure the detector was. Across the eight genuine tracks measured in this
                   project -- near and far on four clips -- the median runs 0.636 to 0.924; the
                   officials sat at 0.237.
      movement     median frame-to-frame travel, in the player's own body heights per second so
                   that frame rate, resolution and how far away they are all divide out. Genuine
                   tracks run 0.48 to 0.90; the officials 0.29, the umpire nothing.
      sway         how far the slot ranged sideways over the clip, in body heights. Genuine
                   tracks run 1.49 to 4.88; a person in a chair does not move.

    The thresholds sit below the genuine range rather than between the two, because a real player
    having a quiet rally should not be flagged. This does not decide anything; it puts the numbers
    next to the coverage figure so that a coverage figure earned by tracking furniture says so.

    ``spans`` names where the rally boundaries came from, because the two callers differ: analyze
    has only its own detected rallies, while the baseline has hand-labelled ones. It reaches the
    note so a saved file never claims to have measured over spans it did not.
    """
    bounds = [(r["start_frame"], r["end_frame"]) for r in rallies]
    if not bounds:
        return {"note": f"no {spans} rallies, so there is nothing to judge the slots over"}
    frames = {f for start, end in bounds for f in range(start, end + 1)}
    out: dict[str, Any] = {}
    warnings: list[str] = []
    for slot in ("near", "far"):
        seq = [(int(r["frame"]), r[slot]) for r in player_rows
               if int(r["frame"]) in frames and r.get(slot)]
        seq.sort()
        if len(seq) < 20:
            out[slot] = {"frames": len(seq), "note": "too few frames to judge"}
            continue
        confs = [p.get("conf", 0.0) for _, p in seq]
        heights = [p["bbox"][3] - p["bbox"][1] for _, p in seq]
        height = sorted(heights)[len(heights) // 2]
        steps = []
        for (fa, pa), (fb, pb) in zip(seq, seq[1:]):
            if fb - fa != 1:            # a gap is the slot being lost, not the player teleporting
                continue
            ax = (pa["bbox"][0] + pa["bbox"][2]) / 2
            bx = (pb["bbox"][0] + pb["bbox"][2]) / 2
            steps.append(((bx - ax) ** 2 + (pb["bbox"][3] - pa["bbox"][3]) ** 2) ** 0.5)
        xs = [(p["bbox"][0] + p["bbox"][2]) / 2 for _, p in seq]
        median_conf = sorted(confs)[len(confs) // 2]
        move = (sorted(steps)[len(steps) // 2] / height * fps) if steps and height else 0.0
        sway = (max(xs) - min(xs)) / height if height else 0.0
        out[slot] = {
            "frames": len(seq),
            "median_detection_confidence": round(median_conf, 3),
            "median_movement_body_heights_per_s": round(move, 3),
            "lateral_range_body_heights": round(sway, 3),
        }
        if median_conf < 0.45:
            warnings.append(
                f"the {slot} slot's median detection confidence is {median_conf:.2f}; every "
                f"genuine track measured in this project sits at 0.63 or above, so this slot is "
                f"probably not holding a player")
        if move < 0.25:
            warnings.append(
                f"the {slot} slot moves {move:.2f} body heights per second during rallies; a "
                f"player runs 0.48 to 0.90, so this slot is probably on something stationary")
        if sway < 0.8:
            warnings.append(
                f"the {slot} slot ranged {sway:.2f} body heights sideways across the whole clip; "
                f"a player covers 1.5 to 4.9, so this slot is probably on a fixed object")
    out["warnings"] = warnings
    out["note"] = (
        f"Measured over {spans} rally frames. Empty warnings mean both slots behave like "
        "players; they do not mean the coverage figure is high."
    )
    return out


def mask_offcourt_shuttle(
    ball: BallTrack, court: CourtROI, width: int, height: int
) -> tuple[BallTrack, int]:
    """Mark shuttle samples outside our court's airspace as not visible.

    The raw values are preserved on the sample so nothing is silently rewritten; they simply
    stop counting as evidence for hit association.
    """
    from dataclasses import replace

    masked = []
    dropped = 0
    for sample in ball:
        if sample.visible and not court.shuttle_contains_normalized(
            sample.x / width, sample.y / height
        ):
            dropped += 1
            masked.append(replace(sample, visible=False))
        else:
            masked.append(sample)
    return BallTrack(masked), dropped


def drop_offcourt_events(events, court: CourtROI, width: int, height: int):
    """Remove shuttle events whose position is over another court, recording each one."""
    kept, dropped = [], []
    for event in events:
        if (
            event.x is not None
            and event.y is not None
            and not court.shuttle_contains_normalized(event.x / width, event.y / height)
        ):
            dropped.append({
                "frame": event.frame,
                "event_type": event.event_type,
                "position": [round(event.x, 1), round(event.y, 1)],
                "position_norm": [round(event.x / width, 4), round(event.y / height, 4)],
                "reason": "outside shuttle airspace polygon (likely a neighbouring court)",
            })
            continue
        kept.append(event)
    return kept, dropped


def _summarise(player_rows, ball, timeline, rallies, assignments, fps: float) -> dict[str, Any]:
    total_frames = len(player_rows)
    both = sum(1 for r in player_rows if r.get("near") and r.get("far"))
    near_any = sum(1 for r in player_rows if r.get("near"))
    far_any = sum(1 for r in player_rows if r.get("far"))
    neither = sum(1 for r in player_rows if not r.get("near") and not r.get("far"))

    hit_events = [e for e in timeline if e.event_type == "hit"]
    near_hits = sum(1 for e in hit_events if e.hitter == "near")
    far_hits = sum(1 for e in hit_events if e.hitter == "far")
    unassigned = sum(1 for e in hit_events if e.hitter is None)
    inferred = sum(
        1 for a in assignments if a.reason == "inferred_by_alternation" and a.hitter is not None
    )

    reasons: dict[str, int] = {}
    for assignment in assignments:
        reasons[assignment.reason] = reasons.get(assignment.reason, 0) + 1

    hit_counts = [r["hit_count"] for r in rallies]
    durations = [r["duration_s"] for r in rallies]
    ball_frames = len(ball)
    visible = ball.visible_count()

    return {
        "frames_analysed": total_frames,
        "two_player_coverage": {
            "both_players_frames": both,
            "both_players_fraction": _frac(both, total_frames),
            "near_detected_fraction": _frac(near_any, total_frames),
            "far_detected_fraction": _frac(far_any, total_frames),
            "no_player_frames": neither,
            "note": (
                "Fraction is over every analysed frame, including frames between rallies and "
                "frames where a passer-by occludes the court. Compare against the active-play "
                "subset in evaluation.json for the acceptance-criteria number."
            ),
        },
        "shuttle_coverage": {
            "trajectory_frames": ball_frames,
            "visible_frames": visible,
            "visible_fraction": _frac(visible, ball_frames),
        },
        "hits": {
            "detected": len(hit_events),
            "near": near_hits,
            "far": far_hits,
            "unassigned": unassigned,
            "assigned_fraction": _frac(near_hits + far_hits, len(hit_events)),
            "inferred_by_alternation": inferred,
            "assignment_reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
        },
        "rallies": {
            "detected": len(rallies),
            "average_hits": round(mean(hit_counts), 2) if hit_counts else 0.0,
            "longest_hits": max(hit_counts) if hit_counts else 0,
            "average_duration_s": round(mean(durations), 2) if durations else 0.0,
            "longest_duration_s": max(durations) if durations else 0.0,
            "total_rally_seconds": round(sum(durations), 2),
        },
        "events_by_type": _counts(timeline),
        "caveat": (
            "All numbers here are pipeline outputs, not accuracy measurements. Precision, "
            "recall, hitter accuracy and rally-boundary accuracy require the manual labels in "
            "ground_truth.json; run 'badminton-poc evaluate' once those are filled in."
        ),
    }


def _counts(timeline) -> dict[str, int]:
    out: dict[str, int] = {}
    for event in timeline:
        out[event.event_type] = out.get(event.event_type, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _frac(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0
