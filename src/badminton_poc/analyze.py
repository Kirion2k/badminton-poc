"""Merge player tracks with shuttle trajectory and events into analysis.json."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

from .assoc import assign_hit, enforce_alternation
from .config import AnalysisConfig, CourtROI
from .parsing import BallTrack, load_ball_csv, load_events_json, load_jsonl
from .players import swing_speed
from .rallies import (
    build_timeline,
    consolidate_events,
    detect_play_spans,
    drop_low_confidence,
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
    width: int,
    height: int,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if fps <= 0:
        raise ValueError("fps must be positive")

    player_rows = load_jsonl(players_jsonl)
    players_by_frame = {int(row["frame"]): row for row in player_rows}
    ball_raw = load_ball_csv(ball_csv)
    ball, offcourt_frames = mask_offcourt_shuttle(ball_raw, court, width, height)
    raw_events = load_events_json(events_json)
    raw_events, offcourt_events = drop_offcourt_events(raw_events, court, width, height)
    raw_counts: dict[str, int] = {}
    for event in raw_events:
        raw_counts[event.event_type] = raw_counts.get(event.event_type, 0) + 1
    play_spans = detect_play_spans(
        ball, fps,
        min_speed_px=cfg.play_gate_min_speed_px,
        smooth_seconds=cfg.play_gate_smooth_seconds,
        bridge_seconds=cfg.play_gate_bridge_seconds,
        min_span_seconds=cfg.play_gate_min_span_seconds,
    ) if cfg.play_gate_min_speed_px > 0 else []
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

    hits = [e for e in events if e.event_type == "hit"]
    swing_by_frame = {}
    for event in hits:
        swing_by_frame[event.frame] = {
            slot: swing_speed(players_by_frame, event.frame, slot, fps=fps)
            for slot in ("near", "far")
        }
    raw_assignments = [
        assign_hit(event, ball, players_by_frame, court, cfg, width=width, height=height)
        for event in hits
    ]
    max_gap = max(1, int(round(cfg.rally_gap_seconds * fps)))
    assignments = enforce_alternation(raw_assignments, max_gap)
    by_frame = {a.frame: a for a in assignments}

    timeline = build_timeline(events, by_frame, ball, fps, cfg.shuttle_absent_seconds,
                              swing_by_frame=swing_by_frame)
    rallies = segment_rallies(
        timeline, fps, gap_seconds=cfg.rally_gap_seconds, min_hits=cfg.min_hits_per_rally
    )

    summary = _summarise(player_rows, ball, timeline, rallies, assignments, fps)
    summary["two_player_coverage"]["during_detected_rallies"] = _rally_coverage(
        player_rows, rallies
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
        "config": {"analysis": vars(cfg), "court": court.to_dict(), "fps": fps},
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
        "offcourt_events_dropped": offcourt_events,
    }
    out = Path(output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


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
