from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

from .io import load_ball_csv, load_events_json, load_jsonl

WRIST_INDICES = (9, 10)  # COCO left/right wrist


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _player_contact_distance(player: dict[str, Any] | None, shuttle: tuple[float, float]) -> float:
    if not player:
        return float("inf")
    bbox = player.get("bbox")
    kpts = player.get("keypoints") or []
    distances: list[float] = []
    for idx in WRIST_INDICES:
        if idx < len(kpts) and len(kpts[idx]) >= 2:
            x, y = float(kpts[idx][0]), float(kpts[idx][1])
            conf = float(kpts[idx][2]) if len(kpts[idx]) > 2 else 1.0
            if conf >= 0.15:
                distances.append(_distance((x, y), shuttle))
    if distances:
        return min(distances)
    if bbox:
        bx1, by1, bx2, by2 = map(float, bbox)
        cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
        return _distance((cx, cy), shuttle)
    return float("inf")


def _nearest_visible_ball(ball: dict[int, dict[str, float | int]], frame: int, radius: int = 3):
    for offset in [0, -1, 1, -2, 2, -3, 3][: 2 * radius + 1]:
        row = ball.get(frame + offset)
        if row and int(row["visibility"]) > 0:
            return row
    return None


def _group_rallies(events: list[dict[str, Any]], fps: float) -> list[dict[str, Any]]:
    meaningful = [e for e in events if e.get("event_type") in {"hit", "landing", "out_of_frame"}]
    if not meaningful:
        return []

    gap_frames = max(1, int(round(3.0 * fps)))
    rallies: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []

    for event in meaningful:
        if current and event["frame"] - current[-1]["frame"] > gap_frames:
            rallies.append(_finalize_rally(current, fps))
            current = []
        current.append(event)
        if event["event_type"] in {"landing", "out_of_frame"} and any(e["event_type"] == "hit" for e in current):
            rallies.append(_finalize_rally(current, fps))
            current = []

    if current and any(e["event_type"] == "hit" for e in current):
        rallies.append(_finalize_rally(current, fps))
    return rallies


def _finalize_rally(events: list[dict[str, Any]], fps: float) -> dict[str, Any]:
    hits = [e for e in events if e["event_type"] == "hit"]
    start = hits[0]["frame"] if hits else events[0]["frame"]
    end = events[-1]["frame"]
    return {
        "start_frame": start,
        "end_frame": end,
        "start_s": start / fps,
        "end_s": end / fps,
        "duration_s": max(0.0, (end - start) / fps),
        "hit_count": len(hits),
        "hits": hits,
        "end_event": events[-1]["event_type"],
    }


def analyze(
    players_jsonl: str,
    ball_csv: str,
    events_json: str,
    output_json: str,
    fps: float,
) -> dict[str, Any]:
    players = {int(r["frame"]): r for r in load_jsonl(players_jsonl)}
    ball = load_ball_csv(ball_csv)
    events = load_events_json(events_json)

    assigned_events: list[dict[str, Any]] = []
    for event in events:
        event = dict(event)
        if event["event_type"] == "hit":
            frame = int(event["frame"])
            ball_row = _nearest_visible_ball(ball, frame)
            player_row = players.get(frame)
            hitter = None
            confidence = None
            distances = None
            if ball_row and player_row:
                shuttle = (float(ball_row["x"]), float(ball_row["y"]))
                near_d = _player_contact_distance(player_row.get("near"), shuttle)
                far_d = _player_contact_distance(player_row.get("far"), shuttle)
                distances = {"near": near_d, "far": far_d}
                if math.isfinite(min(near_d, far_d)):
                    hitter = "near" if near_d <= far_d else "far"
                    other = max(near_d, far_d)
                    best = min(near_d, far_d)
                    # Not a calibrated probability, just a useful POC confidence score.
                    confidence = float(max(0.0, min(1.0, (other - best) / max(other, 1.0))))
            event["hitter"] = hitter
            event["assignment_confidence"] = confidence
            event["player_distances_px"] = distances
        assigned_events.append(event)

    rallies = _group_rallies(assigned_events, fps)

    near_hits = sum(1 for e in assigned_events if e.get("event_type") == "hit" and e.get("hitter") == "near")
    far_hits = sum(1 for e in assigned_events if e.get("event_type") == "hit" and e.get("hitter") == "far")
    unassigned_hits = sum(1 for e in assigned_events if e.get("event_type") == "hit" and not e.get("hitter"))
    all_hit_counts = [r["hit_count"] for r in rallies]
    visible_ball_frames = sum(1 for r in ball.values() if int(r["visibility"]) > 0)
    player_two_track_frames = sum(1 for r in players.values() if r.get("near") and r.get("far"))

    summary = {
        "rallies_detected": len(rallies),
        "hits_detected": near_hits + far_hits + unassigned_hits,
        "near_hits": near_hits,
        "far_hits": far_hits,
        "unassigned_hits": unassigned_hits,
        "average_hits_per_rally": mean(all_hit_counts) if all_hit_counts else 0.0,
        "longest_rally_hits": max(all_hit_counts) if all_hit_counts else 0,
        "ball_visible_frame_fraction": visible_ball_frames / max(len(ball), 1),
        "two_players_detected_frame_fraction": player_two_track_frames / max(len(players), 1),
    }

    output = {"summary": summary, "rallies": rallies, "events": assigned_events}
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(output_json).write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output
