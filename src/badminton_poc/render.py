"""Annotated diagnostic video.

This is the main instrument for judging the pipeline by eye, so it shows the *inputs to the
decisions*, not just the conclusions: which boxes were kept and which were rejected and why,
where the shuttle is and where it has been, which anchor point drove a hit assignment, and
which rally we believe we are in.
"""
from __future__ import annotations

import json
import math
from collections import deque
from pathlib import Path
from typing import Any

import cv2

from .config import CourtROI
from .parsing import load_ball_csv, load_jsonl

NEAR_COLOR = (80, 220, 90)
FAR_COLOR = (240, 170, 60)
REJECT_COLOR = (90, 90, 200)
SHUTTLE_COLOR = (60, 240, 255)
HIT_COLOR = (80, 120, 255)
LANDING_COLOR = (100, 100, 255)
COURT_COLOR = (110, 110, 60)
OFFCOURT_SHUTTLE_COLOR = (130, 130, 130)
PARKED_SHUTTLE_COLOR = (110, 110, 110)
PARKED_TOLERANCE_PX = 14.0
PANEL_BG = (24, 24, 24)

SKELETON = (
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
)


def _fourcc(code: str) -> int:
    # OpenCV 5 moved the helper onto the class; 4.x has the module-level function.
    if hasattr(cv2.VideoWriter, "fourcc"):
        return cv2.VideoWriter.fourcc(*code)
    return cv2.VideoWriter_fourcc(*code)


def _text(frame, message: str, origin: tuple[int, int], color, scale: float = 0.6, thickness: int = 2):
    cv2.putText(frame, message, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, message, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _draw_player(frame, player: dict[str, Any] | None, label: str, color) -> None:
    if not player:
        return
    x1, y1, x2, y2 = (int(v) for v in player["bbox"])
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    track_id = player.get("track_id")
    tag = f"{label} id={track_id} {player.get('conf', 0):.2f}"
    _text(frame, tag, (x1, max(18, y1 - 8)), color, 0.55)

    foot = player.get("foot")
    if foot:
        cv2.drawMarker(frame, (int(foot[0]), int(foot[1])), color, cv2.MARKER_CROSS, 14, 2)

    keypoints = player.get("keypoints") or []
    for a, b in SKELETON:
        if a < len(keypoints) and b < len(keypoints):
            ka, kb = keypoints[a], keypoints[b]
            if ka[2] >= 0.3 and kb[2] >= 0.3:
                cv2.line(frame, (int(ka[0]), int(ka[1])), (int(kb[0]), int(kb[1])), color, 1, cv2.LINE_AA)
    for index in (9, 10):  # wrists drive hit association, so make them stand out
        if index < len(keypoints) and keypoints[index][2] >= 0.3:
            k = keypoints[index]
            cv2.circle(frame, (int(k[0]), int(k[1])), 6, (255, 255, 255), -1)
            cv2.circle(frame, (int(k[0]), int(k[1])), 6, color, 2)


def _draw_rejects(frame, rejected: list[dict[str, Any]]) -> None:
    for item in rejected:
        x1, y1, x2, y2 = (int(v) for v in item["bbox"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), REJECT_COLOR, 1)
        _text(frame, item.get("reason", "?"), (x1, max(12, y1 - 4)), REJECT_COLOR, 0.4, 1)


def _draw_court(frame, court: CourtROI) -> None:
    import numpy as np

    height, width = frame.shape[:2]
    points = np.array([[int(x * width), int(y * height)] for x, y in court.polygon], dtype=np.int32)
    cv2.polylines(frame, [points], True, COURT_COLOR, 1, cv2.LINE_AA)
    left, right = court.net_endpoints_pixels(width, height)
    cv2.line(frame, left, right, COURT_COLOR, 1, cv2.LINE_AA)


def shuttle_state(
    sample,
    court: CourtROI | None,
    fixed_points: list[dict[str, Any]],
    width: int,
    height: int,
    tolerance_px: float = PARKED_TOLERANCE_PX,
) -> str:
    """What the shuttle marker at this frame actually means.

    ``none`` no sample, ``live`` a shuttle we believe in, ``off_court`` a neighbouring court's
    shuttle, ``parked`` the tracker resting on a fixed object in the room.

    TrackNetV3 emits exactly one shuttle per frame whether or not one is in play, so between
    points it holds a single pixel on a ceiling light for seconds at a time. Drawn as a live
    shuttle that reads as a shuttle hovering mid-court, which is the most misleading thing in the
    annotated video. It is still drawn, greyed and labelled, because hiding it would hide a real
    failure mode of the tracker.
    """
    if sample is None or not sample.visible:
        return "none"
    if court is not None and not court.shuttle_contains_normalized(
        sample.x / width, sample.y / height
    ):
        # Named before "parked" because it is the more specific diagnosis.
        return "off_court"
    for point in fixed_points or []:
        if math.hypot(sample.x - float(point["x"]), sample.y - float(point["y"])) <= tolerance_px:
            return "parked"
    return "live"


def _rally_for_frame(rallies: list[dict[str, Any]], frame_index: int) -> dict[str, Any] | None:
    for rally in rallies:
        if rally["start_frame"] <= frame_index <= rally["end_frame"]:
            return rally
    return None


def render(
    video: str | Path,
    players_jsonl: str | Path,
    ball_csv: str | Path,
    analysis_json: str | Path,
    output_video: str | Path,
    *,
    court: CourtROI | None = None,
    trail_length: int = 20,
    flash_frames: int = 12,
    show_rejects: bool = True,
) -> dict[str, Any]:
    players = {int(r["frame"]): r for r in load_jsonl(players_jsonl)}
    ball = load_ball_csv(ball_csv)
    analysis = json.loads(Path(analysis_json).read_text(encoding="utf-8"))
    rallies = analysis.get("rallies", [])
    fps = float(analysis.get("config", {}).get("fps") or 30.0)

    fixed_points = list((analysis.get("shuttle_fixed_points") or []))

    events_by_frame: dict[int, list[dict[str, Any]]] = {}
    for event in analysis.get("events", []):
        events_by_frame.setdefault(int(event["frame"]), []).append(event)

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"could not open {video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output = Path(output_video)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), _fourcc("mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"could not open video writer for {output}")

    trail: deque[tuple[int, tuple[int, int]]] = deque(maxlen=trail_length)
    active_flashes: list[tuple[int, dict[str, Any]]] = []
    written = 0

    try:
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if court is not None:
                _draw_court(frame, court)

            row = players.get(frame_index)
            if row:
                if show_rejects:
                    _draw_rejects(frame, row.get("rejected", []))
                _draw_player(frame, row.get("far"), "FAR", FAR_COLOR)
                _draw_player(frame, row.get("near"), "NEAR", NEAR_COLOR)

            sample = ball.get(frame_index)
            state = shuttle_state(sample, court, fixed_points, width, height)
            if state in ("off_court", "parked"):
                # Drawn, but greyed and kept out of the trail. Both are real failure modes of a
                # tracker that must emit a shuttle every frame, and hiding either would hide
                # them; drawing them like a live shuttle is what made the video misleading.
                point = (int(sample.x), int(sample.y))
                colour = OFFCOURT_SHUTTLE_COLOR if state == "off_court" else PARKED_SHUTTLE_COLOR
                label = "off-court" if state == "off_court" else "tracker parked"
                cv2.drawMarker(frame, point, colour, cv2.MARKER_TILTED_CROSS, 12, 1)
                cv2.circle(frame, point, 6, colour, 1)
                _text(frame, label, (point[0] + 12, point[1] - 9), colour, 0.5, 1)
                if state == "parked":
                    # A parked tracker means no shuttle is in play, so there is no trajectory
                    # left to show. An off-court sample is a brief lock onto the next court and
                    # the trail either side of it is still ours, so that one is left alone.
                    trail.clear()
            elif state == "live":
                point = (int(sample.x), int(sample.y))
                trail.append((frame_index, point))
                cv2.circle(frame, point, 9, (0, 0, 0), -1)
                cv2.circle(frame, point, 7, SHUTTLE_COLOR, -1)
            history = list(trail)
            for i in range(1, len(history)):
                (prev_frame, prev_point), (this_frame, this_point) = history[i - 1], history[i]
                # Only join samples that are actually adjacent in time; drawing across a gap
                # would invent a trajectory the tracker never produced.
                if this_frame - prev_frame > 2:
                    continue
                weight = i / max(1, len(history))
                cv2.line(frame, prev_point, this_point, SHUTTLE_COLOR,
                         max(1, int(1 + 2 * weight)), cv2.LINE_AA)

            for event in events_by_frame.get(frame_index, []):
                active_flashes.append((frame_index, event))
            active_flashes = [(f, e) for f, e in active_flashes if frame_index - f < flash_frames]
            for start, event in active_flashes:
                _draw_event(frame, event, frame_index - start, flash_frames)

            _draw_panel(frame, frame_index, fps, row, sample, _rally_for_frame(rallies, frame_index),
                        events_by_frame.get(frame_index, []))
            writer.write(frame)
            written += 1
            frame_index += 1
    finally:
        capture.release()
        writer.release()

    return {"output": str(output), "frames_written": written, "fps": fps,
            "size": [width, height]}


def _draw_event(frame, event: dict[str, Any], age: int, flash_frames: int) -> None:
    position = event.get("position")
    event_type = event.get("event_type", "event")
    color = HIT_COLOR if event_type == "hit" else LANDING_COLOR
    if position:
        radius = 14 + age * 3
        cv2.circle(frame, (int(position[0]), int(position[1])), radius, color, 2, cv2.LINE_AA)
    label = event_type.upper()
    if event_type == "hit":
        label += f" -> {str(event.get('hitter') or 'UNASSIGNED').upper()}"
    if age < flash_frames // 2 and position:
        _text(frame, label, (int(position[0]) + 18, int(position[1]) - 10), color, 0.7)


def _draw_panel(frame, frame_index: int, fps: float, row, sample, rally, events) -> None:
    height, width = frame.shape[:2]
    panel_h = 96
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, panel_h), PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)

    _text(frame, f"frame {frame_index}  t={frame_index / fps:6.2f}s", (14, 28), (235, 235, 235), 0.7)

    if rally:
        rally_text = (f"RALLY {rally['rally']}  hits={rally['hit_count']} "
                      f"(near {rally['near_hits']} / far {rally['far_hits']} / "
                      f"unassigned {rally['unassigned_hits']})")
        _text(frame, rally_text, (14, 56), (120, 255, 200), 0.62)
    else:
        _text(frame, "between rallies", (14, 56), (170, 170, 170), 0.62)

    near_ok = bool(row and row.get("near"))
    far_ok = bool(row and row.get("far"))
    status = (f"NEAR {'OK ' if near_ok else 'MISS'}   FAR {'OK ' if far_ok else 'MISS'}   "
              f"shuttle {'OK ' if (sample and sample.visible) else 'MISS'}")
    color = (120, 255, 120) if (near_ok and far_ok) else (120, 200, 255)
    _text(frame, status, (14, 84), color, 0.62)

    if row:
        detail = (f"detections {row.get('detections_total', 0)}  "
                  f"on-court {row.get('candidates_on_court', 0)}")
        _text(frame, detail, (width - 330, 28), (200, 200, 200), 0.55)
    if events:
        kinds = ", ".join(sorted({e.get("event_type", "?") for e in events}))
        _text(frame, f"EVENT: {kinds}", (width - 330, 56), (120, 180, 255), 0.6)
