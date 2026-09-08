from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2

from .io import load_ball_csv, load_jsonl


def _draw_player(frame, player: dict[str, Any] | None, label: str) -> None:
    if not player:
        return
    bbox = player.get("bbox")
    if bbox:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    kpts = player.get("keypoints") or []
    for idx in (5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16):
        if idx < len(kpts) and len(kpts[idx]) >= 2:
            x, y = int(kpts[idx][0]), int(kpts[idx][1])
            conf = float(kpts[idx][2]) if len(kpts[idx]) > 2 else 1.0
            if conf >= 0.15:
                cv2.circle(frame, (x, y), 3, (255, 255, 255), -1)


def render(video: str, players_jsonl: str, ball_csv: str, analysis_json: str, output_video: str) -> None:
    players = {int(r["frame"]): r for r in load_jsonl(players_jsonl)}
    ball = load_ball_csv(ball_csv)
    analysis = json.loads(Path(analysis_json).read_text(encoding="utf-8"))
    events_by_frame = {int(e["frame"]): e for e in analysis.get("events", [])}

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    Path(output_video).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    trail: list[tuple[int, int]] = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        player_row = players.get(frame_idx)
        if player_row:
            _draw_player(frame, player_row.get("near"), "NEAR")
            _draw_player(frame, player_row.get("far"), "FAR")

        ball_row = ball.get(frame_idx)
        if ball_row and int(ball_row["visibility"]) > 0:
            pt = (int(float(ball_row["x"])), int(float(ball_row["y"])))
            trail.append(pt)
            trail = trail[-12:]
            cv2.circle(frame, pt, 7, (255, 255, 255), -1)
        for a, b in zip(trail, trail[1:]):
            cv2.line(frame, a, b, (255, 255, 255), 2)

        event = events_by_frame.get(frame_idx)
        if event:
            text = event.get("event_type", "event").upper()
            if event.get("hitter"):
                text += f" - {str(event['hitter']).upper()}"
            cv2.putText(frame, text, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
