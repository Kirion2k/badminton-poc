"""Court ROI calibration helper.

The court polygon is the single most important hand-set parameter in the pipeline: it is what
separates the two match players from spectators and from the neighbouring court. So it needs to
be *seen*, not guessed. This renders a frame with a normalized coordinate grid, and optionally
the current polygon and net line drawn on top, so a wrong ROI is obvious at a glance.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import CourtROI

GRID_COLOR = (90, 90, 90)
POLY_COLOR = (0, 235, 255)
NET_COLOR = (255, 120, 0)


def grab_frame(video: str | Path, frame_index: int):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"cannot read frame {frame_index} from {video}")
        return frame
    finally:
        cap.release()


def draw_grid(frame, step: float = 0.05) -> None:
    height, width = frame.shape[:2]
    ticks = int(round(1.0 / step))
    for i in range(ticks + 1):
        value = i * step
        x = int(value * (width - 1))
        y = int(value * (height - 1))
        major = abs((value * 10) - round(value * 10)) < 1e-6 and round(value * 10) % 2 == 0
        thickness = 2 if major else 1
        cv2.line(frame, (x, 0), (x, height), GRID_COLOR, thickness)
        cv2.line(frame, (0, y), (width, y), GRID_COLOR, thickness)
        if major:
            cv2.putText(frame, f"{value:.1f}", (x + 3, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"{value:.1f}", (4, y - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 1, cv2.LINE_AA)


def draw_court(frame, court: CourtROI) -> None:
    height, width = frame.shape[:2]
    points = np.array(
        [[int(x * width), int(y * height)] for x, y in court.polygon], dtype=np.int32
    )
    overlay = frame.copy()
    cv2.fillPoly(overlay, [points], (40, 90, 40))
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
    cv2.polylines(frame, [points], True, POLY_COLOR, 3, cv2.LINE_AA)
    for index, (px, py) in enumerate(points):
        cv2.circle(frame, (int(px), int(py)), 7, POLY_COLOR, -1)
        cv2.putText(frame, str(index), (int(px) + 10, int(py) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, POLY_COLOR, 2, cv2.LINE_AA)
    left, right = court.net_endpoints_pixels(width, height)
    cv2.line(frame, left, right, NET_COLOR, 2, cv2.LINE_AA)
    cv2.putText(frame, "net floor line (near below / far above)", (12, left[1] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, NET_COLOR, 2, cv2.LINE_AA)


def render_calibration(
    video: str | Path,
    output: str | Path,
    *,
    frame_index: int = 0,
    court: CourtROI | None = None,
    grid: bool = True,
) -> Path:
    frame = grab_frame(video, frame_index)
    if grid:
        draw_grid(frame)
    if court is not None:
        draw_court(frame, court)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), frame):
        raise RuntimeError(f"failed to write {out}")
    return out
