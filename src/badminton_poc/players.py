from __future__ import annotations

import json
from pathlib import Path

from ultralytics import YOLO


def _parse_roi(text: str | None) -> tuple[float, float, float, float]:
    if not text:
        return (0.05, 0.02, 0.95, 0.98)
    vals = tuple(float(v.strip()) for v in text.split(","))
    if len(vals) != 4:
        raise ValueError("ROI must be x1,y1,x2,y2 in normalized 0-1 coordinates")
    x1, y1, x2, y2 = vals
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError("ROI values must satisfy 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1")
    return vals  # type: ignore[return-value]


def extract_players(
    video: str,
    output_jsonl: str,
    model_name: str = "yolo26n-pose.pt",
    roi: str | None = None,
    conf: float = 0.25,
) -> None:
    """Extract near/far player boxes and pose keypoints for a fixed-camera singles video.

    This intentionally uses a simple court ROI + box-size heuristic. The POC is testing whether
    generic pose tracking is good enough before we invest in a badminton-specific person model.
    """
    x1n, y1n, x2n, y2n = _parse_roi(roi)
    model = YOLO(model_name)
    Path(output_jsonl).parent.mkdir(parents=True, exist_ok=True)

    with Path(output_jsonl).open("w", encoding="utf-8") as out:
        results = model.track(
            source=video,
            stream=True,
            persist=True,
            classes=[0],
            conf=conf,
            verbose=False,
        )
        for frame_idx, result in enumerate(results):
            h, w = result.orig_shape
            candidates = []
            boxes = result.boxes
            keypoints = result.keypoints
            if boxes is not None:
                for i in range(len(boxes)):
                    xyxy = boxes.xyxy[i].detach().cpu().tolist()
                    bx1, by1, bx2, by2 = [float(v) for v in xyxy]
                    cx = (bx1 + bx2) / 2.0
                    cy = (by1 + by2) / 2.0
                    if not (x1n * w <= cx <= x2n * w and y1n * h <= cy <= y2n * h):
                        continue
                    area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
                    score = float(boxes.conf[i].detach().cpu()) if boxes.conf is not None else 0.0
                    track_id = None
                    if boxes.id is not None:
                        track_id = int(boxes.id[i].detach().cpu())
                    kpts = []
                    if keypoints is not None and i < len(keypoints.data):
                        kpts = keypoints.data[i].detach().cpu().tolist()
                    candidates.append(
                        {
                            "bbox": [bx1, by1, bx2, by2],
                            "area": area,
                            "bottom_y": by2,
                            "conf": score,
                            "track_id": track_id,
                            "keypoints": kpts,
                        }
                    )

            # Prefer the two largest people inside the court ROI, then assign side by vertical position.
            selected = sorted(candidates, key=lambda c: (c["area"], c["conf"]), reverse=True)[:2]
            selected = sorted(selected, key=lambda c: c["bottom_y"])
            far = selected[0] if len(selected) >= 1 else None
            near = selected[-1] if len(selected) >= 2 else None

            payload = {
                "frame": frame_idx,
                "width": w,
                "height": h,
                "far": far,
                "near": near,
                "candidate_count": len(candidates),
            }
            out.write(json.dumps(payload) + "\n")
