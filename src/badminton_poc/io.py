from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def load_ball_csv(path: str | Path) -> dict[int, dict[str, float | int]]:
    path = Path(path)
    rows: dict[int, dict[str, float | int]] = {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"No CSV header found in {path}")
        lower = {name.lower(): name for name in reader.fieldnames}
        frame_key = lower.get("frame")
        vis_key = lower.get("visibility")
        x_key = lower.get("x")
        y_key = lower.get("y")
        if not all([frame_key, x_key, y_key]):
            raise ValueError(
                "Expected TrackNet-style CSV columns Frame, X, Y and optionally Visibility. "
                f"Found: {reader.fieldnames}"
            )
        for row in reader:
            frame = int(float(row[frame_key]))
            x = float(row[x_key])
            y = float(row[y_key])
            visibility = int(float(row[vis_key])) if vis_key and row.get(vis_key, "") != "" else int(x != 0 or y != 0)
            rows[frame] = {"frame": frame, "x": x, "y": y, "visibility": visibility}
    return rows


def load_events_json(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for key in ("events", "predictions", "results"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError("Events JSON must be a list or an object containing an events list")

    out: list[dict[str, Any]] = []
    for raw in data:
        if not isinstance(raw, dict):
            continue
        frame = raw.get("frame", raw.get("Frame"))
        event_type = raw.get("event_type", raw.get("type", raw.get("event")))
        if frame is None or event_type is None:
            continue
        item = dict(raw)
        item["frame"] = int(frame)
        item["event_type"] = str(event_type).lower()
        out.append(item)
    return sorted(out, key=lambda e: e["frame"])


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
