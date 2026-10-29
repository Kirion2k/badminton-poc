"""Parsers for the artefacts produced by upstream BadmintonTrackNet and by our own stages.

The schemas here are taken from the upstream source, not from documentation:

* ``<stem>_ball.csv`` is written by ``tracknet.data.io.write_pred_csv`` with columns
  ``Frame, Visibility, X, Y`` (plus ``Inpaint_Mask`` when saving InpaintNet training data).
  Inference never writes ``Inpaint_Mask``, and InpaintNet's gap fills arrive with
  ``Visibility = 1`` like real detections; ``load_ball_csv`` recovers which is which from the
  decode grid when given the frame size.
* ``<stem>_events.json`` is written by ``tracknet.inference.events.main`` as a JSON list of
  the candidate dicts built in ``tracknet.events.candidate_generator``:
  ``frame, x, y, event_type, rule, confidence, all_rules, auxiliary_rules, features``.
  Note that ``events.py`` filters the list to ``{"landing", "hit"}`` before writing, so
  ``out_of_frame`` candidates never reach the file even though the generator produces them.

Upstream treats ``Frame`` as a real frame identifier that may start late or contain gaps,
so we key on it rather than on row order.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

KNOWN_EVENT_TYPES = frozenset({"hit", "landing", "out_of_frame"})


class ParseError(ValueError):
    """Raised when an input file does not match the expected upstream schema."""


@dataclass(frozen=True)
class BallSample:
    frame: int
    x: float
    y: float
    visible: bool
    inpainted: bool = False

    def position(self) -> tuple[float, float]:
        return (self.x, self.y)


class BallTrack:
    """Per-frame shuttle samples keyed by real frame id."""

    def __init__(self, samples: Iterable[BallSample]) -> None:
        self._by_frame: dict[int, BallSample] = {}
        for sample in samples:
            if sample.frame in self._by_frame:
                raise ParseError(f"duplicate Frame {sample.frame} in shuttle track")
            self._by_frame[sample.frame] = sample
        self._frames = sorted(self._by_frame)

    def __len__(self) -> int:
        return len(self._by_frame)

    def __iter__(self) -> Iterator[BallSample]:
        for frame in self._frames:
            yield self._by_frame[frame]

    def __contains__(self, frame: object) -> bool:
        return frame in self._by_frame

    @property
    def frames(self) -> list[int]:
        return list(self._frames)

    def get(self, frame: int) -> BallSample | None:
        return self._by_frame.get(frame)

    def visible_count(self) -> int:
        return sum(1 for s in self._by_frame.values() if s.visible)

    def nearest_visible(self, frame: int, radius: int = 3) -> BallSample | None:
        """Closest visible sample to ``frame``, searching outwards and preferring the
        earlier frame on ties so a hit is attributed to the moment before contact."""
        if radius < 0:
            raise ValueError("radius must be >= 0")
        sample = self._by_frame.get(frame)
        if sample is not None and sample.visible:
            return sample
        for offset in range(1, radius + 1):
            for candidate_frame in (frame - offset, frame + offset):
                candidate = self._by_frame.get(candidate_frame)
                if candidate is not None and candidate.visible:
                    return candidate
        return None

    def absence_spans(self, min_length: int) -> list[tuple[int, int]]:
        """Runs of >= ``min_length`` consecutive frames with no visible shuttle.

        Upstream's events CLI discards its own ``out_of_frame`` candidates, so this is how we
        recover "the shuttle stopped being tracked" as a rally-termination signal. It is our
        heuristic, not an upstream detection, and is labelled as such everywhere it surfaces.
        """
        if min_length < 1:
            raise ValueError("min_length must be >= 1")
        spans: list[tuple[int, int]] = []
        start: int | None = None
        previous: int | None = None
        for frame in self._frames:
            visible = self._by_frame[frame].visible
            if not visible:
                if start is None or (previous is not None and frame != previous + 1):
                    if start is not None and previous is not None and previous - start + 1 >= min_length:
                        spans.append((start, previous))
                    start = frame
            else:
                if start is not None and previous is not None and previous - start + 1 >= min_length:
                    spans.append((start, previous))
                start = None
            previous = frame
        if start is not None and previous is not None and previous - start + 1 >= min_length:
            spans.append((start, previous))
        return spans


TRACKNET_INPUT_SIZE = (512, 288)
"""``tracknet.data.io.WIDTH, HEIGHT``. A TrackNet detection is the centre of a bounding box on
this heatmap scaled to the frame, so it always lands on a half-cell grid; InpaintNet's output is
a free float. Checked against upstream's own ``_inpaint_mask`` (``tools/export_inpaint_mask.py``)
on clips 2, 3, 5 and 6: identical on all 30,902 visible samples."""


def _off_decode_grid(x: float, y: float, width: int, height: int) -> bool:
    for value, size, cells in ((x, width, TRACKNET_INPUT_SIZE[0]), (y, height, TRACKNET_INPUT_SIZE[1])):
        q = value / (size / (2 * cells))
        if abs(q - round(q)) > 0.01:
            return True
    return False


def load_ball_csv(path: str | Path, frame_size: tuple[int, int] | None = None) -> BallTrack:
    """``frame_size`` (width, height) lets a file with no ``Inpaint_Mask`` column mark
    InpaintNet's samples as ``inpainted``; without it they read as observed."""
    path = Path(path)
    if not path.exists():
        raise ParseError(f"shuttle CSV not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ParseError(f"no CSV header in {path}")
        columns = {name.strip().lower(): name for name in reader.fieldnames if name}
        frame_key = columns.get("frame")
        x_key = columns.get("x")
        y_key = columns.get("y")
        vis_key = columns.get("visibility")
        mask_key = columns.get("inpaint_mask")
        missing = [n for n, k in (("Frame", frame_key), ("X", x_key), ("Y", y_key)) if k is None]
        if missing:
            raise ParseError(
                f"{path} is missing required TrackNet column(s) {missing}. "
                f"Found columns: {reader.fieldnames}"
            )

        samples: list[BallSample] = []
        for line_no, row in enumerate(reader, start=2):
            try:
                frame = int(float(row[frame_key]))
                x = float(row[x_key] or 0.0)
                y = float(row[y_key] or 0.0)
            except (TypeError, ValueError) as exc:
                raise ParseError(f"{path}:{line_no}: non-numeric Frame/X/Y ({row!r})") from exc
            if vis_key is not None and str(row.get(vis_key, "")).strip() != "":
                visible = float(row[vis_key]) > 0
            else:
                # Upstream writes (0, 0) for a missed detection; treat that as invisible.
                visible = not (x == 0.0 and y == 0.0)
            inpainted = False
            if mask_key is not None and str(row.get(mask_key, "")).strip() != "":
                inpainted = float(row[mask_key]) > 0
            elif visible and frame_size is not None:
                inpainted = _off_decode_grid(x, y, *frame_size)
            samples.append(BallSample(frame=frame, x=x, y=y, visible=visible, inpainted=inpainted))
    if not samples:
        raise ParseError(f"{path} contains a header but no rows")
    return BallTrack(samples)


@dataclass(frozen=True)
class ShuttleEvent:
    frame: int
    event_type: str
    x: float | None
    y: float | None
    confidence: float | None
    rule: str | None
    all_rules: tuple[str, ...]
    source: str = "tracknet"
    raw: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "event_type": self.event_type,
            "x": self.x,
            "y": self.y,
            "detector_confidence": self.confidence,
            "rule": self.rule,
            "all_rules": list(self.all_rules),
            "source": self.source,
        }


def load_events_json(path: str | Path) -> list[ShuttleEvent]:
    path = Path(path)
    if not path.exists():
        raise ParseError(f"events JSON not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ParseError(f"{path} is not valid JSON: {exc}") from exc

    if isinstance(data, dict):
        for key in ("events", "predictions", "results"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise ParseError(
            f"{path} must contain a JSON list of events (or an object with an 'events' list), "
            f"got {type(data).__name__}"
        )

    events: list[ShuttleEvent] = []
    for index, raw in enumerate(data):
        if not isinstance(raw, dict):
            raise ParseError(f"{path}[{index}] is {type(raw).__name__}, expected an object")
        if "frame" not in raw and "Frame" not in raw:
            raise ParseError(f"{path}[{index}] has no 'frame' key: {sorted(raw)}")
        event_type = raw.get("event_type") or raw.get("type") or raw.get("event")
        if event_type is None:
            raise ParseError(f"{path}[{index}] has no 'event_type' key: {sorted(raw)}")
        try:
            frame = int(raw.get("frame", raw.get("Frame")))
        except (TypeError, ValueError) as exc:
            raise ParseError(f"{path}[{index}] has non-integer frame {raw.get('frame')!r}") from exc
        rules = raw.get("all_rules") or ([raw["rule"]] if raw.get("rule") else [])
        events.append(
            ShuttleEvent(
                frame=frame,
                event_type=str(event_type).strip().lower(),
                x=_optional_float(raw.get("x")),
                y=_optional_float(raw.get("y")),
                confidence=_optional_float(raw.get("confidence")),
                rule=str(raw["rule"]) if raw.get("rule") else None,
                all_rules=tuple(str(r) for r in rules),
                raw=raw,
            )
        )
    return sorted(events, key=lambda e: e.frame)


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise ParseError(f"JSONL file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ParseError(f"{path}:{line_no} is not valid JSON: {exc}") from exc
            if not isinstance(obj, dict):
                raise ParseError(f"{path}:{line_no} is {type(obj).__name__}, expected an object")
            rows.append(obj)
    return rows
