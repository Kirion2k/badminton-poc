"""Two-player detection, tracking and near/far identity for a fixed-camera singles clip.

The scaffold this replaces picked the two largest person boxes inside a rectangular ROI on
every frame independently. On the first real clip that fails badly: the camera sits at ground
level beside the court, so foreground spectators produce the largest boxes in the image, and a
rectangle that contains the match court also contains the neighbouring court's players.

What we do instead:

1. Reject on the *foot point* (bottom-centre of the box) against a court polygon. A spectator
   between the camera and the court has their centre over the court but their feet below it,
   and a neighbouring court is a different region of the trapezoid.
2. Track with ByteTrack via Ultralytics and prefer identity continuity over per-frame size, so
   a momentary occlusion does not swap near and far.
3. Split near/far by which side of the net line the foot point falls on, rather than by ranking
   two detections against each other.

Every rejected candidate and every scoring term is written out, because when this misses we
need to be able to see why rather than guess.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from .config import CourtROI, PlayerConfig

# COCO-17 keypoint indices used by Ultralytics pose models.
KEYPOINT_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
LEFT_WRIST, RIGHT_WRIST = 9, 10
WRIST_INDICES = (LEFT_WRIST, RIGHT_WRIST)


@dataclass
class Candidate:
    bbox: tuple[float, float, float, float]
    conf: float
    track_id: int | None
    keypoints: list[list[float]]

    @property
    def foot(self) -> tuple[float, float]:
        x1, _, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, y2)

    @property
    def height(self) -> float:
        return max(0.0, self.bbox[3] - self.bbox[1])

    @property
    def width(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0])

    def compact(self) -> dict[str, Any]:
        return {
            "bbox": [round(v, 1) for v in self.bbox],
            "conf": round(self.conf, 3),
            "track_id": self.track_id,
        }


@dataclass
class Slot:
    """Persistent near/far identity."""

    name: str
    track_id: int | None = None
    last_foot: tuple[float, float] | None = None
    frames_missing: int = 0
    seen_frames: int = 0
    id_switches: int = 0
    history: list[int] = field(default_factory=list)

    def claim(self, candidate: Candidate) -> None:
        if (
            candidate.track_id is not None
            and self.track_id is not None
            and candidate.track_id != self.track_id
        ):
            self.id_switches += 1
        if candidate.track_id is not None:
            self.track_id = candidate.track_id
            if not self.history or self.history[-1] != candidate.track_id:
                self.history.append(candidate.track_id)
        self.last_foot = candidate.foot
        self.frames_missing = 0
        self.seen_frames += 1

    def miss(self, forget_after: int) -> None:
        self.frames_missing += 1
        if self.frames_missing > forget_after:
            self.track_id = None
            self.last_foot = None


def _keypoint_list(raw: Any) -> list[list[float]]:
    points: list[list[float]] = []
    for entry in raw or []:
        values = [float(v) for v in entry]
        while len(values) < 3:
            values.append(1.0)
        points.append([round(values[0], 1), round(values[1], 1), round(values[2], 3)])
    return points


def wrist_points(keypoints: list[list[float]], min_conf: float = 0.15) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for index in WRIST_INDICES:
        if index < len(keypoints):
            x, y, conf = keypoints[index][0], keypoints[index][1], keypoints[index][2]
            if conf >= min_conf:
                out.append((float(x), float(y)))
    return out


def _score(
    candidate: Candidate,
    slot: Slot,
    diagonal: float,
) -> tuple[float, dict[str, float]]:
    """Explicit, inspectable scoring. Continuity outweighs confidence on purpose: a slightly
    worse detection of the player we were already following beats a crisper detection of
    somebody else."""
    continuity = 1.0 if (candidate.track_id is not None and candidate.track_id == slot.track_id) else 0.0
    if slot.last_foot is None:
        proximity = 0.0
    else:
        distance = math.dist(candidate.foot, slot.last_foot)
        proximity = math.exp(-distance / (0.10 * diagonal))
    terms = {
        "conf": round(candidate.conf, 4),
        "continuity": continuity,
        "proximity": round(proximity, 4),
    }
    total = 1.0 * candidate.conf + 1.5 * continuity + 0.8 * proximity
    return total, terms


def select_players(
    candidates: list[Candidate],
    near_slot: Slot,
    far_slot: Slot,
    *,
    court: CourtROI,
    width: int,
    height: int,
    diagonal: float,
    forget_after: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Pick one player per side of the net. Pure function over already-filtered candidates,
    so it can be unit-tested without running any model."""
    sides = {
        id(c): court.side_of(c.foot[0] / width, c.foot[1] / height) for c in candidates
    }
    near_pool = [c for c in candidates if sides[id(c)] == "near"]
    far_pool = [c for c in candidates if sides[id(c)] == "far"]

    chosen: dict[str, dict[str, Any] | None] = {}
    for slot, pool in (("near", near_pool), ("far", far_pool)):
        slot_state = near_slot if slot == "near" else far_slot
        if not pool:
            slot_state.miss(forget_after)
            chosen[slot] = None
            continue
        scored = [(*_score(c, slot_state, diagonal), c) for c in pool]
        scored.sort(key=lambda item: item[0], reverse=True)
        total, terms, best = scored[0]
        runner_up = round(scored[1][0], 4) if len(scored) > 1 else None
        slot_state.claim(best)
        chosen[slot] = {
            "bbox": [round(v, 1) for v in best.bbox],
            "foot": [round(v, 1) for v in best.foot],
            "conf": round(best.conf, 3),
            "track_id": best.track_id,
            "keypoints": best.keypoints,
            "selection": {
                "score": round(total, 4),
                "terms": terms,
                "runner_up_score": runner_up,
                "pool_size": len(pool),
            },
        }
    return chosen["near"], chosen["far"]


class CandidateFilter:
    """Decides which person detections can be a court player, and remembers why not.

    Beyond the geometric tests this tracks how far each ByteTrack id has ever travelled. A
    badminton player moves; a seated spectator, an umpire on a raised chair, and a kit bag do
    not. Without this test the far slot on our clip locked onto a completely static detection
    (positional standard deviation 0.0 px over 1348 frames) in preference to the real opponent,
    because the identity-continuity bonus rewards a detection that never breaks -- and nothing
    is more continuous than furniture.

    A track is only rejected once it has been seen enough times to judge, and a track that has
    ever moved is trusted permanently afterwards, so a player standing still between rallies is
    never discarded.
    """

    def __init__(self, court: CourtROI, cfg: PlayerConfig, width: int, height: int) -> None:
        self.court = court
        self.cfg = cfg
        self.width = width
        self.height = height
        self.min_travel_px = cfg.static_track_min_travel_frac * width
        self._extent: dict[int, list[float]] = {}
        self._seen: dict[int, int] = {}
        self.mobile: set[int] = set()

    def _travel(self, candidate: Candidate) -> float | None:
        """Largest distance this track has ever moved from its own bounding extent.

        Returns None while the track has too few observations to judge.
        """
        track_id = candidate.track_id
        if track_id is None:
            return None
        fx, fy = candidate.foot
        extent = self._extent.get(track_id)
        if extent is None:
            self._extent[track_id] = [fx, fx, fy, fy]
            self._seen[track_id] = 1
            return None
        extent[0] = min(extent[0], fx)
        extent[1] = max(extent[1], fx)
        extent[2] = min(extent[2], fy)
        extent[3] = max(extent[3], fy)
        self._seen[track_id] = self._seen[track_id] + 1
        travel = math.hypot(extent[1] - extent[0], extent[3] - extent[2])
        if travel >= self.min_travel_px:
            self.mobile.add(track_id)
        if self._seen[track_id] < self.cfg.static_track_min_observations:
            return None
        return travel

    def split(self, raw: list[Candidate]) -> tuple[list[Candidate], list[dict[str, Any]]]:
        """Split detections into on-court candidates and rejects.

        Every reject carries its reason *and* the numbers behind it. Reconstructing why a player
        was dropped from a bare reason string means re-deriving the geometry by hand, which is
        exactly the debugging loop this project is supposed to avoid.
        """
        kept: list[Candidate] = []
        rejected: list[dict[str, Any]] = []
        for candidate in raw:
            fx, fy = candidate.foot
            height_frac = candidate.height / self.height
            travel = self._travel(candidate)
            detail = {
                **candidate.compact(),
                "foot_norm": [round(fx / self.width, 4), round(fy / self.height, 4)],
                "height_frac": round(height_frac, 4),
            }
            if height_frac < self.cfg.min_box_height_frac:
                rejected.append({**detail, "reason": "too_small",
                                 "limit": self.cfg.min_box_height_frac})
                continue
            if height_frac > self.cfg.max_box_height_frac:
                rejected.append({**detail, "reason": "too_large_foreground",
                                 "limit": self.cfg.max_box_height_frac})
                continue
            if not self.court.contains_normalized(fx / self.width, fy / self.height):
                rejected.append({**detail, "reason": "foot_outside_court"})
                continue
            if (
                candidate.track_id is not None
                and candidate.track_id not in self.mobile
                and travel is not None
            ):
                rejected.append({**detail, "reason": "static_track",
                                 "travel_px": round(travel, 1),
                                 "limit_px": round(self.min_travel_px, 1),
                                 "observations": self._seen[candidate.track_id]})
                continue
            kept.append(candidate)
        return kept, rejected


def iter_player_frames(
    video: str | Path,
    court: CourtROI,
    cfg: PlayerConfig,
) -> Iterator[dict[str, Any]]:
    """Run pose tracking and yield one record per frame. Importing Ultralytics is deferred so
    that the parsing/analysis test suite never needs torch installed."""
    from ultralytics import YOLO

    model = YOLO(cfg.model)
    near_slot, far_slot = Slot("near"), Slot("far")
    candidate_filter: CandidateFilter | None = None

    stream = model.track(
        source=str(video),
        stream=True,
        persist=True,
        classes=[0],
        conf=cfg.conf,
        iou=cfg.iou,
        imgsz=cfg.imgsz,
        tracker=cfg.tracker,
        verbose=False,
    )

    for frame_index, result in enumerate(stream):
        height, width = result.orig_shape
        raw: list[Candidate] = []
        boxes = result.boxes
        keypoints = result.keypoints
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.detach().cpu().tolist()
            confs = boxes.conf.detach().cpu().tolist() if boxes.conf is not None else [0.0] * len(xyxy)
            ids = boxes.id.detach().cpu().tolist() if boxes.id is not None else [None] * len(xyxy)
            kpt_data = keypoints.data.detach().cpu().tolist() if keypoints is not None else []
            for i, box in enumerate(xyxy):
                raw.append(
                    Candidate(
                        bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                        conf=float(confs[i]),
                        track_id=None if ids[i] is None else int(ids[i]),
                        keypoints=_keypoint_list(kpt_data[i]) if i < len(kpt_data) else [],
                    )
                )

        if candidate_filter is None:
            candidate_filter = CandidateFilter(court, cfg, width, height)
        kept, rejected = candidate_filter.split(raw)
        near, far = select_players(
            kept,
            near_slot,
            far_slot,
            court=court,
            width=width,
            height=height,
            diagonal=math.hypot(width, height),
            forget_after=cfg.identity_smoothing_frames,
        )
        chosen_ids = {
            p["track_id"] for p in (near, far) if p and p.get("track_id") is not None
        }
        not_selected = [
            {
                **candidate.compact(),
                "foot_norm": [round(candidate.foot[0] / width, 4),
                              round(candidate.foot[1] / height, 4)],
                "side": court.side_of(candidate.foot[0] / width, candidate.foot[1] / height),
            }
            for candidate in kept
            if candidate.track_id is None or candidate.track_id not in chosen_ids
        ]
        yield {
            "frame": frame_index,
            "width": width,
            "height": height,
            "near": near,
            "far": far,
            "detections_total": len(raw),
            "candidates_on_court": len(kept),
            "not_selected": not_selected,
            "rejected": rejected,
        }


def extract_players(
    video: str | Path,
    output_jsonl: str | Path,
    court: CourtROI,
    cfg: PlayerConfig,
    *,
    progress_every: int = 200,
) -> dict[str, Any]:
    output = Path(output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)

    frames = 0
    both = 0
    near_only = 0
    far_only = 0
    neither = 0
    with output.open("w", encoding="utf-8") as handle:
        for record in iter_player_frames(video, court, cfg):
            handle.write(json.dumps(record) + "\n")
            frames += 1
            has_near = record["near"] is not None
            has_far = record["far"] is not None
            if has_near and has_far:
                both += 1
            elif has_near:
                near_only += 1
            elif has_far:
                far_only += 1
            else:
                neither += 1
            if progress_every and frames % progress_every == 0:
                print(f"  players: {frames} frames processed", flush=True)

    return {
        "frames": frames,
        "both_players": both,
        "near_only": near_only,
        "far_only": far_only,
        "neither": neither,
        "both_players_fraction": both / frames if frames else 0.0,
        "output": str(output),
    }
