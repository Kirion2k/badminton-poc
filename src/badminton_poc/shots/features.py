"""Turn a shot window into arrays a classifier can eat, with every normalization stated.

Three streams, shaped as BST's dataset produces them:

    joints    (T, 2, 17, 2)   normalized inside each player's bounding box
    position  (T, 2, 2)       foot midpoint
    shuttle   (T, 2)          normalized by the video resolution

The player axis is ordered **far first, then near**. BST orders it Top before Bottom by court
y, and its class names carry a ``Top_`` or ``Bottom_`` prefix, so getting this backwards would
silently swap every prediction's side. Our near/far slots already come from which side of the
net line a foot point falls on, which is a sounder test than comparing raw image y.

Missing data is zero, matching upstream: ``normalize_joints`` keys off ``arr != 0.0`` and their
preparation writes zeros for both players on any frame that failed. We keep that convention so
the arrays are interchangeable, and carry an explicit boolean mask beside them, because a zero
that means "absent" and a zero that means "at the origin" must not be the same thing to us.

**Frame rate.** This is the part that does not transfer for free. ShuttleSet's videos run at 25
or 30 fps, measured across its 104 annotation files by dividing ``frame_num`` by the annotated
timestamp: 46 files at 25 and 58 at 30. Our recordings are genuine 60. A 1.8-second window is
45 to 54 samples for them and 108 for us, and BST's own ``make_seq_len_same`` only decimates
above 100 samples, so raw 60 fps windows would arrive at roughly double the temporal density
its weights were trained on, with every velocity halved per timestep. So the window is resampled
by *time* to ``target_hz`` before anything else touches it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ..parsing import BallTrack
from .windows import ShotSample

N_JOINTS = 17
PLAYER_ORDER = ("far", "near")
"""Index 0 is BST's Top player, index 1 its Bottom player."""

SHUTTLESET_SAMPLE_HZ = (25.0, 30.0)
"""Measured from the BST repository's own ShuttleSet annotations, not assumed."""


@dataclass(frozen=True)
class FeatureConfig:
    target_hz: float = 30.0
    """Sampling rate the window is resampled to. Defaults to the upper of the two rates
    ShuttleSet actually contains, so we lose no more temporal detail than its own worst case."""
    max_samples: int = 100
    """BST's ``seq_len`` for the between-two-hits configuration. A window longer than this is
    truncated at the end, which is what upstream's stride logic effectively does."""
    min_joint_confidence: float = 0.3
    """A keypoint below this is treated as absent rather than trusted. The pose model reports a
    coordinate for every joint whether or not it saw one."""
    fixed_point_tolerance_px: float = 12.0
    """A shuttle sample this close to a known tracker lock-on is not a shuttle."""
    center_align: bool = True
    """Upstream's ``normalize_joints`` default: shift so the bounding-box centre is the origin."""


@dataclass(frozen=True)
class ShotFeatures:
    sample: ShotSample
    times_s: np.ndarray
    """(T,) seconds relative to the contact. Negative before it."""
    frames: np.ndarray
    """(T,) the source frame each sample was drawn from."""
    joints: np.ndarray
    """(T, 2, 17, 2) bounding-box normalized, far player first."""
    joints_mask: np.ndarray
    """(T, 2, 17) True where a joint was actually observed."""
    bbox: np.ndarray
    """(T, 2, 4) raw pixel boxes, kept so a normalization can be re-derived or checked."""
    position: np.ndarray
    """(T, 2, 2) foot midpoint, normalized by frame width and height. NOT court coordinates."""
    position_mask: np.ndarray
    shuttle: np.ndarray
    """(T, 2) normalized by the video resolution, as upstream's ``normalize_shuttlecock``."""
    shuttle_mask: np.ndarray
    hitter_index: int
    """0 or 1, indexing the player axis."""
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return len(self.times_s)

    def coverage(self) -> dict[str, float]:
        """How much of this window was actually observed rather than zero-filled."""
        hitter = self.joints_mask[:, self.hitter_index, :]
        opponent = self.joints_mask[:, 1 - self.hitter_index, :]
        return {
            "hitter_joints": float(hitter.mean()),
            "opponent_joints": float(opponent.mean()),
            "hitter_frames_any_joint": float(hitter.any(axis=-1).mean()),
            "shuttle": float(self.shuttle_mask.mean()),
            "position": float(self.position_mask.mean()),
        }


def resample_times(
    start_frame: int, end_frame: int, hit_frame: int, fps: float, cfg: FeatureConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Frames to sample, and their times relative to the contact.

    Sampling is by time at ``target_hz`` and picks the nearest real frame. Nothing is
    interpolated: upstream decimates with a stride and comments that normalized coordinates
    should not be interpolated, and inventing intermediate poses would be fabricating data.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    if cfg.target_hz <= 0:
        raise ValueError("target_hz must be positive")
    if end_frame < start_frame:
        raise ValueError("window ends before it starts")

    duration = (end_frame - start_frame + 1) / fps
    count = max(1, int(math.floor(duration * cfg.target_hz)))
    count = min(count, cfg.max_samples)
    step = fps / cfg.target_hz
    frames = np.array(
        [min(end_frame, start_frame + int(round(i * step))) for i in range(count)], dtype=int
    )
    times = (frames - hit_frame) / fps
    return frames, times


def normalize_joints(
    keypoints: np.ndarray, bbox: np.ndarray, *, center_align: bool
) -> np.ndarray:
    """Upstream's rule: relative to the box top-left, over the box diagonal, centre-aligned.

    ``keypoints`` (J, 2), ``bbox`` (4,). A zero coordinate is left at zero, which is how
    upstream marks a joint it did not see.
    """
    diagonal = float(np.linalg.norm(bbox[2:] - bbox[:2]))
    if diagonal <= 0:
        return np.zeros_like(keypoints)
    x = keypoints[:, 0]
    y = keypoints[:, 1]
    nx = np.where(x != 0.0, (x - bbox[0]) / diagonal, 0.0)
    ny = np.where(y != 0.0, (y - bbox[1]) / diagonal, 0.0)
    if center_align:
        centre = (bbox[:2] + bbox[2:]) / 2.0
        cx, cy = (centre - bbox[:2]) / diagonal
        nx = np.where(x != 0.0, nx - cx, 0.0)
        ny = np.where(y != 0.0, ny - cy, 0.0)
    return np.stack((nx, ny), axis=-1)


def normalize_shuttle(x: float, y: float, width: int, height: int) -> tuple[float, float]:
    """Upstream's ``normalize_shuttlecock``: straight division by the video resolution."""
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    return x / width, y / height


def _player_arrays(
    player: dict[str, Any] | None, cfg: FeatureConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalized joints, mask and raw box for one player on one frame."""
    joints = np.zeros((N_JOINTS, 2), dtype=np.float32)
    mask = np.zeros(N_JOINTS, dtype=bool)
    bbox = np.zeros(4, dtype=np.float32)
    if not player:
        return joints, mask, bbox
    raw = player.get("keypoints") or []
    box = player.get("bbox")
    if box is None or len(raw) < N_JOINTS:
        return joints, mask, bbox
    bbox = np.asarray(box, dtype=np.float32)
    points = np.zeros((N_JOINTS, 2), dtype=np.float32)
    for j in range(N_JOINTS):
        kx, ky, kc = raw[j][0], raw[j][1], raw[j][2]
        if kc >= cfg.min_joint_confidence and (kx != 0.0 or ky != 0.0):
            points[j] = (kx, ky)
            mask[j] = True
    joints = normalize_joints(points, bbox, center_align=cfg.center_align).astype(np.float32)
    return joints, mask, bbox


def build_features(
    sample: ShotSample,
    players_by_frame: dict[int, dict[str, Any]],
    ball: BallTrack,
    *,
    cfg: FeatureConfig | None = None,
    fixed_points: Sequence[dict[str, Any]] = (),
) -> ShotFeatures:
    """Assemble one shot's arrays. Pure: it reads dictionaries and a track, nothing else."""
    cfg = cfg or FeatureConfig()
    frames, times = resample_times(
        sample.start_frame, sample.end_frame, sample.hit_frame, sample.fps, cfg
    )
    n = len(frames)
    width = sample.width
    height = sample.height or 1

    joints = np.zeros((n, 2, N_JOINTS, 2), dtype=np.float32)
    joints_mask = np.zeros((n, 2, N_JOINTS), dtype=bool)
    bbox = np.zeros((n, 2, 4), dtype=np.float32)
    position = np.zeros((n, 2, 2), dtype=np.float32)
    position_mask = np.zeros((n, 2), dtype=bool)
    shuttle = np.zeros((n, 2), dtype=np.float32)
    shuttle_mask = np.zeros(n, dtype=bool)

    for i, frame in enumerate(frames):
        row = players_by_frame.get(int(frame)) or {}
        for slot_index, slot in enumerate(PLAYER_ORDER):
            player = row.get(slot)
            j, m, b = _player_arrays(player, cfg)
            joints[i, slot_index] = j
            joints_mask[i, slot_index] = m
            bbox[i, slot_index] = b
            foot = (player or {}).get("foot")
            if foot:
                position[i, slot_index] = (foot[0] / width, foot[1] / height)
                position_mask[i, slot_index] = True

        ball_sample = ball.get(int(frame))
        if ball_sample is not None and ball_sample.visible:
            parked = any(
                math.hypot(ball_sample.x - float(p["x"]), ball_sample.y - float(p["y"]))
                <= cfg.fixed_point_tolerance_px
                for p in fixed_points
            )
            if not parked:
                shuttle[i] = normalize_shuttle(ball_sample.x, ball_sample.y, width, height)
                shuttle_mask[i] = True

    hitter_index = PLAYER_ORDER.index(sample.hitter) if sample.hitter in PLAYER_ORDER else 1
    provenance = {
        "target_hz": cfg.target_hz,
        "source_fps": sample.fps,
        "decimation_ratio": round(sample.fps / cfg.target_hz, 3),
        "window_seconds": round(sample.seconds, 3),
        "samples": n,
        "samples_if_not_resampled": sample.frames,
        "player_axis": list(PLAYER_ORDER),
        "joint_format": "coco17",
        "joints_normalization": "bbox top-left origin, divided by bbox diagonal, centre-aligned",
        "shuttle_normalization": "divided by video width and height",
        "position_normalization": "foot point divided by video width and height (image, NOT court)",
        "missing_convention": "zero, with a separate boolean mask",
        "min_joint_confidence": cfg.min_joint_confidence,
    }
    return ShotFeatures(
        sample=sample, times_s=times, frames=frames, joints=joints, joints_mask=joints_mask,
        bbox=bbox, position=position, position_mask=position_mask, shuttle=shuttle,
        shuttle_mask=shuttle_mask, hitter_index=hitter_index, provenance=provenance,
    )
