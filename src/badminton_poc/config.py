"""Pipeline configuration and court ROI handling.

Everything that a human might want to tune for a specific clip lives here so it can be
inspected and version-controlled, rather than being buried in CLI defaults.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = REPO_ROOT / "configs"


class ConfigError(ValueError):
    """Raised when a configuration file is structurally wrong."""


@dataclass(frozen=True)
class CourtROI:
    """Court region as a normalized polygon (0-1 coordinates of the *working* video).

    A polygon rather than a rectangle because the court is a trapezoid in a side-on view;
    a bounding rectangle inevitably swallows neighbouring courts and spectators.

    A person is considered "on court" when their *foot point* (bottom-centre of the box)
    falls inside this polygon. Foot point rather than box centre because spectators in the
    foreground have centres over the court but feet below/outside it.
    """

    polygon: tuple[tuple[float, float], ...]
    net_line: tuple[tuple[float, float], tuple[float, float]] | None = None
    """Two normalized points on the line where the net meets the floor. Near/far is decided
    by which side of this line a player's foot point falls on.

    A line rather than a single y value because the camera is rarely square to the court: in
    our first test clip the net's floor line drops from y=0.52 on the left to y=0.60 on the
    right, so a horizontal split misclassifies players near the sidelines."""
    net_y: float | None = None
    """Legacy horizontal split, used only when net_line is not given."""
    shuttle_polygon: tuple[tuple[float, float], ...] | None = None
    """Airspace above our court, as a normalized polygon.

    TrackNetV3 emits exactly one shuttle position per frame. With more than one court in view it
    happily locks onto a neighbouring court's shuttle: on our test clip that produced 'hits'
    365-760 px from both of our players. This polygon marks which detections can plausibly be
    our shuttle. It is a separate, larger region from the floor polygon because a shuttle flies
    high above the court it belongs to."""

    def __post_init__(self) -> None:
        if len(self.polygon) < 3:
            raise ConfigError("court polygon needs at least 3 points")
        for point in self.polygon:
            if len(point) != 2:
                raise ConfigError(f"court polygon point must be (x, y), got {point!r}")
            x, y = point
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ConfigError(f"court polygon point {point!r} is not normalized to 0-1")
        if self.net_y is not None and not (0.0 <= self.net_y <= 1.0):
            raise ConfigError("net_y must be normalized to 0-1")
        if self.shuttle_polygon is not None and len(self.shuttle_polygon) < 3:
            raise ConfigError("shuttle_polygon needs at least 3 points")
        if self.net_line is not None:
            if len(self.net_line) != 2:
                raise ConfigError("net_line must be exactly two points")
            (x1, y1), (x2, y2) = self.net_line
            if x1 == x2 and y1 == y2:
                raise ConfigError("net_line points must be distinct")

    @classmethod
    def full_frame(cls) -> "CourtROI":
        return cls(polygon=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))

    @classmethod
    def from_obj(cls, obj: Any) -> "CourtROI":
        if isinstance(obj, CourtROI):
            return obj
        if isinstance(obj, dict):
            polygon = obj.get("polygon")
            if polygon is None:
                raise ConfigError("court ROI object must contain 'polygon'")
            raw_shuttle = obj.get("shuttle_polygon")
            shuttle_polygon = None
            if raw_shuttle is not None:
                shuttle_polygon = tuple((float(q[0]), float(q[1])) for q in raw_shuttle)
            raw_net = obj.get("net_line")
            net_line = None
            if raw_net is not None:
                if len(raw_net) != 2:
                    raise ConfigError("net_line must be exactly two [x, y] points")
                net_line = tuple((float(p[0]), float(p[1])) for p in raw_net)  # type: ignore[assignment]
            return cls(
                polygon=tuple((float(p[0]), float(p[1])) for p in polygon),
                net_line=net_line,
                net_y=None if obj.get("net_y") is None else float(obj["net_y"]),
                shuttle_polygon=shuttle_polygon,
            )
        if isinstance(obj, str):
            return cls.from_rect_string(obj)
        raise ConfigError(f"cannot interpret court ROI from {type(obj).__name__}")

    @classmethod
    def from_rect_string(cls, text: str) -> "CourtROI":
        """Backwards-compatible 'x1,y1,x2,y2' rectangle form."""
        parts = [p.strip() for p in text.split(",")]
        if len(parts) != 4:
            raise ConfigError("rectangle ROI must be 'x1,y1,x2,y2' in normalized 0-1 coordinates")
        x1, y1, x2, y2 = (float(p) for p in parts)
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ConfigError("rectangle ROI must satisfy 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1")
        return cls(polygon=((x1, y1), (x2, y1), (x2, y2), (x1, y2)))

    def pixel_polygon(self, width: int, height: int) -> list[tuple[float, float]]:
        return [(x * width, y * height) for x, y in self.polygon]

    def net_y_at(self, x_norm: float) -> float:
        """Normalized y of the net floor line at a given normalized x."""
        if self.net_line is not None:
            (x1, y1), (x2, y2) = self.net_line
            if x1 == x2:
                return (y1 + y2) / 2.0
            return y1 + (y2 - y1) * (x_norm - x1) / (x2 - x1)
        if self.net_y is not None:
            return self.net_y
        ys = [y for _, y in self.polygon]
        return (min(ys) + max(ys)) / 2.0

    def side_of(
        self,
        x_norm: float,
        y_norm: float,
        *,
        bias: str | None = None,
        hysteresis: float = 0.0,
    ) -> str:
        """'near' if the point is on the camera side of the net, else 'far'.

        Bare, this is a hard threshold, and a player standing at the net sits on it: on the
        test clip the far player crossed the line by 0.0004 normalized -- under one pixel --
        and was reclassified as near, emptying the far pool for 32 frames. Within
        ``hysteresis`` of the line the answer is genuinely not knowable from geometry, so
        ``bias`` -- the side that already owns this track -- decides instead. Beyond the band
        geometry always wins, so a player who really does change ends changes sides.
        """
        delta = y_norm - self.net_y_at(x_norm)
        if bias is not None and abs(delta) <= hysteresis:
            return bias
        return "near" if delta >= 0 else "far"

    def net_delta(self, x_norm: float, y_norm: float) -> float:
        """Signed normalized distance from the net floor line; negative is the far side."""
        return y_norm - self.net_y_at(x_norm)

    def net_endpoints_pixels(self, width: int, height: int) -> tuple[tuple[int, int], tuple[int, int]]:
        return (
            (0, int(round(self.net_y_at(0.0) * height))),
            (width, int(round(self.net_y_at(1.0) * height))),
        )

    def contains_normalized(self, x: float, y: float) -> bool:
        return point_in_polygon(x, y, self.polygon)

    def shuttle_contains_normalized(self, x: float, y: float) -> bool:
        """Whether a shuttle position could belong to our court. True everywhere when no
        shuttle polygon is configured, so a single-court clip needs no extra setup."""
        if self.shuttle_polygon is None:
            return True
        return point_in_polygon(x, y, self.shuttle_polygon)

    def to_dict(self) -> dict[str, Any]:
        return {
            "polygon": [list(p) for p in self.polygon],
            "net_line": None if self.net_line is None else [list(p) for p in self.net_line],
            "net_y": self.net_y,
            "shuttle_polygon": (
                None if self.shuttle_polygon is None else [list(p) for p in self.shuttle_polygon]
            ),
        }


def point_in_polygon(x: float, y: float, polygon: Sequence[Sequence[float]]) -> bool:
    """Standard ray-casting test. Points exactly on an edge are not guaranteed either way,
    which is fine for our purposes (a shuttle-width of slack does not change any decision)."""
    inside = False
    count = len(polygon)
    j = count - 1
    for i in range(count):
        xi, yi = polygon[i][0], polygon[i][1]
        xj, yj = polygon[j][0], polygon[j][1]
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


@dataclass(frozen=True)
class PlayerConfig:
    model: str = "yolo11m-pose.pt"
    imgsz: int = 1280
    conf: float = 0.25
    iou: float = 0.5
    tracker: str = "bytetrack.yaml"
    min_box_height_frac: float = 0.02
    """Reject person boxes shorter than this fraction of frame height (noise/far spectators)."""
    max_box_height_frac: float = 0.62
    """Reject person boxes taller than this fraction of frame height.

    This is a weak secondary guard, not the main filter. On a low camera the near player at the
    baseline legitimately fills more than half the frame height (0.53 on our test clip), so
    setting this tightly to exclude foreground spectators throws the near player away instead.
    The foot-point-in-polygon test is what actually separates them."""
    identity_smoothing_frames: int = 15
    """How long a near/far slot keeps claiming a track id after it stops being seen."""
    static_track_min_travel_heights: float = 0.12
    """A track that has never moved this far, measured in its own box heights, is not a player.

    Spectators, umpires and equipment produce perfectly stable detections, and stability is
    exactly what the identity-continuity bonus rewards, so without this they out-compete a real
    player whose track keeps breaking.

    Measured in body heights rather than pixels because of perspective. The far player is half
    the size of the near player in frame and moves a third as many pixels for the same distance
    on court, so an absolute pixel threshold judges the two ends of the court by different
    standards: on the test clip it rejected a real far player who had moved 35.4 px while being
    88 px tall. Scaled, the observed populations separate by an order of magnitude -- real
    players travelled 0.21 to 0.40 box heights over the window in which they were judged,
    against 0.007 to 0.02 for the umpire and a static bag."""
    static_track_min_observations: int = 45
    """Observations required before a track can be judged static, and the length of the
    sliding window its motion is measured over, so a player is never discarded on the strength
    of a couple of frames and a static object never accumulates jitter into apparent motion."""
    static_track_handoff_frames: int = 15
    """How long a proven-mobile track's mobility remains claimable after it disappears.

    ByteTrack broke the far player's identity repeatedly on the test clip. Judging each fresh
    id from scratch is what let a player with 143 px of real movement be rejected as static for
    78 consecutive frames: the new id inherited none of the evidence."""
    static_track_handoff_heights: float = 1.0
    """How close a new track must appear to a vanished mobile track to inherit its mobility,
    in box heights. Spatial, or every new id in the frame inherits a player's mobility."""
    identity_transfer_px_per_frame: float = 30.0
    """How far a slot will accept its identity jumping, per frame since it last saw its track.

    Zero disables the guard. A slot that still remembers a track will not hand itself to a
    different, distant one: a player cannot cross the court in a frame. identity_smoothing_
    frames does not cover this -- it protects a slot against an empty pool, not against a
    substitute standing somewhere else."""
    net_hysteresis_frac: float = 0.01
    """Normalized band around the net floor line inside which track identity, not geometry,
    decides which side a player is on. See ``CourtROI.side_of``."""


@dataclass(frozen=True)
class ShuttleConfig:
    repo: str = "../BadmintonTrackNet"
    python: str | None = None
    """Interpreter used to run upstream. None -> the interpreter running us.
    Set this if upstream ever needs its own environment."""
    tracknet_ckpt: str = "ckpts/TrackNet_best.pt"
    inpaintnet_ckpt: str = "ckpts/InpaintNet_best.pt"
    device: str = "auto"
    batch_size: int = 16
    eval_mode: str = "weight"
    max_sample_num: int = 300
    """Frames upstream samples to build its background median image.

    Upstream's default is 1800. It holds every sample in memory at full resolution and then
    medians them, which costs roughly 2 x frames x width x height x 3 bytes: at 1920x884 that is
    ~18 GB and the process is killed by the OOM killer with an empty stderr. 300 evenly-strided
    samples estimate a static background just as well at ~3 GB."""
    output_video: bool = False
    """Upstream's own overlay video. Off by default; we render our own richer one."""


@dataclass(frozen=True)
class AnalysisConfig:
    hit_search_radius_frames: int = 4
    """How far either side of an event frame we look for a visible shuttle sample."""
    assignment_margin_px: float = 40.0
    """Distance gap below which a near/far assignment is treated as ambiguous."""
    max_assignment_distance_frac: float = 0.35
    """Reject an assignment if the shuttle is further than this fraction of frame
    diagonal from the closest player anchor."""
    rally_gap_seconds: float = 2.5
    """Silence between events longer than this starts a new rally."""
    min_hits_per_rally: int = 2
    trailing_swing_trim: float = 0.0
    """Walk back from a rally's last hit while the swing evidence is below this, in
    body-heights per second.

    Zero disables it. A swing gate on every hit splits rallies by removing real contacts in the
    middle; applied only to the tail it cannot, because it never removes a hit with surviving
    hits after it."""
    rally_end_at_last_hit: bool = False
    """End a rally at its last hit rather than at the terminating landing.

    Landings arrive every 0.8s through dead time on these clips, 47 of 68 gaps sitting exactly
    at the suppression floor, so ending on one puts the boundary wherever the metronome next
    ticks. Ending at the last hit is at worst one stroke early."""
    min_swing_speed: float = 0.0
    """Drop hit candidates whose assigned player shows less racket-arm motion than this, in
    body-heights per second, measured from the pose model.

    Zero disables it. The only measured discriminator that does not come from the event
    detector itself; whether it should be on is a question for labelled clips."""
    play_gate_min_speed_px: float = 0.0
    """Median shuttle speed, in px/frame over a short window, above which play is in progress.

    Zero disables the gate. Rally segmentation is otherwise driven entirely by hits, which is
    circular: a hit fired during dead time keeps a rally alive past its end or invents one.
    TrackNetV3 emits a shuttle on every frame whether or not one is in play and pins it to a
    fixed position between points, so visibility is no signal and motion is."""
    play_gate_smooth_seconds: float = 0.25
    play_gate_bridge_seconds: float = 1.5
    """Play spans separated by less than this are joined. A high clear leaves the top of the
    frame for the better part of a second while play continues."""
    play_gate_min_span_seconds: float = 1.0
    """Shorter bursts of shuttle motion are not rallies -- somebody picking it up off the
    floor moves it too."""
    min_hit_confidence: float = 0.0
    """Drop hit candidates below this upstream confidence before consolidating.

    Upstream's confidence is not a learned score. It is a fixed value per rule -- 0.50
    speed_local_max, 0.75 acceleration_peak, 0.80 vx_reversal, 0.85 vy_reversal -- raised to
    0.90 when two rules agree and 0.95 when three do. So a threshold here selects rules and
    agreement, not certainty: 0.85 means "vy_reversal fired, or at least two rules agreed".

    Zero by default: which rules to trust has to be settled against hand labels, and picking a
    threshold from unlabelled output is fitting a number to a guess. Set it per clip. Every
    dropped candidate is recorded in analysis.json under low_confidence_hits_dropped."""
    min_hit_interval_seconds: float = 0.30
    """Hit candidates closer together than this are treated as one racket contact."""
    min_landing_interval_seconds: float = 0.75
    """Landing candidates closer together than this are treated as one landing. Wider than the
    hit window because a shuttle cannot land twice in quick succession within a rally."""
    shuttle_absent_seconds: float = 1.0
    """Continuous invisible-shuttle span treated as a shuttle-absence (rally-ending) gap."""


@dataclass(frozen=True)
class PipelineConfig:
    court: CourtROI = field(default_factory=CourtROI.full_frame)
    target_width: int = 1920
    target_fps: float | None = None
    """Uniform fps for the normalized working video. None -> measured average of the source."""
    players: PlayerConfig = field(default_factory=PlayerConfig)
    shuttle: ShuttleConfig = field(default_factory=ShuttleConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)

    @classmethod
    def load(cls, path: str | Path | None) -> "PipelineConfig":
        if path is None:
            return cls()
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ConfigError("config file must contain a JSON object")
        known = {"court", "target_width", "target_fps", "players", "shuttle", "analysis"}
        unknown = set(data) - known - {"_comment", "notes"}
        if unknown:
            raise ConfigError(f"unknown config keys: {sorted(unknown)}")
        return cls(
            court=CourtROI.from_obj(data["court"]) if "court" in data else CourtROI.full_frame(),
            target_width=int(data.get("target_width", 1920)),
            target_fps=None if data.get("target_fps") is None else float(data["target_fps"]),
            players=_build(PlayerConfig, data.get("players")),
            shuttle=_build(ShuttleConfig, data.get("shuttle")),
            analysis=_build(AnalysisConfig, data.get("analysis")),
        )

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["court"] = self.court.to_dict()
        return out


def _build(cls, data: Any):
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise ConfigError(f"{cls.__name__} section must be a JSON object")
    valid = {f for f in cls.__dataclass_fields__}
    unknown = set(data) - valid
    if unknown:
        raise ConfigError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**data)
