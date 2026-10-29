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
    mobility_yield_seconds: float = 20.0
    """A slot prefers a candidate that has moved like a player within this many seconds to one
    that has not, however well the stale one scores.

    Mobility is granted permanently, and a per-window motion test applied across thousands of
    frames will eventually grant it to anything: clip 5's seated line judge was promoted by a
    single posture shift at frame 1271 and was then eligible for the rest of the clip. Being
    eligible was enough, because continuity and proximity are worth 2.3 points that an
    incumbent never has to earn again -- he took the near slot at frame 2580 and held it for
    2048 frames, beating the real player 3.19 to 1.05 on every one of them, and the second half
    of that clip reported no rallies at all.

    The horizon separates the two populations by measurement rather than by taste. Real players
    stay motionless for a while between rallies -- the longest measured is 8.90 s, clip 2's far
    player over 534 frames -- while that judge read as moving on 45 of 4628 frames and was
    quiet for 37 s at the moment he took the slot and 98.5 s afterwards. Seconds, not frames:
    a frame count would judge a 60 fps clip and a 34 fps clip by different standards."""
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
    event_source: str = "tracknet"
    """Where hits and landings come from: ``tracknet`` for upstream's events file, ``flight``
    for this project's flight-segment detector.

    Both stay runnable so any claim about the newer detector is a measured comparison rather
    than an assertion. Under ``flight`` the whole upstream filtering chain is skipped -- the
    confidence threshold, burst consolidation, the play gate and the swing gate all exist to
    thin 777 candidates down to something usable, and a segmentation does not produce them."""
    require_hitter: bool = False
    """Discard hits that could not be attributed to either court player.

    A rally shot happens within reach of somebody. A break in the shuttle track with no player
    near it is the tracker wandering, which is what a spurious dead-time rally is made of."""
    require_both_players: bool = False
    """Discard a rally in which only one player ever made contact.

    A rally is an exchange. A run of contacts by one player is somebody knocking the shuttle up,
    handing it back, or bouncing it on a racket between points. On the development clip the two
    chains that are not rallies read ``NNN`` and ``FF`` while all six real ones carry contacts
    from both sides, so this is a gate on what a rally *is* rather than a tuned threshold. It
    asks only that both players appear, not that they alternate: assignment is 92% accurate, so
    a real rally routinely shows runs of one name."""
    hit_search_radius_frames: int = 4
    """How far either side of an event frame we look for a visible shuttle sample."""
    assignment_margin_px: float = 40.0
    """Distance gap below which a near/far assignment is treated as ambiguous."""
    max_assignment_distance_frac: float = 0.35
    """Reject an assignment if the shuttle is further than this fraction of frame
    diagonal from the closest player anchor.

    A coarse outer bound only, and on our clips it never fires: 0.35 of a 1920x884 diagonal is
    740 px, further than the court is wide. The gate that does the work is the next one."""
    max_contact_distance_body_heights: float = 0.9
    """Reject a contact if the shuttle is further than this from the closest player anchor,
    measured in that player's own box heights.

    A racket contact happens at arm plus racket length, which is about nine tenths of a person's
    height, so this is a statement about badminton rather than about our footage. It is in body
    heights and not pixels because the far player is half the near player's size in frame, so the
    same real distance is half as many pixels at the far end -- the same reason the swing
    features are normalised this way.

    Hand labels on 57 detected contacts across two rallies say a quarter of them are not contacts
    at all, and that the shuttle sits a median 75 px from the nearest player at a real contact
    against 238 px at a spurious one. Fitting the threshold on one clip and testing it on the
    other gives 0.75 and 1.08; this is the midpoint. Fitted in pixels instead, the same procedure
    gives 118 and 185 and does not transfer between the two clips."""
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
class FlightConfig:
    """Parameters of the flight-segment event detector, in seconds and pixels.

    Every duration is in seconds and every distance in pixels at ``reference_width``, because
    these clips do not agree on either axis: the frame rates run 28 to 60 fps, so a window fixed
    in frames measures a different amount of time on each, and a speed in px/frame means half as
    much at 60 fps as at 30. Distances are scaled by the actual frame width at run time.
    """

    reference_width: int = 1920
    """Frame width the pixel figures below were measured at. Distances scale with the actual
    working width, so a clip normalized to a different size keeps the same physical meaning."""
    fit_window_seconds: float = 0.12
    """Width of the centred window a quadratic is fitted to. Long enough that noise averages
    out, short enough that a real flight is still locally quadratic under drag."""
    max_fit_residual_px: float = 2.5
    """Fit residual above which the shuttle is not flying smoothly. Measured on the development
    clip: 0.88 px median in a rally away from any hit, 1.49 px at the 90th percentile, against
    5.14 px at a real contact."""
    min_flight_speed_px_s: float = 150.0
    """A smooth track that is barely moving is not flight."""
    min_flight_seconds: float = 0.10
    """Shorter smooth runs are noise that happened to fit."""
    max_link_gap_seconds: float = 1.8
    """How long the shuttle may go unmodelled between two segments and still belong to the same
    rally. Generous because TrackNet loses fast strokes for over a second: three intra-rally
    dropouts on the development clip run 0.8s, 1.53s and 1.55s, and a tighter limit split two
    rallies in half. Dead time between real rallies on that clip is never shorter than 2.3s.

    Counted over the frames of the gap where the tracker actually had a sample, not over
    wall-clock time. A gap the tracker spent blind is not evidence that play stopped -- it is
    evidence of nothing at all -- and measuring it as though it were is what reported clip 5's
    first rally as starting nearly nine seconds late. That rally breaks twice inside itself, at
    284-370 and 451-512, the second by a single frame over this limit, and the tracker carries a
    sample on only 32% and 19% of those frames. Across the three labelled clips every gap that
    really ends a rally is visible for the whole of it, and settled_between fires on all nine of
    them, so nothing is lost by declining to guess from the blind ones.

    What this leaves exposed: two rallies separated by a long stretch the tracker saw nothing of
    would now be joined, because neither this nor settled_between has anything to go on. No gap
    in the three labelled clips is of that shape, so there is no measurement to set a ceiling
    from and none is invented here."""
    min_turn_degrees: float = 60.0
    """Turn between the velocity leaving one segment and entering the next, above which the join
    is a racket contact rather than one flight the tracker briefly lost. Real contacts sit at a
    median 150 degrees; frames inside a flight sit at a median 9."""
    rest_window_seconds: float = 0.20
    rest_tolerance_px: float = 8.0
    min_rest_seconds: float = 0.25
    """A shuttle counts as lying still when it drifts less than ``rest_tolerance_px`` over
    ``rest_window_seconds`` for at least ``min_rest_seconds``."""
    fixed_point_min_runs: int = 3
    fixed_point_tolerance_px: float = 12.0
    """A frozen position reached in this many separate stretches is a fixed object in the room,
    not a shuttle at rest. TrackNetV3 emits one shuttle per frame whether or not one is in play;
    on the development clip InpaintNet fills gaps with one constant point in fourteen separate
    stretches totalling 2340 frames, while the one real shuttle-at-rest appears once for 85."""
    min_serve_quiet_seconds: float = 1.0
    """Quiet needed before a chain's first segment for its start to count as a serve. Without it
    the rally is reported as ``truncated``: the clip began mid-flight and its real start cannot
    be seen. The development clip opens with the far player already mid-stroke."""
    min_hits_per_rally: int = 2
    promote_continuation_within_body_heights: float = 0.9
    """Treat a low-turn join as a contact anyway when it happens this close to a racket.

    `min_turn_degrees` asks whether the shuttle changed direction enough to have been hit. A
    drive, a flat push and a block barely change it at all, so that test alone throws them away
    as tracker dropout: across the two labelled clips 15 of 31 joins classified as continuations
    sit within a racket's reach of a player, and three of the six contacts known to be missing
    are among them, with turns of 42.8, 56.7 and 57.7 degrees against the 60 degree threshold.

    Angle and position answer different questions. A large turn says a contact happened wherever
    it was; being at the end of somebody's racket says one happened however small the turn. This
    is the second test, and it shares its units and threshold with
    `AnalysisConfig.max_contact_distance_body_heights` for the same reasons.

    Set to 0 to disable and judge joins on turn angle alone."""
    gravity_apex_max_speed_ratio: float = 1.0
    """Below this ratio of outgoing to incoming speed, a join that turns the shuttle from rising
    to falling is the top of an arc rather than a racket.

    A lift and a clear reverse the shuttle as sharply as a contact does, and curvature cannot
    tell them apart. Energy can: a racket adds it, gravity does not. Both conditions are
    required, because a smash taken at the top of the bounce turns over the same way -- it just
    leaves three times faster.

    Measured against the two labelled clips at this setting: it discards 2 of clip 2's 12 false
    positives and 1 of clip 3's 8, and none of their 67 true hits. Free on the evidence
    available, and small; the number of false positives it was checked against is three.

    Zero disables it."""
    gravity_apex_window_seconds: float = 0.135
    """How much flight either side of the join the apex test measures its speeds over.

    In seconds rather than frames: it was validated at 8 frames on two 60 fps clips, and a frame
    count would measure a third of a second on a 30 fps recording and an eighth on a 60 fps one.
    Shorter than this and the test sees no turn at all -- at the 3-frame span the turn angle
    uses, it flags nothing on either labelled clip, because a shuttle near the top of its arc
    barely moves."""
    reentry_kink_speed_ratio: float = 3.0
    """How many times faster than it was arriving the shuttle must suddenly move, in the frames
    where it came back into view after the tracker lost it, to count as a racket contact there.

    The case is a stroke taken above the top of the frame: the shuttle reappears falling, a
    racket meets it, and the fit refuses the kinked frames, so the contact sits inside a gap no
    turn test looks into. The two labelled contacts hidden that way (clip 2 at 4317, clip 3 at
    971) jump 7.2x and 8.4x their approach speed; every other blind re-entry on the three clips
    reads 2.3x or less, the worst a tracker teleporting on its first sample back. Three is a
    round number in that gap, not one fitted to it, and the evidence is two positives.

    Zero disables it."""
    contact_frame_rule: str = "exit"
    """Where in the gap between two flight segments to place the contact.

    The contact happened somewhere between the last frame of the incoming flight and the first
    of the outgoing one, and which frame is reported matters downstream: the shuttle leaves the
    racket fast, so a hit placed late already sits nearer the opponent and gets attributed to
    them. All six wrong assignments under a late rule were near-player shots credited to the far
    player.

    ``exit`` is the last frame the shuttle was still unambiguously on its way *to* the racket,
    which is the sample closest to the hitter. It reports a contact a median three frames (50ms)
    early and is the better choice anyway, measured on the development clip:

        rule            precision  recall     F1   hitter
        residual_peak       0.736   0.898  0.809    0.885
        midpoint            0.750   0.915  0.824    0.926
        exit                0.778   0.949  0.855    0.927
    """


@dataclass(frozen=True)
class PipelineConfig:
    court: CourtROI = field(default_factory=CourtROI.full_frame)
    target_width: int = 1920
    target_fps: float | None = None
    """Uniform fps for the normalized working video. None -> measured average of the source."""
    players: PlayerConfig = field(default_factory=PlayerConfig)
    shuttle: ShuttleConfig = field(default_factory=ShuttleConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    flight: FlightConfig = field(default_factory=FlightConfig)

    @classmethod
    def load(cls, path: str | Path | None) -> "PipelineConfig":
        if path is None:
            return cls()
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ConfigError("config file must contain a JSON object")
        known = {"court", "target_width", "target_fps", "players", "shuttle", "analysis",
                 "flight"}
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
            flight=_build(FlightConfig, data.get("flight")),
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
