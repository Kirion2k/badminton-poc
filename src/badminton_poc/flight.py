"""Shuttle events derived from flight segments rather than from per-frame kinematic rules.

Upstream BadmintonTrackNet fires a hit candidate wherever a finite-difference rule trips on the
shuttle track: 777 of them on the development clip, against 60 real rally shots. No threshold on
those candidates reaches the project's rally-boundary target, because the candidates that survive
inside dead time keep a rally alive past its end, and every filter strong enough to remove them
also removes real shots from the middle of rallies.

This module starts from a different observation, measured on the same clip:

    in a rally, away from any hit   0.88 px   median residual of a quadratic fit over 7 frames
    at a real hit                   5.14 px
    dead time, tracker frozen       0.00 px

A shuttle in flight is almost perfectly smooth. A racket contact is where that smoothness breaks.
So instead of scoring frames, we cut the track into *flight segments* -- maximal runs a smooth
motion model actually fits -- and read the events off the segmentation:

* a **hit** is the join between two segments that meet closely in time with a sharp direction
  change;
* a **landing** is how the last segment of a chain ends;
* a **rally** is a maximal chain of segments joined by hits;
* **dead time** is everything no segment covers, which needs no separate detector.

Two joins are possible and telling them apart is what makes the chains hold together. A join with
a large turn is a contact. A join with almost no turn is one flight the tracker briefly lost, and
merging it was worth two rallies on the development clip that a turn-only rule split in half.

Nothing here reads the pose model or upstream's events file. It takes the shuttle track and the
frame rate, which makes it testable against a hand-written track and re-runnable on every clip
without touching the GPU.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from .config import FlightConfig
from .parsing import BallTrack, ShuttleEvent

SOURCE = "flight"


@dataclass(frozen=True)
class FlightSegment:
    """A run of frames over which the shuttle track fits a smooth motion model."""

    start_frame: int
    end_frame: int
    peak_speed_px_s: float
    median_residual_px: float

    @property
    def frames(self) -> int:
        return self.end_frame - self.start_frame + 1

    def to_dict(self, fps: float) -> dict[str, Any]:
        return {
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "frames": self.frames,
            "seconds": round(self.frames / fps, 3),
            "peak_speed_px_s": round(self.peak_speed_px_s, 1),
            "median_residual_px": round(self.median_residual_px, 2),
        }


@dataclass(frozen=True)
class Link:
    """How one flight segment gives way to the next."""

    before: int
    """Index of the earlier segment."""
    after: int
    gap_frames: int
    turn_degrees: float
    kind: str
    """``contact`` (a racket hit) or ``continuation`` (one flight, tracking briefly lost)."""
    hit_frame: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "before_segment": self.before,
            "after_segment": self.after,
            "gap_frames": self.gap_frames,
            "turn_degrees": round(self.turn_degrees, 1),
            "kind": self.kind,
            "hit_frame": self.hit_frame,
        }


@dataclass(frozen=True)
class FlightRally:
    start_frame: int
    end_frame: int
    hit_frames: tuple[int, ...]
    segment_indices: tuple[int, ...]
    start_kind: str
    """``serve`` when a quiet period precedes the first segment, ``truncated`` when the clip or
    the track simply begins mid-flight and the real start is not observable."""
    rest_seen_at: int | None
    """Frame where the shuttle was next seen lying still, when it was seen at all. Evidence that
    the rally really did end, not the boundary itself: on the development clip the shuttle takes
    up to 1.4s to settle, which puts the rest well past the point."""

    def to_dict(self, fps: float) -> dict[str, Any]:
        return {
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "seconds": round((self.end_frame - self.start_frame + 1) / fps, 3),
            "hit_frames": list(self.hit_frames),
            "segments": list(self.segment_indices),
            "start_kind": self.start_kind,
            "rest_seen_at": self.rest_seen_at,
        }


@dataclass(frozen=True)
class FlightModel:
    segments: tuple[FlightSegment, ...]
    links: tuple[Link, ...]
    rallies: tuple[FlightRally, ...]
    events: tuple[ShuttleEvent, ...]
    fixed_points: tuple[dict[str, Any], ...]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, fps: float) -> dict[str, Any]:
        return {
            "segments": [s.to_dict(fps) for s in self.segments],
            "links": [l.to_dict() for l in self.links],
            "rallies": [r.to_dict(fps) for r in self.rallies],
            "fixed_points": list(self.fixed_points),
            "diagnostics": self.diagnostics,
        }


# --------------------------------------------------------------------------------------------
# frame-level measurements


def _quadratic_residual(times: list[float], values: list[float]) -> float:
    """RMS residual of a least-squares quadratic fit, without pulling in numpy.

    Solved through the normal equations on the 3x3 moment matrix. The windows are seven frames
    and centred on zero, so the system is well conditioned and a hand-rolled solve is cheaper
    and easier to test than a matrix library.
    """
    n = len(times)
    if n < 3:
        return 0.0
    powers = [sum(t ** k for t in times) for k in range(5)]
    rhs = [
        sum(v for v in values),
        sum(t * v for t, v in zip(times, values)),
        sum(t * t * v for t, v in zip(times, values)),
    ]
    matrix = [
        [powers[0], powers[1], powers[2]],
        [powers[1], powers[2], powers[3]],
        [powers[2], powers[3], powers[4]],
    ]
    coeffs = _solve3(matrix, rhs)
    if coeffs is None:
        return 0.0
    a, b, c = coeffs
    total = 0.0
    for t, v in zip(times, values):
        error = (a + b * t + c * t * t) - v
        total += error * error
    return math.sqrt(total / n)


def _solve3(matrix: list[list[float]], rhs: list[float]) -> tuple[float, float, float] | None:
    """Gaussian elimination with partial pivoting on a 3x3 system."""
    m = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            return None
        m[col], m[pivot] = m[pivot], m[col]
        for row in range(col + 1, 3):
            factor = m[row][col] / m[col][col]
            for k in range(col, 4):
                m[row][k] -= factor * m[col][k]
    out = [0.0, 0.0, 0.0]
    for row in (2, 1, 0):
        acc = m[row][3] - sum(m[row][k] * out[k] for k in range(row + 1, 3))
        out[row] = acc / m[row][row]
    return (out[0], out[1], out[2])


def _median(values: Iterable[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _percentile(values: Iterable[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = int(round(fraction * (len(ordered) - 1)))
    return ordered[max(0, min(len(ordered) - 1, index))]


PLAUSIBLE_SHUTTLE_SPEED_M_S = 120.0
"""Faster than any shuttle in play, chosen to be generous rather than tight.

The fastest smash ever recorded leaves the racket near 137 m/s and sheds speed immediately;
amateur rally speeds are a small fraction of that. A tracked step implying more than this is not
a shuttle moving, it is the tracker jumping to something else.
"""

ADULT_HEIGHT_M = 1.75
"""What a player's box height stands for, to turn pixels into metres without a homography."""

MIN_COVERAGE = 0.85
MAX_IMPOSSIBLE_SHARE = 0.005
MAX_P99_SHARE_OF_LIMIT = 0.5
"""Screening thresholds, set by looking at four recordings, three sound and one not.

They exist to make a person look at a clip, and nothing in the pipeline branches on the verdict.
Treat a "suspect" as a prompt to watch the annotated video, not as a measurement: with one bad
clip to calibrate against, the gap between 78% of the speed limit and 21% is the only thing
these really know."""


def track_quality(
    ball: BallTrack,
    fps: float,
    *,
    body_height_px: float | None,
    frames: Iterable[int] | None = None,
) -> dict[str, Any]:
    """How much of the shuttle track is physically possible.

    Everything downstream -- segments, contacts, rallies, shot windows -- is a reading of this
    one track, so a track that teleports produces confident nonsense rather than an error. On
    `test-video` 13% of steps exceed 150 px and 89 exceed 400 px within a single frame, and its
    flight segments come out 4 frames long against 16 and 25 on the clips whose tracking is
    sound. Nothing in the pipeline noticed.

    The speed limit is physical, and scaled by the player's own size so it transfers between
    clips and frame rates: a body height is about 1.75 m, which converts pixels to metres
    wherever that player is standing. Without a body height there is no scale, so the verdict is
    "unknown" rather than a guess.
    """
    positions = {s.frame: (s.x, s.y) for s in ball if s.visible}
    wanted = set(frames) if frames is not None else set(positions)
    seen = sorted(f for f in positions if f in wanted)
    out: dict[str, Any] = {
        "frames_considered": len(wanted),
        "frames_with_a_shuttle": len(seen),
        "coverage": round(len(seen) / len(wanted), 4) if wanted else 0.0,
    }
    steps = [
        (f, math.dist(positions[f], positions[f - 1]))
        for f in seen if f - 1 in positions and f - 1 in wanted
    ]
    if not steps or fps <= 0:
        out["verdict"] = "unknown"
        out["why"] = "not enough consecutive samples to measure"
        return out
    values = sorted(v for _, v in steps)
    out["median_step_px"] = round(values[len(values) // 2], 1)
    out["p99_step_px"] = round(values[min(len(values) - 1, int(0.99 * len(values)))], 1)
    if not body_height_px or body_height_px <= 1:
        out["verdict"] = "unknown"
        out["why"] = "no player box to scale pixels by, so no speed limit can be applied"
        return out

    px_per_m = body_height_px / ADULT_HEIGHT_M
    limit_px = PLAUSIBLE_SHUTTLE_SPEED_M_S / fps * px_per_m
    impossible = [f for f, v in steps if v > limit_px]
    share = len(impossible) / len(steps)
    out["speed_limit_px_per_frame"] = round(limit_px, 1)
    out["speed_limit_note"] = (
        f"{PLAUSIBLE_SHUTTLE_SPEED_M_S:.0f} m/s at {fps:.2f} fps, "
        f"scaled by a {body_height_px:.0f} px player standing for {ADULT_HEIGHT_M} m"
    )
    out["impossible_steps"] = len(impossible)
    out["impossible_share"] = round(share, 4)
    out["first_impossible_frames"] = impossible[:12]
    out["p99_share_of_limit"] = round(values[min(len(values) - 1,
                                                 int(0.99 * len(values)))] / limit_px, 3)

    # A jumping track has two very different causes and they need different answers. A shuttle
    # cannot be in two places, so if the impossible steps are long lateral hops that keep
    # returning to the same band, the tracker is alternating between our court and the one next
    # to it, and the fix is a tighter shuttle_polygon rather than a different clip.
    if impossible:
        all_x = sorted(x for x, _ in positions.values())
        mid = all_x[len(all_x) // 2]
        jump_x = sorted(positions[f][0] for f in impossible if f in positions)
        if jump_x:
            band = jump_x[3 * len(jump_x) // 4] - jump_x[len(jump_x) // 4]
            offset = abs(sum(jump_x) / len(jump_x) - mid)
            out["jump_band_iqr_px"] = round(band, 1)
            out["jump_offset_from_median_x_px"] = round(offset, 1)
            # Reported, not judged. Two causes produce impossible steps and they need opposite
            # answers: the tracker taking the neighbouring court's shuttle, which is fixed by a
            # tighter shuttle_polygon, or a track that is simply bad, which is fixed by a
            # different clip. Separating them automatically was tried and the rule could not
            # tell a second court from scatter on synthetic cases, so these two numbers are left
            # for a person: a second court shows as jumps landing in a narrow band well away
            # from where the shuttle otherwise sits.

    # A shuttle is near its top speed for a few frames of a smash and nowhere else, so a track
    # whose 99th percentile sits close to the physical maximum is jumping, not flying.
    faults = []
    if out["coverage"] < MIN_COVERAGE:
        faults.append(f"the shuttle is missing on {1 - out['coverage']:.0%} of rally frames")
    if share >= MAX_IMPOSSIBLE_SHARE:
        faults.append(f"{share:.1%} of steps imply a shuttle faster than "
                      f"{PLAUSIBLE_SHUTTLE_SPEED_M_S:.0f} m/s")
    if out["p99_share_of_limit"] >= MAX_P99_SHARE_OF_LIMIT:
        faults.append(f"the 99th percentile step is {out['p99_share_of_limit']:.0%} of the "
                      f"physical maximum, where a sound track sits near a fifth")
    out["verdict"] = "suspect" if faults else "sound"
    if faults:
        tail = (". Events read off this track may not be measuring the shuttle. Check "
                "jump_band_iqr_px against jump_offset_from_median_x_px first: jumps landing in "
                "a narrow band well away from the rest of the track are the neighbouring "
                "court's shuttle, which a tighter shuttle_polygon fixes without changing clip.")
    else:
        tail = ""
    out["why"] = ("; ".join(faults) + tail if faults
                  else "coverage, step sizes and the speed limit all look like a real shuttle")
    return out


def measure_track(
    ball: BallTrack, fps: float, *, window_frames: int
) -> tuple[dict[int, float], dict[int, float]]:
    """Per-frame fit residual and local speed, over a centred window of visible samples.

    A frame without a *full* window is left out of both maps rather than measured on a partial
    one: a residual taken over four samples is not comparable to one taken over seven, and a
    frame we cannot measure should read as unknown, not as smooth.
    """
    if window_frames < 3 or window_frames % 2 == 0:
        raise ValueError("window_frames must be odd and at least 3")
    half = window_frames // 2
    positions = {s.frame: (s.x, s.y) for s in ball if s.visible}
    residual: dict[int, float] = {}
    speed: dict[int, float] = {}
    for frame in ball.frames:
        window = [f for f in range(frame - half, frame + half + 1) if f in positions]
        if len(window) < window_frames:
            continue
        times = [float(f - frame) for f in window]
        rx = _quadratic_residual(times, [positions[f][0] for f in window])
        ry = _quadratic_residual(times, [positions[f][1] for f in window])
        residual[frame] = math.hypot(rx, ry)
        steps = [
            math.hypot(
                positions[window[i + 1]][0] - positions[window[i]][0],
                positions[window[i + 1]][1] - positions[window[i]][1],
            )
            for i in range(len(window) - 1)
        ]
        speed[frame] = _median(steps) * fps
    return residual, speed


def find_fixed_points(
    ball: BallTrack, fps: float, cfg: FlightConfig, scale: float = 1.0
) -> list[dict[str, Any]]:
    """Positions the tracker locks onto that are not a shuttle lying still.

    TrackNetV3 emits exactly one shuttle per frame whether or not one is in play, so between
    points it settles on whatever else in the room looks like a shuttle. On the development clip
    that is a single ceiling light: the position is pixel-identical across fourteen separate
    stretches totalling 2340 frames, while the one genuine shuttle-at-rest appears once for 85.

    Recurrence is what separates them, and it needs no court geometry: a shuttle can lie still
    in one place for a while, but it does not return to the same pixel again and again across a
    whole recording.
    """
    runs = _rest_runs(ball, fps, cfg, scale)
    tolerance = cfg.fixed_point_tolerance_px * scale
    buckets: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for start, end, (x, y) in runs:
        buckets[(int(x // max(1.0, tolerance)), int(y // max(1.0, tolerance)))].append((start, end))
    fixed: list[dict[str, Any]] = []
    for (bx, by), spans in buckets.items():
        if len(spans) < cfg.fixed_point_min_runs:
            continue
        fixed.append(
            {
                "x": round((bx + 0.5) * tolerance, 1),
                "y": round((by + 0.5) * tolerance, 1),
                "runs": len(spans),
                "frames": sum(end - start + 1 for start, end in spans),
                "first_frame": min(start for start, _ in spans),
                "note": (
                    "the shuttle track froze on this point in several separate stretches, so it "
                    "is a fixed object in the room rather than a shuttle come to rest"
                ),
            }
        )
    fixed.sort(key=lambda item: -item["frames"])
    return fixed


def _rest_runs(
    ball: BallTrack, fps: float, cfg: FlightConfig, scale: float
) -> list[tuple[int, int, tuple[float, float]]]:
    """Stretches where the shuttle position barely moves, with where it sat."""
    window = max(2, int(round(cfg.rest_window_seconds * fps)))
    minimum = max(1, int(round(cfg.min_rest_seconds * fps)))
    tolerance = cfg.rest_tolerance_px * scale
    positions = {s.frame: (s.x, s.y) for s in ball if s.visible}
    still: dict[int, tuple[float, float]] = {}
    for frame in ball.frames:
        span = [f for f in range(frame, frame + window) if f in positions]
        if len(span) < window:
            continue
        anchor = positions[span[0]]
        drift = max(
            math.hypot(positions[f][0] - anchor[0], positions[f][1] - anchor[1]) for f in span
        )
        if drift < tolerance:
            still[frame] = anchor
    runs: list[tuple[int, int, tuple[float, float]]] = []
    current: list[Any] | None = None
    for frame in ball.frames:
        anchor = still.get(frame)
        if anchor is None:
            if current is not None and current[1] - current[0] + 1 >= minimum:
                runs.append((current[0], current[1], current[2]))
            current = None
        elif current is None:
            current = [frame, frame, anchor]
        else:
            current[1] = frame
    if current is not None and current[1] - current[0] + 1 >= minimum:
        runs.append((current[0], current[1], current[2]))
    return runs


# --------------------------------------------------------------------------------------------
# segmentation


def _runs_of(frames: list[int], member: set[int]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    previous: int | None = None
    for frame in frames:
        if frame in member:
            if start is None or (previous is not None and frame != previous + 1):
                if start is not None and previous is not None:
                    runs.append((start, previous))
                start = frame
        else:
            if start is not None and previous is not None:
                runs.append((start, previous))
            start = None
        previous = frame
    if start is not None and previous is not None:
        runs.append((start, previous))
    return runs


def _turn_degrees(
    positions: dict[int, tuple[float, float]], exit_frame: int, entry_frame: int, span: int
) -> float | None:
    """Angle between the velocity leaving one segment and the velocity entering the next."""
    before = [f for f in range(exit_frame - span, exit_frame + 1) if f in positions]
    after = [f for f in range(entry_frame, entry_frame + span + 1) if f in positions]
    if len(before) < 2 or len(after) < 2:
        return None
    v0 = (
        positions[before[-1]][0] - positions[before[0]][0],
        positions[before[-1]][1] - positions[before[0]][1],
    )
    v1 = (
        positions[after[-1]][0] - positions[after[0]][0],
        positions[after[-1]][1] - positions[after[0]][1],
    )
    n0, n1 = math.hypot(*v0), math.hypot(*v1)
    if n0 < 1e-6 or n1 < 1e-6:
        return None
    cosine = (v0[0] * v1[0] + v0[1] * v1[1]) / (n0 * n1)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def segment_flight(
    ball: BallTrack,
    fps: float,
    cfg: FlightConfig,
    *,
    width: int = 1920,
    racket_reach_at: "Callable[[int], float | None] | None" = None,
) -> FlightModel:
    """Cut the shuttle track into flight segments and read the events off the segmentation.

    ``racket_reach_at`` answers, for a frame, how far the shuttle was from the nearest player's
    reach in that player's body heights, or None when there is nobody to measure against. It is
    passed in rather than computed here so this module stays a function of the shuttle track
    alone. Without it, joins are judged on turn angle by itself, which loses every contact that
    barely deflects the shuttle.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    scale = width / cfg.reference_width if cfg.reference_width > 0 else 1.0

    window = max(3, int(round(cfg.fit_window_seconds * fps)))
    if window % 2 == 0:
        window += 1
    residual, speed = measure_track(ball, fps, window_frames=window)

    fixed_points = find_fixed_points(ball, fps, cfg, scale)
    fixed_tolerance = cfg.fixed_point_tolerance_px * scale
    positions = {s.frame: (s.x, s.y) for s in ball if s.visible}
    on_fixed_point = {
        frame
        for frame, (x, y) in positions.items()
        if any(math.hypot(x - p["x"], y - p["y"]) <= fixed_tolerance for p in fixed_points)
    }

    max_residual = cfg.max_fit_residual_px * scale
    min_speed = cfg.min_flight_speed_px_s * scale
    in_flight = {
        frame
        for frame, value in residual.items()
        if value <= max_residual and speed.get(frame, 0.0) >= min_speed
        and frame not in on_fixed_point
    }
    min_length = max(2, int(round(cfg.min_flight_seconds * fps)))
    segments = [
        FlightSegment(
            start_frame=start,
            end_frame=end,
            peak_speed_px_s=_percentile([speed[f] for f in range(start, end + 1) if f in speed], 0.9),
            median_residual_px=_median([residual[f] for f in range(start, end + 1) if f in residual]),
        )
        for start, end in _runs_of(ball.frames, in_flight)
        if end - start + 1 >= min_length
    ]

    rest_runs = _rest_runs(ball, fps, cfg, scale)
    real_rest = [
        (start, end)
        for start, end, (x, y) in rest_runs
        if not any(math.hypot(x - p["x"], y - p["y"]) <= fixed_tolerance for p in fixed_points)
    ]

    def settled_between(first: int, last: int) -> bool:
        """Did the shuttle track come to a standstill anywhere in this gap?

        Every standstill counts here, including the ones on a fixed object. Between points the
        tracker has no shuttle to follow and settles on the furniture, so a stretch on a fixed
        point is as good a sign that play stopped as the shuttle lying on the floor. Measured on
        the development clip, that separates the two kinds of gap exactly: the one gap holding a
        sustained fixed-point stretch is between rallies, and all seventy-odd intra-rally gaps
        hold none, losing the shuttle to invisibility instead."""
        return any(not (end < first or start > last) for start, end, _ in rest_runs)

    def rest_after(frame: int) -> int | None:
        later = [start for start, _ in real_rest if start >= frame]
        return min(later) if later else None

    gap_limit = int(round(cfg.max_link_gap_seconds * fps))
    turn_span = max(2, window // 2)
    links: list[Link] = []
    for index in range(len(segments) - 1):
        exit_frame = segments[index].end_frame
        entry_frame = segments[index + 1].start_frame
        gap = entry_frame - exit_frame - 1
        if gap > gap_limit:
            continue
        if settled_between(exit_frame + 1, entry_frame - 1):
            # The shuttle track stood still between the two flights, so the rally the earlier
            # one belonged to had already ended.
            continue
        turn = _turn_degrees(positions, exit_frame, entry_frame, turn_span)
        if turn is None:
            continue
        hit_frame = _contact_frame(cfg.contact_frame_rule, exit_frame, entry_frame, residual)
        if turn < cfg.min_turn_degrees:
            # A small turn is what a drive, a flat push or a block looks like, and it is also
            # what a tracker blink mid-flight looks like. The angle cannot separate them; where
            # it happened can. A join at the end of somebody's racket is a contact.
            reach = None
            if cfg.promote_continuation_within_body_heights > 0 and racket_reach_at is not None:
                reach = racket_reach_at((exit_frame + entry_frame) // 2)
            if reach is not None and reach <= cfg.promote_continuation_within_body_heights:
                links.append(Link(index, index + 1, gap, turn, "contact_by_reach", hit_frame))
                continue
            links.append(Link(index, index + 1, gap, turn, "continuation", None))
            continue
        links.append(Link(index, index + 1, gap, turn, "contact", hit_frame))

    linked = {link.before: link for link in links}
    chains: list[list[int]] = []
    current: list[int] = [0] if segments else []
    for index in range(len(segments) - 1):
        if index in linked:
            current.append(index + 1)
        else:
            chains.append(current)
            current = [index + 1]
    if current:
        chains.append(current)

    quiet_frames = int(round(cfg.min_serve_quiet_seconds * fps))
    rallies: list[FlightRally] = []
    for chain in chains:
        first, last = segments[chain[0]], segments[chain[-1]]
        contacts = [
            linked[i].hit_frame for i in chain[:-1]
            if i in linked and linked[i].hit_frame is not None
        ]
        previous_end = segments[chain[0] - 1].end_frame if chain[0] > 0 else None
        track_start = ball.frames[0] if ball.frames else 0
        quiet_before = (
            first.start_frame - (previous_end if previous_end is not None else track_start)
        )
        served = quiet_before >= quiet_frames
        hits = ([first.start_frame] if served else []) + contacts
        if len(hits) < cfg.min_hits_per_rally:
            continue
        rallies.append(
            FlightRally(
                # Truncated or not, the rally was in progress from the first flight we can see.
                # What the missing serve costs is a hit, not a boundary.
                start_frame=first.start_frame,
                end_frame=last.end_frame,
                hit_frames=tuple(sorted(hits)),
                segment_indices=tuple(chain),
                start_kind="serve" if served else "truncated",
                rest_seen_at=rest_after(last.end_frame),
            )
        )

    events = _events(rallies, ball, fps)
    diagnostics = {
        "fit_window_frames": window,
        "frames_measured": len(residual),
        "frames_in_flight": len(in_flight),
        "frames_on_a_fixed_point": len(on_fixed_point),
        "segments_before_length_filter": len(_runs_of(ball.frames, in_flight)),
        "links": {
            "contact": sum(1 for l in links if l.kind == "contact"),
            "contact_by_reach": sum(1 for l in links if l.kind == "contact_by_reach"),
            "continuation": sum(1 for l in links if l.kind == "continuation"),
        },
        "chains": len(chains),
        "chains_below_min_hits": len(chains) - len(rallies),
        "rest_runs": len(rest_runs),
        "rest_runs_on_a_fixed_point": len(rest_runs) - len(real_rest),
        "median_residual_px": round(_median(residual.values()), 3) if residual else None,
        "note": (
            "Events come from the flight segmentation in this project, not from upstream's "
            "events file. Every segment, link and rejected chain is listed here so a boundary "
            "can be traced back to the frames that produced it."
        ),
    }
    return FlightModel(
        segments=tuple(segments),
        links=tuple(links),
        rallies=tuple(rallies),
        events=tuple(events),
        fixed_points=tuple(fixed_points),
        diagnostics=diagnostics,
    )


def _contact_frame(
    rule: str, exit_frame: int, entry_frame: int, residual: dict[int, float]
) -> int:
    if rule == "exit":
        return exit_frame
    if rule == "midpoint":
        return (exit_frame + entry_frame) // 2
    if rule == "residual_peak":
        band = [f for f in range(exit_frame, entry_frame + 1) if f in residual]
        return max(band, key=lambda f: residual[f]) if band else (exit_frame + entry_frame) // 2
    raise ValueError(
        f"unknown contact_frame_rule {rule!r}; expected 'midpoint', 'exit' or 'residual_peak'"
    )


def _events(rallies: list[FlightRally], ball: BallTrack, fps: float) -> list[ShuttleEvent]:
    """Hits and landings as ``ShuttleEvent``, so downstream code sees the shape it already knows.

    ``confidence`` is deliberately absent. Upstream's number was a fixed constant per rule
    dressed up as certainty, and this detector has nothing better to offer: a hit is either a
    join between two flights or it is not. The evidence that matters -- the turn angle, the gap,
    the segments either side -- is carried on the rally and segment records instead.
    """
    events: list[ShuttleEvent] = []
    for rally in rallies:
        for frame in rally.hit_frames:
            sample = ball.nearest_visible(frame, radius=3)
            events.append(
                ShuttleEvent(
                    frame=frame,
                    event_type="hit",
                    x=sample.x if sample else None,
                    y=sample.y if sample else None,
                    confidence=None,
                    rule="flight_break",
                    all_rules=("flight_break",),
                    source=SOURCE,
                )
            )
        sample = ball.nearest_visible(rally.end_frame, radius=3)
        events.append(
            ShuttleEvent(
                frame=rally.end_frame,
                event_type="landing",
                x=sample.x if sample else None,
                y=sample.y if sample else None,
                confidence=None,
                rule="flight_end",
                all_rules=("flight_end",),
                source=SOURCE,
            )
        )
    return sorted(events, key=lambda e: (e.frame, 0 if e.event_type == "hit" else 1))
