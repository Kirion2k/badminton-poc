"""Tests for the flight-segment event detector.

Every track here is written by hand, so each test states a piece of shuttle behaviour and the
events it should produce. Nothing loads a fixture or a model.
"""
from __future__ import annotations

import math

import pytest

from badminton_poc.config import FlightConfig
from badminton_poc.flight import (
    _contact_frame,
    _quadratic_residual,
    find_fixed_points,
    measure_track,
    segment_flight,
)
from badminton_poc.parsing import BallSample, BallTrack

FPS = 60.0


def track(points: list[tuple[int, float, float] | tuple[int, float, float, bool]]) -> BallTrack:
    samples = []
    for point in points:
        frame, x, y = point[0], point[1], point[2]
        visible = point[3] if len(point) > 3 else True
        samples.append(BallSample(frame=frame, x=x, y=y, visible=visible))
    return BallTrack(samples)


def straight(start: int, count: int, x0: float, y0: float, dx: float, dy: float):
    return [(start + i, x0 + dx * i, y0 + dy * i) for i in range(count)]


# --------------------------------------------------------------------------------------------
# the fit


def test_quadratic_residual_is_zero_on_a_parabola():
    times = [float(t) for t in range(-3, 4)]
    values = [2.0 + 3.0 * t - 0.5 * t * t for t in times]
    assert _quadratic_residual(times, values) == pytest.approx(0.0, abs=1e-9)


def test_quadratic_residual_grows_with_a_kink():
    times = [float(t) for t in range(-3, 4)]
    smooth = [t * 2.0 for t in times]
    kinked = [abs(t) * 2.0 for t in times]
    assert _quadratic_residual(times, smooth) < _quadratic_residual(times, kinked)


def test_quadratic_residual_ignores_a_constant_offset():
    times = [float(t) for t in range(-3, 4)]
    values = [7.0] * 7
    assert _quadratic_residual(times, values) == pytest.approx(0.0, abs=1e-9)


def test_measure_track_skips_frames_without_a_full_window():
    ball = track(straight(0, 9, 100.0, 100.0, 5.0, 1.0))
    residual, speed = measure_track(ball, FPS, window_frames=7)
    # The first and last three frames cannot be centred in a seven-frame window.
    assert sorted(residual) == [3, 4, 5]
    assert speed[4] == pytest.approx(math.hypot(5.0, 1.0) * FPS)


def test_measure_track_rejects_an_even_window():
    ball = track(straight(0, 9, 0.0, 0.0, 1.0, 1.0))
    with pytest.raises(ValueError):
        measure_track(ball, FPS, window_frames=6)


def test_measure_track_treats_an_invisible_sample_as_missing():
    points = straight(0, 9, 100.0, 100.0, 5.0, 1.0)
    points[4] = (points[4][0], 0.0, 0.0, False)
    ball = track(points)
    residual, _ = measure_track(ball, FPS, window_frames=7)
    # Every window would have to span the invisible frame, so none is complete.
    assert residual == {}


# --------------------------------------------------------------------------------------------
# fixed points


def _cfg(**over) -> FlightConfig:
    return FlightConfig(**over)


def test_a_position_frozen_once_is_not_a_fixed_point():
    points = straight(0, 20, 10.0, 10.0, 30.0, 0.0)
    points += [(20 + i, 700.0, 700.0) for i in range(40)]
    assert find_fixed_points(track(points), FPS, _cfg()) == []


def test_a_position_returned_to_again_and_again_is_a_fixed_point():
    points: list[tuple[int, float, float]] = []
    frame = 0
    for _ in range(4):
        points += [(frame + i, 900.0, 220.0) for i in range(30)]
        frame += 30
        points += [(frame + i, 100.0 + 25.0 * i, 400.0) for i in range(20)]
        frame += 20
    found = find_fixed_points(track(points), FPS, _cfg())
    assert len(found) == 1
    assert found[0]["runs"] == 4
    assert found[0]["x"] == pytest.approx(900.0, abs=12.0)
    assert found[0]["y"] == pytest.approx(220.0, abs=12.0)


# --------------------------------------------------------------------------------------------
# segmentation and events


def test_a_single_smooth_flight_is_one_segment_and_no_rally():
    ball = track(straight(0, 60, 100.0, 400.0, 12.0, 1.0))
    model = segment_flight(ball, FPS, _cfg())
    assert len(model.segments) == 1
    # One flight has nobody hitting it, so there is no rally and no contact.
    assert model.rallies == ()
    assert [e.event_type for e in model.events] == []


def test_a_reversal_between_two_flights_is_a_contact():
    points = straight(0, 40, 100.0, 400.0, 12.0, 0.0)
    points += straight(42, 40, 570.0, 400.0, -12.0, 0.0)
    model = segment_flight(ball := track(points), FPS, _cfg(min_serve_quiet_seconds=1000.0, min_hits_per_rally=1))
    assert len(model.segments) == 2
    kinds = [link.kind for link in model.links]
    assert kinds == ["contact"]
    assert model.links[0].turn_degrees == pytest.approx(180.0, abs=1.0)
    assert ball.frames  # the track really was consumed
    assert [e.event_type for e in model.events] == ["hit", "landing"]


def test_a_dropout_without_a_turn_is_a_continuation_not_a_contact():
    points = straight(0, 40, 100.0, 400.0, 12.0, 0.0)
    # Same heading, resumed 10 frames later where the shuttle would have been.
    points += straight(50, 40, 100.0 + 12.0 * 50, 400.0, 12.0, 0.0)
    model = segment_flight(track(points), FPS, _cfg())
    assert [link.kind for link in model.links] == ["continuation"]
    assert model.links[0].hit_frame is None


def test_a_gap_longer_than_the_limit_breaks_the_chain():
    points = straight(0, 40, 100.0, 400.0, 12.0, 0.0)
    points += straight(300, 40, 570.0, 400.0, -12.0, 0.0)
    model = segment_flight(track(points), FPS, _cfg(max_link_gap_seconds=0.5))
    assert model.links == ()


def test_a_standstill_inside_a_gap_breaks_the_chain():
    """The shuttle was seen lying still, so whatever came next is a new rally."""
    points = straight(0, 30, 100.0, 400.0, 12.0, 0.0)
    points += [(30 + i, 460.0, 400.0) for i in range(30)]      # at rest for half a second
    points += straight(60, 30, 460.0, 400.0, -12.0, 0.0)
    model = segment_flight(track(points), FPS, _cfg(max_link_gap_seconds=2.0))
    assert model.links == ()


def test_a_rally_is_a_chain_of_flights_joined_by_contacts():
    points: list[tuple[int, float, float]] = []
    frame, x, direction = 0, 100.0, 1.0
    for _ in range(4):
        points += straight(frame, 30, x, 400.0, 12.0 * direction, 0.0)
        frame += 32
        x += 12.0 * 29 * direction
        direction *= -1.0
    model = segment_flight(track(points), FPS, _cfg(min_serve_quiet_seconds=1000.0))
    assert len(model.rallies) == 1
    rally = model.rallies[0]
    assert len(rally.hit_frames) == 3
    assert rally.start_kind == "truncated"
    assert rally.start_frame == model.segments[0].start_frame
    assert rally.end_frame == model.segments[-1].end_frame


def test_a_chain_below_the_hit_minimum_is_not_a_rally():
    points = straight(0, 40, 100.0, 400.0, 12.0, 0.0)
    points += straight(42, 40, 570.0, 400.0, -12.0, 0.0)
    model = segment_flight(track(points), FPS, _cfg(min_hits_per_rally=5,
                                                    min_serve_quiet_seconds=1000.0))
    assert model.rallies == ()


def test_quiet_before_the_first_flight_makes_the_start_a_serve():
    points = [(i, 500.0, 700.0) for i in range(120)]           # shuttle held, two seconds
    points += straight(120, 40, 500.0, 700.0, 12.0, -4.0)
    points += straight(162, 40, 980.0, 540.0, -12.0, 4.0)
    model = segment_flight(track(points), FPS, _cfg())
    assert len(model.rallies) == 1
    assert model.rallies[0].start_kind == "serve"
    # The serve itself is reported as a hit, so a two-flight rally carries two contacts.
    assert len(model.rallies[0].hit_frames) == 2


def test_a_track_starting_mid_flight_is_reported_as_truncated():
    points = straight(0, 40, 100.0, 400.0, 12.0, 0.0)
    points += straight(42, 40, 570.0, 400.0, -12.0, 0.0)
    model = segment_flight(track(points), FPS, _cfg(min_hits_per_rally=1))
    assert model.rallies[0].start_kind == "truncated"


def test_a_landing_is_emitted_at_the_end_of_the_last_flight():
    points = straight(0, 40, 100.0, 400.0, 12.0, 0.0)
    points += straight(42, 40, 570.0, 400.0, -12.0, 0.0)
    model = segment_flight(track(points), FPS, _cfg(min_serve_quiet_seconds=1000.0, min_hits_per_rally=1))
    landings = [e for e in model.events if e.event_type == "landing"]
    assert len(landings) == 1
    assert landings[0].frame == model.segments[-1].end_frame
    assert landings[0].source == "flight"
    # Upstream dressed a per-rule constant up as a confidence. This detector offers none.
    assert landings[0].confidence is None


def test_events_carry_this_project_as_their_source():
    points = straight(0, 40, 100.0, 400.0, 12.0, 0.0)
    points += straight(42, 40, 570.0, 400.0, -12.0, 0.0)
    model = segment_flight(track(points), FPS, _cfg(min_serve_quiet_seconds=1000.0, min_hits_per_rally=1))
    assert {e.source for e in model.events} == {"flight"}
    assert {e.rule for e in model.events} == {"flight_break", "flight_end"}


def test_frames_on_a_fixed_point_never_count_as_flight():
    """A tracker parked on a light drifts a pixel at a time and can otherwise look smooth."""
    points: list[tuple[int, float, float]] = []
    frame = 0
    for _ in range(4):
        points += [(frame + i, 900.0, 220.0) for i in range(30)]
        frame += 30
        points += straight(frame, 20, 100.0, 400.0, 25.0, 0.0)
        frame += 20
    model = segment_flight(track(points), FPS, _cfg())
    assert model.fixed_points
    for segment in model.segments:
        assert not (segment.start_frame <= 10 <= segment.end_frame)


def test_scale_follows_the_frame_width():
    """A clip normalized to half the width should keep the same segmentation."""
    wide = track(straight(0, 60, 200.0, 800.0, 24.0, 2.0))
    narrow = track([(f, x / 2.0, y / 2.0) for f, x, y in straight(0, 60, 200.0, 800.0, 24.0, 2.0)])
    a = segment_flight(wide, FPS, _cfg(), width=1920)
    b = segment_flight(narrow, FPS, _cfg(), width=960)
    assert [(s.start_frame, s.end_frame) for s in a.segments] == \
           [(s.start_frame, s.end_frame) for s in b.segments]


def test_fps_is_required_to_be_positive():
    with pytest.raises(ValueError):
        segment_flight(track(straight(0, 20, 0.0, 0.0, 5.0, 0.0)), 0.0, _cfg())


# --------------------------------------------------------------------------------------------
# where the contact is placed


def test_contact_frame_rules():
    residual = {10: 1.0, 11: 9.0, 12: 2.0, 13: 1.0}
    assert _contact_frame("exit", 10, 13, residual) == 10
    assert _contact_frame("midpoint", 10, 13, residual) == 11
    assert _contact_frame("residual_peak", 10, 13, residual) == 11


def test_an_unknown_contact_frame_rule_is_refused():
    with pytest.raises(ValueError):
        _contact_frame("whatever", 10, 13, {})
