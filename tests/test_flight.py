"""Tests for the flight-segment event detector.

Every track here is written by hand, so each test states a piece of shuttle behaviour and the
events it should produce. Nothing loads a fixture or a model.
"""
from __future__ import annotations

import math

import pytest

from badminton_poc.config import FlightConfig
from badminton_poc.flight import (
    track_quality,
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


def test_the_analyze_entry_point_honours_the_flight_config():
    """The CLI dropped this section on the floor and fell back to defaults.

    Three different `contact_frame_rule` settings produced byte-identical output, which read as
    "the rule does not matter" when it actually meant the rule was never applied.
    """
    import inspect

    from badminton_poc import cli

    source = inspect.getsource(cli.main)
    analyze_call = source[source.index('if args.command == "analyze"'):]
    analyze_call = analyze_call[:analyze_call.index("return 0")]
    assert "flight_cfg=config.flight" in analyze_call


# --- contacts that barely deflect the shuttle ------------------------------------------------
# A drive, a flat push and a block turn the shuttle very little, and by turn angle alone they
# are indistinguishable from the tracker blinking mid-flight. Across the two labelled clips 15
# of 31 joins classified as continuations sat within a racket's reach of a player, and three of
# the six contacts known to be missing were among them.


def _straight_track(frames, x0, dx, y0, dy):
    from badminton_poc.parsing import BallSample
    return [BallSample(f, x0 + dx * i, y0 + dy * i, True) for i, f in enumerate(frames)]


def _shallow_turn_track():
    """Two flights with a small direction change and a short gap between them."""
    from badminton_poc.parsing import BallTrack
    out = _straight_track(range(0, 30), 400.0, 12.0, 300.0, 2.0)
    out += _straight_track(range(34, 64), 760.0, 11.0, 360.0, -6.0)
    return BallTrack(out)


def test_a_shallow_turn_beside_a_racket_is_promoted_to_a_contact():
    from badminton_poc.config import FlightConfig
    model = segment_flight(_shallow_turn_track(), 60.0, FlightConfig(),
                           width=1920, racket_reach_at=lambda frame: 0.3)
    kinds = [l.kind for l in model.links]
    assert "contact_by_reach" in kinds
    promoted = [l for l in model.links if l.kind == "contact_by_reach"]
    assert all(l.hit_frame is not None for l in promoted)
    assert all(l.turn_degrees < FlightConfig().min_turn_degrees for l in promoted)


def test_the_same_shallow_turn_far_from_anyone_stays_a_continuation():
    from badminton_poc.config import FlightConfig
    model = segment_flight(_shallow_turn_track(), 60.0, FlightConfig(),
                           width=1920, racket_reach_at=lambda frame: 2.5)
    assert "contact_by_reach" not in [l.kind for l in model.links]


def test_an_unknown_reach_does_not_promote():
    """No players on the frame is not evidence that nobody hit it."""
    from badminton_poc.config import FlightConfig
    model = segment_flight(_shallow_turn_track(), 60.0, FlightConfig(),
                           width=1920, racket_reach_at=lambda frame: None)
    assert "contact_by_reach" not in [l.kind for l in model.links]


def test_promotion_can_be_switched_off():
    from badminton_poc.config import FlightConfig
    cfg = FlightConfig(promote_continuation_within_body_heights=0.0)
    model = segment_flight(_shallow_turn_track(), 60.0, cfg,
                           width=1920, racket_reach_at=lambda frame: 0.1)
    assert "contact_by_reach" not in [l.kind for l in model.links]


def test_segmentation_still_runs_without_being_told_about_players():
    """The module must stay a function of the shuttle track alone."""
    from badminton_poc.config import FlightConfig
    model = segment_flight(_shallow_turn_track(), 60.0, FlightConfig(), width=1920)
    assert model.links


# --- shuttle track quality -------------------------------------------------------------------
# Everything downstream is a reading of one shuttle track, so a track that jumps produces
# confident nonsense instead of an error. Nothing branches on the verdict; it exists to make a
# person look at the clip.


def _clean_track(n=200, fps=60.0):
    from badminton_poc.parsing import BallSample, BallTrack
    out = []
    for f in range(n):
        t = f / fps
        out.append(BallSample(f, 300.0 + 400.0 * t, 400.0 - 200.0 * t + 300.0 * t * t, True))
    return BallTrack(out)


def test_a_clean_track_reads_as_sound():
    q = track_quality(_clean_track(), 60.0, body_height_px=280.0)
    assert q["verdict"] == "sound"
    assert q["impossible_steps"] == 0
    assert q["coverage"] == 1.0


def test_a_track_that_teleports_is_flagged():
    from badminton_poc.parsing import BallSample, BallTrack
    samples = list(_clean_track())
    for f in range(0, 200, 7):                       # jump to the far side of the frame
        samples[f] = BallSample(f, 1800.0, 60.0, True)
    q = track_quality(BallTrack(samples), 60.0, body_height_px=280.0)
    assert q["verdict"] == "suspect"
    assert q["impossible_steps"] > 0
    assert "faster than" in q["why"]


def test_a_track_full_of_holes_is_flagged():
    q = track_quality(_clean_track(), 60.0, body_height_px=280.0,
                      frames=range(0, 400))          # only the first half has any shuttle
    assert q["verdict"] == "suspect"
    assert "missing on" in q["why"]


def test_without_a_player_to_scale_by_the_verdict_is_unknown_not_a_guess():
    q = track_quality(_clean_track(), 60.0, body_height_px=None)
    assert q["verdict"] == "unknown"
    assert "no player box" in q["why"]


def test_the_speed_limit_scales_with_the_player_and_the_frame_rate():
    """Half the frame rate is twice the distance per frame; half the player is half the pixels."""
    a = track_quality(_clean_track(), 60.0, body_height_px=280.0)
    b = track_quality(_clean_track(), 30.0, body_height_px=280.0)
    c = track_quality(_clean_track(), 60.0, body_height_px=140.0)
    assert b["speed_limit_px_per_frame"] == pytest.approx(2 * a["speed_limit_px_per_frame"])
    assert c["speed_limit_px_per_frame"] == pytest.approx(a["speed_limit_px_per_frame"] / 2)


def test_a_track_stolen_by_the_next_court_reports_where_the_jumps_land():
    """The two causes of impossible steps need opposite answers, so the numbers that separate
    them are reported even though the verdict does not try to decide between them."""
    from badminton_poc.parsing import BallSample, BallTrack
    samples = list(_clean_track())
    for f in range(0, 200, 6):                       # hop to a court far to the left
        samples[f] = BallSample(f, 40.0, 380.0, True)
    q = track_quality(BallTrack(samples), 60.0, body_height_px=280.0)
    assert q["verdict"] == "suspect"
    assert q["jump_offset_from_median_x_px"] > 0
    assert "shuttle_polygon" in q["why"]
