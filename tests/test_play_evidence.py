"""Does the play-evidence block describe a rally without pretending to judge one?

Clip 3's detector invented a rally at frames 2017-2122 out of two people hand-throwing shuttles
during an interval, and seven candidate signals were measured to try to gate it out. Every one of
them overlapped on clip 2, the only clip with hand labels, so nothing here suppresses anything.
What it does is put the numbers next to the rally they describe, and say which of them fall
outside the range every hand-labelled rally sits in.

The test that matters most is the last one: this must not grow a verdict.
"""
from __future__ import annotations

import math

from badminton_poc.parsing import BallSample, BallTrack
from badminton_poc.rallies import LABELLED_RALLY_RANGES, play_evidence

FPS = 60.0
WIDTH = 1920
BOX_H = 200.0


def track(start, n, *, lateral_span, speed_px, park_at=None, park_frames=0, invisible=()):
    """A shuttle sweeping side to side at a chosen speed across a chosen width."""
    half = lateral_span / 2.0
    period = max(4, int(round(2 * lateral_span / max(speed_px, 0.01))))
    samples = []
    for i in range(n):
        f = start + i
        phase = (i % period) / period
        x = 960 - half + (2 * half) * (2 * phase if phase < 0.5 else 2 * (1 - phase))
        y = 400 + 6 * math.sin(i / 7.0)
        samples.append(BallSample(frame=f, x=x, y=y, visible=f not in invisible))
    for k in range(park_frames):
        f = start + n + k
        samples.append(BallSample(frame=f, x=park_at[0], y=park_at[1], visible=True))
    return BallTrack(samples)


def rows(start, n, *, near_step, far_step, park_frames=0):
    """One row per frame; each slot's box walks at a fixed number of px per frame."""
    out = []
    for i in range(n + park_frames):
        f = start + i
        nx, fx = 700 + near_step * i, 900 + far_step * i
        out.append({
            "frame": f,
            "near": {"bbox": [nx, 500.0, nx + 90, 500.0 + BOX_H], "conf": 0.9, "track_id": 1},
            "far": {"bbox": [fx, 300.0, fx + 70, 300.0 + BOX_H], "conf": 0.9, "track_id": 2},
        })
    return out


def bh(per_second):
    """px per frame that gives this many body heights per second."""
    return per_second * BOX_H / FPS


N = 192
RALLY_LATERAL = 0.35 * WIDTH


def a_rally():
    return (track(0, N, lateral_span=RALLY_LATERAL, speed_px=7.0),
            rows(0, N, near_step=bh(0.50), far_step=bh(0.45)))


def test_a_rally_shaped_span_falls_outside_nothing():
    ball, prows = a_rally()
    e = play_evidence(ball, prows, 0, N - 1, fps=FPS, width=WIDTH)
    assert e["outside_labelled_range"] == []
    assert 0.30 < e["shuttle_lateral_range_frac"] < 0.40
    assert e["usable_shuttle_fraction"] == 1.0


def test_a_narrow_track_and_a_barely_moving_slot_are_both_reported():
    """The shape of clip 3's invented rally: the shuttle stays put and nobody covers ground."""
    ball = track(0, N, lateral_span=0.06 * WIDTH, speed_px=5.5)
    prows = rows(0, N, near_step=bh(0.33), far_step=bh(0.23))
    e = play_evidence(ball, prows, 0, N - 1, fps=FPS, width=WIDTH)
    named = {o["measurement"] for o in e["outside_labelled_range"]}
    assert named == {"shuttle_lateral_range_frac", "slower_slot_movement_bh_s"}
    assert all(o["direction"] == "below" for o in e["outside_labelled_range"])
    # the two signals that did not separate must not be dragged along by the two that did
    assert "median_shuttle_speed_px" not in named
    assert "usable_shuttle_fraction" not in named


def test_each_reported_signal_carries_the_value_and_the_bound_it_missed():
    ball = track(0, N, lateral_span=0.06 * WIDTH, speed_px=5.5)
    prows = rows(0, N, near_step=bh(0.33), far_step=bh(0.23))
    e = play_evidence(ball, prows, 0, N - 1, fps=FPS, width=WIDTH)
    o = next(o for o in e["outside_labelled_range"] if o["measurement"] == "slower_slot_movement_bh_s")
    assert o["value"] == e["slower_slot_movement_bh_s"]
    assert o["labelled_range"] == list(LABELLED_RALLY_RANGES["slower_slot_movement_bh_s"])


def test_the_slower_slot_is_the_one_reported_not_the_average():
    """A player standing still is not excused by their opponent running."""
    ball = track(0, N, lateral_span=RALLY_LATERAL, speed_px=7.0)
    prows = rows(0, N, near_step=bh(1.40), far_step=bh(0.05))
    e = play_evidence(ball, prows, 0, N - 1, fps=FPS, width=WIDTH)
    assert e["slower_slot_movement_bh_s"] < 0.1
    assert any(o["measurement"] == "slower_slot_movement_bh_s" for o in e["outside_labelled_range"])


def test_a_value_exactly_on_the_bound_is_inside():
    """The bounds are the extremes of ten labelled rallies, so those rallies are inside them."""
    lo, hi = LABELLED_RALLY_RANGES["slower_slot_movement_bh_s"]
    ball = track(0, N, lateral_span=RALLY_LATERAL, speed_px=7.0)
    for value in (lo, hi):
        prows = rows(0, N, near_step=bh(value), far_step=bh(value))
        e = play_evidence(ball, prows, 0, N - 1, fps=FPS, width=WIDTH)
        assert not any(o["measurement"] == "slower_slot_movement_bh_s"
                       for o in e["outside_labelled_range"]), f"{value} sits on the bound"


def test_the_tracker_parked_on_a_fixture_is_not_counted_as_a_still_shuttle():
    """Between points TrackNet settles on a ceiling light. Counting it would report a slow,
    narrow rally as if the shuttle had really been there."""
    park = (915.0, 220.0)
    ball = track(0, N, lateral_span=RALLY_LATERAL, speed_px=7.0, park_at=park, park_frames=120)
    prows = rows(0, N, near_step=bh(0.50), far_step=bh(0.45), park_frames=120)
    fixed = [{"x": park[0], "y": park[1], "runs": 5, "frames": 400}]
    e = play_evidence(ball, prows, 0, N + 119, fps=FPS, width=WIDTH, fixed_points=fixed)
    assert e["usable_shuttle_fraction"] < 0.65, "the parked frames must not count as usable"
    assert 0.30 < e["shuttle_lateral_range_frac"] < 0.40, "and must not widen the lateral range"


def test_a_span_too_short_to_measure_says_so_instead_of_reporting_numbers():
    ball, prows = a_rally()
    e = play_evidence(ball, prows, 0, 12, fps=FPS, width=WIDTH)
    assert "note" in e and "too short" in e["note"]
    assert "shuttle_lateral_range_frac" not in e
    assert e["outside_labelled_range"] == []


def test_too_little_usable_shuttle_reports_that_rather_than_a_narrow_range():
    """A rally the tracker never saw is not a rally with a stationary shuttle."""
    ball = track(0, N, lateral_span=RALLY_LATERAL, speed_px=7.0, invisible=set(range(0, N)))
    prows = rows(0, N, near_step=bh(0.50), far_step=bh(0.45))
    e = play_evidence(ball, prows, 0, N - 1, fps=FPS, width=WIDTH)
    assert e["usable_shuttle_fraction"] == 0.0
    assert "shuttle_lateral_range_frac" not in e
    assert "too little" in e["note"]


def test_lateral_range_is_a_fraction_of_width_so_clips_of_different_size_compare():
    ball, prows = a_rally()
    wide = play_evidence(ball, prows, 0, N - 1, fps=FPS, width=WIDTH)
    narrow = play_evidence(ball, prows, 0, N - 1, fps=FPS, width=WIDTH // 2)
    assert narrow["shuttle_lateral_range_frac"] > wide["shuttle_lateral_range_frac"]


def test_it_reports_and_never_decides():
    """Seven signals were measured and every one overlapped on the clip with hand labels. A
    verdict field here would turn a ten-rally observation into a gate, which is the mistake this
    whole exercise exists to avoid."""
    ball, prows = a_rally()
    e = play_evidence(ball, prows, 0, N - 1, fps=FPS, width=WIDTH)
    forbidden = {"is_play", "play", "suppress", "drop", "verdict", "pass", "ok", "valid",
                 "suspicious", "invented", "rejected", "gate"}
    assert not (forbidden & set(e)), f"play_evidence grew a verdict: {forbidden & set(e)}"
    assert isinstance(e["outside_labelled_range"], list)
    assert "does not decide" in e["note"]
