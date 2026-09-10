"""Tests for the annotated-video renderer."""
from __future__ import annotations

import pytest


# ------------------------------------------------------------------------------------------
# what the shuttle marker means


def _sample(x, y, visible=True):
    from badminton_poc.parsing import BallSample
    return BallSample(frame=0, x=x, y=y, visible=visible)


def _full_court():
    from badminton_poc.config import CourtROI
    return CourtROI.full_frame()


def test_shuttle_state_is_none_without_a_visible_sample():
    from badminton_poc.render import shuttle_state
    assert shuttle_state(None, _full_court(), [], 1920, 884) == "none"
    assert shuttle_state(_sample(10, 10, visible=False), _full_court(), [], 1920, 884) == "none"


def test_a_moving_shuttle_inside_the_airspace_is_live():
    from badminton_poc.render import shuttle_state
    assert shuttle_state(_sample(900, 400), _full_court(), [], 1920, 884) == "live"


def test_a_shuttle_sitting_on_a_known_fixed_point_is_parked():
    """Between points TrackNet parks on a fixed object and holds one pixel for seconds.

    Drawn as a live shuttle it reads as a shuttle hovering mid-court, which is the single most
    misleading thing in the annotated video.
    """
    from badminton_poc.render import shuttle_state
    fixed = [{"x": 918.0, "y": 222.0}]
    assert shuttle_state(_sample(918, 222), _full_court(), fixed, 1920, 884) == "parked"
    assert shuttle_state(_sample(921, 225), _full_court(), fixed, 1920, 884) == "parked"


def test_a_shuttle_well_away_from_a_fixed_point_is_still_live():
    from badminton_poc.render import shuttle_state
    fixed = [{"x": 918.0, "y": 222.0}]
    assert shuttle_state(_sample(600, 500), _full_court(), fixed, 1920, 884) == "live"


def test_off_court_beats_parked_because_it_names_the_more_specific_failure():
    from badminton_poc.config import CourtROI
    from badminton_poc.render import shuttle_state
    court = CourtROI(
        polygon=((0.1, 0.1), (0.5, 0.1), (0.5, 0.9), (0.1, 0.9)),
        shuttle_polygon=((0.1, 0.0), (0.5, 0.0), (0.5, 1.0), (0.1, 1.0)),
    )
    fixed = [{"x": 1800.0, "y": 200.0}]
    assert shuttle_state(_sample(1800, 200), court, fixed, 1920, 884) == "off_court"
