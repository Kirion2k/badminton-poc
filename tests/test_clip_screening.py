"""Does the screening tool tell a moving camera from a few junk frames at the end of a recording?

Both of the clips still available for labelling are phone screen recordings of YouTube, and both
end with the swipe that stopped the recording: clip 3's last 37 frames are an iOS Control Centre
and clip 5's last frame is the same thing. Taking the worst single sample as the drift, the tool
reported 14.9% and 23.0% of frame width and rejected both, while their court footage never moves
more than 0.3px. A screening tool that discards usable clips is worse than none, because the
clips it discards are the ones nobody looks at again.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from screen_clip import summarise_drift, verdict

WIDTH = 1920
FAR_OK = {"far_height_px_at_imgsz": 200, "configured_imgsz": 1920, "far_detected_fraction": 0.99}
BALL_OK = {"parked_fraction": 0.1, "moving_fraction": 0.5}


def frames(n):
    return list(range(0, n * 200, 200))


def test_a_fixed_camera_reports_no_drift():
    d = summarise_drift([0.1] * 25, frames(25), WIDTH)
    assert d["sustained_frac_of_width"] < 0.005
    assert verdict(d, FAR_OK, BALL_OK) == "USABLE"


def test_a_handheld_camera_is_still_rejected():
    """yt-armori: 55 of 60 samples over 2% of frame width. Nothing here may excuse that."""
    d = summarise_drift([130.0] * 22 + [4.0] * 3, frames(25), WIDTH)
    assert d["sustained_frac_of_width"] > 0.005
    assert verdict(d, FAR_OK, BALL_OK).startswith("REJECT")


def test_junk_frames_at_the_end_do_not_condemn_a_steady_clip():
    """Clip 3: steady to within 0.3px for 85 seconds, then the recording UI is swiped in."""
    mags = [0.2] * 24 + [286.8]
    d = summarise_drift(mags, frames(25), WIDTH)
    assert d["max_drift_px"] == 286.8, "the outlier is still reported, not silently dropped"
    assert d["sustained_frac_of_width"] < 0.005
    assert not verdict(d, FAR_OK, BALL_OK).startswith("REJECT")


def test_the_outlier_frames_are_named_so_somebody_looks_at_them():
    """A clip that is steady apart from a few frames has something wrong with those frames."""
    d = summarise_drift([0.2] * 24 + [286.8], frames(25), WIDTH)
    assert d["outlier_frames"] == [4800]
    v = verdict(d, FAR_OK, BALL_OK)
    assert "4800" in v and "1 sample" in v


def test_a_tripod_knocked_once_is_rejected_not_excused():
    """The docstring's own case: still either side of a knock, but one polygon cannot fit both."""
    d = summarise_drift([0.2] * 12 + [90.0] * 13, frames(25), WIDTH)
    assert d["sustained_frac_of_width"] > 0.005
    assert verdict(d, FAR_OK, BALL_OK).startswith("REJECT")


def test_three_junk_frames_are_tolerated_but_four_are_not():
    """The allowance is small and fixed, so a clip that drifts for a second is not waved through."""
    steady_with_3 = summarise_drift([0.2] * 22 + [250.0] * 3, frames(25), WIDTH)
    assert steady_with_3["sustained_frac_of_width"] < 0.005

    drifting_with_4 = summarise_drift([0.2] * 21 + [250.0] * 4, frames(25), WIDTH)
    assert drifting_with_4["sustained_frac_of_width"] > 0.005


def test_drift_is_reported_as_a_fraction_of_this_clip_s_width():
    """720p and 1080p clips must be comparable, so pixels alone cannot be the verdict."""
    small = summarise_drift([0.2] * 22 + [40.0] * 4, frames(25), 1280)
    large = summarise_drift([0.2] * 22 + [40.0] * 4, frames(25), 3840)
    assert small["sustained_frac_of_width"] > large["sustained_frac_of_width"]


def test_an_empty_sample_list_says_so_rather_than_claiming_steadiness():
    d = summarise_drift([], [], WIDTH)
    assert d["sustained_frac_of_width"] is None
    assert "could not" in verdict(d, FAR_OK, BALL_OK).lower()
