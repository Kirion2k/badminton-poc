"""Why a player slot is empty: the states, and that the diagnostic never fills a slot."""
from __future__ import annotations

from badminton_poc.visibility import box_cover, classify, summarise

NEAR = [800.0, 500.0, 1000.0, 1000.0]
FAR = [880.0, 520.0, 920.0, 640.0]  # inside NEAR's box once the near player stands in front


def row(frame, near=NEAR, far=FAR, not_selected=()):
    return {
        "frame": frame, "width": 1920, "height": 1080,
        "near": {"bbox": near} if near else None,
        "far": {"bbox": far} if far else None,
        "not_selected": list(not_selected),
    }


def test_box_cover_is_the_share_of_the_first_box():
    assert box_cover(FAR, NEAR) == 1.0
    assert box_cover(NEAR, FAR) < 0.1
    assert box_cover([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_a_far_player_last_seen_behind_the_near_player_is_occluded():
    states = classify([row(0), row(1, far=None)])
    assert states[0]["far"] == ("observed", None)
    assert states[1]["far"] == ("occluded", 1.0)


def test_a_far_player_lost_in_open_court_is_unknown_not_occluded():
    clear = [100.0, 500.0, 140.0, 620.0]
    assert classify([row(0, far=clear), row(1, far=None)])[1]["far"][0] == "unknown"


def test_a_detected_but_unselected_candidate_is_a_selection_miss():
    states = classify([row(0), row(1, far=None, not_selected=[{"side": "far"}])])
    assert states[1]["far"][0] == "candidate_not_selected"


def test_a_box_last_seen_on_the_border_is_at_frame_edge():
    bottom = [700.0, 600.0, 900.0, 1080.0]
    assert classify([row(0, near=bottom), row(1, near=None)])[1]["near"][0] == "at_frame_edge"


def test_a_slot_never_observed_is_not_guessed_at():
    assert classify([row(0, far=None)])[0]["far"][0] == "never_seen"


def test_both_observed_is_exactly_both_slots_filled():
    """The existing coverage definition: the diagnostic must not move it."""
    rows = [row(0), row(1, far=None), row(2), row(3, near=None)]
    report = summarise(rows, fps=30.0)
    assert report["both_observed_fraction"] == 0.5
    assert report["far"]["observed_fraction"] == 0.75


def test_gaps_are_cut_where_a_frame_restriction_jumps():
    rows = [row(0), row(1, far=None), row(2, far=None), row(3), row(4, far=None)]
    report = summarise(rows, fps=30.0, frames=[0, 1, 3, 4])
    assert report["far"]["gaps"] == 2
    longest = report["far"]["longest_gaps"][0]
    assert longest["state"] == "occluded" and longest["max_cover"] == 1.0
