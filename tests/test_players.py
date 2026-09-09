"""Player selection and identity tests. No model, no GPU: select_players is a pure function."""
from __future__ import annotations

from badminton_poc.config import CourtROI, PlayerConfig
from badminton_poc.players import CandidateFilter, Candidate, Slot, select_players, wrist_points

W, H = 1920, 884
COURT = CourtROI(
    polygon=((0.415, 0.450), (0.805, 0.425), (0.950, 0.880), (0.165, 0.990)),
    net_line=((0.435, 0.520), (0.790, 0.600)),
)


def box(cx_n, foot_n, height_n, conf=0.9, track_id=None, keypoints=None):
    """Build a candidate from normalized centre-x, foot-y and box height."""
    cx, foot, h = cx_n * W, foot_n * H, height_n * H
    return Candidate(bbox=(cx - h * 0.2, foot - h, cx + h * 0.2, foot),
                     conf=conf, track_id=track_id, keypoints=keypoints or [])


def fresh():
    return Slot("near"), Slot("far")


def test_selects_one_player_per_side():
    near_slot, far_slot = fresh()
    near, far = select_players(
        [box(0.575, 0.72, 0.30, track_id=1), box(0.675, 0.515, 0.12, track_id=2)],
        near_slot, far_slot, court=COURT, width=W, height=H, diagonal=2113.0, forget_after=20,
    )
    assert near["track_id"] == 1
    assert far["track_id"] == 2


def test_continuity_beats_a_more_confident_stranger():
    """The scaffold's 'two largest boxes' rule swapped identities here; this must not."""
    near_slot, far_slot = fresh()
    tracked = box(0.575, 0.72, 0.30, conf=0.62, track_id=7)
    select_players([tracked], near_slot, far_slot, court=COURT, width=W, height=H,
                   diagonal=2113.0, forget_after=20)
    assert near_slot.track_id == 7

    intruder = box(0.60, 0.75, 0.34, conf=0.95, track_id=99)
    near, _ = select_players([intruder, tracked], near_slot, far_slot, court=COURT,
                             width=W, height=H, diagonal=2113.0, forget_after=20)
    assert near["track_id"] == 7
    assert near_slot.id_switches == 0


def test_slot_forgets_after_enough_missing_frames():
    near_slot, far_slot = fresh()
    select_players([box(0.575, 0.72, 0.30, track_id=5)], near_slot, far_slot, court=COURT,
                   width=W, height=H, diagonal=2113.0, forget_after=3)
    for _ in range(3):
        select_players([], near_slot, far_slot, court=COURT, width=W, height=H,
                       diagonal=2113.0, forget_after=3)
    assert near_slot.track_id == 5
    select_players([], near_slot, far_slot, court=COURT, width=W, height=H,
                   diagonal=2113.0, forget_after=3)
    assert near_slot.track_id is None


def test_empty_side_yields_none_and_counts_as_missing():
    near_slot, far_slot = fresh()
    near, far = select_players([box(0.675, 0.515, 0.12)], near_slot, far_slot, court=COURT,
                               width=W, height=H, diagonal=2113.0, forget_after=20)
    assert near is None
    assert far is not None
    assert near_slot.frames_missing == 1


def test_selection_records_its_reasoning():
    near_slot, far_slot = fresh()
    near, _ = select_players(
        [box(0.575, 0.72, 0.30, track_id=1), box(0.55, 0.70, 0.28, track_id=2)],
        near_slot, far_slot, court=COURT, width=W, height=H, diagonal=2113.0, forget_after=20,
    )
    selection = near["selection"]
    assert selection["pool_size"] == 2
    assert selection["runner_up_score"] is not None
    assert set(selection["terms"]) == {"conf", "continuity", "proximity"}


def test_filter_rejects_foreground_and_offcourt_and_tiny():
    cfg = PlayerConfig(min_box_height_frac=0.06, max_box_height_frac=0.42)
    candidates = [
        box(0.575, 0.72, 0.30),   # near player: kept
        box(0.30, 0.90, 0.55),    # foreground spectator: too tall
        box(0.32, 0.47, 0.10),    # neighbouring court: foot outside polygon
        box(0.60, 0.60, 0.03),    # noise: too small
    ]
    kept, rejected = CandidateFilter(COURT, cfg, W, H).split(candidates)
    assert len(kept) == 1
    assert sorted(r["reason"] for r in rejected) == [
        "foot_outside_court", "too_large_foreground", "too_small",
    ]
    assert rejected[0]["foot_norm"] and rejected[0]["height_frac"]


def test_filter_rejects_a_track_that_never_moves():
    """The far slot locked onto a detection with 0.0 px positional spread over 1348 frames,
    because identity continuity rewards a detection that never breaks."""
    cfg = PlayerConfig(static_track_min_observations=5, static_track_min_travel_frac=0.02)
    flt = CandidateFilter(COURT, cfg, W, H)
    reasons = []
    for _ in range(12):
        kept, rejected = flt.split([box(0.728, 0.485, 0.16, track_id=7)])
        reasons.append(rejected[0]["reason"] if rejected else None)
    assert reasons[:4] == [None, None, None, None]      # too early to judge
    assert reasons[-1] == "static_track"
    assert 7 not in flt.mobile


def test_filter_keeps_a_track_that_has_ever_moved_even_when_it_stops():
    """A player standing still between rallies must not be discarded."""
    cfg = PlayerConfig(static_track_min_observations=5, static_track_min_travel_frac=0.02)
    flt = CandidateFilter(COURT, cfg, W, H)
    for x in (0.50, 0.55, 0.60, 0.62, 0.58, 0.52):
        flt.split([box(x, 0.72, 0.30, track_id=3)])
    assert 3 in flt.mobile
    for _ in range(30):
        kept, rejected = flt.split([box(0.52, 0.72, 0.30, track_id=3)])
    assert len(kept) == 1 and not rejected


def test_filter_never_rejects_an_untracked_detection_as_static():
    cfg = PlayerConfig(static_track_min_observations=2, static_track_min_travel_frac=0.02)
    flt = CandidateFilter(COURT, cfg, W, H)
    for _ in range(10):
        kept, rejected = flt.split([box(0.575, 0.72, 0.30, track_id=None)])
    assert len(kept) == 1 and not rejected


def test_wrist_points_respects_confidence():
    keypoints = [[0.0, 0.0, 0.0]] * 9 + [[10.0, 20.0, 0.9], [30.0, 40.0, 0.05]]
    assert wrist_points(keypoints) == [(10.0, 20.0)]
    assert wrist_points(keypoints, min_conf=0.01) == [(10.0, 20.0), (30.0, 40.0)]
    assert wrist_points([]) == []
