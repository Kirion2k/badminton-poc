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
    cfg = PlayerConfig(static_track_min_observations=5, static_track_min_travel_heights=0.35)
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
    cfg = PlayerConfig(static_track_min_observations=5, static_track_min_travel_heights=0.35)
    flt = CandidateFilter(COURT, cfg, W, H)
    for x in (0.50, 0.55, 0.60, 0.62, 0.58, 0.52):
        flt.split([box(x, 0.72, 0.30, track_id=3)])
    assert 3 in flt.mobile
    for _ in range(30):
        kept, rejected = flt.split([box(0.52, 0.72, 0.30, track_id=3)])
    assert len(kept) == 1 and not rejected


def test_filter_never_rejects_an_untracked_detection_as_static():
    cfg = PlayerConfig(static_track_min_observations=2, static_track_min_travel_heights=0.35)
    flt = CandidateFilter(COURT, cfg, W, H)
    for _ in range(10):
        kept, rejected = flt.split([box(0.575, 0.72, 0.30, track_id=None)])
    assert len(kept) == 1 and not rejected


def test_wrist_points_respects_confidence():
    keypoints = [[0.0, 0.0, 0.0]] * 9 + [[10.0, 20.0, 0.9], [30.0, 40.0, 0.05]]
    assert wrist_points(keypoints) == [(10.0, 20.0)]
    assert wrist_points(keypoints, min_conf=0.01) == [(10.0, 20.0), (30.0, 40.0)]
    assert wrist_points([]) == []


# --- regressions from the far-player coverage investigation on the first test clip ---
# Attribution of the 390 frames where the far slot was empty: 276 occlusion (an input
# property, not fixable here), 79 static_track false positives, 32 net-line side flips.


def test_a_small_far_player_is_not_called_static_for_moving_few_pixels():
    """Track 5 moved 35.4 px and was rejected by an absolute 38.4 px threshold -- while being
    88 px tall, so it had moved 0.40 of its own body height. Perspective makes the far player
    move a third of the near player's pixels for the same court distance, so an absolute
    threshold judges the two ends of the court by different standards."""
    cfg = PlayerConfig(static_track_min_observations=5, static_track_min_travel_heights=0.35)
    flt = CandidateFilter(COURT, cfg, W, H)
    # 0.10 frame-heights tall = 88 px; moving 0.0184 normalized x = 35.4 px = 0.40 heights.
    for x in (0.600, 0.606, 0.612, 0.6184, 0.612, 0.600):
        flt.split([box(x, 0.52, 0.10, track_id=5)])
    assert 5 in flt.mobile
    for _ in range(20):
        kept, rejected = flt.split([box(0.600, 0.52, 0.10, track_id=5)])
    assert len(kept) == 1 and not rejected


def test_an_umpire_sized_wobble_is_still_static_at_the_same_scale():
    """The other half of the same threshold: the umpire moved 0.067 body-heights all clip."""
    cfg = PlayerConfig(static_track_min_observations=5, static_track_min_travel_heights=0.35)
    flt = CandidateFilter(COURT, cfg, W, H)
    for x in (0.728, 0.7283, 0.7286, 0.728):
        flt.split([box(x, 0.485, 0.10, track_id=123)])
    for _ in range(10):
        kept, rejected = flt.split([box(0.728, 0.485, 0.10, track_id=123)])
    assert 123 not in flt.mobile
    assert rejected and rejected[0]["reason"] == "static_track"


def test_mobility_survives_a_bytetrack_id_break():
    """ByteTrack gave the far player three ids. Re-judging each fresh id from scratch is what
    let track 168 -- 143 px of real movement -- be rejected as static for 78 frames in a row
    after an id break left it standing still for its first observations under the new id."""
    cfg = PlayerConfig(static_track_min_observations=5, static_track_min_travel_heights=0.35)
    flt = CandidateFilter(COURT, cfg, W, H)
    for x in (0.600, 0.620, 0.640, 0.660, 0.640, 0.620):
        flt.split([box(x, 0.52, 0.15, track_id=168)])
    assert 168 in flt.mobile

    # ByteTrack loses it and re-issues a new id in the same place; it then stands still.
    for _ in range(20):
        kept, rejected = flt.split([box(0.620, 0.52, 0.15, track_id=176)])
    assert 176 in flt.mobile
    assert len(kept) == 1 and not rejected


def test_mobility_is_not_inherited_by_a_track_somewhere_else():
    """Handoff must be spatial, or every new id in the frame inherits a player's mobility."""
    cfg = PlayerConfig(static_track_min_observations=5, static_track_min_travel_heights=0.35)
    flt = CandidateFilter(COURT, cfg, W, H)
    for x in (0.600, 0.620, 0.640, 0.660, 0.640, 0.620):
        flt.split([box(x, 0.52, 0.15, track_id=168)])
    for _ in range(10):
        kept, rejected = flt.split([box(0.728, 0.485, 0.10, track_id=200)])
    assert 200 not in flt.mobile
    assert rejected and rejected[0]["reason"] == "static_track"


def test_net_line_hysteresis_keeps_the_far_player_far_at_the_net():
    """The far player advancing to the net crossed the line by 0.0004 normalized -- under one
    pixel -- and flipped to 'near', emptying the far pool on 32 frames."""
    assert COURT.side_of(0.6412, 0.5669) == "near"          # bare geometry, unchanged
    assert COURT.side_of(0.6412, 0.5669, bias="far", hysteresis=0.01) == "far"
    assert COURT.side_of(0.6412, 0.5669, bias="near", hysteresis=0.01) == "near"


def test_net_line_hysteresis_does_not_survive_a_real_crossing():
    """A player who genuinely changes ends must change sides. The far player's typical
    distance from the line is 0.04 normalized, four times the band."""
    assert COURT.side_of(0.6412, 0.62, bias="far", hysteresis=0.01) == "near"


def test_far_slot_keeps_its_player_at_the_net_line():
    near_slot, far_slot = fresh()
    select_players([box(0.575, 0.72, 0.30, track_id=1), box(0.60, 0.50, 0.15, track_id=69)],
                   near_slot, far_slot, court=COURT, width=W, height=H, diagonal=2113.0,
                   forget_after=20, net_hysteresis=0.01)
    assert far_slot.track_id == 69

    # advances to the net: 0.0004 past the line
    near, far = select_players(
        [box(0.575, 0.72, 0.30, track_id=1), box(0.6412, 0.5669, 0.15, track_id=69)],
        near_slot, far_slot, court=COURT, width=W, height=H, diagonal=2113.0,
        forget_after=20, net_hysteresis=0.01)
    assert far is not None and far["track_id"] == 69
    assert near["track_id"] == 1


def test_a_proven_player_outranks_an_unproven_static_detection():
    """Track 250 -- 4.5 px of movement across the whole clip -- held the far slot for 44
    frames while the real far player (track 247) was an available candidate on every one of
    them, because nothing is more continuous than furniture."""
    near_slot, far_slot = fresh()
    far_slot.track_id = 250
    far_slot.last_foot = (0.7285 * W, 0.4864 * H)
    furniture = box(0.7285, 0.4864, 0.10, conf=0.85, track_id=250)
    player = box(0.62, 0.52, 0.15, conf=0.80, track_id=247)

    _, far = select_players([furniture, player], near_slot, far_slot, court=COURT, width=W,
                            height=H, diagonal=2113.0, forget_after=20, mobile={247})
    assert far["track_id"] == 247


def test_mobility_preference_does_not_override_a_proven_players_identity():
    """Both proven: continuity must still win, or the slots swap on every frame."""
    near_slot, far_slot = fresh()
    far_slot.track_id = 69
    far_slot.last_foot = (0.62 * W, 0.52 * H)
    ours = box(0.62, 0.52, 0.15, conf=0.70, track_id=69)
    other = box(0.66, 0.50, 0.15, conf=0.95, track_id=311)

    _, far = select_players([ours, other], near_slot, far_slot, court=COURT, width=W, height=H,
                            diagonal=2113.0, forget_after=20, mobile={69, 311})
    assert far["track_id"] == 69


def test_a_slow_creep_over_many_frames_is_still_static():
    """Mobility must be judged on recent motion, not on a track's whole life. The umpire's
    detection drifted 12.4 px over 1348 frames -- enough to cross any fixed threshold
    eventually, because a lifetime extent only ever grows. A player moves that far in a
    fraction of a second; furniture never does, however long you watch it."""
    cfg = PlayerConfig(static_track_min_observations=5, static_track_min_travel_heights=0.12)
    flt = CandidateFilter(COURT, cfg, W, H)
    for step in range(60):                      # 1 px per frame: 60 px of lifetime drift
        kept, rejected = flt.split([box(0.700 + step / W, 0.485, 0.10, track_id=42)])
    assert 42 not in flt.mobile
    assert rejected and rejected[0]["reason"] == "static_track"


def test_an_unproven_track_fills_a_slot_only_when_nothing_proven_is_offered():
    """Preference, not a gate. Refusing the slot outright was tried on the test clip and
    measured: it cost 101 frames of real player to save 40 frames of furniture, because a
    player waiting to serve -- or re-emerging from a 60-frame occlusion under a fresh id -- is
    genuinely motionless and indistinguishable from a bag on motion evidence alone. So an
    unproven pick is taken and flagged, not withheld."""
    near_slot, far_slot = fresh()
    unproven = box(0.7285, 0.4864, 0.10, conf=0.85, track_id=250)
    _, far = select_players([unproven], near_slot, far_slot, court=COURT, width=W, height=H,
                            diagonal=2113.0, forget_after=20, mobile=set())
    assert far["track_id"] == 250
    assert far["selection"]["proven_mobile"] is False


def test_an_untracked_detection_is_not_withheld():
    """With no track id there is no mobility evidence to be had, and the geometric filters have
    already passed. Withholding it would drop a detection for a reason it cannot answer."""
    near_slot, far_slot = fresh()
    _, far = select_players([box(0.675, 0.515, 0.12, track_id=None)], near_slot, far_slot,
                            court=COURT, width=W, height=H, diagonal=2113.0, forget_after=20,
                            mobile=set())
    assert far is not None
