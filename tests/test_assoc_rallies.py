"""Hitter assignment and rally segmentation."""
from __future__ import annotations

import pytest

from badminton_poc.assoc import HitAssignment, assign_hit, enforce_alternation, player_anchors
from badminton_poc.config import AnalysisConfig, CourtROI
from badminton_poc.parsing import BallSample, BallTrack, ShuttleEvent
from badminton_poc.rallies import TimelineEvent, build_timeline, derive_absence_events, segment_rallies

W, H = 1920, 884
FPS = 28.0
COURT = CourtROI(
    polygon=((0.415, 0.450), (0.805, 0.425), (0.950, 0.880), (0.165, 0.990)),
    net_line=((0.435, 0.520), (0.790, 0.600)),
)
CFG = AnalysisConfig()


def player(x1, y1, x2, y2, wrist=None):
    keypoints = [[0.0, 0.0, 0.0] for _ in range(17)]
    if wrist:
        keypoints[9] = [wrist[0], wrist[1], 0.9]
    return {"bbox": [x1, y1, x2, y2], "keypoints": keypoints}


def hit(frame):
    return ShuttleEvent(frame=frame, event_type="hit", x=None, y=None,
                        confidence=0.8, rule="vy_reversal", all_rules=("vy_reversal",))


def test_anchor_ordering_prefers_wrist():
    p = player(1000, 500, 1100, 800, wrist=(1050, 520))
    anchors = player_anchors(p, (1055, 525))
    assert anchors[0].name == "wrist_0"
    assert anchors[0].distance == pytest.approx(7.07, abs=0.1)
    assert {a.name for a in anchors} == {"wrist_0", "box_top_centre", "box_centre"}


def test_anchor_falls_back_to_box_when_pose_missing():
    anchors = player_anchors({"bbox": [0, 0, 100, 200], "keypoints": []}, (50, 10))
    assert anchors[0].name == "box_top_centre"
    assert player_anchors(None, (0, 0)) == []


def test_assigns_hit_to_the_nearer_wrist():
    ball = BallTrack([BallSample(100, 1055.0, 525.0, True)])
    players = {100: {"near": player(1000, 500, 1100, 800, wrist=(1050, 520)),
                     "far": player(1300, 400, 1360, 500, wrist=(1330, 430))}}
    result = assign_hit(hit(100), ball, players, COURT, CFG, width=W, height=H)
    assert result.hitter == "near"
    assert result.reason == "closest_anchor"
    assert result.evidence["near"]["anchor"] == "wrist_0"
    assert result.evidence["distance_gap_px"] > CFG.assignment_margin_px


def test_refuses_to_choose_when_both_players_are_equally_close():
    ball = BallTrack([BallSample(10, 960.0, 500.0, True)])
    players = {10: {"near": player(900, 480, 1000, 700, wrist=(950, 500)),
                    "far": player(920, 460, 1010, 560, wrist=(970, 500))}}
    result = assign_hit(hit(10), ball, players, COURT, CFG, width=W, height=H)
    assert result.hitter is None
    assert result.reason == "ambiguous_both_players_close"


def test_rejects_assignment_when_shuttle_is_far_from_everyone():
    ball = BallTrack([BallSample(10, 50.0, 50.0, True)])
    players = {10: {"near": player(1500, 700, 1600, 870, wrist=(1550, 720)), "far": None}}
    result = assign_hit(hit(10), ball, players, COURT, CFG, width=W, height=H)
    assert result.hitter is None
    assert result.reason == "closest_player_too_far"


def test_uses_neighbouring_frame_when_shuttle_invisible_on_the_event_frame():
    ball = BallTrack([BallSample(9, 1055.0, 525.0, True), BallSample(10, 0.0, 0.0, False)])
    players = {10: {"near": player(1000, 500, 1100, 800, wrist=(1050, 520)), "far": None}}
    result = assign_hit(hit(10), ball, players, COURT, CFG, width=W, height=H)
    assert result.evidence["shuttle_frame_offset"] == -1
    assert result.hitter == "near"


def test_falls_back_to_the_event_coordinate():
    ball = BallTrack([BallSample(10, 0.0, 0.0, False)])
    event = ShuttleEvent(frame=10, event_type="hit", x=1055.0, y=525.0, confidence=0.8,
                         rule="r", all_rules=("r",))
    players = {10: {"near": player(1000, 500, 1100, 800, wrist=(1050, 520)), "far": None}}
    result = assign_hit(event, ball, players, COURT, CFG, width=W, height=H)
    assert result.evidence["shuttle_source"] == "event_coordinate"
    assert result.hitter == "near"


def test_reports_why_it_could_not_assign():
    empty = BallTrack([BallSample(10, 0.0, 0.0, False)])
    assert assign_hit(hit(10), empty, {}, COURT, CFG, width=W, height=H).reason == "no_shuttle_position"
    ball = BallTrack([BallSample(10, 900.0, 600.0, True)])
    assert assign_hit(hit(10), ball, {}, COURT, CFG, width=W, height=H).reason == "no_player_row_for_frame"
    assert assign_hit(hit(10), ball, {10: {"near": None, "far": None}}, COURT, CFG,
                      width=W, height=H).reason == "no_players_detected"


def test_alternation_fills_a_single_gap():
    assignments = [
        HitAssignment(10, "near", 0.5, "closest_anchor"),
        HitAssignment(30, None, None, "ambiguous_both_players_close"),
        HitAssignment(50, "near", 0.5, "closest_anchor"),
    ]
    filled = enforce_alternation(assignments, max_gap_frames=70)
    assert filled[1].hitter == "far"
    assert filled[1].reason == "inferred_by_alternation"
    assert filled[1].confidence is None
    assert filled[1].evidence["original_reason"] == "ambiguous_both_players_close"


def test_alternation_declines_when_neighbours_disagree_or_are_distant():
    disagree = [HitAssignment(10, "near", 0.5, "a"), HitAssignment(30, None, None, "b"),
                HitAssignment(50, "far", 0.5, "a")]
    assert enforce_alternation(disagree, 70)[1].hitter is None

    distant = [HitAssignment(10, "near", 0.5, "a"), HitAssignment(300, None, None, "b"),
               HitAssignment(600, "near", 0.5, "a")]
    assert enforce_alternation(distant, 70)[1].hitter is None


def test_absence_events_are_labelled_as_ours_not_upstream():
    ball = BallTrack([BallSample(i, 0, 0, i < 10 or i > 50) for i in range(60)])
    events = derive_absence_events(ball, FPS, 1.0)
    assert len(events) == 1
    assert events[0].frame == 10
    assert events[0].source == "shuttle_absence"
    assert "not an upstream detection" in events[0].detail["note"]


def timeline(*specs):
    return [TimelineEvent(frame=f, event_type=t, source="tracknet", hitter=h)
            for f, t, h in specs]


def test_segment_rallies_splits_on_landing():
    events = timeline((10, "hit", "near"), (25, "hit", "far"), (40, "landing", None),
                      (200, "hit", "near"), (215, "hit", "far"), (230, "landing", None))
    rallies = segment_rallies(events, FPS, gap_seconds=2.5, min_hits=2)
    assert len(rallies) == 2
    assert rallies[0]["start_frame"] == 10 and rallies[0]["end_frame"] == 40
    assert rallies[0]["hit_count"] == 2
    assert rallies[0]["near_hits"] == 1 and rallies[0]["far_hits"] == 1
    assert rallies[0]["terminating_event"] == "landing"
    assert rallies[1]["rally"] == 2


def test_segment_rallies_splits_on_time_gap_without_a_terminal_event():
    events = timeline((10, "hit", "near"), (25, "hit", "far"),
                      (500, "hit", "near"), (515, "hit", "far"))
    rallies = segment_rallies(events, FPS, gap_seconds=2.5, min_hits=2)
    assert len(rallies) == 2
    assert rallies[0]["termination_reason"] == "time_gap_or_end_of_clip"
    assert rallies[0]["terminating_event"] is None


def test_segment_rallies_drops_runs_below_min_hits():
    events = timeline((10, "hit", "near"), (40, "landing", None))
    assert segment_rallies(events, FPS, gap_seconds=2.5, min_hits=2) == []


def test_segment_rallies_ignores_terminal_events_with_no_preceding_hits():
    events = timeline((5, "landing", None), (10, "hit", "near"), (25, "hit", "far"),
                      (40, "landing", None))
    rallies = segment_rallies(events, FPS, gap_seconds=2.5, min_hits=2)
    assert len(rallies) == 1
    assert rallies[0]["start_frame"] == 10


def test_segment_rallies_records_ordered_hits():
    events = timeline((10, "hit", "near"), (25, "hit", "far"), (40, "hit", "near"),
                      (55, "landing", None))
    rally = segment_rallies(events, FPS, gap_seconds=2.5, min_hits=2)[0]
    assert [h["hitter"] for h in rally["hits"]] == ["near", "far", "near"]
    assert [h["index"] for h in rally["hits"]] == [0, 1, 2]
    assert rally["duration_s"] == pytest.approx(45 / FPS, abs=1e-3)


def test_build_timeline_merges_upstream_and_derived_events():
    ball = BallTrack([BallSample(i, 5.0, 5.0, not (60 <= i <= 120)) for i in range(200)])
    events = [hit(10), ShuttleEvent(50, "landing", 1.0, 2.0, 0.9, "r", ("r",))]
    assignments = {10: HitAssignment(10, "near", 0.4, "closest_anchor")}
    merged = build_timeline(events, assignments, ball, FPS, shuttle_absent_seconds=1.0)
    kinds = [(e.frame, e.event_type, e.source) for e in merged]
    assert kinds == [(10, "hit", "tracknet"), (50, "landing", "tracknet"),
                     (60, "shuttle_absent", "shuttle_absence")]
    assert merged[0].hitter == "near"
    assert merged[0].detail["assignment_reason"] == "closest_anchor"


def test_consolidate_hits_merges_bursts_and_keeps_the_best():
    from badminton_poc.rallies import consolidate_events as consolidate_hits

    def h(frame, conf):
        return ShuttleEvent(frame, "hit", 1.0, 2.0, conf, "r", ("r",))

    events = [h(10, 0.75), h(13, 0.90), h(16, 0.85), h(40, 0.80), h(70, 0.70)]
    merged, audit = consolidate_hits(events, FPS, 0.30)  # 0.30s * 28fps -> 8 frames
    assert [e.frame for e in merged] == [13, 40, 70]
    assert audit[0]["merged_frames"] == [10, 13, 16]
    assert audit[0]["kept_frame"] == 13
    assert len(audit) == 1


def test_consolidate_hits_keeps_non_hit_events_untouched():
    from badminton_poc.rallies import consolidate_events as consolidate_hits

    events = [
        ShuttleEvent(10, "hit", 1.0, 2.0, 0.8, "r", ("r",)),
        ShuttleEvent(12, "hit", 1.0, 2.0, 0.9, "r", ("r",)),
        ShuttleEvent(30, "landing", 3.0, 4.0, 0.9, "r", ("r",)),
    ]
    merged, audit = consolidate_hits(events, FPS, 0.30)
    assert [(e.frame, e.event_type) for e in merged] == [(12, "hit"), (30, "landing")]


def test_consolidate_hits_is_a_noop_when_hits_are_far_apart():
    from badminton_poc.rallies import consolidate_events as consolidate_hits

    events = [ShuttleEvent(f, "hit", 1.0, 2.0, 0.8, "r", ("r",)) for f in (10, 40, 80)]
    merged, audit = consolidate_hits(events, FPS, 0.30)
    assert [e.frame for e in merged] == [10, 40, 80]
    assert audit == []


def test_a_spurious_landing_mid_rally_does_not_split_it():
    """The failure this restructure fixes: on our test clip the detector emitted 250 landing
    candidates for 15 rallies, and splitting on every one shredded the segmentation."""
    from badminton_poc.rallies import segment_rallies

    events = timeline(
        (10, "hit", "near"), (25, "hit", "far"),
        (30, "landing", None),          # spurious: speed dip mid-rally
        (45, "hit", "near"), (60, "hit", "far"),
        (75, "landing", None),          # the real one, after the last hit
    )
    rallies = segment_rallies(events, FPS, gap_seconds=2.5, min_hits=2)
    assert len(rallies) == 1
    assert rallies[0]["hit_count"] == 4
    assert rallies[0]["start_frame"] == 10
    assert rallies[0]["end_frame"] == 75
    assert rallies[0]["terminating_event"] == "landing"


def test_terminator_must_fall_before_the_next_rally_starts():
    from badminton_poc.rallies import segment_rallies

    events = timeline(
        (10, "hit", "near"), (25, "hit", "far"),
        (400, "hit", "near"), (415, "hit", "far"),
        (500, "landing", None),
    )
    rallies = segment_rallies(events, FPS, gap_seconds=2.5, min_hits=2)
    assert len(rallies) == 2
    assert rallies[0]["terminating_event"] is None
    assert rallies[0]["termination_reason"] == "time_gap_or_end_of_clip"
    assert rallies[1]["terminating_event"] == "landing"
    assert rallies[1]["end_frame"] == 500


def _landing(frame, conf=0.9):
    return ShuttleEvent(frame, "landing", 1.0, 2.0, conf, "speed_step", ("speed_step",))


def test_consolidate_events_handles_landings_with_their_own_window():
    from badminton_poc.rallies import consolidate_events

    # the exact pathology from the test clip: candidates spaced four frames apart. 0.75s at
    # 28 fps is a 21-frame window, and this run sits inside it.
    events = [_landing(f) for f in range(100, 120, 4)]
    merged, audit = consolidate_events(events, FPS, 0.75, "landing")
    assert len(merged) == 1
    assert audit[0]["event_type"] == "landing"
    assert len(audit[0]["merged_frames"]) == 5


def test_a_landing_run_longer_than_its_own_window_is_not_one_landing():
    """This case used to merge into a single landing, and that was the chaining bug rather
    than a feature: min_landing_interval_seconds says two real landings can be 0.75s apart, so
    a run spanning 1.0s admits two by the parameter's own definition. Merging it regardless is
    what let a hit burst swallow eight seconds of a rally elsewhere. The rule cannot be
    generous here and strict there -- it is the same rule."""
    from badminton_poc.rallies import consolidate_events

    events = [_landing(f) for f in range(100, 132, 4)]   # 28 frames = 1.0s at 28 fps
    merged, _ = consolidate_events(events, FPS, 0.75, "landing")
    assert [e.frame for e in merged] == [100, 124]


def test_consolidation_does_not_chain_an_entire_rally_into_one_hit():
    """The clip-2 defect. Merging compared each candidate to the *previous* one, so a dense
    stream chained without bound: one merge swallowed 84 candidates across 8.15 seconds and
    reported them as a single hit, and 33 of 39 merges spanned longer than a shot. A burst may
    not span longer than the minimum time between two real hits, because if it does it contains
    more than one of them.
    """
    from badminton_poc.rallies import consolidate_events

    def h(frame):
        return ShuttleEvent(frame, "hit", 1.0, 2.0, 0.8, "r", ("r",))

    # A candidate every 6 frames for 8 seconds at 60 fps. Every consecutive pair is inside the
    # 18-frame window, which is exactly what let the old rule swallow the lot.
    events = [h(f) for f in range(0, 480, 6)]
    merged, audit = consolidate_events(events, 60.0, 0.30)

    assert len(merged) == 20
    spans = [max(a["merged_frames"]) - min(a["merged_frames"]) for a in audit]
    assert max(spans) <= 18


def test_consolidated_hits_are_never_closer_than_the_minimum_interval():
    """The parameter's whole meaning: two real hits cannot be closer than this, so no two
    consolidated hits may be either.

    Splitting a candidate stream into fixed windows and keeping each window's best does not
    give this: if one window's winner sits at its end and the next window's at its start, the
    two survive a few frames apart. On clip 2 that put 5% of consolidated hits within 0.12s of
    each other, well inside the 0.3s floor. Confidences here rise across each window so the
    winners are pushed together, which equal confidences would hide.
    """
    from badminton_poc.rallies import consolidate_events

    # Windows fall at [0..18] and [24..42]; the confident candidate in the first sits at its
    # end and in the second at its start, so window-local winners land 6 frames apart.
    peaks = {18, 24}
    events = [
        ShuttleEvent(f, "hit", 1.0, 2.0, 0.95 if f in peaks else 0.5, "r", ("r",))
        for f in range(0, 480, 6)
    ]
    merged, _ = consolidate_events(events, 60.0, 0.30)
    frames = [e.frame for e in merged]
    assert frames, "consolidation dropped everything"
    closest = min((b - a for a, b in zip(frames, frames[1:])), default=999)
    assert closest > 18, f"two hits {closest} frames apart, inside the 18-frame floor"


def test_low_confidence_hit_candidates_are_dropped_before_consolidation():
    """Clip 2's residual over-firing: 613 of 777 raw candidates came from one upstream rule at
    a single confidence of 0.75. Filtering has to happen before suppression, or a weak
    candidate that survives only because nothing stronger is near it still becomes a hit."""
    from badminton_poc.rallies import drop_low_confidence

    events = [
        ShuttleEvent(10, "hit", 1.0, 2.0, 0.75, "acceleration_peak", ("acceleration_peak",)),
        ShuttleEvent(60, "hit", 1.0, 2.0, 0.90, "vy_reversal", ("vy_reversal",)),
        ShuttleEvent(90, "landing", 1.0, 2.0, 0.50, "speed_step", ("speed_step",)),
    ]
    kept, dropped = drop_low_confidence(events, 0.85, "hit")
    assert [e.frame for e in kept] == [60, 90]
    assert [d["frame"] for d in dropped] == [10]
    assert dropped[0]["confidence"] == 0.75
    assert dropped[0]["rule"] == "acceleration_peak"


def test_a_zero_threshold_drops_nothing():
    """The default. A threshold has to be chosen against hand labels, not assumed."""
    from badminton_poc.rallies import drop_low_confidence

    events = [ShuttleEvent(f, "hit", 1.0, 2.0, 0.5, "r", ("r",)) for f in (10, 20)]
    kept, dropped = drop_low_confidence(events, 0.0, "hit")
    assert len(kept) == 2 and dropped == []


def test_the_confidence_filter_leaves_other_event_types_alone():
    """Landings have their own confidence distribution; a hit threshold says nothing about
    them."""
    from badminton_poc.rallies import drop_low_confidence

    events = [ShuttleEvent(10, "landing", 1.0, 2.0, 0.5, "speed_step", ("speed_step",))]
    kept, dropped = drop_low_confidence(events, 0.85, "hit")
    assert len(kept) == 1 and dropped == []


def _track(points):
    """BallTrack from {frame: (x, y) or None}."""
    from badminton_poc.parsing import BallSample, BallTrack
    return BallTrack([
        BallSample(frame=f, visible=p is not None, x=(p[0] if p else 0.0),
                   y=(p[1] if p else 0.0), inpainted=False)
        for f, p in sorted(points.items())
    ])


def test_a_frozen_false_shuttle_is_not_play():
    """Between rallies TrackNetV3 still emits a shuttle every frame, pinned at a fixed
    position -- on clip 2 it sat at exactly y=219 with a median speed of 0.00 px/frame across
    every dead period. A visible shuttle is not evidence that a rally is happening; a moving
    one is."""
    from badminton_poc.rallies import detect_play_spans

    frozen = {f: (900.0, 219.0) for f in range(0, 600)}
    spans = detect_play_spans(_track(frozen), fps=60.0, min_speed_px=4.0)
    assert spans == []


def test_a_moving_shuttle_is_play():
    from badminton_poc.rallies import detect_play_spans

    moving = {f: (100.0 + 8.0 * f, 300.0) for f in range(0, 600)}
    spans = detect_play_spans(_track(moving), fps=60.0, min_speed_px=4.0)
    assert len(spans) == 1
    assert spans[0][0] <= 30 and spans[0][1] >= 570


def test_a_shuttle_leaving_the_frame_does_not_end_a_rally():
    """A high clear goes above the top of frame and TrackNet loses it. On clip 2 the shuttle
    exited at y=6, was absent for 0.80s, and returned at y=12 -- play never stopped. A gap
    shorter than the bridge must not split the span."""
    from badminton_poc.rallies import detect_play_spans

    points = {}
    for f in range(0, 200):
        points[f] = (100.0 + 8.0 * f, 300.0)
    for f in range(200, 248):          # 0.80s out of frame
        points[f] = None
    for f in range(248, 500):
        points[f] = (100.0 + 8.0 * (f - 48), 300.0)
    spans = detect_play_spans(_track(points), fps=60.0, min_speed_px=4.0,
                              bridge_seconds=1.5)
    assert len(spans) == 1, f"the out-of-frame gap split the span: {spans}"


def test_a_gap_longer_than_the_bridge_does_split():
    from badminton_poc.rallies import detect_play_spans

    points = {f: (100.0 + 8.0 * f, 300.0) for f in range(0, 200)}
    points.update({f: (900.0, 219.0) for f in range(200, 500)})     # long dead stretch
    points.update({f: (100.0 + 8.0 * (f - 300), 300.0) for f in range(500, 700)})
    spans = detect_play_spans(_track(points), fps=60.0, min_speed_px=4.0,
                              bridge_seconds=1.5)
    assert len(spans) == 2


def test_a_flicker_of_motion_is_not_a_rally():
    """A couple of fast frames in dead time -- someone picking the shuttle up -- must not
    become a play span."""
    from badminton_poc.rallies import detect_play_spans

    points = {f: (900.0, 219.0) for f in range(0, 600)}
    for f in range(300, 306):
        points[f] = (900.0 + 20.0 * (f - 300), 219.0)
    spans = detect_play_spans(_track(points), fps=60.0, min_speed_px=4.0,
                              min_span_seconds=1.0)
    assert spans == []


def test_hits_outside_a_play_span_are_dropped_and_recorded():
    """Hits fired during dead time are what kept rallies alive past their end and invented one
    outright. Rejecting them needs a signal independent of the hits themselves."""
    from badminton_poc.rallies import reject_hits_outside_play

    events = [ShuttleEvent(f, "hit", 1.0, 2.0, 0.9, "r", ("r",)) for f in (50, 150, 400)]
    events.append(ShuttleEvent(160, "landing", 1.0, 2.0, 0.9, "r", ("r",)))
    kept, dropped = reject_hits_outside_play(events, [(100, 200)], fps=60.0)

    assert [e.frame for e in kept] == [150, 160]      # the landing survives regardless
    assert [d["frame"] for d in dropped] == [50, 400]
    assert dropped[0]["reason"] == "outside detected play"


def test_no_play_spans_means_no_gate():
    """If shuttle motion never says play is happening, refusing every hit would be worse than
    saying nothing. The gate is off rather than total."""
    from badminton_poc.rallies import reject_hits_outside_play

    events = [ShuttleEvent(50, "hit", 1.0, 2.0, 0.9, "r", ("r",))]
    kept, dropped = reject_hits_outside_play(events, [], fps=60.0)
    assert len(kept) == 1 and dropped == []


def test_hits_without_a_swing_behind_them_can_be_dropped():
    """The last lever on rally boundaries. Ends run long because false hits during dead time
    keep a rally alive, so the question is whether removing weakly-evidenced hits pulls the
    ends back without breaking the rallies apart."""
    from badminton_poc.rallies import drop_weak_swings

    events = [
        ShuttleEvent(10, "hit", 1.0, 2.0, 0.9, "r", ("r",)),
        ShuttleEvent(60, "hit", 1.0, 2.0, 0.9, "r", ("r",)),
        ShuttleEvent(90, "landing", 1.0, 2.0, 0.9, "r", ("r",)),
    ]
    swing = {10: {"near": {"peak_wrist_speed": 8.0}}, 60: {"near": {"peak_wrist_speed": 0.4}}}
    hitters = {10: "near", 60: "near"}
    kept, dropped = drop_weak_swings(events, swing, hitters, 3.0, fps=60.0)
    assert [e.frame for e in kept] == [10, 90]        # the landing is never gated
    assert dropped[0]["frame"] == 60
    assert dropped[0]["peak_wrist_speed"] == 0.4


def test_a_hit_with_no_measurable_swing_is_kept():
    """No pose, no evidence -- and absence of evidence is not evidence of absence. Dropping
    these would silently penalise frames where the player was occluded."""
    from badminton_poc.rallies import drop_weak_swings

    events = [ShuttleEvent(10, "hit", 1.0, 2.0, 0.9, "r", ("r",))]
    kept, dropped = drop_weak_swings(events, {10: {"near": {"peak_wrist_speed": None}}},
                                     {10: "near"}, 3.0, fps=60.0)
    assert len(kept) == 1 and dropped == []


def test_a_zero_threshold_gates_nothing():
    from badminton_poc.rallies import drop_weak_swings

    events = [ShuttleEvent(10, "hit", 1.0, 2.0, 0.9, "r", ("r",))]
    kept, dropped = drop_weak_swings(events, {10: {"near": {"peak_wrist_speed": 0.1}}},
                                     {10: "near"}, 0.0, fps=60.0)
    assert len(kept) == 1 and dropped == []


def test_a_rally_can_end_at_its_last_hit_instead_of_a_landing():
    """Landings arrive every 0.8s through dead time -- 47 of 68 gaps on the development clip
    sit exactly at the suppression floor, the signature of a detector free-running while
    suppression samples it. Ending on one puts the rally end wherever the metronome next ticks,
    which is why no amount of hit filtering moved rally 1's end off frame 493.

    Ending at the last hit trades an error we cannot reason about for one we can: the rally
    ends when the last stroke was played, which is at worst one shot early."""
    from badminton_poc.rallies import segment_rallies

    events = timeline((10, "hit", "near"), (40, "hit", "far"), (300, "landing", None))
    at_landing = segment_rallies(events, FPS, gap_seconds=2.5, min_hits=2)
    at_last_hit = segment_rallies(events, FPS, gap_seconds=2.5, min_hits=2, end_at_last_hit=True)

    assert at_landing[0]["end_frame"] == 300
    assert at_last_hit[0]["end_frame"] == 40
    assert at_last_hit[0]["termination_reason"] == "last_hit"
    # the terminal event is still recorded, it just no longer sets the boundary
    assert at_last_hit[0]["terminating_event"] == "landing"


def test_ending_at_the_last_hit_still_reports_a_duration():
    from badminton_poc.rallies import segment_rallies

    events = timeline((10, "hit", "near"), (40, "hit", "far"), (300, "landing", None))
    r = segment_rallies(events, FPS, gap_seconds=2.5, min_hits=2, end_at_last_hit=True)[0]
    assert r["duration_s"] == pytest.approx(30 / FPS, abs=1e-3)


def test_trailing_weak_hits_are_trimmed_without_splitting_the_rally():
    """A swing gate applied to every hit removes real ones mid-rally and cuts the rally in two.
    Applied only to the tail it cannot split anything: it walks back from the last hit while
    the evidence is weak and stops at the first hit worth believing."""
    from badminton_poc.rallies import trim_trailing_weak_hits

    rally = {"rally": 1, "start_frame": 10, "end_frame": 300,
             "hits": [{"frame": 10, "swing_peak_wrist_speed": 8.0},
                      {"frame": 60, "swing_peak_wrist_speed": 0.5},   # weak, but mid-rally
                      {"frame": 120, "swing_peak_wrist_speed": 9.0},
                      {"frame": 200, "swing_peak_wrist_speed": 1.0},  # weak tail
                      {"frame": 260, "swing_peak_wrist_speed": 0.8}]} # weak tail
    out, dropped = trim_trailing_weak_hits(rally, 3.0, fps=60.0)
    assert [h["frame"] for h in out["hits"]] == [10, 60, 120]
    assert out["end_frame"] == 120
    assert [d["frame"] for d in dropped] == [200, 260]
    assert out["hit_count"] == 3


def test_trimming_stops_at_the_first_strong_hit():
    from badminton_poc.rallies import trim_trailing_weak_hits

    rally = {"rally": 1, "start_frame": 10, "end_frame": 300,
             "hits": [{"frame": 10, "swing_peak_wrist_speed": 8.0},
                      {"frame": 200, "swing_peak_wrist_speed": 9.0}]}
    out, dropped = trim_trailing_weak_hits(rally, 3.0, fps=60.0)
    assert dropped == [] and out["end_frame"] == 300


def test_trimming_never_empties_a_rally():
    """If every hit looks weak the rally is still a rally; something happened there."""
    from badminton_poc.rallies import trim_trailing_weak_hits

    rally = {"rally": 1, "start_frame": 10, "end_frame": 300,
             "hits": [{"frame": 10, "swing_peak_wrist_speed": 0.2},
                      {"frame": 60, "swing_peak_wrist_speed": 0.3}]}
    out, dropped = trim_trailing_weak_hits(rally, 3.0, fps=60.0)
    assert len(out["hits"]) >= 1


# ------------------------------------------------------------------------------------------
# rallies built from flight-segment chains


class _Chain:
    def __init__(self, start, end, hits, kind="serve", rest=None, segments=(0,)):
        self.start_frame = start
        self.end_frame = end
        self.hit_frames = tuple(hits)
        self.segment_indices = tuple(segments)
        self.start_kind = kind
        self.rest_seen_at = rest


class _Model:
    def __init__(self, chains):
        self.rallies = chains


def _flight_hit(frame, hitter):
    return TimelineEvent(frame=frame, event_type="hit", source="flight", hitter=hitter)


def _flight_landing(frame):
    return TimelineEvent(frame=frame, event_type="landing", source="flight")


def test_flight_rally_span_comes_from_the_chain_not_the_hits():
    from badminton_poc.rallies import rallies_from_flight

    timeline = [_flight_hit(120, "near"), _flight_hit(180, "far"), _flight_landing(400)]
    model = _Model([_Chain(100, 400, [120, 180])])
    rallies, dropped = rallies_from_flight(model, timeline, 60.0)
    assert dropped == []
    assert (rallies[0]["start_frame"], rallies[0]["end_frame"]) == (100, 400)
    assert rallies[0]["hit_count"] == 2
    assert rallies[0]["termination_reason"] == "last_flight_ended"
    assert rallies[0]["start_kind"] == "serve"


def test_require_hitter_drops_unattributed_hits_but_keeps_the_span():
    from badminton_poc.rallies import rallies_from_flight

    timeline = [_flight_hit(120, "near"), _flight_hit(150, None), _flight_hit(180, "far"), _flight_landing(400)]
    model = _Model([_Chain(100, 400, [120, 150, 180])])
    rallies, dropped = rallies_from_flight(model, timeline, 60.0, require_hitter=True)
    assert rallies[0]["hit_count"] == 2
    assert (rallies[0]["start_frame"], rallies[0]["end_frame"]) == (100, 400)
    assert [d["frame"] for d in dropped] == [150]


def test_a_chain_only_one_player_ever_touched_is_not_a_rally():
    from badminton_poc.rallies import rallies_from_flight

    timeline = [_flight_hit(120, "near"), _flight_hit(150, "near"), _flight_hit(180, "near"), _flight_landing(400)]
    model = _Model([_Chain(100, 400, [120, 150, 180])])
    rallies, dropped = rallies_from_flight(model, timeline, 60.0, require_both_players=True)
    assert rallies == []
    assert {d["reason"] for d in dropped} == {
        "only one player made contact, so this is not an exchange"
    }


def test_both_players_gate_keeps_a_rally_with_a_run_of_one_name():
    from badminton_poc.rallies import rallies_from_flight

    timeline = [_flight_hit(120, "near"), _flight_hit(150, "near"), _flight_hit(180, "far"), _flight_landing(400)]
    model = _Model([_Chain(100, 400, [120, 150, 180])])
    rallies, _ = rallies_from_flight(model, timeline, 60.0, require_both_players=True)
    assert len(rallies) == 1


def test_a_chain_below_the_hit_minimum_is_dropped_with_a_reason():
    from badminton_poc.rallies import rallies_from_flight

    timeline = [_flight_hit(120, "near"), _flight_landing(400)]
    model = _Model([_Chain(100, 400, [120])])
    rallies, dropped = rallies_from_flight(model, timeline, 60.0, min_hits=2)
    assert rallies == []
    assert dropped and "fewer than 2" in dropped[0]["reason"]


# ------------------------------------------------------------------------------------------
# events must not outlive the rally they belonged to


def test_flight_events_for_a_rejected_chain_do_not_reach_the_timeline(tmp_path):
    """A chain the rally gates throw out must take its landing with it.

    Otherwise the annotated video flashes LANDING in the middle of dead time for a rally that
    the pipeline decided was not a rally.
    """
    from badminton_poc.analyze import prune_to_rallies
    from badminton_poc.rallies import TimelineEvent

    timeline = [
        TimelineEvent(frame=100, event_type="hit", source="flight", hitter="near"),
        TimelineEvent(frame=150, event_type="hit", source="flight", hitter="far"),
        TimelineEvent(frame=200, event_type="landing", source="flight"),
        TimelineEvent(frame=400, event_type="hit", source="flight", hitter="near"),
        TimelineEvent(frame=480, event_type="landing", source="flight"),
        TimelineEvent(frame=600, event_type="shuttle_absent", source="shuttle_absence"),
    ]
    rallies = [{"start_frame": 90, "end_frame": 200,
                "hits": [{"frame": 100}, {"frame": 150}]}]
    kept, dropped = prune_to_rallies(timeline, rallies)
    assert [(e.frame, e.event_type) for e in kept] == [
        (100, "hit"), (150, "hit"), (200, "landing"), (600, "shuttle_absent"),
    ]
    assert sorted((d["frame"], d["event_type"]) for d in dropped) == [
        (400, "hit"), (480, "landing"),
    ]


def test_pruning_keeps_shuttle_absence_which_belongs_to_no_rally():
    from badminton_poc.analyze import prune_to_rallies
    from badminton_poc.rallies import TimelineEvent

    timeline = [TimelineEvent(frame=600, event_type="shuttle_absent", source="shuttle_absence")]
    kept, dropped = prune_to_rallies(timeline, [])
    assert len(kept) == 1 and dropped == []


def test_timeline_keeps_each_event_own_source():
    """A flight event stamped "tracknet" would misreport where the timeline came from."""
    from badminton_poc.rallies import build_timeline

    events = [
        ShuttleEvent(10, "hit", 1.0, 2.0, None, "flight_break", ("flight_break",), source="flight"),
        ShuttleEvent(40, "landing", 3.0, 4.0, 0.9, "speed_step", ("speed_step",)),
    ]
    ball = BallTrack([BallSample(frame=f, x=1.0, y=2.0, visible=True) for f in range(60)])
    timeline = build_timeline(events, {}, ball, 60.0, 1.0)
    assert [(e.frame, e.source) for e in timeline] == [(10, "flight"), (40, "tracknet")]


# --- racket reach gate -------------------------------------------------------------------
# A quarter of the contacts the detector reports are not contacts. Hand labels on 57 of them
# put the shuttle a median 75 px from the nearest player at a real contact and 238 px at a
# spurious one, so reach is the signal that separates them -- measured in the player's own
# body heights, because the far player is half the near player's size in frame.


def test_a_contact_out_of_racket_reach_is_refused():
    """300 px box, nearest anchor 300 px away: a full body height, past arm plus racket."""
    ball = BallTrack([BallSample(10, 1050.0, 200.0, True)])
    players = {10: {"near": player(1000, 500, 1100, 800, wrist=(1050, 520)), "far": None}}
    result = assign_hit(hit(10), ball, players, COURT, CFG, width=W, height=H)
    assert result.hitter is None
    assert result.reason == "out_of_racket_reach"
    assert result.evidence["distance_body_heights"] == pytest.approx(1.0, abs=0.01)


def test_a_contact_within_racket_reach_is_kept():
    ball = BallTrack([BallSample(10, 1050.0, 380.0, True)])
    players = {10: {"near": player(1000, 500, 1100, 800, wrist=(1050, 520)), "far": None}}
    result = assign_hit(hit(10), ball, players, COURT, CFG, width=W, height=H)
    assert result.hitter == "near"
    assert result.evidence["distance_body_heights"] < CFG.max_contact_distance_body_heights


def test_the_far_player_is_judged_on_their_own_scale_not_the_near_player_s():
    """The same real reach. The far player is half the size, so half the pixels.

    In pixels the far contact looks twice as close and a pixel threshold would treat the two
    differently; in body heights they are the same contact and both survive.
    """
    near_ball = BallTrack([BallSample(10, 1050.0, 400.0, True)])          # 100 px off a 300 px box
    near_only = {10: {"near": player(1000, 500, 1100, 800, wrist=(1050, 520)), "far": None}}
    far_ball = BallTrack([BallSample(10, 1330.0, 350.0, True)])           # 50 px off a 150 px box
    far_only = {10: {"near": None, "far": player(1300, 400, 1360, 550, wrist=(1330, 430))}}

    a = assign_hit(hit(10), near_ball, near_only, COURT, CFG, width=W, height=H)
    b = assign_hit(hit(10), far_ball, far_only, COURT, CFG, width=W, height=H)
    assert a.hitter == "near" and b.hitter == "far"
    assert a.evidence["near"]["distance_px"] == pytest.approx(2 * b.evidence["far"]["distance_px"],
                                                              rel=0.1)
    assert a.evidence["distance_body_heights"] == pytest.approx(
        b.evidence["distance_body_heights"], rel=0.1)


def test_reach_is_not_judged_without_a_body_height_to_scale_by():
    """No box means no depth estimate. Refusing on a guess would drop real contacts."""
    ball = BallTrack([BallSample(10, 1050.0, 240.0, True)])
    keypoints = [[0.0, 0.0, 0.0] for _ in range(17)]
    keypoints[9] = [1050.0, 520.0, 0.9]
    players = {10: {"near": {"bbox": None, "keypoints": keypoints}, "far": None}}
    result = assign_hit(hit(10), ball, players, COURT, CFG, width=W, height=H)
    assert result.hitter == "near"
    assert "distance_body_heights" not in result.evidence


def test_alternation_does_not_revive_a_contact_nobody_could_reach():
    """The reach gate and the alternation pass disagree, and the gate wins.

    Filling these was undoing the gate: every spurious contact it removed came back with a
    hitter attached, because its neighbours happened to name the same player.
    """
    out = enforce_alternation([
        HitAssignment(0, "near", 0.5, "closest_anchor"),
        HitAssignment(10, None, None, "out_of_racket_reach"),
        HitAssignment(20, "near", 0.5, "closest_anchor"),
    ], max_gap_frames=100)
    assert out[1].hitter is None
    assert out[1].reason == "out_of_racket_reach"


def test_alternation_still_fills_a_contact_that_was_merely_ambiguous():
    out = enforce_alternation([
        HitAssignment(0, "near", 0.5, "closest_anchor"),
        HitAssignment(10, None, None, "ambiguous_both_players_close"),
        HitAssignment(20, "near", 0.5, "closest_anchor"),
    ], max_gap_frames=100)
    assert out[1].hitter == "far"
    assert out[1].reason == "inferred_by_alternation"
