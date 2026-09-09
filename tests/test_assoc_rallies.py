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


def test_consolidate_events_handles_landings_with_their_own_window():
    from badminton_poc.rallies import consolidate_events

    def landing(frame, conf):
        return ShuttleEvent(frame, "landing", 1.0, 2.0, conf, "speed_step", ("speed_step",))

    # the exact pathology from the test clip: candidates spaced four frames apart
    events = [landing(f, 0.9) for f in range(100, 132, 4)]
    merged, audit = consolidate_events(events, FPS, 0.75, "landing")
    assert len(merged) == 1
    assert audit[0]["event_type"] == "landing"
    assert len(audit[0]["merged_frames"]) == 8
