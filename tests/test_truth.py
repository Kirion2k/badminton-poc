"""Detector-independent ground truth, and scoring predictions against it.

The old format stored one row per *detected candidate* with an is_real_hit flag, which makes
the labels a property of the configuration that produced them: "candidate at frame X was not a
hit" means nothing to a run that never emitted X. This format stores only what is true of the
video -- the racket contacts that happened, and when rallies ran -- so any configuration can be
scored against one labelling.
"""
from __future__ import annotations

import pytest

from badminton_poc.truth import (
    match_contacts,
    migrate_legacy_ground_truth,
    score_hit_detection,
    score_hitter_assignment,
    score_rallies,
)

FPS = 60.0


def _c(frame, hitter="near"):
    return {"frame": frame, "hitter": hitter}


# --- matching -------------------------------------------------------------------------

def test_a_prediction_close_to_a_contact_matches_it():
    result = match_contacts([100], [_c(103)], fps=FPS, tolerance_seconds=0.25)
    assert result["matched"] == [(100, 103)]
    assert result["false_positives"] == [] and result["missed"] == []


def test_a_prediction_beyond_the_tolerance_is_a_false_positive():
    """0.25s at 60fps is 15 frames. A contact 20 frames away is a different event."""
    result = match_contacts([100], [_c(120)], fps=FPS, tolerance_seconds=0.25)
    assert result["matched"] == []
    assert result["false_positives"] == [100] and result["missed"] == [120]


def test_two_predictions_near_one_contact_yield_one_hit_and_one_false_positive():
    """A burst that survives consolidation must not score twice against a single contact."""
    result = match_contacts([100, 108], [_c(103)], fps=FPS, tolerance_seconds=0.25)
    assert result["matched"] == [(100, 103)]          # 100 is nearer, wins
    assert result["false_positives"] == [108]
    assert result["missed"] == []


def test_matching_prefers_the_closest_pair_not_the_earliest():
    result = match_contacts([100, 200], [_c(198), _c(104)], fps=FPS, tolerance_seconds=0.25)
    assert sorted(result["matched"]) == [(100, 104), (200, 198)]


def test_a_contact_no_prediction_reaches_is_missed():
    result = match_contacts([100], [_c(103), _c(500)], fps=FPS, tolerance_seconds=0.25)
    assert result["missed"] == [500]


def test_matching_is_stable_when_nothing_is_predicted():
    result = match_contacts([], [_c(10), _c(20)], fps=FPS, tolerance_seconds=0.25)
    assert result["matched"] == [] and result["missed"] == [10, 20]
    assert result["false_positives"] == []


# --- hit detection scoring ------------------------------------------------------------

def test_hit_detection_reports_tp_fp_fn_and_f1():
    result = score_hit_detection([100, 300], [_c(103), _c(500)], fps=FPS, tolerance_seconds=0.25)
    assert result["true_positives"] == 1
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 1
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.5)
    assert result["f1"] == pytest.approx(0.5)


def test_f1_is_zero_rather_than_undefined_when_nothing_matches():
    result = score_hit_detection([100], [_c(500)], fps=FPS, tolerance_seconds=0.25)
    assert result["f1"] == 0.0


# --- hitter assignment ----------------------------------------------------------------

def test_hitter_accuracy_is_scored_only_over_matched_pairs():
    """A hit the pipeline never emitted cannot have its hitter judged, and a false positive has
    no true hitter to be judged against."""
    predictions = {100: "near", 300: "far"}
    truth = [_c(103, "near"), _c(500, "far")]
    result = score_hitter_assignment(predictions, truth, fps=FPS, tolerance_seconds=0.25)
    assert result["matched_hits"] == 1
    assert result["correct"] == 1 and result["incorrect"] == 0
    assert result["accuracy"] == pytest.approx(1.0)


def test_an_unassigned_prediction_is_counted_but_not_scored_as_wrong():
    predictions = {100: None, 200: "far"}
    truth = [_c(103, "near"), _c(204, "near")]
    result = score_hitter_assignment(predictions, truth, fps=FPS, tolerance_seconds=0.25)
    assert result["matched_hits"] == 2
    assert result["unassigned"] == 1
    assert result["correct"] == 0 and result["incorrect"] == 1
    assert result["accuracy"] == pytest.approx(0.0)      # 0 of 1 assigned pairs correct


def test_a_contact_with_no_labelled_hitter_is_not_scored():
    predictions = {100: "near"}
    result = score_hitter_assignment(predictions, [_c(103, None)], fps=FPS, tolerance_seconds=0.25)
    assert result["matched_hits"] == 1 and result["scored"] == 0
    assert result["accuracy"] is None


# --- rally segmentation ---------------------------------------------------------------

def _r(start, end):
    return {"start_frame": start, "end_frame": end}


def test_rally_scoring_reports_signed_boundary_errors():
    """Signed, because the failure on the development clip is one-directional: ends run long,
    never short, and a median of absolute values would hide that."""
    result = score_rallies([_r(100, 260)], [_r(100, 200)], fps=FPS, tolerance_seconds=1.0)
    row = result["detail"][0]
    assert row["start_error_s"] == pytest.approx(0.0)
    assert row["end_error_s"] == pytest.approx(1.0)
    assert result["median_end_error_s"] == pytest.approx(1.0)
    assert result["median_abs_end_error_s"] == pytest.approx(1.0)


def test_a_rally_within_tolerance_at_both_edges_is_recovered():
    result = score_rallies([_r(95, 205)], [_r(100, 200)], fps=FPS, tolerance_seconds=1.0)
    assert result["recovered"] == 1
    assert result["accuracy"] == pytest.approx(1.0)


def test_a_split_rally_is_not_recovered():
    result = score_rallies([_r(100, 140), _r(160, 200)], [_r(100, 200)],
                           fps=FPS, tolerance_seconds=1.0)
    assert result["recovered"] == 0
    assert result["split"] == 1
    assert result["detail"][0]["verdict"] == "split across several predicted rallies"


def test_a_predicted_rally_matching_nothing_is_spurious():
    result = score_rallies([_r(100, 200), _r(900, 950)], [_r(100, 200)],
                           fps=FPS, tolerance_seconds=1.0)
    assert result["spurious"] == 1 and result["recovered"] == 1


def test_a_labelled_rally_with_no_prediction_is_missed():
    result = score_rallies([], [_r(100, 200)], fps=FPS, tolerance_seconds=1.0)
    assert result["missed"] == 1 and result["accuracy"] == pytest.approx(0.0)


def test_medians_are_none_when_no_rally_matched_one_to_one():
    result = score_rallies([], [_r(100, 200)], fps=FPS, tolerance_seconds=1.0)
    assert result["median_end_error_s"] is None


# --- migration ------------------------------------------------------------------------

def test_migration_keeps_real_contacts_and_discards_candidate_verdicts():
    """The 128 'this candidate was false' rows on the development clip are the detector-specific
    half: they describe a configuration's output, not the video. Under a truth-first format they
    are re-derived for any configuration, so migration drops them and keeps only the contacts a
    human asserted were real."""
    legacy = {
        "labelled": True, "fps": 60.0,
        "active_play_ranges": [[100, 500]],
        "missed_hits": [{"frame": 250, "true_hitter": "far", "notes": "not detected"}],
        "hits": [
            {"frame": 110, "is_real_hit": True, "true_hitter": "near", "notes": "clear"},
            {"frame": 140, "is_real_hit": False, "true_hitter": None, "notes": "junk"},
        ],
        "rallies": [{"rally": 1, "boundary_ok": False}],
    }
    truth = migrate_legacy_ground_truth(legacy, clip="dev")

    assert [c["frame"] for c in truth["contacts"]] == [110, 250]
    assert truth["contacts"][0]["hitter"] == "near"
    assert truth["contacts"][1]["hitter"] == "far"
    assert truth["rallies"] == [{"rally": 1, "start_frame": 100, "end_frame": 500, "notes": ""}]
    assert truth["labelled"] is True
    assert truth["migrated_from"]["discarded_candidate_verdicts"] == 1


def test_migration_refuses_an_unlabelled_file():
    with pytest.raises(ValueError, match="not labelled"):
        migrate_legacy_ground_truth({"labelled": False, "hits": [], "fps": 60.0}, clip="dev")


# --- player tracking ------------------------------------------------------------------

def _p(frame, near=True, far=True, near_id=1, far_id=2):
    row = {"frame": frame, "near": None, "far": None}
    if near:
        row["near"] = {"bbox": [0, 0, 10, 10], "track_id": near_id}
    if far:
        row["far"] = {"bbox": [0, 0, 10, 10], "track_id": far_id}
    return row


def test_tracking_coverage_is_measured_over_labelled_rallies_only():
    """Coverage between rallies is not what the criterion is about, and including it lets a
    clip with long gaps flatter or punish the tracker for frames nobody cares about."""
    from badminton_poc.truth import score_player_tracking

    rows = [_p(f) for f in range(0, 100)] + [_p(f, near=False, far=False) for f in range(100, 200)]
    result = score_player_tracking(rows, [{"start_frame": 0, "end_frame": 99}], fps=FPS)
    assert result["active_play_frames"] == 100
    assert result["both_players_fraction"] == pytest.approx(1.0)
    assert result["near_fraction"] == pytest.approx(1.0)
    assert result["far_fraction"] == pytest.approx(1.0)


def test_near_and_far_coverage_are_reported_separately():
    """They fail for different reasons -- the far player is smaller and more often occluded --
    so a combined figure hides which one needs work."""
    from badminton_poc.truth import score_player_tracking

    rows = [_p(f, far=(f % 2 == 0)) for f in range(0, 100)]
    result = score_player_tracking(rows, [{"start_frame": 0, "end_frame": 99}], fps=FPS)
    assert result["near_fraction"] == pytest.approx(1.0)
    assert result["far_fraction"] == pytest.approx(0.5)
    assert result["both_players_fraction"] == pytest.approx(0.5)


def test_identity_breaks_count_track_id_changes_within_active_play():
    """A slot that changes track id mid-rally has lost the player and re-acquired them, which
    is invisible in a coverage figure: the slot was filled the whole time."""
    from badminton_poc.truth import score_player_tracking

    rows = [_p(f, far_id=2 if f < 50 else 7) for f in range(0, 100)]
    result = score_player_tracking(rows, [{"start_frame": 0, "end_frame": 99}], fps=FPS)
    assert result["far_identity_breaks"] == 1
    assert result["near_identity_breaks"] == 0


def test_a_slot_holding_the_other_slots_track_is_a_swap():
    """The failure that put an official in the far slot for 23 seconds was invisible to every
    metric. A near/far id exchange is at least detectable."""
    from badminton_poc.truth import score_player_tracking

    rows = [_p(f, near_id=1, far_id=2) for f in range(0, 50)]
    rows += [_p(f, near_id=2, far_id=1) for f in range(50, 100)]
    result = score_player_tracking(rows, [{"start_frame": 0, "end_frame": 99}], fps=FPS)
    assert result["identity_swaps"] == 1


def test_tracking_needs_labelled_rallies():
    from badminton_poc.truth import score_player_tracking

    result = score_player_tracking([_p(1)], [], fps=FPS)
    assert result["active_play_frames"] == 0
    assert result["both_players_fraction"] is None


# --- composite report -----------------------------------------------------------------

def test_baseline_report_covers_all_four_metric_groups():
    """One call produces the whole Step 4 shape, so a baseline is a single artefact rather
    than four numbers assembled by hand."""
    from badminton_poc.truth import evaluate_against_truth

    analysis = {
        "config": {"fps": FPS},
        "events": [
            {"event_type": "hit", "frame": 100, "time_s": 1.67, "hitter": "near"},
            {"event_type": "hit", "frame": 900, "time_s": 15.0, "hitter": "far"},
        ],
        "rallies": [{"rally": 1, "start_frame": 100, "end_frame": 260}],
    }
    truth = {
        "labelled": True, "fps": FPS, "clip": "dev",
        "contacts": [{"frame": 103, "hitter": "near"}, {"frame": 200, "hitter": "far"}],
        "rallies": [{"rally": 1, "start_frame": 100, "end_frame": 200}],
    }
    players = [_p(f) for f in range(100, 201)]
    report = evaluate_against_truth(analysis, truth, players)

    assert report["hit_detection"]["true_positives"] == 1
    assert report["hit_detection"]["false_positives"] == 1     # frame 900 matches nothing
    assert report["hit_detection"]["false_negatives"] == 1     # contact 200 unmatched
    assert report["hitter_assignment"]["correct"] == 1
    assert report["rally_segmentation"]["median_end_error_s"] == pytest.approx(1.0)
    assert report["player_tracking"]["both_players_fraction"] == pytest.approx(1.0)
    assert report["clip"] == "dev"


def test_the_report_refuses_an_unlabelled_truth_file():
    from badminton_poc.truth import evaluate_against_truth

    with pytest.raises(ValueError, match="not labelled"):
        evaluate_against_truth({"config": {"fps": FPS}, "events": [], "rallies": []},
                               {"labelled": False, "contacts": [], "rallies": []}, [])


# --- draft seeding --------------------------------------------------------------------

def test_a_seeded_draft_marks_every_contact_unreviewed():
    """Typing forty frame numbers by hand is not a workflow, so the draft is seeded from what
    the detector emitted. That is a labour-saving suggestion, not evidence: every row starts
    unreviewed and the file cannot be scored until a human has been through them."""
    from badminton_poc.truth import draft_truth_from_analysis

    analysis = {
        "config": {"fps": FPS},
        "events": [{"event_type": "hit", "frame": 100, "time_s": 1.67, "hitter": "near"},
                   {"event_type": "hit", "frame": 300, "time_s": 5.0, "hitter": "far"}],
        "rallies": [{"rally": 1, "start_frame": 90, "end_frame": 320}],
    }
    draft = draft_truth_from_analysis(analysis, clip="c3")
    assert [c["frame"] for c in draft["contacts"]] == [100, 300]
    assert all(c["reviewed"] is False for c in draft["contacts"])
    assert draft["labelled"] is False
    assert draft["rallies"][0]["reviewed"] is False


def test_an_unreviewed_contact_blocks_scoring():
    """The failure this prevents: a seeded draft returned unedited and scored as if it were
    ground truth, which makes the detector agree with itself by construction."""
    from badminton_poc.truth import evaluate_against_truth

    truth = {"labelled": True, "fps": FPS, "clip": "c3",
             "contacts": [{"frame": 100, "hitter": "near", "reviewed": False}],
             "rallies": [{"start_frame": 90, "end_frame": 200}]}
    with pytest.raises(ValueError, match="unreviewed"):
        evaluate_against_truth({"config": {"fps": FPS}, "events": [], "rallies": []}, truth, [])


def test_a_hand_added_contact_needs_no_reviewed_flag():
    """A contact a human typed in is reviewed by definition -- they saw it happen."""
    from badminton_poc.truth import evaluate_against_truth

    truth = {"labelled": True, "fps": FPS, "clip": "c3",
             "contacts": [{"frame": 100, "hitter": "near"}],
             "rallies": [{"start_frame": 90, "end_frame": 200}]}
    report = evaluate_against_truth(
        {"config": {"fps": FPS}, "events": [], "rallies": []}, truth, [])
    assert report["labelled_contacts"] == 1


# --- contacts that are not rally shots -------------------------------------------------

def test_a_contact_outside_a_rally_is_neither_a_hit_nor_a_miss():
    """Handing the shuttle back to the server is a real racket contact and not a rally shot.
    Counting it as a missed hit punishes the pipeline for not reporting something that should
    never appear in a rally timeline; counting a detection of it as a false positive punishes
    the pipeline for seeing something that genuinely happened. It belongs in neither column."""
    result = score_hit_detection(
        [100, 400], [_c(103), {"frame": 398, "hitter": "near", "in_rally": False}],
        fps=FPS, tolerance_seconds=0.25)
    assert result["true_positives"] == 1
    assert result["false_positives"] == 0          # 400 matched a real, non-rally contact
    assert result["false_negatives"] == 0
    assert result["out_of_rally_matched"] == 1
    assert result["precision"] == pytest.approx(1.0)


def test_an_undetected_out_of_rally_contact_is_not_a_miss():
    result = score_hit_detection(
        [100], [_c(103), {"frame": 900, "hitter": "near", "in_rally": False}],
        fps=FPS, tolerance_seconds=0.25)
    assert result["false_negatives"] == 0
    assert result["recall"] == pytest.approx(1.0)
    assert result["out_of_rally_missed"] == 1


def test_out_of_rally_contacts_do_not_affect_hitter_accuracy():
    result = score_hitter_assignment(
        {400: "far"}, [{"frame": 398, "hitter": "near", "in_rally": False}],
        fps=FPS, tolerance_seconds=0.25)
    assert result["scored"] == 0 and result["accuracy"] is None


def test_contacts_default_to_being_rally_shots():
    """Existing labels have no in_rally field and every one of them is a rally shot."""
    result = score_hit_detection([100], [_c(103)], fps=FPS, tolerance_seconds=0.25)
    assert result["true_positives"] == 1 and result["out_of_rally_matched"] == 0
