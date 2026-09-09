from __future__ import annotations

import pytest

from badminton_poc.evaluation import NotLabelledError, build_template, evaluate

ANALYSIS = {
    "summary": {"shuttle_coverage": {"visible_fraction": 0.61}},
    "events": [
        {"frame": 10, "time_s": 0.36, "event_type": "hit", "hitter": "near"},
        {"frame": 30, "time_s": 1.07, "event_type": "hit", "hitter": "far"},
        {"frame": 45, "time_s": 1.61, "event_type": "landing"},
    ],
    "rallies": [{"rally": 1, "start_frame": 10, "end_frame": 45}],
}


def test_template_prefills_detections():
    template = build_template(ANALYSIS, 28.0)
    assert template["labelled"] is False
    assert len(template["hits"]) == 2
    assert template["hits"][0]["true_hitter"] == "near"
    assert template["hits"][0]["is_real_hit"] is True
    assert len(template["rallies"]) == 1
    assert template["rallies"][0]["boundary_ok"] is None


def test_write_template_never_clobbers_existing_labels(tmp_path):
    from badminton_poc.evaluation import write_template

    path = tmp_path / "ground_truth.json"
    path.write_text('{"labelled": true, "hits": []}', encoding="utf-8")
    write_template(path, ANALYSIS, 28.0)
    assert path.read_text(encoding="utf-8") == '{"labelled": true, "hits": []}'


def test_evaluate_refuses_unlabelled_data():
    with pytest.raises(NotLabelledError, match="labelled=false"):
        evaluate(ANALYSIS, {"labelled": False, "hits": []}, [])
    with pytest.raises(NotLabelledError, match="no hit rows"):
        evaluate(ANALYSIS, {"labelled": True, "hits": []}, [])


def test_evaluate_scores_the_six_poc_metrics():
    ground_truth = {
        "labelled": True,
        "active_play_ranges": [[0, 9]],
        "missed_hits": [{"frame": 77}],
        "hits": [
            {"frame": 10, "detected_hitter": "near", "true_hitter": "near", "is_real_hit": True},
            {"frame": 30, "detected_hitter": "far", "true_hitter": "near", "is_real_hit": True},
            {"frame": 40, "detected_hitter": "near", "true_hitter": None, "is_real_hit": False},
        ],
        "rallies": [{"boundary_ok": True}, {"boundary_ok": False}],
    }
    players = [{"frame": i, "near": {"bbox": []} if i < 9 else None, "far": {"bbox": []}}
               for i in range(10)]
    # The labels have to describe this run, or precision and recall are withheld by design.
    analysis = dict(
        ANALYSIS,
        rallies=[{"rally": 1, "start_frame": 0, "end_frame": 9}],
        events=[{"event_type": "hit", "frame": f, "time_s": f / 60.0} for f in (10, 30, 40)],
    )
    result = evaluate(analysis, ground_truth, players)

    assert result["hit_detection"]["precision"] == pytest.approx(2 / 3, abs=1e-4)
    assert result["hit_detection"]["recall"] == pytest.approx(2 / 3, abs=1e-4)
    assert result["hitter_assignment"]["accuracy"] == pytest.approx(0.5)
    # Scored against active_play_ranges, not the labeller's flags: the one real rally is
    # covered by exactly one detected rally with both edges exact.
    assert result["rally_boundaries"]["accuracy"] == pytest.approx(1.0)
    assert result["rally_boundaries"]["spurious"] == 0
    assert result["rally_boundaries"]["manual_boundary_ok"] == {
        "judged": 2, "marked_ok": 1,
        "note": "The labeller's own per-detected-rally flags, kept for cross-checking.",
    }
    assert result["two_player_tracking"]["both_players_fraction"] == pytest.approx(0.9)
    assert result["targets"]["hitter_assignment_accuracy >= 0.90"] == "FAIL"


def test_coverage_is_not_reported_without_active_play_ranges():
    ground_truth = {
        "labelled": True,
        "hits": [{"frame": 10, "detected_hitter": "near", "true_hitter": "near", "is_real_hit": True}],
        "rallies": [],
    }
    result = evaluate(ANALYSIS, ground_truth, [{"frame": 0, "near": {}, "far": {}}])
    assert result["two_player_tracking"]["both_players_fraction"] is None
    assert "cannot be computed" in result["two_player_tracking"]["note"]
    assert result["targets"]["two_player_coverage_active_play >= 0.90"].startswith("not measurable")
    assert result["rally_boundaries"]["accuracy"] is None


def test_active_play_ranges_are_validated():
    ground_truth = {"labelled": True, "active_play_ranges": [[10]],
                    "hits": [{"frame": 1, "is_real_hit": True, "true_hitter": "near",
                              "detected_hitter": "near"}], "rallies": []}
    with pytest.raises(ValueError, match=r"\[start, end\]"):
        evaluate(ANALYSIS, ground_truth, [])


def test_rally_boundaries_are_scored_against_the_labelled_play_ranges():
    """A manual boundary_ok flag cannot see the two failures that matter. A rally the pipeline
    invented has no real rally to be flagged against, and a real rally split in two gets two
    flags for one error. Scoring against active_play_ranges catches both."""
    from badminton_poc.evaluation import score_rally_boundaries

    real = [[100, 200], [500, 700], [1000, 1100]]
    detected = [
        {"rally": 1, "start_frame": 100, "end_frame": 205},   # good
        {"rally": 2, "start_frame": 500, "end_frame": 560},   # split, first half
        {"rally": 3, "start_frame": 600, "end_frame": 700},   # split, second half
        {"rally": 4, "start_frame": 800, "end_frame": 900},   # spurious
        # [1000, 1100] is missed entirely
    ]
    result = score_rally_boundaries(detected, real, fps=60.0, tolerance_seconds=1.0)

    assert result["real_rallies"] == 3
    assert result["approximately_correct"] == 1
    assert result["accuracy"] == pytest.approx(1 / 3)
    assert result["spurious"] == 1
    assert result["split"] == 1
    assert result["missed"] == 1


def test_a_boundary_inside_the_tolerance_counts_as_correct():
    from badminton_poc.evaluation import score_rally_boundaries

    result = score_rally_boundaries(
        [{"rally": 1, "start_frame": 90, "end_frame": 240}], [[100, 236]],
        fps=60.0, tolerance_seconds=1.0)
    assert result["approximately_correct"] == 1
    assert result["spurious"] == 0


def test_an_end_far_beyond_the_tolerance_is_not_correct():
    """The commonest failure on the test clip: the rally is found but runs on past the point
    play stopped, because a stray hit candidate keeps it alive."""
    from badminton_poc.evaluation import score_rally_boundaries

    result = score_rally_boundaries(
        [{"rally": 1, "start_frame": 83, "end_frame": 493}], [[83, 236]],
        fps=60.0, tolerance_seconds=1.0)
    assert result["approximately_correct"] == 0
    assert result["detail"][0]["end_error_s"] == pytest.approx(4.28, abs=0.02)


def test_scoring_needs_labelled_ranges():
    from badminton_poc.evaluation import score_rally_boundaries

    result = score_rally_boundaries([{"rally": 1, "start_frame": 1, "end_frame": 2}], [],
                                    fps=60.0, tolerance_seconds=1.0)
    assert result["accuracy"] is None
    assert result["real_rallies"] == 0


def test_evaluation_refuses_to_score_hits_the_labels_do_not_describe():
    """Labels are tied to the run that produced them. Change a threshold and the pipeline emits
    a different set of hits, but ground_truth.json still holds the old rows -- and scoring them
    yields a precision figure for a configuration nobody labelled. That is worse than no number
    at all, so it has to be visible."""
    analysis = dict(
        ANALYSIS,
        rallies=[],
        events=[{"event_type": "hit", "frame": f, "time_s": f / 60.0} for f in (10, 20, 30, 40)],
    )
    ground_truth = {
        "labelled": True, "fps": 60.0, "active_play_ranges": [[0, 50]],
        "hits": [{"frame": 10, "detected_hitter": "near", "true_hitter": "near",
                  "is_real_hit": True}],
        "rallies": [],
    }
    result = evaluate(analysis, ground_truth, [])

    assert result["hit_detection"]["scorable"] is False
    assert result["hit_detection"]["unlabelled_hits"] == 3
    assert "precision" not in result["hit_detection"]


def test_hits_are_scored_when_the_labels_match_the_run():
    analysis = dict(
        ANALYSIS, rallies=[],
        events=[{"event_type": "hit", "frame": f, "time_s": f / 60.0} for f in (10, 20)],
    )
    ground_truth = {
        "labelled": True, "fps": 60.0, "active_play_ranges": [[0, 50]],
        "hits": [
            {"frame": 10, "detected_hitter": "near", "true_hitter": "near", "is_real_hit": True},
            {"frame": 20, "detected_hitter": "far", "true_hitter": None, "is_real_hit": False},
        ],
        "rallies": [],
    }
    result = evaluate(analysis, ground_truth, [])
    assert result["hit_detection"]["scorable"] is True
    assert result["hit_detection"]["precision"] == pytest.approx(0.5)


def test_untouched_active_play_ranges_are_flagged():
    """The template seeds active_play_ranges from the detected rallies so labelling starts from
    a draft. If they come back byte-identical, they were not edited, and every number measured
    over them -- tracking coverage and every rally boundary -- is scored against the pipeline's
    own opinion. That happened once and produced a boundary score that contradicted its own
    ranges."""
    analysis = dict(
        ANALYSIS,
        rallies=[{"rally": 1, "start_frame": 100, "end_frame": 200}],
        events=[{"event_type": "hit", "frame": 150, "time_s": 2.5}],
    )
    ground_truth = {
        "labelled": True, "fps": 60.0,
        "active_play_ranges": [[100, 200]],          # exactly the detected rally
        "hits": [{"frame": 150, "detected_hitter": "near", "true_hitter": "near",
                  "is_real_hit": True}],
        "rallies": [],
    }
    result = evaluate(analysis, ground_truth, [])
    assert result["rally_boundaries"]["ranges_look_unedited"] is True
    assert "not edited" in result["rally_boundaries"]["warning"]


def test_edited_ranges_are_not_flagged():
    analysis = dict(
        ANALYSIS,
        rallies=[{"rally": 1, "start_frame": 100, "end_frame": 200}],
        events=[{"event_type": "hit", "frame": 150, "time_s": 2.5}],
    )
    ground_truth = {
        "labelled": True, "fps": 60.0,
        "active_play_ranges": [[100, 160]],          # a human moved the end
        "hits": [{"frame": 150, "detected_hitter": "near", "true_hitter": "near",
                  "is_real_hit": True}],
        "rallies": [],
    }
    result = evaluate(analysis, ground_truth, [])
    assert result["rally_boundaries"]["ranges_look_unedited"] is False
    assert "warning" not in result["rally_boundaries"]


def test_refresh_keeps_judgements_already_made():
    """Changing a threshold changes which hits are emitted. Re-seeding the template from
    scratch would throw away hours of labelling; refusing to re-seed leaves the labels
    describing a run that no longer exists. Neither is acceptable, so judgements are carried
    across by frame and only genuinely new hits come back unjudged."""
    from badminton_poc.evaluation import refresh_template

    existing = {
        "labelled": True, "fps": 60.0,
        "active_play_ranges": [[100, 500]],
        "missed_hits": [{"frame": 250}],
        "hits": [
            {"frame": 110, "detected_hitter": "near", "true_hitter": "far",
             "is_real_hit": True, "notes": "clear"},
            {"frame": 400, "detected_hitter": "far", "true_hitter": None,
             "is_real_hit": False, "notes": ""},
        ],
        "rallies": [{"rally": 1, "boundary_ok": True}],
    }
    analysis = {
        "rallies": [{"rally": 1, "start_frame": 100, "end_frame": 500}],
        "events": [
            {"event_type": "hit", "frame": 110, "time_s": 1.83, "hitter": "near"},
            {"event_type": "hit", "frame": 250, "time_s": 4.17, "hitter": "far"},
            {"event_type": "hit", "frame": 400, "time_s": 6.67, "hitter": "far"},
        ],
    }
    out = refresh_template(existing, analysis, 60.0)

    kept = {h["frame"]: h for h in out["hits"]}
    assert kept[110]["true_hitter"] == "far" and kept[110]["notes"] == "clear"
    assert kept[400]["is_real_hit"] is False
    assert kept[250]["is_real_hit"] is None                    # new, unjudged
    assert out["active_play_ranges"] == [[100, 500]]           # hand-corrected, preserved
    assert out["refresh"]["carried_over"] == 2
    assert out["refresh"]["new_to_judge"] == 1


def test_refresh_drops_labels_for_hits_no_longer_detected():
    from badminton_poc.evaluation import refresh_template

    existing = {"labelled": True, "fps": 60.0, "active_play_ranges": [], "missed_hits": [],
                "hits": [{"frame": 999, "is_real_hit": True, "true_hitter": "near"}],
                "rallies": []}
    analysis = {"rallies": [], "events": [{"event_type": "hit", "frame": 110, "time_s": 1.8}]}
    out = refresh_template(existing, analysis, 60.0)
    assert [h["frame"] for h in out["hits"]] == [110]
    assert out["refresh"]["dropped_no_longer_detected"] == 1


def test_unjudged_hits_make_the_evaluation_unscorable():
    """A row carried across is a judgement; a row added by a refresh is not. Scoring the two
    together reports a precision that half of the data never agreed to."""
    analysis = dict(
        ANALYSIS, rallies=[],
        events=[{"event_type": "hit", "frame": f, "time_s": f / 60.0} for f in (10, 20)],
    )
    ground_truth = {
        "labelled": True, "fps": 60.0, "active_play_ranges": [[0, 50]],
        "hits": [
            {"frame": 10, "detected_hitter": "near", "true_hitter": "near", "is_real_hit": True},
            {"frame": 20, "detected_hitter": "far", "true_hitter": None, "is_real_hit": None},
        ],
        "rallies": [],
    }
    result = evaluate(analysis, ground_truth, [])
    assert result["hit_detection"]["scorable"] is False
    assert result["hit_detection"]["unjudged_hits"] == 1
