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
    result = evaluate(ANALYSIS, ground_truth, players)

    assert result["hit_detection"]["precision"] == pytest.approx(2 / 3, abs=1e-4)
    assert result["hit_detection"]["recall"] == pytest.approx(2 / 3, abs=1e-4)
    assert result["hitter_assignment"]["accuracy"] == pytest.approx(0.5)
    assert result["rally_boundaries"]["accuracy"] == pytest.approx(0.5)
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
