"""Tests for shot ground-truth parsing and the evaluation metrics."""
from __future__ import annotations

import json

import pytest

from badminton_poc.shots import evaluate as E
from badminton_poc.shots import truth as T
from badminton_poc.shots.runner import categorise_failure
from badminton_poc.shots.windows import ShotSample


def _sample(shot_id, hitter="near", frame=100, first=False):
    return ShotSample(
        clip="c", shot_id=shot_id, rally=1, index_in_rally=shot_id, hit_frame=frame,
        hit_time_s=frame / 60.0, hitter=hitter, prev_hit_frame=None, next_hit_frame=None,
        start_frame=frame - 30, end_frame=frame + 30, fps=60.0, width=1920, height=884,
        is_first_in_rally=first, is_last_in_rally=False,
        rally_start_frame=0, rally_end_frame=500, rally_start_kind="serve",
    )


def _truth(labels, labelled=True, **overrides):
    doc = T.empty_truth(
        "c", [_sample(i, frame=100 + 50 * i) for i in range(len(labels))], 60.0)
    for row, label in zip(doc["shots"], labels):
        row["shot_type"] = label
        row["reviewed"] = True
    doc["labelled"] = labelled
    doc.update(overrides)
    return doc


# --------------------------------------------------------------------------------------------
# the labelling file


def test_a_seeded_file_has_no_shot_types_in_it():
    """Seeding a label from a prediction measures agreement with that prediction, not accuracy."""
    doc = T.empty_truth("c", [_sample(0), _sample(1)], 60.0)
    assert all(row["shot_type"] is None for row in doc["shots"])
    assert all(row["reviewed"] is False for row in doc["shots"])
    assert doc["labelled"] is False
    assert "classifier" in doc["_seeded_from"]


def test_the_class_guide_covers_every_class():
    doc = T.empty_truth("c", [_sample(0)], 60.0)
    assert set(doc["_class_guide"]) == set(doc["_classes"])


def test_scoring_an_unfinished_file_is_refused():
    with pytest.raises(T.NotLabelledError):
        T.labelled_rows(_truth(["smash"], labelled=False))


def test_scoring_is_refused_while_any_row_is_unreviewed():
    doc = _truth(["smash", "clear"])
    doc["shots"][1]["reviewed"] = False
    with pytest.raises(T.NotLabelledError):
        T.labelled_rows(doc)


def test_a_reviewed_row_with_no_label_is_refused_rather_than_dropped():
    doc = _truth(["smash", None])
    with pytest.raises(T.NotLabelledError):
        T.labelled_rows(doc)


def test_a_label_outside_the_taxonomy_is_refused():
    doc = _truth(["smash", "backhand-flick"])
    with pytest.raises(ValueError):
        T.labelled_rows(doc)


def test_a_contact_marked_not_real_is_excluded_from_scoring():
    doc = _truth(["smash", "clear"])
    doc["shots"][1]["not_a_real_contact"] = True
    rows = T.labelled_rows(doc)
    assert [r["shot_type"] for r in rows] == ["smash"]


def test_refresh_carries_judgements_across_by_hit_frame():
    doc = _truth(["smash", "clear"])
    doc["shots"][0]["notes"] = "clearly a smash"
    fresh = T.refresh(doc, [_sample(0, frame=100), _sample(1, frame=150)], 60.0)
    assert fresh["shots"][0]["shot_type"] == "smash"
    assert fresh["shots"][0]["notes"] == "clearly a smash"


def test_a_judgement_whose_hit_disappeared_is_parked_not_lost():
    doc = _truth(["smash"])
    fresh = T.refresh(doc, [_sample(0, frame=999)], 60.0)
    assert fresh["orphaned"] and fresh["orphaned"][0]["shot_type"] == "smash"
    assert fresh["labelled"] is False


def test_summarise_counts_the_flags():
    doc = _truth(["smash", "clear", "clear"])
    doc["shots"][0]["uncertain"] = True
    doc["shots"][1]["hitter_looks_wrong"] = True
    summary = T.summarise(doc)
    assert summary["uncertain"] == 1 and summary["hitter_looks_wrong"] == 1
    assert summary["class_counts"]["clear"] == 2


def test_the_file_round_trips_through_json(tmp_path):
    doc = _truth(["smash"])
    path = T.save(tmp_path / "gt.json", doc)
    assert T.load(path)["shots"][0]["shot_type"] == "smash"


def test_loading_a_foreign_file_is_refused(tmp_path):
    path = tmp_path / "gt.json"
    path.write_text(json.dumps({"_format": "something/else"}), encoding="utf-8")
    with pytest.raises(ValueError):
        T.load(path)


# --------------------------------------------------------------------------------------------
# metrics


def _records(pairs, hitters=None, scores=None):
    out = []
    for i, (truth, prediction) in enumerate(pairs):
        out.append({
            "truth": truth, "prediction": prediction,
            "hitter": (hitters or ["near"] * len(pairs))[i],
            "broad_scores": (scores or [None] * len(pairs))[i],
        })
    return out


def test_a_perfect_classifier_scores_one():
    result = E.evaluate(_records([("smash", "smash"), ("clear", "clear")]))
    assert result["accuracy"] == 1.0 and result["macro_f1"] == 1.0


def test_per_class_precision_and_recall_are_computed_separately():
    result = E.evaluate(_records([
        ("smash", "smash"), ("clear", "smash"), ("smash", "clear"),
    ]))
    smash = result["per_class"]["smash"]
    assert smash["support"] == 2 and smash["predicted"] == 2
    assert smash["precision"] == pytest.approx(0.5)
    assert smash["recall"] == pytest.approx(0.5)


def test_macro_f1_ignores_classes_nobody_was_asked_to_find():
    """A class with no examples has no recall, and scoring it as zero punishes nothing."""
    result = E.evaluate(_records([("smash", "smash")]))
    assert result["macro_f1"] == 1.0


def test_the_confusion_matrix_has_every_class_even_at_zero():
    result = E.evaluate(_records([("smash", "clear")]))
    assert set(result["confusion"]) == set(E.BROAD_CLASSES)
    assert result["confusion"]["smash"]["clear"] == 1


def test_class_counts_are_reported_beside_the_accuracy():
    result = E.evaluate(_records([("lift", "lift")] * 9 + [("smash", "lift")]))
    assert result["class_counts_truth"] == {"smash": 1, "lift": 9}
    assert result["largest_class_share"] == pytest.approx(0.9)


def test_results_are_split_by_near_and_far_player():
    result = E.evaluate(_records(
        [("smash", "smash"), ("smash", "clear")], hitters=["near", "far"]))
    assert result["by_hitter"]["near"]["accuracy"] == 1.0
    assert result["by_hitter"]["far"]["accuracy"] == 0.0


def test_top_two_accuracy_uses_the_score_ranking():
    scores = [{"smash": 0.4, "clear": 0.45, "net": 0.15}]
    result = E.evaluate(_records([("smash", "clear")], scores=scores))
    assert result["accuracy"] == 0.0
    assert result["top2_accuracy"] == 1.0


def test_top_k_is_none_when_no_model_offered_scores():
    assert E.evaluate(_records([("smash", "clear")]))["top2_accuracy"] is None


def test_the_abstention_rate_counts_predictions_of_other():
    result = E.evaluate(_records([("smash", "other"), ("clear", "clear")]))
    assert result["abstention_rate"] == pytest.approx(0.5)


def test_most_confused_ignores_the_diagonal():
    result = E.evaluate(_records([("lift", "clear")] * 3 + [("clear", "clear")] * 5))
    top = E.most_confused(result)
    assert top[0] == {"truth": "lift", "predicted": "clear", "count": 3}


def test_an_empty_result_says_so_rather_than_dividing_by_zero():
    assert E.evaluate([])["shots"] == 0


def test_acceptance_reports_each_target_against_its_value():
    result = E.evaluate(_records([("smash", "smash")] * 10))
    gates = E.acceptance(result)
    assert gates["overall_accuracy"]["target"] == 0.75
    assert gates["overall_accuracy"]["pass"] is True
    assert gates["smash_f1"]["value"] == 1.0


def test_acceptance_fails_rather_than_passing_on_a_missing_figure():
    result = E.evaluate(_records([("lift", "lift")]))
    gates = E.acceptance(result)
    assert gates["smash_f1"]["pass"] is False        # no smash in the labels at all
    assert gates["top2_accuracy"]["pass"] is False   # no scores offered


# --------------------------------------------------------------------------------------------
# failure attribution


def test_a_missing_pose_explains_an_error_before_the_model_does():
    cause = categorise_failure(
        {"shot_type": "clear", "uncertain": False},
        {"broad_prediction": "smash"},
        {"coverage": {"hitter_joints": 0.4, "shuttle": 1.0}},
    )
    assert cause == "pose incomplete"


def test_a_missing_shuttle_track_is_named_next():
    cause = categorise_failure(
        {"shot_type": "clear"}, {"broad_prediction": "smash"},
        {"coverage": {"hitter_joints": 1.0, "shuttle": 0.2}},
    )
    assert cause == "shuttle track incomplete"


def test_a_suspect_hitter_is_blamed_on_upstream():
    cause = categorise_failure(
        {"shot_type": "clear", "hitter_looks_wrong": True},
        {"broad_prediction": "smash"}, {"coverage": {"hitter_joints": 1.0, "shuttle": 1.0}},
    )
    assert cause == "upstream hit timing or attribution"


def test_clear_versus_lift_is_named_as_a_similar_pair():
    cause = categorise_failure(
        {"shot_type": "clear", "hitter": "near"}, {"broad_prediction": "lift"},
        {"coverage": {"hitter_joints": 1.0, "shuttle": 1.0}},
    )
    assert cause == "genuinely similar shot types"


def test_anything_touching_other_is_a_taxonomy_gap():
    cause = categorise_failure(
        {"shot_type": "other", "hitter": "near"}, {"broad_prediction": "smash"},
        {"coverage": {"hitter_joints": 1.0, "shuttle": 1.0}},
    )
    assert cause == "taxonomy gap"


def test_a_clean_near_player_error_falls_through_to_the_model():
    cause = categorise_failure(
        {"shot_type": "clear", "hitter": "near"}, {"broad_prediction": "net"},
        {"coverage": {"hitter_joints": 1.0, "shuttle": 1.0}},
    )
    assert cause == "domain shift or model failure"
