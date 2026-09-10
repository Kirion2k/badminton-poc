"""Tests for prior-shift correction and the source prior it uses."""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from badminton_poc.shots.calibration import (
    class_prior,
    decompose_gap,
    prior_shift,
    shuttleset_prior,
)
from badminton_poc.shots.taxonomy import BROAD_CLASSES


def test_class_prior_sums_to_one_and_covers_every_class():
    prior = class_prior(["smash", "clear", "clear"])
    assert set(prior) == set(BROAD_CLASSES)
    assert sum(prior.values()) == pytest.approx(1.0)
    assert prior["clear"] == pytest.approx(2 / 3)
    assert prior["net"] == 0.0


def test_an_empty_label_set_gives_zeros_rather_than_dividing_by_zero():
    assert sum(class_prior([]).values()) == 0.0


def test_prior_shift_moves_mass_toward_classes_the_target_favours():
    scores = {"smash": 0.5, "lift": 0.5}
    source = {"smash": 0.5, "lift": 0.5}
    target = {"smash": 0.1, "lift": 0.9}
    out = prior_shift(scores, source, target)
    assert out["lift"] > out["smash"]
    assert sum(out.values()) == pytest.approx(1.0)


def test_prior_shift_is_the_identity_when_the_distributions_agree():
    scores = {"smash": 0.3, "lift": 0.7}
    prior = {"smash": 0.4, "lift": 0.6}
    out = prior_shift(scores, prior, prior)
    assert out["smash"] == pytest.approx(0.3)
    assert out["lift"] == pytest.approx(0.7)


def test_dividing_by_the_source_alone_flattens_rather_than_retargets():
    """The deployable version. It removes the source base rates and puts nothing back."""
    scores = {"smash": 0.5, "lift": 0.5}
    source = {"smash": 0.1, "lift": 0.9}
    out = prior_shift(scores, source)
    assert out["smash"] > out["lift"]        # the rare class is boosted, not aimed anywhere


def test_a_class_absent_from_the_source_does_not_explode():
    out = prior_shift({"smash": 1.0}, {"smash": 0.0})
    assert sum(out.values()) == pytest.approx(1.0)


def test_all_zero_scores_fall_back_to_other():
    out = prior_shift({"smash": 0.0, "other": 0.0}, {"smash": 0.5, "other": 0.5})
    assert out["other"] == 1.0


def test_shuttleset_prior_counts_the_repository_annotations(tmp_path):
    root = tmp_path / "ShuttleSet" / "set" / "a_match"
    root.mkdir(parents=True)
    with (root / "set1.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rally", "type"])
        writer.writeheader()
        for stroke in ["殺球", "殺球", "挑球", "長球"]:
            writer.writerow({"rally": 1, "type": stroke})
    prior = shuttleset_prior(tmp_path)
    assert prior["smash"] == pytest.approx(0.5)
    assert prior["lift"] == pytest.approx(0.25)
    assert prior["clear"] == pytest.approx(0.25)
    assert sum(prior.values()) == pytest.approx(1.0)


def test_a_missing_annotation_tree_is_reported_not_guessed(tmp_path):
    with pytest.raises(FileNotFoundError):
        shuttleset_prior(tmp_path)


def test_an_empty_annotation_tree_is_refused(tmp_path):
    (tmp_path / "ShuttleSet" / "set").mkdir(parents=True)
    with pytest.raises(ValueError):
        shuttleset_prior(tmp_path)


def test_the_gap_decomposition_attributes_each_step_and_leaves_a_remainder():
    result = decompose_gap(
        [("as measured", 0.349), ("oracle prior", 0.460), ("confident labels", 0.548)],
        target=0.75,
    )
    assert result["steps"][0]["gain"] is None
    assert result["steps"][1]["gain"] == pytest.approx(0.111)
    assert result["unexplained"] == pytest.approx(0.202)
    assert "not achievable" in result["note"]
