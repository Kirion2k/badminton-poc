"""Tests for the linear probe on BST's frozen representation.

The probe itself is arithmetic over feature matrices, so it is tested without loading any model.
"""
from __future__ import annotations

import numpy as np
import pytest

from badminton_poc.shots.probe import (
    ProbeResult,
    _scores,
    fit_probe,
    majority_baseline,
    with_hitter,
)


def test_the_hitter_flag_is_appended_as_one_column():
    matrix = np.zeros((3, 4), dtype=np.float32)
    out = with_hitter(matrix, ["near", "far", None])
    assert out.shape == (3, 5)
    assert list(out[:, -1]) == [1.0, 0.0, 0.0]


def test_scores_are_computed_per_class_and_averaged_over_present_ones():
    accuracy, macro, f1, support = _scores(
        ["smash", "smash", "clear"], ["smash", "clear", "clear"])
    assert accuracy == pytest.approx(2 / 3)
    assert support == {"smash": 2, "clear": 1}
    assert f1["smash"] == pytest.approx(2 / 3)
    assert f1["clear"] == pytest.approx(2 / 3)
    assert macro == pytest.approx(2 / 3)
    assert "net" not in support          # absent classes are not averaged in


def test_scores_of_an_empty_set_do_not_divide_by_zero():
    accuracy, macro, _, support = _scores([], [])
    assert accuracy == 0.0 and macro == 0.0 and support == {}


def test_a_probe_learns_a_separable_signal():
    rng = np.random.default_rng(0)
    train_x = np.vstack([rng.normal(0, 0.1, (20, 3)), rng.normal(5, 0.1, (20, 3))])
    train_y = ["smash"] * 20 + ["clear"] * 20
    test_x = np.vstack([rng.normal(0, 0.1, (10, 3)), rng.normal(5, 0.1, (10, 3))])
    test_y = ["smash"] * 10 + ["clear"] * 10
    result = fit_probe(train_x, train_y, test_x, test_y,
                       name="p", trained_on="a", tested_on="b")
    assert result.accuracy == 1.0
    assert result.n_train == 40 and result.n_test == 20


def test_a_probe_on_noise_does_not_learn_anything():
    rng = np.random.default_rng(1)
    train_x = rng.normal(0, 1, (60, 5))
    train_y = (["smash", "clear", "net"] * 20)
    test_x = rng.normal(0, 1, (30, 5))
    test_y = (["smash", "clear", "net"] * 10)
    result = fit_probe(train_x, train_y, test_x, test_y,
                       name="p", trained_on="a", tested_on="b")
    assert result.accuracy < 0.6          # chance on three balanced classes is 0.33


def test_the_majority_baseline_answers_the_commonest_training_class():
    result = majority_baseline(["lift"] * 8 + ["smash"] * 2, ["lift"] * 3 + ["smash"] * 1,
                               trained_on="a", tested_on="b")
    assert "lift" in result.name
    assert result.accuracy == pytest.approx(0.75)
    assert result.macro_f1 < result.accuracy      # it never gets the minority class


def test_a_result_serialises_with_its_provenance():
    result = ProbeResult("p", "clip a", "clip b", 0.5, 0.4, {"smash": 0.5}, {"smash": 4}, 40, 20)
    out = result.to_dict()
    assert out["trained_on"] == "clip a" and out["tested_on"] == "clip b"
    assert out["accuracy"] == 0.5 and out["n_train"] == 40
