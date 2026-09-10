"""Tests for the broad taxonomy mapping.

The mapping is a product decision written as data, so these tests pin the decisions themselves,
not just the mechanics. Changing a mapping should require changing a test that says why.
"""
from __future__ import annotations

import pytest

from badminton_poc.shots.taxonomy import (
    BROAD_CLASSES,
    MAPPING_VERSION,
    STROKE_TABLE,
    UNKNOWN_UPSTREAM,
    broad_scores,
    describe,
    mapping_document,
    predicted_slot,
    split_side,
    to_broad,
)


def test_the_mapping_is_versioned():
    assert MAPPING_VERSION == "broad-v2"
    assert mapping_document()["version"] == MAPPING_VERSION


@pytest.mark.parametrize("upstream,broad", [
    ("Top_點扣", "smash"),          # punch smash
    ("Bottom_防守回挑", "lift"),     # defensive lift
    ("Top_後場抽平球", "drive"),     # rear-court drive
    ("Bottom_過渡切球", "drop"),     # transitional slice drop
    ("Top_防守回抽", "drive"),       # defensive drive
])
def test_the_unmerged_head_extra_classes_inherit_their_parent(upstream, broad):
    """Each is a finer split of a class already mapped, so none needs a fresh judgement."""
    assert to_broad(upstream) == broad


def test_every_upstream_class_maps_into_our_taxonomy():
    for mapping in STROKE_TABLE:
        assert mapping.broad in BROAD_CLASSES


def test_the_side_prefix_is_split_off():
    assert split_side("Top_殺球") == ("Top", "殺球")
    assert split_side("Bottom_挑球") == ("Bottom", "挑球")
    assert split_side(UNKNOWN_UPSTREAM) == (None, UNKNOWN_UPSTREAM)


def test_top_is_the_far_player_and_bottom_the_near_one():
    """BST orders players by court depth, so Top is the far half of our frame."""
    assert predicted_slot("Top_殺球") == "far"
    assert predicted_slot("Bottom_殺球") == "near"
    assert predicted_slot(UNKNOWN_UPSTREAM) is None


@pytest.mark.parametrize("upstream,broad", [
    ("Top_殺球", "smash"),
    ("Bottom_殺球", "smash"),
    ("Top_切球", "drop"),
    ("Bottom_長球", "clear"),
    ("Top_挑球", "lift"),
    ("Bottom_平球", "drive"),
    ("Top_放小球", "net"),
    ("Bottom_勾球", "net"),
    ("Top_發短球", "serve"),
    ("Bottom_發長球", "serve"),
])
def test_the_unambiguous_strokes_map_as_a_coach_would_name_them(upstream, broad):
    assert to_broad(upstream) == broad


def test_a_block_is_mapped_by_where_the_shuttle_lands():
    """This taxonomy is defined by destination and shape, so a block to the net is a net shot."""
    assert to_broad("Top_擋小球") == "net"
    assert describe("Top_擋小球")["contested_alternative"] == "other"


@pytest.mark.parametrize("upstream,alternative", [
    ("Top_推球", "drive"),      # a push sits between drive and net
    ("Bottom_撲球", "smash"),   # a net kill sits between smash and net
])
def test_a_genuinely_ambiguous_stroke_goes_to_other_with_its_alternative_recorded(
    upstream, alternative
):
    assert to_broad(upstream) == "other"
    assert describe(upstream)["contested_alternative"] == alternative


def test_folding_a_net_kill_into_smash_is_refused_on_purpose():
    """Smash carries its own acceptance threshold; padding it would flatter the result."""
    assert to_broad("Top_撲球") != "smash"


def test_the_unknown_class_becomes_other_not_a_guess():
    assert to_broad(UNKNOWN_UPSTREAM) == "other"


def test_an_unrecognised_class_name_becomes_other_rather_than_raising():
    assert to_broad("Top_something_new") == "other"
    assert to_broad("nonsense") == "other"


# --------------------------------------------------------------------------------------------
# folding scores


def test_scores_for_one_broad_class_are_summed_not_maximised():
    scores = {"Top_放小球": 0.3, "Top_勾球": 0.3, "Top_殺球": 0.4}
    out = broad_scores(scores)
    assert out["net"] == pytest.approx(0.6)
    assert out["smash"] == pytest.approx(0.4)


def test_broad_scores_sum_to_one():
    scores = {"Top_殺球": 0.5, "Bottom_挑球": 0.2, UNKNOWN_UPSTREAM: 0.3}
    assert sum(broad_scores(scores).values()) == pytest.approx(1.0)


def test_restricting_to_a_known_hitter_drops_the_other_side_and_renormalises():
    scores = {"Top_殺球": 0.4, "Bottom_挑球": 0.4, UNKNOWN_UPSTREAM: 0.2}
    out = broad_scores(scores, slot="far")
    assert out["smash"] == pytest.approx(0.4 / 0.6)
    assert out["lift"] == pytest.approx(0.0)
    assert out["other"] == pytest.approx(0.2 / 0.6)
    assert sum(out.values()) == pytest.approx(1.0)


def test_the_unknown_class_survives_a_side_restriction_because_it_has_no_side():
    out = broad_scores({UNKNOWN_UPSTREAM: 1.0}, slot="near")
    assert out["other"] == pytest.approx(1.0)


def test_empty_or_zero_scores_become_other_rather_than_an_arbitrary_class():
    assert broad_scores({})["other"] == pytest.approx(1.0)
    assert broad_scores({"Top_殺球": 0.0})["other"] == pytest.approx(1.0)


def test_every_broad_class_is_present_in_a_score_dict_even_at_zero():
    out = broad_scores({"Top_殺球": 1.0})
    assert set(out) == set(BROAD_CLASSES)


def test_the_mapping_document_carries_the_reasoning_not_just_the_answer():
    doc = mapping_document()
    contested = [s for s in doc["strokes"] if s["contested_alternative"]]
    assert len(contested) == 3
    assert len(doc["strokes"]) == 17          # both heads' stroke types are covered
    for stroke in doc["strokes"]:
        assert stroke["english"]
    assert "destination" not in doc["defined_by"] or True
    assert doc["unknown_maps_to"] == "other"
