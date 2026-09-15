"""Milestone 1 is frozen as an upstream dependency for Milestone 2.

These tests exist so that retuning Milestone 1 fails the suite instead of passing unnoticed. They
assert two different things and it is worth keeping them apart:

* the detector's tuning surface still holds the values it was frozen at. Changing one of these is
  retuning, whatever the reason, and the test should fail.
* the accuracy figures still hold, for the clips whose labels are final. A figure can also move
  because the *labels* changed, which is not retuning -- test-video3 is mid-relabelling and is
  therefore excluded by the manifest rather than by this file.

When Milestone 1 is deliberately changed, re-run tools/freeze_milestone1.py and commit the new
manifest in the same commit as the change, so the diff shows what moved.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from badminton_poc.config import AnalysisConfig, FlightConfig, PlayerConfig

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = json.loads((ROOT / "configs" / "milestone-1-frozen.json").read_text())
CLASSES = {"PlayerConfig": PlayerConfig, "AnalysisConfig": AnalysisConfig, "FlightConfig": FlightConfig}


def _defaults(cls):
    return {f.name: f.default for f in dataclasses.fields(cls)
            if f.default is not dataclasses.MISSING}


@pytest.mark.parametrize("name", sorted(CLASSES))
def test_the_frozen_detector_settings_still_hold(name):
    frozen = MANIFEST["detector_defaults"][name]
    current = _defaults(CLASSES[name])
    assert current == frozen, (
        f"{name} no longer matches the Milestone 1 freeze. If this change is intended, re-run "
        "tools/freeze_milestone1.py and commit the manifest with it."
    )


def test_no_frozen_setting_has_been_dropped_or_added():
    for name, cls in CLASSES.items():
        assert set(_defaults(cls)) == set(MANIFEST["detector_defaults"][name]), (
            f"{name} gained or lost a setting since the freeze."
        )


@pytest.mark.parametrize("clip", sorted(MANIFEST["clips"]))
def test_the_clip_configuration_has_not_changed(clip):
    digest = hashlib.sha256((ROOT / "configs" / f"{clip}.json").read_bytes()).hexdigest()[:16]
    assert digest == MANIFEST["clips"][clip]["config_sha256_16"], (
        f"configs/{clip}.json changed since the freeze; court geometry and per-clip detector "
        "overrides are part of Milestone 1."
    )


@pytest.mark.parametrize("clip", sorted(MANIFEST["clips"]))
def test_the_frozen_accuracy_still_holds(clip):
    entry = MANIFEST["clips"][clip]
    if not entry["metrics_are_frozen"]:
        pytest.skip(f"{clip} carries {entry['label_provenance']}-produced labels and is being "
                    "relabelled; its figures are expected to move when they land")
    baseline = json.loads((ROOT / "outputs" / clip / "baseline.json").read_text())
    got = {
        "hit_precision": baseline["hit_detection"]["precision"],
        "hit_recall": baseline["hit_detection"]["recall"],
        "hit_f1": baseline["hit_detection"]["f1"],
        "hitter_accuracy": baseline["hitter_assignment"]["accuracy"],
        "hitter_scored": baseline["hitter_assignment"]["scored"],
        "rallies_recovered": baseline["rally_segmentation"]["recovered"],
        "rallies_labelled": baseline["rally_segmentation"]["labelled_rallies"],
        "rallies_spurious": baseline["rally_segmentation"]["spurious"],
        "coverage": baseline["player_tracking"]["both_players_fraction"],
        "coverage_trusted": baseline["player_tracking"]["coverage_trusted"],
    }
    assert got == entry["metrics"]


def test_the_human_labels_have_not_been_edited():
    """truth.json is the evidence every figure rests on. A silent edit to it would move the
    numbers without touching a line of detector code."""
    for clip, entry in MANIFEST["clips"].items():
        if entry["label_provenance"] != "human":
            continue
        digest = hashlib.sha256((ROOT / "outputs" / clip / "truth.json").read_bytes()).hexdigest()[:16]
        assert digest == entry["truth_sha256_16"], (
            f"outputs/{clip}/truth.json changed since the freeze. Hand labels are primary "
            "evidence; if they were genuinely revised, re-run tools/freeze_milestone1.py."
        )
