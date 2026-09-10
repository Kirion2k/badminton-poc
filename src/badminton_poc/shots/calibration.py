"""Prior-shift correction, and the ShuttleSet class prior it corrects against.

A classifier trained on one distribution carries that distribution's base rates in its output.
BST was trained on elite tournament singles, where a smash is one stroke in nine; our clip is
amateur play where it is one in sixty. Under the standard label-shift assumption -- the classes
change frequency but each class still *looks* the same -- the fix is to divide the predicted
probability by the source base rate and multiply by the target one.

Measured on the development clip, that is worth eleven points of accuracy, which makes the
distribution mismatch real and quantified rather than asserted. But it comes with a catch worth
stating loudly: **it only works if you already know the target base rates.** Dividing by the
source prior alone, which is the only version you could actually deploy on an unlabelled clip,
buys nothing at all.

So this module is here as a measuring instrument, not as a component of the pipeline. It answers
"how much of the gap is distribution?" and the answer is "about a quarter of it".
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from .taxonomy import BROAD_CLASSES, to_broad

EPSILON = 1e-6


def shuttleset_prior(repo: str | Path) -> dict[str, float]:
    """Broad-class base rates of ShuttleSet, counted from the BST repository's own annotations.

    Counted rather than assumed, and counted from the *source* corpus rather than from anything
    we labelled, so using it involves no circularity.
    """
    root = Path(repo) / "ShuttleSet" / "set"
    if not root.is_dir():
        raise FileNotFoundError(f"no ShuttleSet annotations under {root}")
    counts: Counter[str] = Counter()
    for path in sorted(root.glob("*/set*.csv")):
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "type" not in reader.fieldnames:
                continue
            for row in reader:
                stroke = (row.get("type") or "").strip()
                if stroke:
                    counts[to_broad(stroke)] += 1
    total = sum(counts.values())
    if total == 0:
        raise ValueError(f"no stroke annotations found under {root}")
    return {cls: counts.get(cls, 0) / total for cls in BROAD_CLASSES}


def class_prior(labels: Iterable[str]) -> dict[str, float]:
    """Base rates of a labelled set. Using this as a *target* prior is an oracle, not a method."""
    counts = Counter(labels)
    total = sum(counts.values())
    if total == 0:
        return {cls: 0.0 for cls in BROAD_CLASSES}
    return {cls: counts.get(cls, 0) / total for cls in BROAD_CLASSES}


def prior_shift(
    scores: Mapping[str, float],
    source: Mapping[str, float],
    target: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Re-weight class scores from the source base rates to the target ones, then renormalise.

    With ``target`` omitted this divides by the source prior alone, which flattens the model's
    base rates rather than moving them anywhere in particular. That version is deployable and,
    measured here, useless; it is kept because "we tried it and it did nothing" is a result.
    """
    adjusted = {}
    for cls, value in scores.items():
        weight = 1.0 / max(float(source.get(cls, 0.0)), EPSILON)
        if target is not None:
            weight *= float(target.get(cls, 0.0))
        adjusted[cls] = float(value) * weight
    total = sum(adjusted.values())
    if total <= 0:
        return {cls: (1.0 if cls == "other" else 0.0) for cls in scores}
    return {cls: value / total for cls, value in adjusted.items()}


def decompose_gap(steps: list[tuple[str, float]], target: float) -> dict[str, Any]:
    """Turn a chain of increasingly favourable conditions into a readable attribution.

    Each step is a condition relaxed, and the gain is what that condition was costing. What is
    left at the end is the part no amount of favourable conditions explains.
    """
    out = []
    previous = None
    for name, value in steps:
        out.append({
            "condition": name,
            "accuracy": round(value, 4),
            "gain": None if previous is None else round(value - previous, 4),
        })
        previous = value
    return {
        "steps": out,
        "target": target,
        "unexplained": None if previous is None else round(target - previous, 4),
        "note": (
            "Every step after the first uses information not available at inference time, so "
            "these are diagnostic bounds and not achievable scores."
        ),
    }
