"""Scoring a shot classifier against hand labels.

Every figure is reported beside its class counts. A single accuracy number on a set where one
class holds half the shots says almost nothing, and this taxonomy is badly imbalanced by nature:
a rally has one serve and a dozen other strokes.

Results are also split by near and far player, because the far player is a fraction of the
pixels and any honest read of "does this work" has to say whether it works for both.
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

from .taxonomy import BROAD_CLASSES


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or (precision + recall) == 0:
        return 0.0 if (precision is not None and recall is not None) else None
    return 2 * precision * recall / (precision + recall)


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def confusion(pairs: Sequence[tuple[str, str]], classes: Sequence[str]) -> dict[str, dict[str, int]]:
    """``matrix[truth][prediction]``, every class present even at zero."""
    matrix = {t: {p: 0 for p in classes} for t in classes}
    for truth, prediction in pairs:
        if truth in matrix and prediction in matrix[truth]:
            matrix[truth][prediction] += 1
    return matrix


def per_class(pairs: Sequence[tuple[str, str]], classes: Sequence[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for cls in classes:
        tp = sum(1 for t, p in pairs if t == cls and p == cls)
        fp = sum(1 for t, p in pairs if t != cls and p == cls)
        fn = sum(1 for t, p in pairs if t == cls and p != cls)
        support = tp + fn
        precision = _ratio(tp, tp + fp)
        recall = _ratio(tp, support)
        out[cls] = {
            "support": support,
            "predicted": tp + fp,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": None if precision is None else round(precision, 4),
            "recall": None if recall is None else round(recall, 4),
            "f1": None if support == 0 and (tp + fp) == 0
                  else round(_f1(precision or 0.0, recall or 0.0), 4),
        }
    return out


def macro_f1(per_class_stats: dict[str, Any]) -> float | None:
    """Averaged over the classes that actually occur in the labels.

    A class with no examples has no recall to average, and including it as a zero would drag the
    figure down for a class nobody was asked to find.
    """
    present = [s["f1"] for s in per_class_stats.values() if s["support"] > 0 and s["f1"] is not None]
    return round(sum(present) / len(present), 4) if present else None


def top_k_accuracy(records: Sequence[dict[str, Any]], k: int) -> float | None:
    """Fraction of shots whose true class is in the model's top ``k`` broad classes."""
    usable = [r for r in records if r.get("broad_scores")]
    if not usable:
        return None
    hits = 0
    for record in usable:
        ranked = sorted(record["broad_scores"].items(), key=lambda kv: -kv[1])
        if record["truth"] in [c for c, _ in ranked[:k]]:
            hits += 1
    return round(hits / len(usable), 4)


def evaluate(
    records: Sequence[dict[str, Any]],
    *,
    classes: Iterable[str] = BROAD_CLASSES,
    abstain_class: str = "other",
) -> dict[str, Any]:
    """Score a list of ``{truth, prediction, hitter, broad_scores, ...}`` records."""
    classes = list(classes)
    if not records:
        return {"shots": 0, "note": "nothing to score"}

    pairs = [(r["truth"], r["prediction"]) for r in records]
    stats = per_class(pairs, classes)
    correct = sum(1 for t, p in pairs if t == p)

    def slice_for(hitter: str) -> dict[str, Any]:
        subset = [r for r in records if r.get("hitter") == hitter]
        if not subset:
            return {"shots": 0}
        sub_pairs = [(r["truth"], r["prediction"]) for r in subset]
        sub_stats = per_class(sub_pairs, classes)
        return {
            "shots": len(subset),
            "accuracy": round(sum(1 for t, p in sub_pairs if t == p) / len(subset), 4),
            "macro_f1": macro_f1(sub_stats),
            "top2_accuracy": top_k_accuracy(subset, 2),
            "class_counts": {c: sub_stats[c]["support"] for c in classes
                             if sub_stats[c]["support"]},
            "abstention_rate": round(
                sum(1 for _, p in sub_pairs if p == abstain_class) / len(subset), 4),
        }

    truth_counts = {c: stats[c]["support"] for c in classes if stats[c]["support"]}
    return {
        "shots": len(records),
        "accuracy": round(correct / len(records), 4),
        "macro_f1": macro_f1(stats),
        "top2_accuracy": top_k_accuracy(records, 2),
        "top3_accuracy": top_k_accuracy(records, 3),
        "abstention_rate": round(
            sum(1 for _, p in pairs if p == abstain_class) / len(records), 4),
        "class_counts_truth": truth_counts,
        "class_counts_predicted": {c: stats[c]["predicted"] for c in classes
                                   if stats[c]["predicted"]},
        "largest_class_share": round(max(truth_counts.values()) / len(records), 4)
                               if truth_counts else None,
        "per_class": stats,
        "confusion": confusion(pairs, classes),
        "by_hitter": {"near": slice_for("near"), "far": slice_for("far")},
        "note": (
            "Read the class counts before the accuracy. This taxonomy is imbalanced by nature: "
            "a rally contains one serve and a dozen other strokes."
        ),
    }


def most_confused(result: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
    """The class pairs costing the most, worst first, ignoring the diagonal."""
    out = []
    for truth, row in (result.get("confusion") or {}).items():
        for prediction, count in row.items():
            if prediction != truth and count:
                out.append({"truth": truth, "predicted": prediction, "count": count})
    out.sort(key=lambda item: -item["count"])
    return out[:limit]


def acceptance(result: dict[str, Any]) -> dict[str, Any]:
    """The Milestone 2 targets, fixed before any result was seen."""
    per = result.get("per_class") or {}
    smash = (per.get("smash") or {}).get("f1")
    discriminative = {
        cls: (per.get(cls) or {}).get("f1")
        for cls in ("clear", "lift", "drop")
    }
    near = (result.get("by_hitter") or {}).get("near") or {}
    far = (result.get("by_hitter") or {}).get("far") or {}
    near_acc, far_acc = near.get("accuracy"), far.get("accuracy")
    return {
        "overall_accuracy": {"target": 0.75, "value": result.get("accuracy"),
                             "pass": _at_least(result.get("accuracy"), 0.75)},
        "macro_f1": {"target": 0.65, "value": result.get("macro_f1"),
                     "pass": _at_least(result.get("macro_f1"), 0.65)},
        "top2_accuracy": {"target": 0.90, "value": result.get("top2_accuracy"),
                          "pass": _at_least(result.get("top2_accuracy"), 0.90)},
        "smash_f1": {"target": 0.80, "value": smash, "pass": _at_least(smash, 0.80)},
        "clear_lift_drop_discriminate": {
            "target": "each shows useful discriminative ability rather than collapsing",
            "value": discriminative,
            "pass": all(v is not None and v >= 0.30 for v in discriminative.values()),
            "note": "read as each of the three reaching an F1 of 0.30 rather than one absorbing "
                    "the others",
        },
        "far_player_not_catastrophic": {
            "target": "far-player accuracy within 20 points of near",
            "value": {"near": near_acc, "far": far_acc},
            "pass": (near_acc is not None and far_acc is not None
                     and far_acc >= near_acc - 0.20),
        },
    }


def _at_least(value: float | None, threshold: float) -> bool:
    return value is not None and value >= threshold
