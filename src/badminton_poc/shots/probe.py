"""A linear probe on BST's frozen representation, to ask whether the model or the head is wrong.

Pretrained BST scores about 38% on our footage, cross-validated over two clips. That number
cannot say *why*. Two very different worlds produce it:

* the representation carries what a shot type looks like on amateur footage, and only the output
  layer is aimed at the wrong distribution -- in which case adaptation is cheap and the question
  is how much data to collect;
* the representation does not carry it, and no head on top will help -- in which case BST is the
  wrong starting point and fine-tuning it would be throwing good effort after bad.

A linear probe separates them. Freeze the backbone, take the 300-dimensional vector it hands to
its own classifier, and fit a logistic regression on one clip's labels. Test on the *other* clip,
never on the clip it was fitted to, so nothing leaks. If the probe beats zero-shot by a wide
margin the representation is fine and the head was the problem.

Two controls make the answer readable. A majority-class predictor says what the class imbalance
alone is worth. A probe on the raw input arrays, the same pose and shuttle numbers BST is given,
says whether the transformer earns its keep over the features it started from.

With 116 labelled shots across two clips this is a small experiment and its error bars are wide.
It is a direction-finder, not a result.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .bst_adapter import BSTConfig, bone_pairs, load_bst, to_model_inputs
from .features import ShotFeatures
from .taxonomy import BROAD_CLASSES

REPRESENTATION_DIM = 300
"""``d_model * 3``: the two players' conclusions concatenated with the shuttle token."""


@dataclass(frozen=True)
class ProbeResult:
    name: str
    trained_on: str
    tested_on: str
    accuracy: float
    macro_f1: float
    per_class_f1: dict[str, float]
    support: dict[str, int]
    n_train: int
    n_test: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "probe": self.name,
            "trained_on": self.trained_on,
            "tested_on": self.tested_on,
            "accuracy": round(self.accuracy, 4),
            "macro_f1": round(self.macro_f1, 4),
            "per_class_f1": {k: round(v, 4) for k, v in self.per_class_f1.items()},
            "support": dict(self.support),
            "n_train": self.n_train,
            "n_test": self.n_test,
        }


def representations(
    features: Sequence[ShotFeatures],
    cfg: BSTConfig | None = None,
    base: Path = Path("."),
    batch_size: int = 64,
) -> np.ndarray:
    """The 300-d vector BST hands its own classifier, one row per shot.

    Captured with a forward hook on ``mlp_head`` rather than by copying the forward pass, so it
    stays correct if upstream changes and cannot drift out of step with the real model.
    """
    import torch

    cfg = cfg or BSTConfig()
    if not features:
        return np.zeros((0, REPRESENTATION_DIM), dtype=np.float32)
    net, _, device = load_bst(cfg, base)
    pose, shuttle, lengths = to_model_inputs(features, bone_pairs(cfg, base), cfg.seq_len)

    captured: list[np.ndarray] = []
    handle = net.mlp_head.register_forward_pre_hook(
        lambda module, inputs: captured.append(inputs[0].detach().cpu().numpy())
    )
    try:
        with torch.no_grad():
            for i in range(0, len(features), batch_size):
                net(torch.from_numpy(pose[i:i + batch_size]).to(device),
                    torch.from_numpy(shuttle[i:i + batch_size]).to(device),
                    torch.from_numpy(lengths[i:i + batch_size]).to(device))
    finally:
        handle.remove()
    return np.concatenate(captured).astype(np.float32)


def raw_inputs(features: Sequence[ShotFeatures], cfg: BSTConfig | None = None,
               base: Path = Path(".")) -> np.ndarray:
    """The control: BST's own input arrays, flattened, with no transformer in between."""
    cfg = cfg or BSTConfig()
    if not features:
        return np.zeros((0, 1), dtype=np.float32)
    pose, shuttle, lengths = to_model_inputs(features, bone_pairs(cfg, base), cfg.seq_len)
    flat = pose.reshape(len(features), -1)
    return np.concatenate(
        [flat, shuttle.reshape(len(features), -1), lengths.reshape(-1, 1).astype(np.float32)],
        axis=1,
    )


def with_hitter(matrix: np.ndarray, hitters: Sequence[str | None]) -> np.ndarray:
    """Append which player hit. Milestone 1 knows this at inference time, so it is fair game."""
    flag = np.array([[1.0 if h == "near" else 0.0] for h in hitters], dtype=np.float32)
    return np.concatenate([matrix, flag], axis=1)


def _scores(truth: Sequence[str], predicted: Sequence[str]) -> tuple[float, float, dict, dict]:
    accuracy = float(np.mean([t == p for t, p in zip(truth, predicted)])) if truth else 0.0
    f1: dict[str, float] = {}
    support: dict[str, int] = {}
    for cls in BROAD_CLASSES:
        tp = sum(1 for t, p in zip(truth, predicted) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(truth, predicted) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(truth, predicted) if t == cls and p != cls)
        support[cls] = tp + fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1[cls] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    present = [f1[c] for c in BROAD_CLASSES if support[c] > 0]
    macro = float(np.mean(present)) if present else 0.0
    return accuracy, macro, f1, {k: v for k, v in support.items() if v}


def fit_probe(
    train_x: np.ndarray, train_y: Sequence[str],
    test_x: np.ndarray, test_y: Sequence[str],
    *, name: str, trained_on: str, tested_on: str,
    balanced: bool = True, seed: int = 0,
) -> ProbeResult:
    """Multinomial logistic regression, standardised inputs, no tuning against the test clip."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=4000, C=1.0, random_state=seed,
            class_weight="balanced" if balanced else None,
        ),
    )
    model.fit(train_x, list(train_y))
    predicted = list(model.predict(test_x))
    accuracy, macro, f1, support = _scores(list(test_y), predicted)
    return ProbeResult(name, trained_on, tested_on, accuracy, macro, f1, support,
                       len(train_y), len(test_y))


def majority_baseline(
    train_y: Sequence[str], test_y: Sequence[str], *, trained_on: str, tested_on: str
) -> ProbeResult:
    """Always answer the commonest class in the training clip. The floor any probe must clear."""
    from collections import Counter

    top = Counter(train_y).most_common(1)[0][0] if train_y else "other"
    predicted = [top] * len(test_y)
    accuracy, macro, f1, support = _scores(list(test_y), predicted)
    return ProbeResult(f"majority class ({top})", trained_on, tested_on, accuracy, macro,
                       f1, support, len(train_y), len(test_y))
