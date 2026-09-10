"""Mapping BST's stroke classes onto this project's broad shot taxonomy.

BST's merged ShuttleSet head has 25 outputs: an unknown class, then twelve stroke types for the
Top player and the same twelve for the Bottom player. The side prefix is part of the class, so a
prediction carries an opinion about *who* hit as well as *what* was hit. We keep that: it is an
independent check on Milestone 1's hitter assignment, which nothing else in the pipeline offers.

Our own taxonomy is deliberately coarser and is defined by **where the shuttle goes and what
shape the flight has**, not by where the player was standing:

    serve   the shot that starts a rally
    smash   steep and fast, downward
    drop    from the rear court, falling into the forecourt
    clear   high and deep to the rear court
    lift    upward from the forecourt to the rear court
    drive   flat and fast, roughly parallel to the floor
    net     played in the forecourt and staying there
    other   anything the mapping cannot place without guessing

Two upstream classes are deliberately unmapped. ``推球`` (push) is a flat forecourt shot sent to
the opponent's midcourt, which sits between ``drive`` and ``net``; ``撲球`` (net kill) is a steep
downward putaway taken at the net, which sits between ``smash`` and ``net``. Calling either one
would be an unsupported decision, and folding a net kill into ``smash`` in particular would
quietly inflate the one class with its own acceptance threshold. Both go to ``other`` and the
alternative reading is recorded beside them, so changing our minds is a one-line edit plus a
re-score rather than an argument.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MAPPING_VERSION = "broad-v2"

BROAD_CLASSES = ("serve", "smash", "drop", "clear", "lift", "drive", "net", "other")

UNKNOWN_UPSTREAM = "未知球種"

TOP = "Top"
BOTTOM = "Bottom"

SIDE_TO_SLOT = {TOP: "far", BOTTOM: "near"}
"""BST orders players Top before Bottom by court depth. Top is the far half of our frame."""


@dataclass(frozen=True)
class StrokeMapping:
    upstream: str
    broad: str
    english: str
    note: str = ""
    contested_alternative: str = ""


STROKE_TABLE: tuple[StrokeMapping, ...] = (
    StrokeMapping("發短球", "serve", "short serve", "starts a rally, low over the net"),
    StrokeMapping("發長球", "serve", "long serve", "starts a rally, high to the rear court"),
    StrokeMapping("殺球", "smash", "smash", "steep, fast, downward"),
    StrokeMapping("切球", "drop", "slice drop", "from the rear court into the forecourt"),
    StrokeMapping("長球", "clear", "clear", "high and deep to the rear court"),
    StrokeMapping("挑球", "lift", "lift", "upward from the forecourt to the rear court"),
    StrokeMapping("平球", "drive", "drive", "flat and fast"),
    StrokeMapping("放小球", "net", "net shot", "played in the forecourt, stays there"),
    StrokeMapping("勾球", "net", "cross-court net shot", "forecourt to the opposite forecourt"),
    StrokeMapping(
        "擋小球", "net", "block to the net",
        "a soft block off a smash that drops into the forecourt; mapped by where the shuttle "
        "lands, which is what this taxonomy is defined by",
        contested_alternative="other",
    ),
    StrokeMapping(
        "推球", "other", "push",
        "flat forecourt shot pushed to the opponent's midcourt: between drive and net, and "
        "neither without guessing",
        contested_alternative="drive",
    ),
    StrokeMapping(
        "撲球", "other", "net kill",
        "steep downward putaway taken at the net: between smash and net. Folding it into smash "
        "would inflate the class carrying its own acceptance threshold",
        contested_alternative="smash",
    ),
    # The five below exist only in the unmerged 35-class head. Each is a finer split of a class
    # already in the table above, so each inherits that class without a new judgement.
    StrokeMapping("點扣", "smash", "punch smash", "a flatter, shorter-swing smash"),
    StrokeMapping("防守回挑", "lift", "defensive lift", "a lift played under pressure"),
    StrokeMapping("後場抽平球", "drive", "rear-court drive", "a flat drive struck from the back"),
    StrokeMapping("過渡切球", "drop", "transitional slice drop", "a slice drop played to regain position"),
    StrokeMapping("防守回抽", "drive", "defensive drive", "a flat drive played under pressure"),
)

_BY_UPSTREAM = {m.upstream: m for m in STROKE_TABLE}


MERGED_CLASSES = 25
UNMERGED_CLASSES = 35
"""BST publishes two heads. They differ in more than width: the merged head puts the unknown
class **first**, at index 0, while the unmerged head puts it **last**, at index 34. Reading a
prediction against the wrong list silently shifts every label by one."""


def split_side(class_name: str) -> tuple[str | None, str]:
    """Split ``Top_殺球`` into its side and its stroke. The unknown class has no side."""
    if class_name == UNKNOWN_UPSTREAM:
        return None, UNKNOWN_UPSTREAM
    for side in (TOP, BOTTOM):
        prefix = f"{side}_"
        if class_name.startswith(prefix):
            return side, class_name[len(prefix):]
    return None, class_name


def to_broad(class_name: str) -> str:
    """The broad class for one upstream class name. Anything unrecognised is ``other``."""
    _, stroke = split_side(class_name)
    mapping = _BY_UPSTREAM.get(stroke)
    return mapping.broad if mapping else "other"


def predicted_slot(class_name: str) -> str | None:
    """Which of our player slots the upstream class claims played the shot."""
    side, _ = split_side(class_name)
    return SIDE_TO_SLOT.get(side) if side else None


def describe(class_name: str) -> dict[str, Any]:
    side, stroke = split_side(class_name)
    mapping = _BY_UPSTREAM.get(stroke)
    return {
        "upstream": class_name,
        "side": side,
        "slot": SIDE_TO_SLOT.get(side) if side else None,
        "stroke": stroke,
        "english": mapping.english if mapping else ("unknown stroke type"
                                                    if stroke == UNKNOWN_UPSTREAM else stroke),
        "broad": mapping.broad if mapping else "other",
        "note": mapping.note if mapping else "",
        "contested_alternative": mapping.contested_alternative if mapping else "",
    }


def broad_scores(scores: dict[str, float], *, slot: str | None = None) -> dict[str, float]:
    """Fold upstream class scores into broad-class scores by summing.

    Summing rather than taking a maximum because the broad class is the union of the upstream
    classes: two net-shot classes each at 0.3 mean the shot is very probably a net shot.

    Pass ``slot`` to restrict to the classes claiming that player. Milestone 1 already knows who
    hit, at 92.7% accuracy, and a prediction for the wrong side carries no information about the
    stroke. The remaining mass is renormalised so the result still sums to one.
    """
    kept = {
        name: value for name, value in scores.items()
        if slot is None or predicted_slot(name) in (None, slot)
    }
    total = sum(kept.values())
    out = {c: 0.0 for c in BROAD_CLASSES}
    if total <= 0:
        out["other"] = 1.0
        return out
    for name, value in kept.items():
        out[to_broad(name)] += value / total
    return out


def mapping_document() -> dict[str, Any]:
    """The whole mapping as data, for writing next to any result that used it."""
    return {
        "version": MAPPING_VERSION,
        "broad_classes": list(BROAD_CLASSES),
        "side_to_slot": dict(SIDE_TO_SLOT),
        "unknown_upstream": UNKNOWN_UPSTREAM,
        "unknown_maps_to": "other",
        "defined_by": (
            "where the shuttle goes and the shape of its flight, not where the player stood"
        ),
        "strokes": [
            {
                "upstream": m.upstream, "english": m.english, "broad": m.broad,
                "note": m.note, "contested_alternative": m.contested_alternative,
            }
            for m in STROKE_TABLE
        ],
    }
