"""Turn Milestone 1 hits into shot samples: a window of frames around each contact.

The window is not chosen by us. It is the ``between_2_hits_with_max_limits`` rule from the BST
repository's ``ShuttleSet/gen_my_dataset.py``, which is what its published checkpoints were
trained on, restated in seconds:

    half   = 0.5s     fallback reach when a neighbouring hit is missing
    eps    = 0.25s    extension past the next hit, so the shuttle's arrival is inside the window
    limit  = 1.5s     hard cap either side of the contact

    start = previous hit, or contact - half, but never earlier than contact - limit
    end   = next hit + eps, or contact + half, but never later than contact + limit + eps

Upstream writes those as ``fps // 2``, ``t // 2`` and ``fps * 3 // 2``, so the rule is already
expressed in time and transfers to our frame rate unchanged. What does *not* transfer is the
number of samples inside it, which is what :mod:`badminton_poc.shots.features` deals with.

A sample stores frame references, never pixels. Everything needed to rebuild the window is on
disk already in ``players.jsonl`` and the shuttle CSV.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

BST_HALF_SECONDS = 0.5
BST_EPS_SECONDS = 0.25
BST_LIMIT_SECONDS = 1.5


@dataclass(frozen=True)
class ShotWindowConfig:
    strategy: str = "between_2_hits_with_max_limits"
    """Which of upstream's clipping rules to reproduce.

    ``between_2_hits_with_max_limits`` is the rule behind the sequence-100 checkpoints and is
    described above. ``middle_in_a_sec`` is the rule behind the sequence-30 checkpoints and is
    far simpler: exactly ``half_seconds`` either side of the contact, ignoring the neighbours
    entirely. Both come from ``ShuttleSet/gen_my_dataset.py``; neither was chosen by us."""
    half_seconds: float = BST_HALF_SECONDS
    eps_seconds: float = BST_EPS_SECONDS
    limit_seconds: float = BST_LIMIT_SECONDS
    require_hitter: bool = True
    """Skip contacts Milestone 1 could not attribute. A shot classifier keyed on the hitter has
    nothing to say about a hit with no hitter, and guessing one here would launder an unknown
    into a label."""


@dataclass(frozen=True)
class ShotSample:
    clip: str
    shot_id: int
    rally: int
    index_in_rally: int
    hit_frame: int
    hit_time_s: float
    hitter: str | None
    prev_hit_frame: int | None
    next_hit_frame: int | None
    start_frame: int
    end_frame: int
    fps: float
    width: int
    height: int
    is_first_in_rally: bool
    is_last_in_rally: bool
    rally_start_frame: int
    rally_end_frame: int
    rally_start_kind: str | None
    shot_type: str | None = None
    """The labeller's own word for this stroke, carried through untouched. None when the clip's
    contacts were labelled without one. Never written by a model."""

    @property
    def frames(self) -> int:
        return self.end_frame - self.start_frame + 1

    @property
    def seconds(self) -> float:
        return self.frames / self.fps

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["frames"] = self.frames
        out["seconds"] = round(self.seconds, 3)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ShotSample":
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in fields})


def window_for(
    hit_frame: int,
    prev_hit_frame: int | None,
    next_hit_frame: int | None,
    *,
    fps: float,
    cfg: ShotWindowConfig,
    first_frame: int = 0,
    last_frame: int | None = None,
) -> tuple[int, int]:
    """Inclusive frame span around one contact, per the BST rule above."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    half = int(round(cfg.half_seconds * fps))
    eps = int(round(cfg.eps_seconds * fps))
    limit = int(round(cfg.limit_seconds * fps))

    if cfg.strategy == "middle_in_a_sec":
        start, end = hit_frame - half, hit_frame + half
        start = max(start, first_frame)
        if last_frame is not None:
            end = min(end, last_frame)
        return start, max(start, end)
    if cfg.strategy != "between_2_hits_with_max_limits":
        raise ValueError(
            f"unknown clipping strategy {cfg.strategy!r}; expected "
            "'between_2_hits_with_max_limits' or 'middle_in_a_sec'"
        )

    start = prev_hit_frame if prev_hit_frame is not None else hit_frame - half
    end = (next_hit_frame + eps) if next_hit_frame is not None else hit_frame + half
    start = max(start, hit_frame - limit)
    end = min(end, hit_frame + limit + eps)

    start = max(start, first_frame)
    if last_frame is not None:
        end = min(end, last_frame)
    if end < start:
        end = start
    return start, end


def build_samples(
    analysis: dict[str, Any],
    *,
    clip: str,
    width: int,
    height: int,
    cfg: ShotWindowConfig | None = None,
    last_frame: int | None = None,
) -> tuple[list[ShotSample], list[dict[str, Any]]]:
    """One sample per Milestone 1 hit, plus a record of every hit deliberately skipped.

    ``width`` and ``height`` are required rather than read from the analysis document, which
    does not carry them. Everything downstream normalizes by frame size, so a silent default
    would turn every "normalized" coordinate into raw pixels without anything failing.
    """
    cfg = cfg or ShotWindowConfig()
    fps = float(analysis.get("config", {}).get("fps") or 0.0)
    if fps <= 0:
        raise ValueError("analysis.json carries no usable fps")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive; they normalize every coordinate")
    metadata = analysis.get("metadata") or {}
    if last_frame is None and metadata.get("frame_count"):
        last_frame = int(metadata["frame_count"]) - 1

    samples: list[ShotSample] = []
    skipped: list[dict[str, Any]] = []
    shot_id = 0
    for rally in analysis.get("rallies", []):
        hits = rally.get("hits") or []
        frames = [int(h["frame"]) for h in hits]
        for index, hit in enumerate(hits):
            hitter = hit.get("hitter")
            if cfg.require_hitter and hitter not in ("near", "far"):
                skipped.append({
                    "rally": rally.get("rally"),
                    "frame": int(hit["frame"]),
                    "reason": "Milestone 1 could not attribute this contact to a player",
                })
                continue
            frame = int(hit["frame"])
            start, end = window_for(
                frame,
                frames[index - 1] if index > 0 else None,
                frames[index + 1] if index + 1 < len(frames) else None,
                fps=fps, cfg=cfg, last_frame=last_frame,
            )
            samples.append(ShotSample(
                clip=clip,
                shot_id=shot_id,
                rally=int(rally.get("rally", 0)),
                index_in_rally=index,
                hit_frame=frame,
                hit_time_s=round(frame / fps, 3),
                hitter=hitter,
                prev_hit_frame=frames[index - 1] if index > 0 else None,
                next_hit_frame=frames[index + 1] if index + 1 < len(frames) else None,
                start_frame=start,
                end_frame=end,
                fps=fps,
                width=width,
                height=height,
                is_first_in_rally=index == 0,
                is_last_in_rally=index == len(hits) - 1,
                rally_start_frame=int(rally.get("start_frame", frame)),
                rally_end_frame=int(rally.get("end_frame", frame)),
                rally_start_kind=rally.get("start_kind"),
            ))
            shot_id += 1
    return samples, skipped


def build_samples_from_truth(
    truth: dict[str, Any],
    *,
    clip: str,
    width: int,
    height: int,
    cfg: ShotWindowConfig | None = None,
    last_frame: int | None = None,
) -> tuple[list[ShotSample], list[dict[str, Any]]]:
    """One sample per hand-labelled racket contact, plus a record of every one left out.

    The sibling :func:`build_samples` reads Milestone 1's own hits, which is the right source
    when the question is what the pipeline produced. It is the wrong source here. Milestone 2
    asks how well a classifier reads a stroke, so a sample has to come from a contact a human
    confirmed; building from detector output would put Milestone 1's false positives into the
    dataset and score the classifier on strokes that never happened.

    Three kinds of contact are deliberately left out, each recorded with its reason:

    * ``in_rally: false`` -- handing the shuttle back, a knock-up between points. A real racket
      contact and not a stroke to classify.
    * outside ``labelled_contact_spans`` -- clip 5 is labelled in parts, and outside those spans
      its contact list is incomplete by declaration, so a window there would take its neighbours
      from contacts nobody checked.
    * no hitter -- the labeller could not tell who played it, and the window is built around a
      player.

    A contact that is left out still counts as a neighbour for the windows either side of it,
    because it happened: the shuttle really did change direction there.
    """
    cfg = cfg or ShotWindowConfig()
    fps = float(truth.get("fps") or 0.0)
    if fps <= 0:
        raise ValueError("truth file carries no usable fps")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive; they normalize every coordinate")

    spans = truth.get("labelled_contact_spans") or []

    def inside_spans(frame: int) -> bool:
        return not spans or any(
            int(s["start_frame"]) <= frame <= int(s["end_frame"]) for s in spans
        )

    rallies = sorted(
        (
            {
                "rally": int(r.get("rally", i + 1)),
                "start": int(r["start_frame"]),
                "end": int(r["end_frame"]),
            }
            for i, r in enumerate(truth.get("rallies") or [])
        ),
        key=lambda r: r["start"],
    )
    contacts = sorted((truth.get("contacts") or []), key=lambda c: int(c["frame"]))

    samples: list[ShotSample] = []
    skipped: list[dict[str, Any]] = []
    shot_id = 0

    for rally in rallies:
        in_rally = [c for c in contacts if rally["start"] <= int(c["frame"]) <= rally["end"]]
        frames = [int(c["frame"]) for c in in_rally]
        kept = 0
        for index, contact in enumerate(in_rally):
            frame = int(contact["frame"])
            reason = None
            if contact.get("in_rally") is False:
                reason = "a real contact but not a rally shot, so not a stroke to classify"
            elif contact.get("uncertain"):
                reason = "the labeller could not tell whether this was a contact at all"
            elif not inside_spans(frame):
                reason = (
                    "outside labelled_contact_spans, where the contact list is incomplete by "
                    "declaration"
                )
            elif contact.get("hitter") not in ("near", "far"):
                reason = "the labeller could not tell which player made this contact"
            if reason is not None:
                skipped.append({"clip": clip, "rally": rally["rally"], "frame": frame,
                                "reason": reason})
                continue

            start, end = window_for(
                frame,
                frames[index - 1] if index > 0 else None,
                frames[index + 1] if index + 1 < len(frames) else None,
                fps=fps, cfg=cfg, last_frame=last_frame,
            )
            samples.append(ShotSample(
                clip=clip,
                shot_id=shot_id,
                rally=rally["rally"],
                index_in_rally=kept,
                hit_frame=frame,
                hit_time_s=round(frame / fps, 3),
                hitter=contact.get("hitter"),
                prev_hit_frame=frames[index - 1] if index > 0 else None,
                next_hit_frame=frames[index + 1] if index + 1 < len(frames) else None,
                start_frame=start,
                end_frame=end,
                fps=fps,
                width=width,
                height=height,
                is_first_in_rally=index == 0,
                is_last_in_rally=index == len(in_rally) - 1,
                rally_start_frame=rally["start"],
                rally_end_frame=rally["end"],
                rally_start_kind=None,
                shot_type=contact.get("shot_type"),
            ))
            shot_id += 1
            kept += 1

    inside_any = {int(c["frame"]) for r in rallies
                  for c in contacts if r["start"] <= int(c["frame"]) <= r["end"]}
    for contact in contacts:
        frame = int(contact["frame"])
        if frame not in inside_any:
            skipped.append({"clip": clip, "rally": None, "frame": frame,
                            "reason": "not inside any labelled rally span"})

    skipped.sort(key=lambda s: s["frame"])
    return samples, skipped
