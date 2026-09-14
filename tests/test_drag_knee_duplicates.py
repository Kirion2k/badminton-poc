"""Does one racket contact produce one hit, or two?

A struck shuttlecock sheds its horizontal speed to drag before it has built vertical speed, so
for a few frames it travels slower than `min_flight_speed_px_s` and the segmenter stops calling
it flight. That cut splits one physical flight in two, and the join across it turns by 60-80
degrees -- above `min_turn_degrees`, whose own docstring puts real contacts at a median 150 and
frames inside a flight at 9. The knee sits between the two, so the link classifier reads it as a
racket contact and a second hit is reported about 0.3s after the real one.

On clip 3 that happens at frames 106/123, 810/831, 881/899 and 1619/1636, and on clip 2 at
7054/7071. `consolidate_events` exists to collapse exactly this and was never run on the flight
path: analyze.py took `events = raw_events` whenever a flight model was present.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from badminton_poc.analyze import analyze
from badminton_poc.config import AnalysisConfig, CourtROI, FlightConfig
from badminton_poc.flight import segment_flight
from badminton_poc.parsing import BallSample, BallTrack
from badminton_poc.rallies import consolidate_events

FPS = 60.0
WIDTH, HEIGHT = 1920, 1080


def knee_points():
    """Approach, a racket reversal, the drag knee, then the terminal fall.

    Shaped after clip 3 frames 85-160: the struck flight runs left at 46px/frame decaying to
    about 10, then the shuttle crawls for six frames before falling straight down.
    """
    pts = []
    x, y = 1000.0, 340.0
    for f in range(0, 26):
        pts.append((f, x, y)); x += 15.0; y += 6.0
    vx, vy = -46.0, 0.4
    for f in range(26, 40):
        pts.append((f, x, y)); x += vx; y += vy; vx *= 0.90
    for f in range(40, 47):
        pts.append((f, x, y)); x += -1.2; y += 1.2
    vy = 1.8
    for f in range(47, 96):
        pts.append((f, x, y)); x += -0.7; y += vy; vy = min(3.6, vy + 0.15)
    return pts


def knee_track():
    return BallTrack([BallSample(frame=f, x=x, y=y, visible=True) for f, x, y in knee_points()])


def test_the_drag_knee_splits_one_flight_and_reports_a_second_hit():
    """The root cause, pinned so a change to the segmenter cannot quietly alter it."""
    model = segment_flight(knee_track(), FPS, FlightConfig(), width=WIDTH)
    hits = sorted(e.frame for e in model.events if e.event_type == "hit")
    assert len(hits) == 2, f"expected the knee to split the flight, got {hits}"
    assert hits[0] == 23, "the racket contact"
    gap = (hits[1] - hits[0]) / FPS
    assert 0.25 < gap < 0.31, f"the knee follows the contact by about 0.3s, got {gap:.3f}s"
    knee = next(l for l in model.links if l.hit_frame == hits[1])
    assert knee.turn_degrees > 60.0, "the knee turns far enough to read as a contact"


def test_consolidation_keeps_the_contact_and_drops_the_knee():
    model = segment_flight(knee_track(), FPS, FlightConfig(), width=WIDTH)
    kept, audit = consolidate_events(list(model.events), FPS, 0.30, "hit")
    hits = sorted(e.frame for e in kept if e.event_type == "hit")
    assert hits == [23], "the earlier event is the racket contact; the knee follows it"
    assert audit and audit[0]["merged_frames"] == [23, 40], "and the merge is recorded, not silent"


def test_two_genuine_contacts_a_third_of_a_second_apart_survive():
    """The closest pair of hand-labelled contacts in the project is 0.350s (clip 2, 7227/7248).
    A window that merged those would be destroying real contacts to tidy up false ones."""
    from badminton_poc.parsing import ShuttleEvent
    events = [ShuttleEvent(frame=f, event_type="hit", source="flight", confidence=None,
                           x=0.0, y=0.0, rule="", all_rules=())
              for f in (100, 100 + round(0.350 * FPS))]
    kept, _ = consolidate_events(events, FPS, 0.30, "hit")
    assert sorted(e.frame for e in kept) == [100, 121]


def _write_clip(tmp_path: Path):
    ball = tmp_path / "ball.csv"
    lines = ["Frame,Visibility,X,Y"]
    lines += [f"{f},1,{x},{y}" for f, x, y in knee_points()]
    ball.write_text("\n".join(lines))
    (tmp_path / "events.json").write_text("[]")
    rows = []
    for f, x, y in knee_points():
        rows.append(json.dumps({
            "frame": f, "width": WIDTH, "height": HEIGHT,
            "near": {"bbox": [900.0, 700.0, 1040.0, 1000.0], "conf": 0.9, "track_id": 1,
                     "keypoints": []},
            "far": {"bbox": [860.0, 300.0, 950.0, 480.0], "conf": 0.9, "track_id": 2,
                    "keypoints": []},
        }))
    (tmp_path / "players.jsonl").write_text("\n".join(rows))
    return ball


COURT = CourtROI(
    polygon=[(0.05, 0.20), (0.95, 0.20), (0.98, 0.98), (0.02, 0.98)],
    net_line=[(0.05, 0.55), (0.95, 0.55)],
)


def run(tmp_path, **overrides):
    ball = _write_clip(tmp_path)
    cfg = AnalysisConfig(event_source="flight", require_hitter=False,
                         require_both_players=False, min_hits_per_rally=1, **overrides)
    return analyze(
        tmp_path / "players.jsonl", ball, tmp_path / "events.json", tmp_path / "out.json",
        fps=FPS, court=COURT, cfg=cfg, flight_cfg=FlightConfig(),
        width=WIDTH, height=HEIGHT,
    )


def test_the_flight_path_consolidates_hits(tmp_path):
    """The regression this exists for: analyze took `events = raw_events` on the flight path, so
    min_hit_interval_seconds was never applied to anything the flight detector produced."""
    out = run(tmp_path)
    hits = sorted(h["frame"] for r in out["rallies"] for h in r["hits"])
    assert hits == [23], f"the knee should have been merged into the contact, got {hits}"


def test_the_merge_is_audited_on_the_flight_path_too(tmp_path):
    out = run(tmp_path)
    merged = [a for a in out["hit_consolidation"] if a["event_type"] == "hit"]
    assert merged, "hit_consolidation must record the merge so raw output stays auditable"
    assert 40 in merged[0]["merged_frames"]


def test_setting_the_window_to_zero_leaves_the_flight_detector_alone(tmp_path):
    """Zero has to mean off here as it does on the other path, or the knobs disagree."""
    out = run(tmp_path, min_hit_interval_seconds=0.0)
    hits = sorted(h["frame"] for r in out["rallies"] for h in r["hits"])
    assert hits == [23, 40]
