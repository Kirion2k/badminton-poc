"""The COURTSIDE port: unit rescaling, the kink pass and nearest-wrist attribution."""
from __future__ import annotations

from badminton_poc.experiments import courtside as cs


def test_scale_keeps_durations_and_frame_shares():
    s = cs.Scale(fps=60.0, width=1920.0)
    assert s.frames(4) == 8                      # 4 frames at 30 fps is 8 at 60
    assert s.speed(10.0) == 7.5                  # 1.5x the pixels, half the time per frame


def test_a_reversal_is_one_hit_and_a_straight_flight_is_none():
    s = cs.Scale(fps=30.0, width=1280.0)
    out = {f: (100.0 + 20 * f, 300.0) for f in range(0, 20)}
    back = {f: (100.0 + 20 * 19 - 20 * (f - 19), 300.0) for f in range(20, 40)}
    hits = cs.detect_hits({**out, **back}, {}, 0, 39, s, wrist_pass=False)
    kinks = [h for h in hits if h["pass"] == "kink"]
    assert len(kinks) == 1 and abs(kinks[0]["frame"] - 19) <= 1
    straight = cs.detect_hits(out, {}, 0, 19, s, wrist_pass=False)
    assert not [h for h in straight if h["pass"] == "kink"]


def player(wrist_xy, conf=0.9):
    kps = [[0.0, 0.0, 0.0] for _ in range(17)]
    kps[9] = [wrist_xy[0], wrist_xy[1], conf]
    return {"bbox": [0, 0, 10, 10], "keypoints": kps}


def test_attribution_is_the_nearest_confident_wrist():
    players = {5: {"near": player((100, 500)), "far": player((400, 200))}}
    assert cs.attribute({"frame": 5, "x": 390.0, "y": 210.0}, players) == "far"
    assert cs.attribute({"frame": 5, "x": 110.0, "y": 490.0}, players) == "near"


def test_attribution_falls_back_to_the_box_centre_without_a_confident_wrist():
    players = {5: {"near": player((100, 500), conf=0.1), "far": None}}
    assert cs.attribute({"frame": 5, "x": 5.0, "y": 5.0}, players) == "near"
