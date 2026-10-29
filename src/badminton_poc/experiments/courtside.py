"""EXPERIMENTAL. COURTSIDE's contact detector and hitter attribution, ported to our inputs.

Source: jayclim/BadmintonAI (MIT), ``src/badminton/hits.py`` at the commit cloned 2026-10-29.
Nothing in the pipeline imports this; ``tools/benchmark_courtside.py`` scores it against the
hand labels beside our own detector.

The algorithm, as written upstream:

1. **Kink pass.** Interpolate visibility gaps up to 6 frames, then score every frame on the
   velocity in over the last 4 frames against the velocity out over the next 4: the larger of
   ``|v_out - v_in| / MIN_DV`` (a speed change, so a smash off a falling lob counts) and
   ``(1 - cos turn) * min speed * 4 / MIN_TURN`` for turns past 60 degrees. A score of 1 is a hit;
   peaks are kept greedily with an 8-frame minimum gap.
2. **Gap pass.** Across a raw visibility gap of 2-14 frames, compare 2-frame velocities either
   side; a hit is placed mid-gap.
3. **Serve pass.** The first 3-frame run moving at 2 x MIN_SPEED is a serve, once per window.
4. **Wrist pass.** A wrist moving at 10 px/frame or more with the raw shuttle within 60 px is a
   contact the shuttle alone missed.

Attribution is the nearest confident wrist (bbox centre if none) over the hit frame +-1.

Upstream tuned every constant on 1280x720 broadcast video at 30 fps. Here frame counts are
rescaled to the same durations and pixel thresholds to the same share of frame width, with
px/frame speeds scaled for both. That is the only adaptation; no constant was re-tuned.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

REF_FPS, REF_WIDTH = 30.0, 1280.0
DELTA, MIN_SPEED, MIN_DV, MIN_TURN, MIN_TURN_COS = 4, 3.0, 30.0, 20.0, 0.5
MIN_GAP, GAP_INTERP, GAP_MIN, GAP_MAX = 8, 6, 2, 14
WRIST_IDX, WRIST_CONF, WRIST_PK, WRIST_SH_PX = (9, 10), 0.3, 10.0, 60.0


class Scale:
    def __init__(self, fps: float, width: float) -> None:
        self.t = fps / REF_FPS        # frames per reference frame
        self.px = width / REF_WIDTH   # pixels per reference pixel

    def frames(self, ref: int) -> int:
        return max(1, round(ref * self.t))

    def speed(self, ref_px_per_frame: float) -> float:
        return ref_px_per_frame * self.px / self.t


def _series(ball: dict[int, tuple[float, float]], f0: int, f1: int) -> np.ndarray:
    xy = np.full((f1 - f0 + 1, 2), np.nan)
    for f, p in ball.items():
        if f0 <= f <= f1:
            xy[f - f0] = p
    return xy


def _interpolate(xy: np.ndarray, limit: int) -> np.ndarray:
    """pandas ``interpolate(limit=..., limit_area='inside')``: fill interior gaps up to limit."""
    out = xy.copy()
    seen = np.flatnonzero(~np.isnan(xy[:, 0]))
    for a, b in zip(seen, seen[1:]):
        if 1 < b - a <= limit + 1:
            for i in range(a + 1, b):
                out[i] = xy[a] + (xy[b] - xy[a]) * (i - a) / (b - a)
    return out


def _score(v_in, v_out, s: Scale, delta: int) -> float:
    sp_in, sp_out = float(np.linalg.norm(v_in)), float(np.linalg.norm(v_out))
    s_dv = float(np.linalg.norm(v_out - v_in)) / s.speed(MIN_DV)
    s_turn = 0.0
    if sp_in > s.speed(1.0) and sp_out > s.speed(1.0):
        cos = float(np.dot(v_in, v_out) / (sp_in * sp_out))
        if cos < MIN_TURN_COS:
            s_turn = (1.0 - cos) * min(sp_in, sp_out) * delta / (MIN_TURN * s.px)
    return max(s_dv, s_turn)


def wrists(player: dict[str, Any] | None) -> list[tuple[float, float]]:
    if not player:
        return []
    kps = player.get("keypoints") or []
    return [(kps[j][0], kps[j][1]) for j in WRIST_IDX if len(kps) > j and kps[j][2] > WRIST_CONF]


def detect_hits(
    ball: dict[int, tuple[float, float]],
    players: dict[int, dict[str, Any]],
    f0: int, f1: int, s: Scale, *, wrist_pass: bool = True,
) -> list[dict[str, Any]]:
    """Hit candidates in [f0, f1]. ``ball`` holds only samples we treat as shuttle."""
    raw = _series(ball, f0, f1)
    xy = _interpolate(raw, s.frames(GAP_INTERP))
    n, d, gap = len(xy), s.frames(DELTA), s.frames(MIN_GAP)
    score = np.zeros(n)
    for i in range(d, n - d):
        if np.isnan(xy[i - d: i + d + 1]).any():
            continue
        score[i] = _score((xy[i] - xy[i - d]) / d, (xy[i + d] - xy[i]) / d, s, d)
    hits: list[dict[str, Any]] = []
    taken = np.zeros(n, dtype=bool)
    for i in np.argsort(-score):
        if score[i] < 1.0:
            break
        if taken[max(0, i - gap): i + gap + 1].any():
            continue
        taken[i] = True
        hits.append({"frame": f0 + int(i), "x": float(xy[i, 0]), "y": float(xy[i, 1]),
                     "score": float(score[i]), "pass": "kink"})

    vis = ~np.isnan(raw[:, 0])
    seen = np.flatnonzero(vis)
    k = s.frames(2)
    for a, b in zip(seen, seen[1:]):
        if not (s.frames(GAP_MIN) <= b - a <= s.frames(GAP_MAX)) or a < k or b > n - k - 1:
            continue
        if not (vis[a - k: a + 1].all() and vis[b: b + k + 1].all()):
            continue
        sc = _score((raw[a] - raw[a - k]) / k, (raw[b + k] - raw[b]) / k, s, d)
        mid = (a + b) // 2
        if sc >= 1.0 and not any(abs(f0 + mid - h["frame"]) < gap for h in hits):
            hits.append({"frame": f0 + int(mid), "x": float((raw[a, 0] + raw[b, 0]) / 2),
                         "y": float((raw[a, 1] + raw[b, 1]) / 2), "score": sc, "pass": "gap"})

    step = np.linalg.norm(np.diff(raw, axis=0), axis=1)
    for i in range(len(step) - 2):
        run = step[i: i + 3]
        if not np.isnan(run).any() and (run >= 2 * s.speed(MIN_SPEED)).all():
            if not any(abs(f0 + i - h["frame"]) <= 2 * gap for h in hits):
                hits.append({"frame": f0 + i, "x": float(raw[i, 0]), "y": float(raw[i, 1]),
                             "score": float(run.sum()), "pass": "serve"})
            break

    if wrist_pass:
        hits += wrist_hits(ball, players, f0, f1, [h["frame"] for h in hits], s)
    return sorted(hits, key=lambda h: h["frame"])


def wrist_hits(
    ball: dict[int, tuple[float, float]], players: dict[int, dict[str, Any]],
    f0: int, f1: int, existing: list[int], s: Scale,
) -> list[dict[str, Any]]:
    """Pass 4 on its own, so it can also be laid over our detector's hits."""
    gap, peak, reach = s.frames(MIN_GAP), s.speed(WRIST_PK), WRIST_SH_PX * s.px
    found: list[dict[str, Any]] = []
    for slot in ("near", "far"):
        w = {f: wrists(players.get(f, {}).get(slot)) for f in range(f0, f1 + 1)}
        w = {f: p for f, p in w.items() if p}
        frames = sorted(w)
        for a, b in zip(frames, frames[1:]):
            if b - a > s.frames(2):
                continue
            speed = min(math.dist(p, q) for p in w[a] for q in w[b]) / (b - a)
            if speed < peak or any(abs(b - g) < gap for g in existing) \
                    or any(abs(b - h["frame"]) < gap for h in found):
                continue
            close = [
                (min(math.dist(ball[g], p) for p in w.get(b, []) + w.get(b - 1, [])), g)
                for g in range(b - s.frames(2), b + s.frames(2) + 1) if g in ball
            ]
            if close and min(close)[0] <= reach:
                dist, g = min(close)
                found.append({"frame": b, "x": ball[g][0], "y": ball[g][1],
                              "score": speed / peak, "pass": "wrist", "slot": slot})
    return found


def attribute(hit: dict[str, Any], players: dict[int, dict[str, Any]]) -> str | None:
    """Nearest confident wrist, else bbox centre, over the hit frame +-1."""
    best, best_d = None, math.inf
    for f in (hit["frame"] - 1, hit["frame"], hit["frame"] + 1):
        for slot in ("near", "far"):
            player = players.get(f, {}).get(slot)
            if not player:
                continue
            points = wrists(player)
            if not points:
                x1, y1, x2, y2 = player["bbox"]
                points = [((x1 + x2) / 2, (y1 + y2) / 2)]
            for p in points:
                dist = math.dist(p, (hit["x"], hit["y"]))
                if dist < best_d:
                    best, best_d = slot, dist
    return best
