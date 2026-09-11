"""Propose a court polygon and net line for a new clip from where people were seen standing.

Fitting these by eye off a coordinate grid is slow and the net's *floor* line is the part hardest
to read, because the net tape sits well above it and the far player's legs are occluded. Their
feet are not: a pose pass over the clip says where people actually stood, and the two players
separate cleanly by apparent size, since a box's height in frame is a direct read on depth.

Proposal only. It is written to configs/<clip>.proposed.json for a person to check against a
calibration frame, because nothing here can tell our court from the one next to it except by
assuming the near player never leaves ours.
"""
import json, statistics as st, sys
from pathlib import Path

def feet(clip):
    out = []
    for line in (Path("outputs") / clip / "players.jsonl").open():
        r = json.loads(line)
        W, H = r["width"], r["height"]
        seen = [r.get("near"), r.get("far")] + list(r.get("not_selected") or []) + \
               list(r.get("rejected") or [])
        for p in seen:
            b = p.get("bbox") if p else None
            if not b:
                continue
            # A box clipped by the frame edge is not short because it is far away, it is short
            # because we cannot see all of it. Spectators in the foreground are the usual case
            # and they sit exactly where the far player should be in the size ordering.
            if b[1] <= 2 or b[3] >= H - 2 or b[0] <= 2 or b[2] >= W - 2:
                continue
            out.append(((b[0] + b[2]) / 2 / W, b[3] / H, (b[3] - b[1]) / H))
    return out

def propose(clip):
    obs = feet(clip)
    if len(obs) < 200:
        raise SystemExit(f"{clip}: only {len(obs)} detections, too few to fit")
    heights = sorted(h for _, _, h in obs)
    tall_cut = heights[int(0.80 * len(heights))]          # the near player dominates the big end
    near = [(x, y) for x, y, h in obs if h >= tall_cut]
    nx = sorted(x for x, _ in near); ny = sorted(y for _, y in near)
    x_lo, x_hi = nx[int(0.01 * len(nx))], nx[int(0.99 * len(nx))]
    near_front = ny[int(0.02 * len(ny))]

    # Far-sized people inside the near player's own x span: our opponent, plus whoever is at the
    # same depth on a neighbouring court, which is why the x span is taken from the near player.
    far = [(x, y) for x, y, h in obs
           if x_lo - 0.05 <= x <= x_hi + 0.05 and 0.35 * tall_cut <= h < 0.75 * tall_cut]
    if not far:
        raise SystemExit(f"{clip}: found nobody at far-court depth")
    fy = sorted(y for _, y in far); fx = sorted(x for x, _ in far)
    far_back = fy[int(0.95 * len(fy))]
    if far_back >= near_front:
        raise SystemExit(
            f"{clip}: no clean gap between the two depths -- the deepest far-court feet sit at "
            f"y={far_back:.3f} and the nearest near-player feet at y={near_front:.3f}. Something "
            f"other than the two players is being taken for one of them, so fit this clip by "
            f"hand rather than trusting a number from here.")
    net_y = round((far_back + near_front) / 2, 3)
    f_lo, f_hi = fx[int(0.02 * len(fx))], fx[int(0.98 * len(fx))]
    top = round(fy[int(0.02 * len(fy))] - 0.02, 3)
    bottom = round(min(0.995, ny[int(0.995 * len(ny))] + 0.02), 3)
    pad = 0.04
    court = [[round(max(0.0, f_lo - pad), 3), top], [round(min(1.0, f_hi + pad), 3), top],
             [round(min(1.0, x_hi + 2 * pad), 3), bottom], [round(max(0.0, x_lo - 2 * pad), 3), bottom]]
    air_lo = round(max(0.0, min(f_lo, x_lo) - 0.06), 3)
    air_hi = round(min(1.0, max(f_hi, x_hi) + 0.06), 3)
    return {
        "court": {
            "polygon": court,
            "net_line": [[court[0][0], net_y], [court[1][0], net_y]],
            "shuttle_polygon": [[air_lo, 0.0], [air_hi, 0.0], [air_hi, 1.0], [air_lo, 1.0]],
        },
        "_evidence": {
            "detections": len(obs),
            "near_player_x_span": [round(x_lo, 3), round(x_hi, 3)],
            "near_player_front_foot_y": round(near_front, 3),
            "far_depth_back_foot_y": round(far_back, 3),
            "net_line_y": net_y,
            "gap_the_net_line_sits_in": [round(far_back, 3), round(near_front, 3)],
            "warning": ("The x span assumes the near player stays on our court. On a multi-court "
                        "venue check shuttle_polygon against a calibration frame before trusting "
                        "any result from this clip."),
        },
    }

for clip in sys.argv[1:]:
    p = propose(clip)
    ev = p.pop("_evidence")
    base = json.loads(Path("configs/test-video2.json").read_text())
    base["court"] = p["court"]
    base["notes"] = (f"Court proposed by tools/propose_court.py from {ev['detections']} pose "
                     f"detections. Net line at y={ev['net_line_y']}, fitted into the gap "
                     f"{ev['gap_the_net_line_sits_in']} between the deepest far-court feet and "
                     f"the nearest near-player feet. {ev['warning']}")
    out = Path("configs") / f"{clip}.proposed.json"
    out.write_text(json.dumps(base, indent=2))
    print(f"{clip}: net y {ev['net_line_y']}, gap {ev['gap_the_net_line_sits_in']}, "
          f"court x {p['court']['polygon'][3][0]}-{p['court']['polygon'][2][0]} -> {out}")
