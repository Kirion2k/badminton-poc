"""Score COURTSIDE's contact detector and attribution beside ours, on the hand-labelled clips.

Every variant is scored by the project's own ``score_hit_detection`` / ``score_hitter_assignment``
over the same windows, so the numbers differ only by the detector:

    ours                 the frozen pipeline's hits and hitters (analysis.json)
    courtside            kink + gap + serve + wrist passes, nearest-wrist attribution
    courtside_no_wrist   the same without the wrist pass
    ours+wrist           our hits plus COURTSIDE's wrist-pass recoveries
    ours/cs_attribution  our hits, attributed by nearest wrist instead of assign_hit

Two window sets. ``labelled`` is what COURTSIDE reports on: each labelled rally +-20 reference
frames, so rally segmentation is given. ``detected`` is our own detected rallies, which is what
a real run would have.

    python tools/benchmark_courtside.py --analysis-dir <dir with <clip>.json> [--out report.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from badminton_poc.analyze import mask_offcourt_shuttle  # noqa: E402
from badminton_poc.config import PipelineConfig  # noqa: E402
from badminton_poc.experiments import courtside as cs  # noqa: E402
from badminton_poc.parsing import load_ball_csv, load_jsonl  # noqa: E402
from badminton_poc.truth import score_hit_detection, score_hitter_assignment  # noqa: E402

CLIPS = ("test-video2", "test-video3", "test-video5")


def intersect(windows, spans):
    if not spans:
        return windows
    out = []
    for a, b in windows:
        for s in spans:
            lo, hi = max(a, s["start_frame"]), min(b, s["end_frame"])
            if lo <= hi:
                out.append((lo, hi))
    return out


def run_clip(clip: str, analysis: dict) -> dict:
    root = Path("outputs") / clip
    meta = json.loads((root / "metadata.json").read_text())["working"]
    fps, width, height = float(meta["fps"]), int(meta["width"]), int(meta["height"])
    config = PipelineConfig.load(f"configs/{clip}.json")
    truth = json.loads((root / "truth.json").read_text())
    ball, _ = mask_offcourt_shuttle(
        load_ball_csv(root / "tracknet/work_ball.csv", (width, height)), config.court, width, height)
    fixed = analysis.get("shuttle_fixed_points") or []
    tol = config.flight.fixed_point_tolerance_px * width / config.flight.reference_width
    shuttle = {s.frame: (s.x, s.y) for s in ball if s.visible
               and not any(math.hypot(s.x - p["x"], s.y - p["y"]) <= tol for p in fixed)}
    players = {int(r["frame"]): r for r in load_jsonl(root / "players.jsonl")}
    scale = cs.Scale(fps, width)
    pad = scale.frames(20)
    last = max(players)

    ours = {e["frame"]: e.get("hitter") for e in analysis["events"] if e.get("event_type") == "hit"}
    contacts = truth["contacts"]
    spans = truth.get("labelled_contact_spans") or []
    window_sets = {
        "labelled": [(max(0, r["start_frame"] - pad), min(last, r["end_frame"] + pad))
                     for r in truth["rallies"]],
        "detected": [(r["start_frame"], r["end_frame"]) for r in analysis["rallies"]],
    }
    out = {}
    for name, windows in window_sets.items():
        windows = intersect(windows, spans)
        span_dicts = [{"start_frame": a, "end_frame": b} for a, b in windows]
        full, no_wrist = [], []
        for a, b in windows:
            full += cs.detect_hits(shuttle, players, a, b, scale)
            no_wrist += cs.detect_hits(shuttle, players, a, b, scale, wrist_pass=False)
        recovered = []
        for a, b in windows:
            recovered += cs.wrist_hits(shuttle, players, a, b, [f for f in ours if a <= f <= b], scale)
        variants = {
            "ours": ours,
            "courtside": {h["frame"]: cs.attribute(h, players) for h in full},
            "courtside_no_wrist": {h["frame"]: cs.attribute(h, players) for h in no_wrist},
            "ours+wrist": {**ours, **{h["frame"]: h["slot"] for h in recovered}},
            # A hit with no shuttle sample on its own frame is left unassigned, not guessed.
            "ours/cs_attribution": {
                f: cs.attribute({"frame": f, "x": shuttle[f][0], "y": shuttle[f][1]}, players)
                if f in shuttle else None
                for f in ours
            },
        }
        out[name] = {}
        for variant, hitters in variants.items():
            det = score_hit_detection(hitters.keys(), contacts, fps=fps, spans=span_dicts)
            hit = score_hitter_assignment(
                {f: h for f, h in hitters.items() if any(a <= f <= b for a, b in windows)},
                contacts, fps=fps)
            out[name][variant] = {k: det[k] for k in ("precision", "recall", "f1",
                                                     "true_positives", "false_positives",
                                                     "false_negatives")}
            out[name][variant].update(hitter_accuracy=hit["accuracy"], hitter_scored=hit["scored"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analysis-dir", required=True,
                    help="directory holding a fresh <clip>.json analysis for each clip")
    ap.add_argument("--out")
    args = ap.parse_args()
    report = {}
    for clip in CLIPS:
        analysis = json.loads((Path(args.analysis_dir) / f"{clip}.json").read_text())
        report[clip] = run_clip(clip, analysis)
        for window, rows in report[clip].items():
            for variant, m in rows.items():
                print(f"{clip:12} {window:9} {variant:20} P {m['precision']:.3f} R {m['recall']:.3f} "
                      f"F1 {m['f1']:.3f}  hitter {m['hitter_accuracy']} (n={m['hitter_scored']})")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
