"""Phase G: build the Milestone 2 dataset from hand-verified contacts, and report it.

Every sample comes from a contact a human confirmed. Nothing here reads a detector's hits, a
model's predictions, or a model-produced label file, and a contact that cannot be used is
recorded with the reason rather than dropped quietly.

Writes one shot_samples.jsonl per clip and one dataset report over all of them.

    python tools/shot_dataset_report.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from badminton_poc.config import PipelineConfig  # noqa: E402
from badminton_poc.shots.runner import load_clip, write_samples  # noqa: E402

BROAD = ["smash", "drop", "clear", "lift", "drive", "net", "serve", "other", "uncertain"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", default="test-video2,test-video3,test-video5")
    ap.add_argument("--out", default="outputs/bst_reference/shot_dataset.json")
    args = ap.parse_args()

    clips = [c.strip() for c in args.clips.split(",") if c.strip()]
    per_clip, all_rows, excluded = {}, [], []

    for clip in clips:
        cfg = PipelineConfig.load(str(ROOT / "configs" / f"{clip}.json"))
        shots = load_clip(clip, output_root=ROOT / "outputs", config=cfg, source="truth")
        write_samples(shots)

        rows = []
        for sample, feature in zip(shots.samples, shots.features):
            cover = feature.coverage()
            rows.append({
                "clip": clip,
                "shot_id": sample.shot_id,
                "rally": sample.rally,
                "hit_frame": sample.hit_frame,
                "hit_time_s": sample.hit_time_s,
                "hitter": sample.hitter,
                "shot_type": sample.shot_type,
                "is_first_in_rally": sample.is_first_in_rally,
                "window_frames": [sample.start_frame, sample.end_frame],
                "window_seconds": round(sample.seconds, 3),
                "samples_in_window": feature.n_samples,
                "hitter_joint_coverage": round(cover["hitter_joints"], 3),
                "shuttle_coverage": round(cover["shuttle"], 3),
            })
        all_rows += rows
        for s in shots.skipped:
            excluded.append({**s, "clip": clip})

        labelled = [r for r in rows if r["shot_type"]]
        per_clip[clip] = {
            "shots_built": len(rows),
            "with_a_shot_type": len(labelled),
            "without_a_shot_type": len(rows) - len(labelled),
            "excluded": len([s for s in shots.skipped]),
            "by_class": dict(Counter(r["shot_type"] for r in labelled).most_common()),
            "by_hitter": dict(Counter(r["hitter"] for r in rows).most_common()),
            "rallies": len({r["rally"] for r in rows}),
            "median_window_seconds": round(
                sorted(r["window_seconds"] for r in rows)[len(rows) // 2], 3) if rows else None,
            "median_hitter_joint_coverage": round(
                sorted(r["hitter_joint_coverage"] for r in rows)[len(rows) // 2], 3)
            if rows else None,
            "median_shuttle_coverage": round(
                sorted(r["shuttle_coverage"] for r in rows)[len(rows) // 2], 3) if rows else None,
        }

    labelled = [r for r in all_rows if r["shot_type"]]
    serves = [r for r in labelled if r["shot_type"] == "serve"]
    by_class = Counter(r["shot_type"] for r in labelled)
    report = {
        "_what": "Phase G: the Milestone 2 dataset, built only from hand-verified racket "
                 "contacts. No detector hits, no model-produced labels, no predictions.",
        "clips": per_clip,
        "totals": {
            "shots_built": len(all_rows),
            "with_a_shot_type": len(labelled),
            "without_a_shot_type": len(all_rows) - len(labelled),
            "excluded": len(excluded),
            "recordings_with_labels": len([c for c, v in per_clip.items()
                                           if v["with_a_shot_type"]]),
        },
        "by_class": dict(by_class.most_common()),
        "by_hitter": dict(Counter(r["hitter"] for r in labelled).most_common()),
        "class_balance": {
            "largest_class": by_class.most_common(1)[0] if by_class else None,
            "always_predict_largest": round(
                by_class.most_common(1)[0][1] / len(labelled), 4) if labelled else None,
            "classes_with_fewer_than_5": {k: v for k, v in by_class.items() if v < 5},
        },
        "serves": {
            "count": len(serves),
            "note": "Serve is derivable from rally structure (is_first_in_rally) and is not "
                    "evidence that a visual classifier works. Reported separately and excluded "
                    "from the headline figures in Phase H.",
            "all_first_in_rally": all(r["is_first_in_rally"] for r in serves),
        },
        "uncertain": len([r for r in labelled if r["shot_type"] == "uncertain"]),
        "excluded_detail": excluded,
        "shots": all_rows,
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n")

    print(f"{'clip':14} {'shots':>6} {'labelled':>9} {'rallies':>8} {'excluded':>9}  classes")
    for clip, v in per_clip.items():
        print(f"{clip:14} {v['shots_built']:6d} {v['with_a_shot_type']:9d} {v['rallies']:8d} "
              f"{v['excluded']:9d}  {v['by_class']}")
    print()
    print("totals:", json.dumps(report["totals"]))
    print("by class:", json.dumps(report["by_class"]))
    print("by hitter:", json.dumps(report["by_hitter"]))
    print("always-predict-largest baseline:", report["class_balance"]["always_predict_largest"])
    print("classes with fewer than 5 examples:", report["class_balance"]["classes_with_fewer_than_5"])
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
