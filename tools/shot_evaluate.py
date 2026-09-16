"""Phase H: score the pretrained BST checkpoint against our hand-labelled contacts.

No training, no fine-tuning, no threshold fitting. The checkpoint is the authors' own, the
contacts are the ones a human confirmed, and the shot types are the labeller's own words.

Serve is reported separately throughout. It is derivable from rally structure -- it is the first
contact of a rally -- so counting it towards a visual classifier's score would flatter the model
for something the pipeline already knows.

    python tools/shot_evaluate.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from badminton_poc.config import PipelineConfig  # noqa: E402
from badminton_poc.shots.baseline import predict as predict_rules  # noqa: E402
from badminton_poc.shots.bst_adapter import BSTConfig, predict  # noqa: E402
from badminton_poc.shots.runner import load_clip  # noqa: E402
from badminton_poc.shots.taxonomy import MAPPING_VERSION  # noqa: E402


def score(rows: list[dict]) -> dict:
    """Accuracy, balanced accuracy, macro F1, top-2, and the per-class table."""
    if not rows:
        return {"n": 0}
    truth = [r["true"] for r in rows]
    pred = [r["pred"] for r in rows]
    classes = sorted(set(truth) | set(pred))
    per = {}
    recalls = []
    for c in classes:
        tp = sum(1 for t, p in zip(truth, pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(truth, pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(truth, pred) if t == c and p != c)
        support = sum(1 for t in truth if t == c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per[c] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
                  "support": support, "predicted": tp + fp}
        if support:
            recalls.append(rec)
    scored = [c for c in classes if per[c]["support"]]
    confusion = defaultdict(int)
    for t, p in zip(truth, pred):
        if t != p:
            confusion[f"{t} -> {p}"] += 1
    return {
        "n": len(rows),
        "accuracy": round(sum(t == p for t, p in zip(truth, pred)) / len(rows), 4),
        "balanced_accuracy": round(sum(recalls) / len(recalls), 4) if recalls else None,
        "macro_f1": round(sum(per[c]["f1"] for c in scored) / len(scored), 4) if scored else None,
        "top2_accuracy": round(sum(r["top2_hit"] for r in rows) / len(rows), 4),
        "per_class": per,
        "confusion_pairs": dict(sorted(confusion.items(), key=lambda kv: -kv[1])),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # Clip 2 joined on 2026-09-16, when its 58 contacts got shot types. It was excluded
    # before only because it had none, not because anything was wrong with it -- though
    # its labels are the weakest of the three and the manifest says why.
    ap.add_argument("--clips", default="test-video2,test-video3,test-video5")
    ap.add_argument("--out", default="outputs/bst_reference/phase_h.json")
    args = ap.parse_args()

    bst = BSTConfig()
    rows, unlabelled, uncertain = [], 0, 0
    for clip in [c.strip() for c in args.clips.split(",") if c.strip()]:
        cfg = PipelineConfig.load(str(ROOT / "configs" / f"{clip}.json"))
        shots = load_clip(clip, output_root=ROOT / "outputs", config=cfg, source="truth")
        # "uncertain" is a label saying the labeller could not tell, which the labelling sheet
        # invites in place of a guess. Scoring it would punish the model for failing to
        # reproduce an answer no human was willing to give, so it is dropped here and counted.
        # Clips 3 and 5 carry none; clip 2 carries two, which is why this had never mattered.
        labelled = [(s, f) for s, f in zip(shots.samples, shots.features)
                    if s.shot_type and s.shot_type != "uncertain"]
        uncertain += sum(1 for s in shots.samples if s.shot_type == "uncertain")
        unlabelled += sum(1 for s in shots.samples if not s.shot_type)
        if not labelled:
            continue
        feats = [f for _, f in labelled]
        preds = predict(feats, cfg=bst, base=ROOT)
        rules = predict_rules(feats, depth=shots.depth)
        for (sample, feature), p, rb in zip(labelled, preds, rules):
            broad = p["broad_scores"] if "broad_scores" in p else p.get("scores", {})
            ordered = sorted(broad.items(), key=lambda kv: -kv[1])
            top2 = [k for k, _ in ordered[:2]]
            rows.append({
                "clip": clip,
                "shot_id": sample.shot_id,
                "rally": sample.rally,
                "hit_frame": sample.hit_frame,
                "hitter": sample.hitter,
                "true": sample.shot_type,
                "pred": p["broad_prediction"] if "broad_prediction" in p else top2[0],
                "top2": top2,
                "top2_hit": sample.shot_type in top2,
                "upstream_prediction": p["source_prediction"],
                "upstream_score": round(max(p["source_scores"].values()), 4),
                "upstream_top3": p["source_top3"],
                "slot_agrees_with_milestone1": p["source_slot_agrees_with_milestone1"],
                "broad_scores": {k: round(v, 4) for k, v in ordered},
                "baseline_pred": rb["broad_prediction"],
                "baseline_rule": rb["rule"],
                "window_seconds": round(sample.seconds, 3),
                "hitter_joint_coverage": round(feature.coverage()["hitter_joints"], 3),
                "shuttle_coverage": round(feature.coverage()["shuttle"], 3),
            })

    non_serve = [r for r in rows if r["true"] != "serve"]
    base_rows = [{**r, "pred": r["baseline_pred"], "top2_hit": r["true"] == r["baseline_pred"]}
                 for r in non_serve]
    report = {
        "_what": "Phase H: pretrained BST against hand-labelled contacts. No training, no "
                 "fine-tuning, no threshold fitting.",
        "checkpoint": Path(bst.weight).name,
        "architecture": "BST_0",
        "mapping_version": MAPPING_VERSION,
        "shots_scored": len(rows),
        "shots_without_a_label_skipped": unlabelled,
        "shots_labelled_uncertain_skipped": uncertain,
        "uncertain_note": "Contacts the labeller marked uncertain rather than guessing. They "
                          "are recorded in the truth file and left out of scoring, so they "
                          "neither flatter nor penalise the model.",
        "headline_excludes_serve": True,
        "serve_note": "Serve is the first contact of a rally and derivable from rally structure, "
                      "so it is scored separately rather than counted as evidence the visual "
                      "classifier works.",
        "overall_including_serve": score(rows),
        "overall": score(non_serve),
        "by_clip": {c: score([r for r in non_serve if r["clip"] == c])
                    for c in sorted({r["clip"] for r in non_serve})},
        "by_hitter": {h: score([r for r in non_serve if r["hitter"] == h])
                      for h in sorted({r["hitter"] for r in non_serve})},
        "serve_only": score([r for r in rows if r["true"] == "serve"]),
        "baseline_overall": score(base_rows),
        "baseline_by_clip": {c: score([r for r in base_rows if r["clip"] == c])
                             for c in sorted({r["clip"] for r in base_rows})},
        "baseline_note": "Phase J: hand-written badminton geometry, no learning. Its top-2 "
                         "column is meaningless -- the rules make one hard decision -- and is "
                         "reported only so the table lines up.",
        "milestone1_hitter_agreement": round(
            sum(r["slot_agrees_with_milestone1"] for r in rows) / len(rows), 4) if rows else None,
        "predictions": rows,
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n")

    o = report["overall"]
    print(f"checkpoint {report['checkpoint']}")
    print(f"scored {o['n']} shots (serve excluded; {len(rows) - o['n']} serves reported apart)\n")
    print(f"  accuracy           {o['accuracy']}")
    print(f"  balanced accuracy  {o['balanced_accuracy']}")
    print(f"  macro F1           {o['macro_f1']}")
    print(f"  top-2 accuracy     {o['top2_accuracy']}\n")
    print(f"  {'class':8} {'prec':>6} {'rec':>6} {'F1':>6} {'support':>8}")
    for c, v in sorted(o["per_class"].items(), key=lambda kv: -kv[1]["support"]):
        print(f"  {c:8} {v['precision']:6.3f} {v['recall']:6.3f} {v['f1']:6.3f} {v['support']:8d}")
    print("\n  most confused:")
    for pair, n in list(o["confusion_pairs"].items())[:8]:
        print(f"    {pair:26} {n}")
    print("\n  by clip:  ", {c: v["accuracy"] for c, v in report["by_clip"].items()})
    print("  by hitter:", {h: v["accuracy"] for h, v in report["by_hitter"].items()})
    print(f"  serve only: {report['serve_only'].get('accuracy')} over "
          f"{report['serve_only'].get('n')}")
    print(f"  BST's own side prediction agrees with Milestone 1's hitter: "
          f"{report['milestone1_hitter_agreement']}")
    b = report["baseline_overall"]
    print(f"\n  --- Phase J, the rule baseline on the same {b['n']} shots ---")
    print(f"  accuracy {b['accuracy']}   balanced {b['balanced_accuracy']}   "
          f"macro F1 {b['macro_f1']}")
    print("  by clip: ", {c: v["accuracy"] for c, v in report["baseline_by_clip"].items()})
    print(f"  BST minus baseline: {round(o['accuracy'] - b['accuracy'], 4):+}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
