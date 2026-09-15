"""Turn a hand-written labels.tsv into a truth file.

Typing JSON by hand is a poor use of a labeller's attention, so labels are written as one line
per contact and converted here. The conversion is deliberately dumb: it does not consult the
detector, does not fill in a hitter that was left as "?", and does not move a frame number to
where something was detected.

    python tools/ingest_labels.py --clip test-video5 --labels outputs/test-video5/labels.tsv \
        --span 1674 2462 --span 3501 3884
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from badminton_poc.truth import empty_truth  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--span", nargs=2, type=int, action="append", metavar=("START", "END"),
                    required=True, help="a span labelled exhaustively; repeat for more")
    ap.add_argument("--keep-rallies-from", help="a truth file whose rallies should be reused")
    ap.add_argument("--output-root", default="outputs")
    args = ap.parse_args()

    clip_dir = Path(args.output_root) / args.clip
    fps = float(json.loads((clip_dir / "metadata.json").read_text())["working"]["fps"])

    contacts, rallies, problems = [], [], []
    for number, raw in enumerate(Path(args.labels).read_text().splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        kind = parts[0].lower()
        try:
            if kind == "c":
                frame, hitter = int(parts[1]), parts[2].lower()
                if hitter not in ("near", "far", "?"):
                    problems.append(f"line {number}: hitter must be near, far or ? -- got {hitter!r}")
                    continue
                contacts.append({"frame": frame,
                                 "hitter": None if hitter == "?" else hitter,
                                 "notes": " ".join(parts[3:])})
            elif kind == "r":
                rallies.append({"start_frame": int(parts[1]), "end_frame": int(parts[2]),
                                "notes": " ".join(parts[3:])})
            else:
                problems.append(f"line {number}: expected a line starting c or r -- got {kind!r}")
        except (IndexError, ValueError):
            problems.append(f"line {number}: could not read {raw.strip()!r}")

    spans = [{"start_frame": a, "end_frame": b} for a, b in args.span]
    stray = [c["frame"] for c in contacts
             if not any(s["start_frame"] <= c["frame"] <= s["end_frame"] for s in spans)]
    if stray:
        problems.append(f"contacts outside every labelled span: {sorted(stray)}")
    frames = [c["frame"] for c in contacts]
    twice = sorted({f for f in frames if frames.count(f) > 1})
    if twice:
        problems.append(f"two contacts on one frame: {twice}")

    if problems:
        print("Nothing written. Fix these first:")
        for p in problems:
            print("   " + p)
        return 1

    truth = empty_truth(args.clip, fps)
    truth["labelled"] = True
    truth["contacts"] = sorted(contacts, key=lambda c: c["frame"])
    truth["labelled_contact_spans"] = spans
    if rallies:
        truth["rallies"] = [{"rally": i, **r} for i, r in enumerate(sorted(
            rallies, key=lambda r: r["start_frame"]), start=1)]
    elif args.keep_rallies_from:
        truth["rallies"] = json.loads(Path(args.keep_rallies_from).read_text())["rallies"]

    out = clip_dir / "truth.json"
    out.write_text(json.dumps(truth, indent=1))
    known = sum(1 for c in contacts if c["hitter"])
    print(f"{out}: {len(contacts)} contacts ({known} with a hitter, {len(contacts) - known} left "
          f"as ?), {len(truth['rallies'])} rallies, {len(spans)} labelled spans")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
