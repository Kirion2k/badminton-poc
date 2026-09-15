"""Merge hand-written shot types into a clip's truth file.

Milestone 1 already knows where each contact is and who hit it. This adds only what stroke it
was, keyed by frame, and refuses anything it cannot match rather than guessing.

Deliberately dumb, like the contact ingest beside it: it does not consult a classifier, does not
fill in a blank, and does not move a frame to where something was detected.

    python tools/ingest_shot_types.py --clip test-video2 \
        --labels outputs/test-video2/shot_labels.tsv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SHOT_TYPES = {"smash", "drop", "clear", "lift", "drive", "net", "serve", "other", "uncertain"}
ROOT = Path(__file__).resolve().parent.parent


def parse(text: str) -> tuple[dict[int, tuple[str, str]], list[str]]:
    out, problems = {}, []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if parts[0].lower() != "t":
            problems.append(f"line {number}: expected a line starting t -- got {parts[0]!r}")
            continue
        if len(parts) < 3:
            problems.append(f"line {number}: no shot type given for frame {parts[1:] or '?'}")
            continue
        try:
            frame = int(parts[1])
        except ValueError:
            problems.append(f"line {number}: {parts[1]!r} is not a frame number")
            continue
        shot = parts[2].lower()
        if shot not in SHOT_TYPES:
            problems.append(
                f"line {number}: {shot!r} is not one of {' '.join(sorted(SHOT_TYPES))}")
            continue
        if frame in out:
            problems.append(f"line {number}: frame {frame} appears twice")
            continue
        out[frame] = (shot, " ".join(parts[3:]))
    return out, problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--output-root", default="outputs")
    args = ap.parse_args()

    truth_path = Path(args.output_root) / args.clip / "truth.json"
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    labels, problems = parse(Path(args.labels).read_text(encoding="utf-8"))

    by_frame = {int(c["frame"]): c for c in truth["contacts"]}
    unknown = sorted(f for f in labels if f not in by_frame)
    if unknown:
        problems.append(
            f"no contact at frame(s) {', '.join(map(str, unknown))} in {truth_path}. Shot types "
            "attach to contacts a human already confirmed; they cannot introduce new ones."
        )
    if problems:
        print("Nothing written. Fix these first:")
        for p in problems:
            print("   " + p)
        return 1

    for frame, (shot, note) in labels.items():
        contact = by_frame[frame]
        contact["shot_type"] = shot
        if note:
            contact["shot_type_note"] = note
    truth["_shot_type_provenance"] = (
        f"Shot types hand-written by the project owner in {args.labels}, one line per contact, "
        "from preview clips that carry the frame, the rally and the hitter and no model output "
        "of any kind. Keyed to contacts Milestone 1's hand labels already confirmed; this pass "
        "added no contacts and moved no frames. Nothing in Milestone 1 reads the field."
    )
    truth_path.write_text(json.dumps(truth, indent=1))

    missing = [f for f in by_frame if f not in labels]
    from collections import Counter
    print(f"{truth_path}: {len(labels)} shot types merged into {len(by_frame)} contacts")
    print("  by class:", dict(Counter(s for s, _ in labels.values()).most_common()))
    if missing:
        print(f"  {len(missing)} contacts still have no shot type: "
              f"{', '.join(map(str, sorted(missing)[:12]))}"
              f"{' ...' if len(missing) > 12 else ''}")
    print("\nNow re-run: .venv/bin/python tools/freeze_milestone1.py   (the truth file changed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
