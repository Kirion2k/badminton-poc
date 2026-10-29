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

# Optional fourth field on a contact line. Nothing in Milestone 1 reads it; it exists so a shot
# type can be captured in the same pass rather than costing a second viewing of the same video.
SHOT_TYPES = {"smash", "drop", "clear", "lift", "drive", "net", "serve", "other", "uncertain"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--span", nargs=2, type=int, action="append", metavar=("START", "END"),
                    help="a span labelled exhaustively; repeat for more. Omit when the whole "
                         "clip was gone through, which is what an empty list means.")
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
            if kind in ("c", "k", "u"):
                frame, hitter = int(parts[1]), parts[2].lower()
                if hitter not in ("near", "far", "?"):
                    problems.append(f"line {number}: hitter must be near, far or ? -- got {hitter!r}")
                    continue
                rest = parts[3:]
                shot = None
                if rest and rest[0].lower() in SHOT_TYPES:
                    shot = rest[0].lower()
                    rest = rest[1:]
                entry = {"frame": frame,
                         "hitter": None if hitter == "?" else hitter,
                         "notes": " ".join(([shot] if shot else []) + rest)}
                if shot:
                    # Recorded as its own field as well as in the notes, so a later shot-type
                    # experiment can read it without parsing prose. Nothing in Milestone 1 reads it.
                    entry["shot_type"] = shot
                if kind == "k":
                    # A real racket contact that is not a rally shot: handing the shuttle back,
                    # a knock-up between points. Scored as neither a hit nor a miss.
                    entry["in_rally"] = False
                if kind == "u":
                    # Maybe a contact; the video does not show it. Scored as neither hit nor miss.
                    entry["uncertain"] = True
                contacts.append(entry)
            elif kind == "r":  # a rally span
                rallies.append({"start_frame": int(parts[1]), "end_frame": int(parts[2]),
                                "notes": " ".join(parts[3:])})
            else:
                problems.append(
                    f"line {number}: expected a line starting c, k, u or r -- got {kind!r}")
        except (IndexError, ValueError):
            problems.append(f"line {number}: could not read {raw.strip()!r}")

    spans = [{"start_frame": a, "end_frame": b} for a, b in (args.span or [])]
    stray = [c["frame"] for c in contacts
             if spans and not any(s["start_frame"] <= c["frame"] <= s["end_frame"] for s in spans)]
    if not contacts:
        problems.append("no contact lines: refusing to write a labelled truth file with nothing in it")
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
    truth["_provenance"] = (
        f"HAND LABELS, written by the project owner as one line per contact in {args.labels}, "
        "which is kept beside this file as the primary record; this file is derived from it. "
        "Labelled against a video carrying frame numbers and nothing else -- no boxes, no hit "
        "markers, no rally spans, no classifier output -- so that the labelling is independent "
        "of anything the pipeline decided."
    )
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
