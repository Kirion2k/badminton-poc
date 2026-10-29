"""Turn kwyoke's amateur-singles hit labels into our truth files, for external evaluation.

Source: the am_singles annotations of github.com/kwyoke/Badminton-hit-detection (Google Drive,
linked from its README). The repository carries no licence, so the data and anything derived
from it stay under data/input/ (gitignored) and are never committed.

Their ``<rally>.mp4_player_hit.csv`` has one row per frame: ``player_hit`` is 0, 1 or 2, and
``rally_bound`` marks a rally's start and/or end. 1 is the near player and 2 the far one --
their notebook's class_vocab says so, and frames 80 and 117 of match_china2/singles0 were
checked by eye. A hit is a single frame.

The conversion does not consult our detector. A rally is the whole clip, from the first
labelled hit to the last rally_bound after it (the clip's end if there is none).

    python tools/ingest_kwyoke.py --clip kw-china2-singles0 \\
        --labels data/input/kwyoke/am_singles/match_china2/player_hit/singles0.mp4_player_hit.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from badminton_poc.truth import empty_truth  # noqa: E402

HITTER = {"1": "near", "2": "far"}


def convert(rows: list[dict[str, str]], clip: str, fps: float, frame_count: int) -> dict:
    if len(rows) != frame_count:
        raise ValueError(f"{len(rows)} label rows for a {frame_count}-frame working video: "
                         "frames would not line up")
    contacts = [{"frame": int(r["frame"]), "hitter": HITTER[r["player_hit"]], "notes": ""}
                for r in rows if r["player_hit"] != "0"]
    if not contacts:
        raise ValueError("no hits labelled")
    bounds = [int(r["frame"]) for r in rows if r.get("rally_bound", "0") != "0"]
    end = max([b for b in bounds if b > contacts[-1]["frame"]], default=frame_count - 1)
    truth = empty_truth(clip, fps)
    truth.update({
        "labelled": True,
        "_provenance": ("EXTERNAL HAND LABELS by the authors of kwyoke/Badminton-hit-detection "
                        "(am_singles), converted by tools/ingest_kwyoke.py. Independent of this "
                        "pipeline. No licence: not for redistribution."),
        "contacts": contacts,
        "rallies": [{"rally": 1, "start_frame": contacts[0]["frame"], "end_frame": end,
                     "notes": "the whole clip is one labelled rally"}],
    })
    return truth


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out-dir", default="data/input/kwyoke/truth")
    args = ap.parse_args()
    meta = json.loads((Path("outputs") / args.clip / "metadata.json").read_text())["working"]
    if meta.get("source_frame_offset"):
        raise SystemExit("working video is trimmed; label frames would be offset")
    rows = list(csv.DictReader(open(args.labels, newline="")))
    truth = convert(rows, args.clip, float(meta["fps"]), int(meta["frame_count"]))
    out = Path(args.out_dir) / f"{args.clip}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(truth, indent=1))
    print(f"{out}: {len(truth['contacts'])} contacts, rally {truth['rallies'][0]['start_frame']}"
          f"-{truth['rallies'][0]['end_frame']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
