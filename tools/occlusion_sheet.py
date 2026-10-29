"""Show the longest empty-slot gaps so the visibility states can be checked by eye.

One row per gap: the last frame the slot was observed (its box in green), the middle of the gap
and the end of the gap (that last box in red, the other player's current box in yellow). If the
middle panel shows the player behind the yellow box, ``occluded`` was right.

This draws pipeline output, so it is a check on the diagnostic, never labelling material.

    python tools/occlusion_sheet.py --clip test-video6 --slot far --out sheet.jpg
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from badminton_poc.visibility import summarise


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--slot", choices=("near", "far"), default="far")
    ap.add_argument("--gaps", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    clip = Path("outputs") / args.clip
    rows = {r["frame"]: r for r in map(json.loads, (clip / "players.jsonl").open())}
    other = "near" if args.slot == "far" else "far"
    fps = cv2.VideoCapture(str(clip / "work.mp4")).get(cv2.CAP_PROP_FPS)
    gaps = summarise(list(rows.values()), fps, examples=args.gaps)[args.slot]["longest_gaps"]

    video = cv2.VideoCapture(str(clip / "work.mp4"))
    strips = []
    for gap in gaps:
        last_seen = gap["start_frame"] - 1
        box = rows[last_seen][args.slot]["bbox"] if rows.get(last_seen, {}).get(args.slot) else None
        panels = []
        for frame in (last_seen, (gap["start_frame"] + gap["end_frame"]) // 2, gap["end_frame"]):
            video.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, image = video.read()
            if not ok:
                image = np.zeros((360, 640, 3), np.uint8)
            if box:
                colour = (0, 200, 0) if frame == last_seen else (0, 0, 255)
                cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), colour, 3)
            if frame != last_seen and rows.get(frame, {}).get(other):
                b = rows[frame][other]["bbox"]
                cv2.rectangle(image, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 220, 255), 3)
            cv2.putText(image, f"{frame}  {gap['state']}  cover {gap['max_cover']}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
            panels.append(cv2.resize(image, (640, int(640 * image.shape[0] / image.shape[1]))))
        strips.append(np.hstack(panels))
    cv2.imwrite(args.out, np.vstack(strips))
    print(f"{len(strips)} gaps -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
