"""Tile frames around candidate contacts so a labeller can judge many of them at once.

Labelling a clip means answering one question a hundred times: at this frame, did a racket
strike the shuttle? Scrubbing a video to find out is slow and leaves no record. This renders a
strip per candidate -- the frames before, at and after it, cropped around where the raw shuttle
CSV says the shuttle was -- so the answer is a glance and the evidence is a file.

The only thing drawn on the image is the shuttle marker, straight from the upstream CSV. Player
boxes, hit verdicts and rally spans are deliberately not drawn: the labels have to describe the
video, and a labeller who can see what the pipeline decided is no longer an independent witness.

    python tools/contact_sheet.py --clip test-video5 --frames 213,251,286 --out sheet.png
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2


def load_ball(clip_dir: Path) -> dict[int, tuple[int, float, float]]:
    csv_path = next(clip_dir.glob("tracknet/*_ball.csv"))
    ball = {}
    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            ball[int(row["Frame"])] = (int(row["Visibility"]), float(row["X"]), float(row["Y"]))
    return ball


def crop(frame, cx, cy, w, h):
    fh, fw = frame.shape[:2]
    x1 = max(0, min(int(cx - w // 2), fw - w))
    y1 = max(0, min(int(cy - h // 2), fh - h))
    return frame[y1:y1 + h, x1:x1 + w], x1, y1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", required=True, help="clip stem, i.e. outputs/<clip>/")
    ap.add_argument("--frames", required=True, help="comma-separated candidate frames")
    ap.add_argument("--out", required=True)
    ap.add_argument("--offsets", default="-4,0,4", help="frames to show either side")
    ap.add_argument("--window", default="560x400", help="crop size in working-video pixels")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--trace", type=int, default=0,
                    help="draw the raw shuttle path this many frames either side of the cell")
    ap.add_argument("--output-root", default="outputs")
    args = ap.parse_args()

    clip_dir = Path(args.output_root) / args.clip
    frames = [int(f) for f in args.frames.split(",") if f.strip()]
    offsets = [int(o) for o in args.offsets.split(",")]
    cw, ch = (int(v) for v in args.window.lower().split("x"))

    ball = load_ball(clip_dir)
    cap = cv2.VideoCapture(str(clip_dir / "work.mp4"))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {clip_dir / 'work.mp4'}")

    wanted = sorted({f + o for f in frames for o in offsets})
    grabbed: dict[int, "cv2.Mat"] = {}
    for f in wanted:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if ok:
            grabbed[f] = frame
    cap.release()

    rows = []
    for f in frames:
        # Centre every crop in the strip on the shuttle at the candidate frame, so the shuttle
        # moving across a fixed window is what the eye sees.
        vis, bx, by = ball.get(f, (0, 0.0, 0.0))
        if not vis:
            for o in offsets:
                v2, x2, y2 = ball.get(f + o, (0, 0.0, 0.0))
                if v2:
                    bx, by = x2, y2
                    break
        cells = []
        for o in offsets:
            frame = grabbed.get(f + o)
            if frame is None:
                continue
            cell, ox, oy = crop(frame, bx, by, cw, ch)
            cell = cell.copy()
            if args.trace:
                # The raw track either side of this frame. A racket contact reverses the
                # shuttle, so the shape of the path is the physical evidence; it is drawn from
                # the upstream CSV, which is the same data a labeller would scrub for.
                path = []
                for t in range(f + o - args.trace, f + o + args.trace + 1):
                    tv, tx, ty = ball.get(t, (0, 0.0, 0.0))
                    path.append((int(tx - ox), int(ty - oy), t) if tv else None)
                for a, b in zip(path, path[1:]):
                    if a is None or b is None:
                        continue
                    before = a[2] < f + o
                    cv2.line(cell, a[:2], b[:2],
                             (255, 160, 0) if before else (0, 200, 255), 2, cv2.LINE_AA)
            v, sx, sy = ball.get(f + o, (0, 0.0, 0.0))
            if v:
                cv2.circle(cell, (int(sx - ox), int(sy - oy)), 13, (0, 255, 255), 2)
            label = f"{f + o}" + ("" if o else "  <-- candidate")
            cv2.putText(cell, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(cell, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255) if o else (0, 255, 255), 1, cv2.LINE_AA)
            cells.append(cell)
        if cells:
            rows.append(cv2.hconcat(cells))

    if not rows:
        raise SystemExit("no frames could be read")
    width = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 2, 2, 0, width - r.shape[1], cv2.BORDER_CONSTANT, value=(40, 40, 40))
            for r in rows]
    sheet = cv2.vconcat(rows)
    if args.scale != 1.0:
        sheet = cv2.resize(sheet, None, fx=args.scale, fy=args.scale,
                           interpolation=cv2.INTER_AREA if args.scale < 1 else cv2.INTER_LANCZOS4)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, sheet)
    print(args.out, sheet.shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
