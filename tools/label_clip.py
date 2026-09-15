"""Render a span of a working video for a human to label, with frame numbers and nothing else.

Labels reference working-video frame indices, so a labeller needs to read those indices off the
picture. This burns them in and trims to the span in question, slowed down by default because a
racket contact at 34 fps is two or three frames long.

Nothing else is drawn. No boxes, no hit markers, no rally spans: a labeller who can see what the
pipeline decided is not an independent witness, and the whole value of a hand labelling is that
it was arrived at separately.

    python tools/label_clip.py --clip test-video5 --span 1674 2462 --out label-rally2.mp4
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--span", nargs=2, type=int, metavar=("START", "END"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--speed", type=float, default=0.33,
                    help="playback speed; 0.33 means a third of real time (default)")
    ap.add_argument("--output-root", default="outputs")
    args = ap.parse_args()

    clip_dir = Path(args.output_root) / args.clip
    source = clip_dir / "work.mp4"
    meta = json.loads((clip_dir / "metadata.json").read_text())
    fps = float(meta["working"]["fps"])
    start, end = args.span
    out_fps = max(1.0, fps * args.speed)

    # The burned-in number is the index in work.mp4, which is what every label refers to.
    # Every colon inside an eif expression has to be escaped or drawtext reads it as the end of
    # the option, and a comma anywhere in here ends the whole filter.
    text = (f"frame %{{eif\\:n+{start}\\:d}}     "
            f"%{{eif\\:(n+{start})/{fps:.6f}\\:d}}s")
    vf = (
        f"select='between(n\\,{start}\\,{end})',"
        f"setpts=N/({out_fps:.6f}*TB),"
        f"drawtext=text='{text}':fontsize=42:fontcolor=yellow:borderw=3:bordercolor=black:"
        f"x=24:y=20"
    )
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-i", str(source),
           "-vf", vf, "-r", f"{out_fps:.6f}", "-an", args.out]
    subprocess.run(cmd, check=True)
    frames = end - start + 1
    print(f"{args.out}: frames {start}-{end} ({frames} frames, {frames / fps:.1f}s of play) "
          f"at {args.speed:g}x -> {frames / out_fps:.1f}s to watch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
