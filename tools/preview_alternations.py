"""Cut one preview per impossible alternation, so a human can say what is wrong with it.

A singles rally alternates, so two adjacent contacts credited to the same player cannot both be
right. The rule says that much and no more: it cannot tell a hitter given to the wrong player
from a real contact between the two that nobody recorded. Only someone watching can.

So the preview shows the pair together rather than one contact at a time, which is what the
shot-type previews do and why they are the wrong tool here. It runs from a little before the
first contact to a little after the second, marks each one where it falls, and marks the stretch
between them -- the place a missing contact would have to be -- as the thing to look at.

The hitter this file currently claims is shown, because the question is whether that claim is
right. Nothing else is: no prediction, no detector output, no suggested answer.

    python tools/preview_alternations.py --clip test-video2
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from badminton_poc.truth import find_impossible_alternations  # noqa: E402


def _escape(text: str) -> str:
    """ffmpeg drawtext eats colons and single quotes."""
    return text.replace("\\", "\\\\").replace(":", r"\:").replace("'", r"\'")


def cut(video: Path, target: Path, *, start: float, end: float, first_at: float,
        second_at: float, header: str, first_label: str, second_label: str,
        width: int = 720, mark_seconds: float = 0.25) -> None:
    a_lo, a_hi = max(0.0, first_at - mark_seconds), first_at + mark_seconds
    b_lo, b_hi = max(0.0, second_at - mark_seconds), second_at + mark_seconds
    # The gap is what the second question is about, so it gets its own band rather than being
    # the absence of a marking. It starts and ends where the contact markings stop.
    gap_lo, gap_hi = a_hi, b_lo
    filters = ",".join([
        f"scale={width}:-2",
        f"drawtext=text='{_escape(header)}':x=8:y=6:fontsize=15:fontcolor=white:box=1:"
        "boxcolor=black@0.6:boxborderw=4",
        # First contact: cyan.
        f"drawbox=x=0:y=0:w=iw:h=ih:color=cyan@0.9:t=6:"
        f"enable='between(t,{a_lo:.3f},{a_hi:.3f})'",
        f"drawtext=text='{_escape(first_label)}':x=(w-tw)/2:y=h-36:fontsize=19:fontcolor=cyan:"
        f"box=1:boxcolor=black@0.65:boxborderw=5:enable='between(t,{a_lo:.3f},{a_hi:.3f})'",
        # Second contact: magenta. Two colours so they cannot be mixed up on a rewatch.
        f"drawbox=x=0:y=0:w=iw:h=ih:color=magenta@0.9:t=6:"
        f"enable='between(t,{b_lo:.3f},{b_hi:.3f})'",
        f"drawtext=text='{_escape(second_label)}':x=(w-tw)/2:y=h-36:fontsize=19:"
        f"fontcolor=magenta:box=1:boxcolor=black@0.65:boxborderw=5:"
        f"enable='between(t,{b_lo:.3f},{b_hi:.3f})'",
        f"drawtext=text='{_escape('between the two -- is there a contact here?')}':"
        f"x=(w-tw)/2:y=h-36:fontsize=19:fontcolor=yellow:box=1:boxcolor=black@0.6:boxborderw=5:"
        f"enable='between(t,{gap_lo:.3f},{gap_hi:.3f})'",
    ])
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
         "-i", str(video), "-vf", filters, "-an", str(target)],
        check=True,
    )


SHEET_HEADER = """\
# Impossible alternations in {clip}. One block per pair, {n} of them.
#
# A singles rally alternates, so the labelling below cannot be right as it stands. Exactly one
# of these is true of each pair, and watching is the only way to tell which:
#
#     a <pair>  hitter <first|second> <near|far>   the named contact's hitter is wrong
#     a <pair>  missing <frame>                    a real contact between them was never
#                                                  labelled; give roughly where, +/- 30 frames
#                                                  is fine
#     a <pair>  spurious <frame>                   one of the two is not a real contact
#     a <pair>  unsure                             recorded and left alone rather than guessed
#
# Add a note after any of them. Preview clips are in {previews}.
# The first contact is marked CYAN, the second MAGENTA, the gap between them YELLOW.
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--output-root", default="outputs")
    ap.add_argument("--pad-seconds", type=float, default=0.7)
    args = ap.parse_args()

    clip_dir = ROOT / args.output_root / args.clip
    truth = json.loads((clip_dir / "truth.json").read_text(encoding="utf-8"))
    fps = float(truth["fps"])
    video = clip_dir / "work.mp4"
    if not video.exists():
        raise SystemExit(f"working video not found at {video}")

    violations = find_impossible_alternations(truth)
    if not violations:
        print(f"{args.clip}: no impossible alternations. Nothing to cut.")
        return 0

    out_dir = clip_dir / "alternations"
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.mp4"):
        stale.unlink()

    sheet = [SHEET_HEADER.format(clip=args.clip, n=len(violations),
                                 previews=out_dir.relative_to(ROOT))]
    for v in violations:
        first, second, hitter = v["first_frame"], v["second_frame"], v["hitter"]
        start = max(0.0, first / fps - args.pad_seconds)
        end = second / fps + args.pad_seconds
        name = f"pair_{first}_{second}_{hitter}.mp4"
        header = (f"frames {first} and {second}   both labelled {hitter}   "
                  f"gap {v['gap_seconds']:.2f}s   rally {v['rally']}")
        cut(video, out_dir / name, start=start, end=end,
            first_at=first / fps - start, second_at=second / fps - start,
            header=header,
            first_label=f"1st contact  frame {first}  labelled {hitter}",
            second_label=f"2nd contact  frame {second}  labelled {hitter}")
        print(f"  {name}")
        sheet.append(
            f"\n# frames {first} -> {second}, both labelled {hitter}, gap "
            f"{v['gap_seconds']:.2f}s, rally {v['rally']}"
            f"\n#   {v['first_shot_type'] or '?'} then {v['second_shot_type'] or '?'}"
            f"\n#   preview {name}"
            f"\na {first}-{second} "
        )

    sheet_path = clip_dir / "alternation_answers.tsv"
    sheet_path.write_text("\n".join(sheet) + "\n", encoding="utf-8")
    print(f"\n{len(violations)} previews in {out_dir}")
    print(f"answer sheet: {sheet_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
