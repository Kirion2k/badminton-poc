"""Answer, before anyone hand-fits a court, whether a downloaded clip can work at all.

Fitting a court takes half an hour of squinting at a coordinate grid, and on most of the clips we
pulled off YouTube it would have been wasted: the camera was handheld, or the framed court was
empty because the match was being played off the edge of the frame, or the far player was too
small for the pose model to see at the input size we run. None of those are visible in a summary
of pipeline output -- they look like "the pipeline found nothing" -- so each one cost a fit before
it was found.

Three checks, in the order that rules a clip out soonest:

  camera drift   phase-correlate frames against the first one. The pipeline assumes one court
                 polygon for the whole video, so a frame that moves makes every coordinate mean
                 something different at the end of the clip than at the start.
  far player     how tall is the smaller of the two players, and does the pose model see them?
                 Confidence is reported at the configured input size and at 3200, because the
                 gap between those two numbers is the difference between a clip that fails the
                 tracking criterion and one that passes it.
  shuttle        how much of the clip the tracker spends parked on a fixed object rather than
                 following anything, which is what a clip with no rallies usually turns out to be.

What it does not catch: whether the court in frame is the one being played on. yt-irish
passes the camera check and reports two players, because there are two people in shot --
they are on the next court along, and the court the camera frames is empty for the whole
clip. Look at a frame before trusting a pass.

Usage: screen_clip.py <clip> [<clip> ...]
"""
import csv, json, sys
from pathlib import Path

import cv2
import numpy as np

DRIFT_SAMPLES = 25
PROXY_WIDTH = 960


def camera_drift(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ref = None
    worst = 0.0
    worst_frame = 0
    for i in np.linspace(0, max(0, n - 1), DRIFT_SAMPLES).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, img = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(cv2.resize(img, (PROXY_WIDTH, int(PROXY_WIDTH * img.shape[0] / img.shape[1]))),
                         cv2.COLOR_BGR2GRAY).astype(np.float32)
        if ref is None:
            ref = g
            continue
        (dx, dy), _ = cv2.phaseCorrelate(ref, g)
        mag = float(np.hypot(dx, dy)) * width / PROXY_WIDTH
        if mag > worst:
            worst, worst_frame = mag, int(i)
    cap.release()
    return {"max_drift_px": round(worst, 1), "at_frame": worst_frame,
            "frac_of_width": round(worst / max(1, width), 4)}


def far_player(clip: str, cfg: dict, src_w: int, src_h: int) -> dict:
    """The smaller player's apparent size, and whether the detector actually sees them.

    Taken from the recorded pose pass rather than re-running it: every candidate the model
    produced is in players.jsonl, including the ones selection threw away.
    """
    path = Path("outputs") / clip / "players.jsonl"
    if not path.exists():
        return {"note": "no players.jsonl; run the players stage first"}
    heights, far_seen, frames = [], 0, 0
    for line in path.open():
        r = json.loads(line)
        frames += 1
        if r.get("far"):
            far_seen += 1
            b = r["far"]["bbox"]
            heights.append((b[3] - b[1]) / r["height"])
    imgsz = (cfg.get("players") or {}).get("imgsz")
    # The 10th percentile, not the median. These heights come only from frames where the far
    # player WAS found, so they are already biased towards the frames the detector handled well;
    # taking the median on top of that would report the easy case twice over. The deep positions
    # are the ones that get lost, and they are the small end of this distribution.
    frac = float(np.percentile(heights, 10)) if heights else None
    # Ultralytics resizes the long side to imgsz, so what the detector actually sees is the
    # player's height after that scaling, not their share of the source frame. It is the only
    # one of the two numbers that predicts anything: melbourne's far player is 8% of frame
    # height and melbourne is 1080p, which puts her at 86 px and below the confidence threshold;
    # ccaa's is 14% of a 720p frame, which is already 151 px and gains nothing from more.
    px = None
    if frac is not None and imgsz and src_w:
        px = round(frac * src_h * imgsz / src_w)
    return {
        "far_detected_fraction": round(far_seen / max(1, frames), 3),
        "far_box_height_frac_p10": round(frac, 4) if frac is not None else None,
        "far_height_px_at_imgsz": px,
        "configured_imgsz": imgsz,
    }


def shuttle_parking(clip: str) -> dict:
    csvs = sorted((Path("outputs") / clip / "tracknet").glob("*_ball.csv"))
    if not csvs:
        return {"note": "no tracknet ball csv; run the shuttle stage first"}
    rows = list(csv.DictReader(csvs[0].open()))
    xy = np.array([[float(r["X"] or 0), float(r["Y"] or 0)] for r in rows])
    step = np.r_[0.0, np.hypot(*np.diff(xy, axis=0).T)] if len(xy) > 1 else np.zeros(1)
    return {"frames": len(rows),
            "parked_fraction": round(float((step < 1.0).mean()), 3),
            "moving_fraction": round(float((step > 5.0).mean()), 3)}


def verdict(drift: dict, far: dict, ball: dict) -> str:
    if drift["frac_of_width"] > 0.005:
        return (f"REJECT: the camera moves {drift['max_drift_px']:.0f}px "
                f"({drift['frac_of_width']*100:.1f}% of frame width) by frame {drift['at_frame']}. "
                f"One court polygon cannot describe the whole clip.")
    notes = []
    px = far.get("far_height_px_at_imgsz")
    if px is not None and px < 120:
        want = int(far["configured_imgsz"] * 140 / max(1, px))
        notes.append(f"the far player is only {px} px tall at imgsz {far['configured_imgsz']}, "
                     f"below the ~120 px the pose model needs; try imgsz {want//64*64}")
    if far.get("far_detected_fraction", 1.0) < 0.9:
        notes.append(f"the far player is found on only {far['far_detected_fraction']*100:.0f}% of "
                     f"frames, below the 90% tracking criterion")
    if ball.get("parked_fraction", 0) > 0.5:
        notes.append(f"the shuttle tracker is parked on a fixed object for "
                     f"{ball['parked_fraction']*100:.0f}% of frames")
    return "USABLE" if not notes else "NEEDS WORK: " + "; ".join(notes)


for clip in sys.argv[1:]:
    work = Path("outputs") / clip / "work.mp4"
    if not work.exists():
        print(f"{clip}: no outputs/{clip}/work.mp4")
        continue
    cfg_path = Path("configs") / f"{clip}.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    drift = camera_drift(work)
    cap = cv2.VideoCapture(str(work))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    far = far_player(clip, cfg, src_w, src_h)
    ball = shuttle_parking(clip)
    print(f"\n{clip}")
    print(f"  camera   max drift {drift['max_drift_px']:.1f}px "
          f"({drift['frac_of_width']*100:.2f}% of width) at frame {drift['at_frame']}")
    print(f"  players  far detected {far.get('far_detected_fraction')}, "
          f"far box height p10 {far.get('far_box_height_frac_p10')} of frame "
          f"= {far.get('far_height_px_at_imgsz')} px at imgsz {far.get('configured_imgsz')}")
    print(f"  shuttle  parked {ball.get('parked_fraction')}, moving {ball.get('moving_fraction')}")
    print(f"  -> {verdict(drift, far, ball)}")
