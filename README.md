# Badminton video-analysis technical POC

This is intentionally **not** the app. It is a local experiment to answer one question:

> Can a fixed-camera singles badminton recording be converted into a usable machine-readable
> rally timeline with player tracks and shuttle events?

It is built to be *inspected*. Every stage writes its intermediate output, every rejected
detection records why it was rejected, and every hit assignment records the evidence it used.
Nothing reports an accuracy number that has not been measured against hand labels.

## Pipeline

```
source video
  │  preprocess   ffmpeg: bake rotation, uniform fps, one frame per source frame
  ├─ players      Ultralytics pose + ByteTrack, court-polygon ROI, near/far identity slots
  ├─ shuttle      upstream BadmintonTrackNet (TrackNetV3 + InpaintNet) via subprocess
  │  analyze      hit -> player association, rally segmentation, coverage metrics
  └─ render       annotated diagnostic video
```

Shuttle inference and player inference never touch each other. They meet only in `analyze`,
which reads files from disk.

## One command

```bash
badminton-poc run --video data/input/test01.mp4 --config configs/test-video.json
```

Individual stages (`preprocess`, `players`, `shuttle`, `analyze`, `render`) can be re-run with
`--stages`, which is what you want during iteration:

```bash
badminton-poc run --video data/input/test01.mp4 --config configs/... --stages analyze,render
```

## Outputs

`outputs/<clip-stem>/`

| File | What it is |
|---|---|
| `work.mp4` | normalized working video — the frame-index reference for every other file |
| `metadata.json` | source probe, working-video properties, true source frame timestamps, warnings |
| `court_roi.png` | the court polygon drawn on a frame, so a bad ROI is visible immediately |
| `players.jsonl` | one row per frame: near/far boxes, keypoints, track ids, scoring terms, rejected candidates with reasons |
| `tracknet/` | upstream artefacts, unmodified: `*_ball.csv`, `*_events.json`, `*_events.csv` |
| `analysis.json` | merged event timeline, rallies, coverage summary, hit-consolidation audit |
| `annotated.mp4` | diagnostic overlay video |
| `ground_truth.json` | labelling template, pre-filled with detections for you to correct |
| `evaluation.json` | written by `badminton-poc evaluate` once labelling is done |
| `logs/` | the exact upstream command, its stdout/stderr, and the run report |

## Environment

**Use a real CPython 3.11 release.** Ubuntu 22.04's `python3.11` package is `3.11.0rc1`, which
predates `sys.get_int_max_str_digits`. `torch._dynamo` assumes that attribute exists, so
`import torchvision` — and therefore Ultralytics — fails on it. Plain `torch` works, which makes
this fail late and confusingly.

```bash
# get a real 3.11 without touching the system Python
uv python install 3.11
uv venv --python 3.11 .venv
uv pip install -r requirements.txt -e .
```

CUDA is used when available and CPU is a working fallback; `--device` is passed through to
upstream.

## BadmintonTrackNet

Clone it **next to** this repository. We never copy its source in and never patch it.

```bash
cd ..
git clone https://github.com/ZSHYC/BadmintonTrackNet.git
```

Download the official TrackNetV3 checkpoints and put them in `BadmintonTrackNet/ckpts/`:

- source: <https://drive.google.com/file/d/1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA/view?usp=sharing>
  (linked from [qaz812345/TrackNetV3 § Inference](https://github.com/qaz812345/TrackNetV3#inference))
- unzip `TrackNetV3_ckpts.zip`, then place `TrackNet_best.pt` and `InpaintNet_best.pt` in `ckpts/`

These are pickled PyTorch checkpoints. Load them only from that official source. The pipeline
will not download them for you; it tells you exactly what is missing and stops.

We invoke upstream as a subprocess:

```
python -m tracknet.inference.events --video_file … --tracknet_file … --inpaintnet_file …
       --save_dir … --device auto --batch_size … --eval_mode weight
```

If upstream ever needs its own dependency set, point `shuttle.python` in the config at a second
interpreter. Nothing else changes.

### Two upstream behaviours worth knowing

- `tracknet/inference/events.py` filters its own events down to `{"landing", "hit"}` before
  writing the file, so **`out_of_frame` never reaches us** even though the candidate generator
  produces it. We derive an equivalent "shuttle disappeared" signal ourselves from trajectory
  visibility and tag it `source: "shuttle_absence"` so it is never mistaken for an upstream
  detection.
- Everything upstream reads about the video (width, height, **fps**) comes from OpenCV's
  `VideoCapture` properties, and those feed `BounceDetector`'s kinematic thresholds. On a
  variable-frame-rate phone recording OpenCV reports the *timebase*, not the real rate. This is
  why `preprocess` exists.

## Court ROI

The court polygon is the most important hand-set parameter: it is what separates the two match
players from spectators, from the neighbouring court, and from the umpire. So look at it.

```bash
badminton-poc calibrate --video outputs/<clip>/work.mp4 --output /tmp/grid.png   # with coordinate grid
badminton-poc calibrate --video outputs/<clip>/work.mp4 --output /tmp/roi.png \
    --config configs/<clip>.json --no-grid                                       # with the polygon
```

A polygon, not a rectangle, because the court is a trapezoid — a bounding rectangle around one
court also contains the next one. Players are tested by their **foot point** (bottom-centre of
the box), not the box centre, because a spectator standing between the camera and the court has
their centre over the court but their feet in front of it.

`net_line` is two points on the line where the net meets the floor. It is a line rather than a
single `y` because the camera is rarely square to the court.

## Recording conditions

For a clip that will actually work:

- singles, landscape, fixed camera, whole court visible
- camera behind a baseline and **elevated** if possible
- nobody walking between the camera and the court
- 1080p, 60 fps genuine constant frame rate
- 1–5 minutes
- the original file, not a screen recording of one

## Evaluation

`analysis.json` reports what the pipeline *produced*. It never reports accuracy. To measure
accuracy:

1. Watch `annotated.mp4`.
2. Edit `ground_truth.json`: fill in `active_play_ranges`, mark false positives with
   `is_real_hit: false`, correct `true_hitter`, add `missed_hits`, set `boundary_ok` per rally.
3. Set `labelled: true`.
4. `badminton-poc evaluate --video data/input/<clip>.mp4`

That produces the six numbers the POC is judged on, with PASS/FAIL against the targets:

| Metric | Target |
|---|---|
| two-player tracking coverage during active play | ≥ 90% |
| shuttle trajectory coherence through rallies | visual judgement |
| hit precision / recall | measured, no target yet |
| hitter-assignment accuracy on clean hits | ≥ 90% |
| rally-boundary accuracy | ≥ 85% |

`evaluate` refuses to run on an unlabelled file rather than produce a number from nothing.

## Tests

```bash
.venv/bin/python -m pytest
```

Covers TrackNet CSV parsing (including real frame ids with gaps, inpaint masks, malformed
input), event JSON parsing, nearest-shuttle lookup, court geometry, player selection and
identity continuity, hitter assignment, hit consolidation, rally segmentation, and the
evaluation scorer. None of it needs a GPU or a model.

## Not in this milestone

smash/drop/clear classification · winner/error inference · LLM coaching · scoreboard OCR ·
doubles · court homography · any UI · any cloud

## Findings from the first test clip

Recorded here because they are the point of the exercise, and because several are properties of
the *input*, not the code.

**The clip is a phone screen recording of a YouTube video** ("Men's Singles R64 – Ricky Tang vs
Jayden, Australian Nationals 2025"), not original footage. That single fact explains most of
what follows.

| Problem | Evidence | Handling |
|---|---|---|
| Stored portrait with 90° display rotation | ffprobe Display Matrix; stored 1320×2868, displays 2868×1320 | rotation baked into `work.mp4` |
| Variable frame rate: advertises 60 fps, delivers 28 | 3121 frames over 111.44 s; gaps alternate 1/60, 2/60, 3/60 s | resampled to a uniform grid; true timestamps kept in `metadata.json` |
| Last 96 frames are the YouTube player UI, then the iOS Control Centre | phase correlation collapses at frame 3026 | `--end-frame 3026` |
| Detector emits ~4 hits/second and 250 landings for 12 rallies | 215 of 250 landing gaps are exactly 4 frames | see below |
| TrackNetV3 tracks the neighbouring court's shuttle | "hits" 365–760 px from *both* players, at x≈0.28 | `shuttle_polygon` airspace ROI; **30% of shuttle frames were off-court** |
| Spectators walk between camera and court | ~18 separate occlusion episodes | recorded, not hidden; visible in `annotated.mp4` |
| Three courts and an umpire in frame | far slot locked onto a detection with **0.0 px** positional spread over 1348 frames | court polygon + static-track rejection |

### Why the event detector over-fires

The VFR resampling is the cause. The source's frames were captured 2/60 and 3/60 s apart in an
alternating pattern; placed on a uniform grid, the real time between consecutive frames
alternates 33 ms / 50 ms. Measured speed therefore oscillates ~20% with a period of two frames
(autocorrelation peaks at lag 2), and every upstream rule that reasons about frame-to-frame
velocity — `speed_step`, `vy_reversal`, `acceleration_peak` — trips repeatedly.

We consolidate bursts of candidates into single events and record every merge in
`analysis.json`. The proper fix is a genuine constant-frame-rate source.

### Two design decisions worth knowing

**Rally boundaries are driven by hits, not by landings.** Splitting on every landing collapsed
under 250 spurious landing candidates. A terminating event is attached afterwards if one falls
between the last hit of a rally and the first hit of the next.

**We do not use "which side of the net is the shuttle on" to constrain hit assignment.** The net
line is where the net meets the *floor*, which is only meaningful for a point on the ground. At
the moment of a hit the shuttle is in the air and sits high in frame regardless of which half it
is over — that test disagreed with 33 of 50 assignments. Doing this properly needs court
homography plus a height estimate.

### Where it stands

Pipeline output on the trimmed clip (3026 frames, 108 s) — **these are counts, not accuracy**:

| | |
|---|---|
| two players detected, all frames | 75.9% |
| two players detected, during detected rallies | 81.9% (near 95.0%, far 86.9%) |
| shuttle visible and on our court | 65.3% |
| shuttle frames rejected as another court's | 30.3% |
| hits (from 153 raw candidates) | 58 — near 34, far 14, unassigned 10 |
| rallies | 10, mean 5.4 hits, longest 13 |

The far player is the weak link: smaller in frame, and their ByteTrack id breaks repeatedly
(three separate ids cover most of the clip). Precision, recall, hitter accuracy and
rally-boundary accuracy are **not** reported, because `ground_truth.json` has not been labelled.
