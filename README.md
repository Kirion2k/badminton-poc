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

Every failure diagnosed on the test clip has a test that reproduces it with the real numbers:
the far player rejected for moving 0.40 of their own body height, the umpire's slow lifetime
creep, mobility surviving an id break but not transferring to a track somewhere else, the
sub-pixel net-line crossing, and the unproven track that fills a slot only when nothing proven
is offered.

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
| Three courts and an umpire in frame | far slot locked onto a detection with **0.0 px** positional spread over 1348 frames | court polygon + static-track rejection; see the far-player section below for how that filter then misfired |

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

### Why the far player was missing, and what was actually wrong

The first pass reported the far player as the weak link and blamed their ByteTrack id breaks.
The id breaks are real — twelve id runs across the clip — but attributing all 390 frames where
the far slot was empty showed that they cause only a fifth of the loss:

| Cause | Frames | Share | Code defect? |
|---|---|---|---|
| Occluded, never detected | 276 | 70.8% | no — a property of the recording |
| `static_track` rejected the real player | 79 | 20.3% | yes |
| Net-line side misclassification | 32 | 8.2% | yes |
| Foot just outside the far baseline | 3 | 0.8% | marginal, config |

**Occlusion dominates and is not fixable here.** On frame 1610 the pipeline reports eight
detections, one on-court, `FAR MISS` — while a spectator carrying a racket bag walks straight
through the far half of the court. There is no detection to select, reject or re-identify.

**The static-track filter was rejecting the player it was built to protect.** It judged
mobility per track id, over the track's whole life, in raw pixels against a threshold scaled to
frame width (38.4 px). Three things were wrong with that, and all three are visible in the data:

- *Per id.* A fresh id starts with no evidence, so every id break re-ran the warm-up. Track 168
  — 143 px of real movement — was judged at 27.9 px and rejected for 78 consecutive frames.
- *Whole life.* A bounding extent only grows, so a static detection's jitter accumulates until
  it crosses any fixed threshold: the umpire drifted 12.4 px over 1348 frames.
- *Raw pixels.* Perspective makes the far player move a third as many pixels as the near player
  for the same distance on court (median per-frame foot movement 1.2 px against 3.4 px), so a
  frame-width threshold judged the two ends of the court by different standards. 33% of any
  45-frame window of genuine far-player motion fell below it, against 10% for near.

Mobility is now measured over a **sliding window**, in **box heights**, and is **inherited by a
replacement track id** that appears where a proven one just vanished. Recent motion in body
heights separates the populations by an order of magnitude: real players 0.21–0.40 over the
window in which they were judged, the umpire and a static bag 0.007–0.02.

Scaling by height was overcorrected on the first attempt — 0.35 box heights rejected the *near*
player, whose box is 265 px tall, for 17 frames. The coefficient came from lifetime travel
while the decision is made at judgement time. Both directions had to be measured before the
threshold was defensible.

**The net line is a sub-pixel threshold.** `side_of` is a hard test, and a player at the net
stands on it: the far player crossed by 0.0001–0.0009 normalized — under one pixel — and was
reclassified as near, emptying the far pool. Track identity was available and thrown away,
because the side was decided geometrically before scoring. Within a configurable band around
the line, the side that already owns the track now decides; beyond it geometry always wins, so
a player who really changes ends changes sides.

**A static bag held the far slot for 40 frames** during its warm-up, simply by being the only
candidate on that side. Refusing to fill a slot without a proven-mobile candidate was tried and
measured: it cost 101 frames of real player to save those 40, because a player waiting to serve
— or re-emerging from a 60-frame occlusion under a fresh id — is genuinely motionless and
indistinguishable from a bag on motion evidence alone. Proven-mobile is therefore a preference,
not a gate, and an unproven pick is recorded as `proven_mobile: false` rather than hidden.

### Consolidation was chaining whole rallies into one hit

A second clip made this unmissable. Bursts of candidates were merged by comparing each one to
the *previous* candidate, which makes merging transitive: a dense enough stream chains without
limit. On clip 2 one merge swallowed 84 candidates across 8.15 seconds and reported them as a
single hit; 33 of 39 merges spanned longer than a shot, and 1430 of 1437 candidates were
absorbed. That is why 772 raw candidates became 30 hits and rallies averaged three shots.

It was present on clip 1 too — one merge there spanned 2 seconds — but masked. At 28 fps with a
shuttle track that was only 65% usable there were just 153 candidates, so the chains stayed
short. Clip 2's genuine 60 fps and 97% shuttle visibility produce five times the candidate
density, which turned a latent bug into the dominant one. **Better input made this failure
worse, not better:** the tracker improved and the event layer collapsed behind it.

Bounding each burst by its *first* candidate instead fixes the chaining but not the guarantee
that matters. Fixed windows each keep their own winner, so a winner at the end of one window
and a winner at the start of the next survive a few frames apart — 5% of consolidated hits
ended up within 0.12s of each other, well inside the 0.3s floor the parameter claims to
enforce. The test that was supposed to catch this passed, because every candidate in it had the
same confidence and the tie-break hid the defect.

Consolidation is now **non-maximum suppression**: take the most confident candidate, suppress
everything within one interval of it, repeat. Suppression is defined by distance from what was
kept, so no two survivors can be closer than the interval and no chain can form. Both
properties now hold on real data — the closest pair of hits on either clip is 0.32s, and the
widest hit merge is 0.583s against a healthy limit of 0.6s.

### Measured accuracy on a labelled clip

Clip 2 is hand-labelled: 88 detected hits judged real or false, 7 missed hits added, and the
six true rally spans marked. `badminton-poc evaluate` scores it.

| Metric | Result | Target | |
|---|---|---|---|
| Two-player tracking, active play | 95.5% | ≥ 90% | **PASS** |
| Hitter assignment | 100% (54/54) | ≥ 90% | **PASS** |
| Rally boundaries | 66.7% (4/6) | ≥ 85% | **FAIL** |
| Hit precision | 62.5% | — | |
| Hit recall | 88.7% | — | |

Hitter assignment is exact on every real, detected hit. Tracking clears its bar. Everything
that fails traces back to one thing: **hit precision.**

### Rally boundaries are downstream of hit precision

With no confidence threshold the rally *structure* is exactly right — six detected for six
real, none split, none invented, none missed — and every single end runs long, by a median
3.69s and never short. Starts are already good at a median −0.21s. The ends run long because
false-positive hits during dead time are pulled into the rally before its terminating event is
looked for. 22 of 33 false positives fall outside real play entirely.

So rally-boundary accuracy and hit precision are one problem. Five discriminators were measured
against the labels, and four of them do not work:

| Signal | Real vs false | Verdict |
|---|---|---|
| Upstream confidence | 57% precision at ≥0.85, 75% at ≥0.90 | **No.** It is a rule id, not a score |
| Shuttle speed (play gate) | ≥4 px/f covers 78% of play frames and 22% of dead ones | **No.** Not a gate |
| Landing events | 47 of 68 gaps are 47–49 frames | **No.** A metronome, not events |
| Trajectory smoothness | residual median 12.4 real vs 18.7 false | **No.** Overlapping |
| **Player wrist speed** | median 4.30 vs 1.98 body-heights/second | **Best lead.** 85% precision at 61% recall |

The landing result is worth dwelling on. `min_landing_interval_seconds` is 0.75s and the
suppression floor is 45 frames; 47 of 68 landing gaps are 47–49 frames. The detector fires
continuously through dead time and suppression simply samples it at the floor. "A landing sits
within 0.75s of the true rally end" is therefore true *everywhere* and carries no information —
a spurious correlation that a rally-trimming fix was nearly built on.

Wrist speed is the first signal that clearly beats the 63% baseline, and the only one
independent of the shuttle track. Every hit now carries it as `swing_evidence`, and the report
says how many hits have little or no swing behind them. It is **not** used to discard anything:
it buys precision with recall, and every filter tried against these labels has cost more than
it gained.

Measured per *second*, not per frame. Per frame it is not comparable across clips -- the same
swing at 28 fps covers twice the distance per frame as at 60, and clip 1 read a median 0.113
against clip 2's 0.051 before the correction. Per second they are 3.17 and 3.08: the same
swing. Any signal normalised by frame has this problem, and this project has now been bitten by
it twice, the other being the static-track travel threshold.

### The confidence threshold, and why it is a rule filter

Upstream's confidence is a fixed value per rule, raised when rules agree: 0.50
`speed_local_max`, 0.75 `acceleration_peak`, 0.80 `vx_reversal`, 0.85 `vy_reversal`, 0.90 for
two rules agreeing, 0.95 for three. A threshold selects which rules to trust, not how certain a
detection is.

Labelling settled its value. `min_hit_confidence = 0.85` on clip 2:

- caused **all 7 missed hits** — every recall failure is self-inflicted;
- caused the one **split rally**: the real hits at 1803 and 1886 were discarded, turning a
  0.97s gap into 3.13s, past the 2.5s rally break;
- removed **none** of the false positives that matter — all 33 are at 0.85 or 0.90, and 32 come
  from `vy_reversal`, the same rule producing 55 of the true positives.

Raising it to 0.88 is meaningless: confidence is discrete, so 0.88 behaves as 0.90 and costs 31
real shots to remove 24 false ones. It is left at 0.85 only because that is the configuration
the labels describe; `evaluate` now withholds precision and recall when the labels do not match
the run, which is what caught a 62.5% figure being reported for a 189-hit run scored against 88
labelled rows.

### Where it stands

| | clip 1 | clip 2 |
|---|---|---|
| source | screen recording of YouTube, VFR 60→28 fps | download, genuine 60 fps |
| labelled | no | **yes** |
| two players, active play | — | **95.5%** PASS |
| hitter assignment | — | **100%** PASS |
| rally boundaries | — | 66.7% FAIL |
| hit precision / recall | — | 62.5% / 88.7% |
| shuttle visible and on our court | 65.3% | 97.3% |

Two of the three POC criteria pass on a real, labelled clip. The third fails, the cause is
understood, and the remaining lever is a hit-quality signal that does not come from the
upstream event detector.

`badminton-poc report --video <clip>` renders any clip's `analysis.json` as a page, with every
figure carrying the caveat it rests on.
