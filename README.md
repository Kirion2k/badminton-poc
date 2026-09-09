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

### Measured accuracy on a fully labelled clip

All 189 hits the pipeline can emit on clip 2 are judged, so any configuration can be scored
against one labelling: a labelled real hit a run no longer emits counts as a miss, which is
what recall means.

| Metric | Result | Target | |
|---|---|---|---|
| Two-player tracking, active play | 92.7% | ≥ 90% | **PASS** |
| Hitter assignment | 98.1% (53/54) | ≥ 90% | **PASS** |
| Rally boundaries | 66.7% (4/6) | ≥ 85% | **FAIL** |
| Hit precision / recall | 62.5% / 88.7% | — | |

**61 of 189 candidates are real hits.** Two out of three the detector emits are not.

### One frame of absence could hand a slot to a bystander

Coverage read 95.5% before this was found, and that number was partly fiction. Spotted by hand
while labelling -- "the far player is being registered as someone on the sideline" -- and much
larger than it looked: an official standing beside the far court held the far slot for **1373
frames, 23 seconds**, while the real far player held it for 38.

The mechanism is a single undetected frame. At frame 1918 the far player was absent from the
detector's output entirely; the official was the only other far-side candidate and took the
slot by default. Identity continuity then held it for the next 1373 frames, and on 1085 of them
(79%) the real player was an available candidate and lost every comparison -- continuity 1.0 and
proximity 0.999 against a stranger's zero.

``identity_smoothing_frames`` exists to survive exactly this and does not: it protects a slot
against an *empty pool*, not against a *substitute*. A player cannot cross the court in one
frame, so a slot that still remembers a track now refuses a different one further away than the
player could plausibly have moved, and reports itself empty instead. The allowance grows with
how long the track has been missing, so a genuine ByteTrack id break -- the same player
reappearing a few pixels away under a new number -- still transfers freely.

The intruder went from 1373 frames to 196, the real far player from 38 to 1123, and the four
tracks that now hold the far slot longest are all within the real player's x range. Reported
coverage *fell* from 95.5% to 92.7%, because the slot now says "empty" where it used to say
"an official". That is the number getting more honest, not worse.

### The confidence threshold, measured

Upstream confidence is a fixed value per rule -- 0.50 `speed_local_max`, 0.75
`acceleration_peak`, 0.80 `vx_reversal`, 0.85 `vy_reversal` -- raised to 0.90 when two rules
agree and 0.95 when three do. A threshold selects which rules to trust, not how certain a
detection is. Scored against the full labelling:

| `min_hit_confidence` | hits | precision | recall | F1 | rally boundaries |
|---|---|---|---|---|---|
| 0.00 | 189 | 32% | 98% | 0.49 | 0.17 |
| 0.80 | 96 | 61% | 95% | **0.75** | 0.50 |
| **0.85** | 88 | 62% | 89% | 0.73 | **0.67** |
| 0.90 | 33 | 73% | 39% | 0.51 | 0.17 |

0.85 is kept: 0.80 scores marginally better on hits and four real hits more, but 0.85 is
materially better on rally boundaries, which is the criterion that is actually failing.

An earlier reading of this project's own notes said the threshold did net harm, on the grounds
that it caused all seven recall failures and one rally split. Both facts are true and the
conclusion was wrong: it also removes about ninety-five false positives, and dropping it takes
precision from 62% to 32%. That reading came from labelling that covered only the hits the
threshold had already let through, which is exactly the trap `evaluate` now refuses to fall
into.

### The swing signal, re-tested

Peak wrist speed of the assigned player, from the pose model, independent of the shuttle track.
Real hits sit at a median 4.31 body-heights per second against 2.24 for false ones.

Its first measurement -- 85% precision at 61% recall -- was taken on the 88 hits the threshold
let through, a population already filtered to one rule. Against the full 189 it reads 53%
precision at 63% recall against a 34% baseline. Roughly +19 points of precision either way, so
the effect is real and consistent, but the absolute number depends entirely on what it is
applied to.

Applied where it belongs -- on top of the 0.85 threshold -- it is the strongest result the
project has:

| swing ≥ | precision | recall (of all 61 real hits) |
|---|---|---|
| off | 62% | 90% |
| 3.0 | 79% | 56% |
| **3.6** | **85%** | 54% |

It is still not applied. Buying 23 points of precision with 36 points of recall is not
obviously right, and that trade should be made against a second clip rather than the one the
threshold was chosen on.

### Where it stands

Two of three POC criteria pass on a real, fully labelled clip. Rally boundaries fail, and the
cause is understood: rally ends run long because false-positive hits during dead time are pulled
into the rally before its terminating event is looked for. Boundary accuracy and hit precision
are one problem.

**Every number here comes from one recording.** Clip 1 is a poor clip -- occlusion alone costs
it ten points of far-player coverage, so it cannot pass the tracking criterion however good the
code is -- and is kept as a regression clip, not an acceptance one. Nothing has been
cross-validated on a second usable recording, and the swing threshold in particular was
measured on the same clip that set the confidence threshold.

`badminton-poc report --video <clip>` renders any clip's `analysis.json` as a page, with every
figure carrying the caveat it rests on. `badminton-poc refresh-labels` re-seeds a ground-truth
file against a new run while keeping judgements already made.
