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
  │  flight       shuttle track -> flight segments -> hits, landings, rallies
  │  analyze      hit -> player association, rally assembly, coverage metrics
  └─ render       annotated diagnostic video
```

Shuttle inference and player inference never touch each other. They meet only in `analyze`,
which reads files from disk.

`flight` takes only the shuttle CSV and the frame rate, so it needs no GPU and a change to it can
be scored against every clip in seconds. Setting `analysis.event_source` to `tracknet` restores
the older path, which reads upstream's events file instead; both stay runnable so any claim about
one is a measurement against the other.

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

Two of those are worth more than they look, because four downloaded clips died on them.

**Fixed means measurably fixed.** A clip that looks steady can still drift a fifth of the frame
width, and a tripod can be knocked once and hold still either side of the knock. One polygon
describes the whole video or it does not; `tools/screen_clip.py` measures the drift so the answer
is a number rather than an impression.

**Whole court visible means the court being played on.** A camera pointed at an empty court beside
the match passes every automatic check there is.

And one thing the list cannot ask for, because it works against the others: the further back and
higher the camera, the smaller the far player, and below about 10% of frame height `yolo11m-pose`
starts missing them at `imgsz: 1920`. Raise `players.imgsz` for those clips rather than move the
camera in. Check the size before spending the runtime, though — yt-ccaa loses its far player just
as often and more pixels do not help, because at 16% of frame height she is large enough already
and what beats the detector there is motion blur at 30 fps.

## Evaluation

`analysis.json` reports what the pipeline *produced*. It never reports accuracy. Accuracy comes
from hand labels, and there are two label files for historical reasons.

**`truth.json` and `badminton-poc baseline` are the live path.** The file records only what
happened in the video, so one labelling scores every configuration; `baseline.json` is a clip's
accuracy record. How it works and why the earlier format was wrong is below, under *Ground truth
is what happened in the video*.

```bash
badminton-poc truth --video data/input/<clip>.mp4    # create or migrate the truth file
badminton-poc baseline --video data/input/<clip>.mp4 # score the current run
```

**`ground_truth.json` and `badminton-poc evaluate` are the earlier, detector-coupled path.** One
row per detected candidate, judged real or false. Label it by watching `annotated.mp4`, filling
in `active_play_ranges` and `missed_hits`, marking false positives with `is_real_hit: false`,
correcting `true_hitter`, setting `boundary_ok` per rally, then `labelled: true`. It still runs,
and `badminton-poc refresh-labels` carries judgements across a configuration change. On
test-video2 its labels describe the upstream event detector, and its rally ends are the ones
later corrected in `truth.json`. So `evaluate` reports that clip's hit and hitter figures as
unscorable rather than scoring one detector's hits against another's labels, and fails its rally
boundaries against ends that have since been superseded. That is the format's original flaw
showing, not a regression: `baseline.json` is that clip's record, and it has all six rallies
recovered.

Either way the targets are:

| Metric | Target |
|---|---|
| two-player tracking coverage during active play | ≥ 90% |
| shuttle trajectory coherence through rallies | visual judgement |
| hit precision / recall | measured, no target yet |
| hitter-assignment accuracy on clean hits | ≥ 90% |
| rally-boundary accuracy | ≥ 85% |

`evaluate` refuses to run on an unlabelled file rather than produce a number from nothing.

The coverage figure counts frames where a slot was filled, which is not the same as frames where
it held a player. Every scoring path now carries slot-plausibility measurements beside it and
withholds trust in the coverage number when they look wrong: `coverage_trusted` in
`baseline.json`, `NOT TRUSTED` in place of `PASS` in `evaluation.json`, and
`summary.tracked_player_plausibility` in `analysis.json`. See below for what is measured.

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

### Ground truth is what happened in the video, not what a detector said

The first labelling format stored one row per *detected candidate* with an `is_real_hit` flag.
That makes the labels a property of the configuration that produced them, and it went wrong
three separate times:

- a precision figure was reported for a configuration nobody had labelled, because the labels
  covered only the hits an earlier threshold had let through;
- every threshold change needed a frame-matching migration command to carry judgements across;
- the template's `detected_hitter` field was seeded once and never refreshed, so hitter accuracy
  was being scored against a run that no longer existed. Eight assignments had drifted, and the
  reported 98.1% was really **92.6%**.

`truth.json` stores only what is true of the video: the racket contacts that happened with the
player who made each one, the frame spans where rallies ran, and optional notes about visible
tracking failures. Nothing about any detector. A false positive is not recorded because it is
not a property of the video -- it is re-derived for whatever configuration is being scored, as
any prediction matching no contact. One labelling therefore scores every configuration.

Predictions are matched to contacts closest-pair-first within a tolerance (0.25s by default,
comfortably inside the 0.3s minimum interval between two real contacts, so it cannot merge
them). Each contact is claimed once, so a burst surviving consolidation scores one true positive
and one false positive rather than two hits.

```bash
badminton-poc truth --video <clip>.mp4 [--migrate]   # create or migrate the truth file
badminton-poc baseline --video <clip>.mp4            # score the current run, save baseline.json
```

Clip 2's labels migrated losslessly: 62 contacts, 6 rallies, 128 candidate verdicts discarded as
detector-specific.

### Baseline, clip 2, current configuration

| | |
|---|---|
| hit detection | P **0.625** / R **0.902** / F1 **0.738** — 55 TP, 33 FP, 6 FN |
| hitter assignment | **92.6%** (50 correct, 4 wrong, 1 unassigned, of 55 matched) |
| rally segmentation | 4 of 6 recovered; 1 split, 1 spurious, 0 missed |
| — median start error | −0.18s |
| — median end error | **+0.52s** (median absolute 0.68s) |
| two-player coverage, active play | **92.7%** (near 100%, far 92.7%) |
| far identity breaks | 7 · swaps 0 |

Every figure comes from one recording, and both the confidence and swing thresholds were chosen
on it. This is a development baseline, not a validation.

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

### Three attempts at rally boundaries, all measured, all failed

Rally-boundary accuracy is the failing criterion: 4 of 6 on the development clip against a
target of 85%. Ends run long by a median 3.7s and never short. Three routes were tried against
the labelled clip; none improved it.

**A play gate on shuttle motion.** Shuttle speed of 4 px/frame or more covers 78% of play
frames but also 22% of dead ones. Retried as an end-trimming rule rather than a per-frame gate
-- searching forward from a known rally start for the first sustained quiet period -- it
recovers 4 of 6, exactly what the current pipeline manages, and introduces two new failures: a
rally whose shuttle track drops out mid-point ends 1.55s early, and the last rally never finds
a quiet period because the clip ends first.

**Scoreboard OCR.** A score change is an independent signal that a rally ended, immune to the
hit detector. It is legible on one of the three clips. Clip 2 has no scoreboard in frame at
all, clip 3's is a 30px LED panel that resolves to coloured blobs, and only clip 5's digits are
readable. A method that works on a third of recordings is not a solution to this.

**A swing gate on hits.** The only measured discriminator that does not come from the event
detector. It works on precision, 0.62 to 0.84, and it makes rally boundaries **worse**: 0.67 to
0.33, with splits rising from one to three. Removing hits opens gaps longer than the rally
break inside real rallies and cuts them in half.

The reason all three fail is the same, and it is now precise. A rally ends at its *terminating
event*, and the terminating event is a landing. Landings arrive every 0.8 seconds throughout
dead time -- 47 of 68 gaps on the development clip sit at exactly the suppression floor, which
is the signature of a detector firing continuously while suppression samples it. Rally 1's end
stays at frame 493 under every configuration tried, +4.28s past the truth, because no amount of
hit filtering moves an end that a metronome sets.

### Not every racket contact is a rally shot

The labels contained a contradiction: rally 1 runs frames 83-236, and frame 395 is marked a
real racket contact, 2.65 seconds past the end of the rally it sits in. Asked about it, the
labeller's answer was that both are correct -- frame 395 is the near player *handing the shuttle
back* to the server. A real contact, and not a rally shot.

That is a flaw in the schema rather than in the labelling. Scoring a handback as a missed hit
punishes the pipeline for not reporting something that must never appear in a rally timeline;
scoring a detection of it as a false positive punishes the pipeline for seeing something that
genuinely happened. Contacts now carry `in_rally`, and a contact marked false is counted in
neither column, with detections and misses of them reported separately.

The same labelling session identified what the pipeline saw at frame 232: the shuttle landing
on the floor, after which TrackNet jumped to tracking a ceiling light. That is a false-positive
category worth naming.

**Ending a rally at its last hit instead** was the obvious follow-up and does not help either:
4 of 6, unchanged. Rally 1's end moves only from frame 493 to 482, because its last *hit* is
itself a false positive during dead time. The landing metronome was not the whole story -- the
rally genuinely contains spurious hits reaching past the true end, and both the last hit and
the next landing sit beyond it.

Nor does the combination. Every pairing of end-at-last-hit with a swing gate scores 4 of 6 or
worse, and the aggressive pairings reach 1 of 6:

| rally end | swing gate | precision | recall | recovered | median end error |
|---|---|---|---|---|---|
| landing | off | 0.62 | 0.90 | **4/6** | +0.52s |
| landing | 3.6 | 0.84 | 0.67 | 2/6 | +0.52s |
| last hit | off | 0.62 | 0.90 | **4/6** | +0.42s |
| last hit | 3.6 | 0.84 | 0.67 | 1/6 | −0.68s |

**Rally boundaries cannot be fixed downstream of this event detector.** Four routes were
measured and none beats 4 of 6. The ceiling is set by false-positive hits *inside* rallies, and
every available signal that removes enough of them also splits the rallies apart. Fixing this
means fixing hit detection at source, which means replacing the detector.

### The labels were wrong about where rallies end

Every attempt at rally boundaries had been scored against labelled spans that nobody had gone
back and checked. Rally 2 ran 773-1098, and the shuttle track showed the tracker frozen on one
pixel from frame 1005 to 1080. Pulling the frames out of `work.mp4` and looking at them settled
it: at 933 the far player is lunging low at the net with the point live, and by 941 both players
are upright and walking. The labelled end was 2.7 seconds into dead time.

So all six ends were re-measured the same way, at 8-frame granularity, each bracketed by one
frame with the point visibly live and one with both players disengaged. Five of six had been
labelled 20 to 40 frames late and rally 2 by 160, which is a convention drifting rather than a
mistake: a rally ends when the shuttle lands, not when the players finish walking.

| rally | was | now | what the frames show |
|---|---|---|---|
| 1 | 83-236 | **0**-198 | the clip opens mid-rally, far player already mid-stroke at frame 0 |
| 2 | 773-1098 | 773-**938** | live at 933, both players upright by 941 |
| 3 | 1502-2767 | 1502-**2740** | shuttle visible falling 2721-2737, lands about 2740 |
| 4 | 3821-5217 | **3810**-**5175** | near player airborne at the net at 5163, turned away by 5179 |
| 5 | 6185-6439 | **6171**-**6378** | near player lunging at 6364, both upright by 6376 |
| 6 | 7024-7355 | **7013**-**7375** | still live at 7355; the clip ends at 7375 and cuts the rally off |

The correction was applied to all six before any of them was scored, and it is not a correction
that flatters anything. It moves two ends earlier than the new detector reports and four later,
it made a rest-based end rule fail that the uncorrected labels had passed, and starts were
aligned to the labelled serve contact, moving three of them 11 to 14 frames earlier against a
detector whose starts already ran late. Scored on the corrected labels, the **old** pipeline
drops from 4 of 6 to 1 of 6.

Two rallies turn out to be truncated by the recording itself. That is worth having in the schema
rather than as a footnote: a clip that starts mid-point has a rally whose start cannot be
observed, and scoring a detector against a start that is not in the video is meaningless.

### Replacing the event detector

The conclusion of the previous session was that rally boundaries cannot be fixed downstream of
upstream's event detector. That turned out to be right, and the way past it was to stop scoring
frames and start segmenting the track.

The measurement that opens the door, on the development clip:

| where | median residual of a quadratic fit over 7 frames |
|---|---|
| in a rally, away from any hit | 0.88 px |
| at a real hit | 5.14 px |
| dead time, tracker frozen | 0.00 px |

A shuttle in flight is almost perfectly smooth, and a racket contact is where that smoothness
breaks. `flight.py` cuts the track into **flight segments** -- maximal runs a smooth motion model
fits at real speed -- and reads everything off the segmentation. A hit is a join between two
segments. A landing is how the last segment of a chain ends. A rally is a chain of segments
joined by hits. Dead time is whatever no segment covers, which needs no detector of its own.

Four things had to be got right, and each was measured rather than guessed:

**Not every join is a contact.** A join with a large turn is a hit; a join with almost no turn is
one flight the tracker briefly lost. Treating the second kind as a rally break split two rallies
in half. Real contacts turn a median 150 degrees, frames inside a flight a median 9.

**Gaps have to be generous, and something else has to close them.** Three intra-rally dropouts run
0.8s, 1.53s and 1.55s, so a tight gap limit splits rallies. What separates a long intra-rally
dropout from the gap between two rallies is not its length but whether the shuttle track came to
a standstill inside it. On this clip that separates them exactly: the one gap holding a sustained
standstill is between rallies, and all seventy-odd intra-rally gaps hold none, losing the shuttle
to invisibility instead.

**A frozen position is usually furniture.** TrackNetV3 emits one shuttle per frame whether or not
one is in play, and between points it settles on a ceiling light -- pixel-identical across
fourteen separate stretches totalling 2340 frames, while the one genuine shuttle-at-rest appears
once for 85. Recurrence tells them apart and needs no court geometry.

**Where the contact is placed decides who gets blamed for it.** The shuttle leaves the racket
fast, so a hit reported a few frames late already sits nearer the opponent. Every wrong
assignment under a late rule was a near-player shot credited to the far player. Putting the
contact on the last frame of the incoming flight -- the last moment the shuttle is still
unambiguously on its way *to* the racket -- fixes it, at the cost of reporting contacts a median
three frames early:

| contact placed at | precision | recall | F1 | hitter |
|---|---|---|---|---|
| the worst-fitting frame | 0.736 | 0.898 | 0.809 | 0.885 |
| the middle of the gap | 0.750 | 0.915 | 0.824 | 0.926 |
| **the last incoming frame** | **0.778** | **0.949** | **0.855** | **0.927** |

A separate fix in `assoc.py` helped both detectors. A hit landing in a one- or two-frame dropout
of one player left the only remaining candidate to win by default, and which frame a detector
lands on is close to chance. Nobody crosses a court in 60ms, so the nearest frame that has both
players is now borrowed. That took the old pipeline's hitter accuracy from 92.3% to 94.1%.

The last piece is what a rally *is*. Two detected chains on this clip are not rallies -- one is
the handback before a serve, the other somebody knocking a shuttle about during a break -- and
both read as contacts by one player alone, `NNN` and `FF`, while all six real rallies carry
contacts from both sides. Requiring both players removes both without touching anything real. It
asks only that both appear, not that they alternate, because at 93% assignment accuracy a real
rally routinely shows runs of one name.

### Milestone 1, measured

Clip 2, corrected labels, `event_source: flight`:

| criterion | target | result |
|---|---|---|
| both players tracked in active play | >= 90% | **92.9%** |
| hitter assignment on detected hits | >= 90% | **92.5%** |
| rally boundaries approximately correct | >= 85% | **6 of 6** |
| hit detection scorable | precision/recall | **0.81 / 0.95**, F1 0.875 |

Two commits have moved those numbers since they were first recorded, and re-scoring every
intermediate commit against the same labels says exactly which. `Gate contacts on racket reach`
left three hits with no hitter rather than a guessed one, taking hitter accuracy from 92.7% to
92.5% over 53 scored hits instead of 55. `Recover contacts that barely deflect the shuttle` added
one prediction on this clip and it was a false positive, taking precision from 0.824 to 0.812 and
F1 from 0.882 to 0.875; recall did not move, so the contact it recovered was not one of this
clip's three missed hits. Both were judged worth it for reasons measured elsewhere, and the table
above is the current run, not the best one.

Against the same labels and the same clip, upstream's detector scores 0.62 / 0.90 on hits, 94.1%
on hitters and 1 of 6 on rallies. The old path is still runnable with `event_source: tracknet`,
so this is a comparison rather than a claim.

Rally ends still run a median 0.51s late, because the last flight segment includes the shuttle's
descent after the final shot. That is a bias with a known cause, well inside the one-second
tolerance, and it is the obvious next thing to shave.

**This is one clip.** Nothing here is cross-validated. Clips 3 and 5 have seeded truth drafts
sitting at `labelled: false` and until somebody works through them the flight detector's
parameters rest entirely on the recording they were measured against. What it does on the
unlabelled clips, for whoever picks that up: clip 1 six rallies, clip 3 five against a six-rally
draft, clip 5 four against an eight-rally draft with one detected rally running 23.5 seconds,
which is either a long point or two rallies merged and only the video can say which. All four
clips now run the flight detector, which on three of them is a preference argued from clip 2
rather than evidence about those clips.

### What the annotated video was showing that was not there

Watching the output, the shuttle marker sat frozen mid-frame between rallies with LANDING
pulsing over it. Three separate causes, and none of them was the shuttle.

**The marker.** TrackNetV3 must emit a shuttle on every frame whether or not one is in play, so
between points it settles on a ceiling light and holds one pixel. The renderer drew that like any
other detection: a bright filled dot, mid-court, apparently hovering. The detector already knew
better, because the same recurring-freeze test that keeps a light out of the flight segmentation
identifies the position, and it is the same fixture in all four clips at (918, 222), accounting
for 860 to 2991 frames each. `analysis.json` now carries `shuttle_fixed_points` whichever detector
produced its events, and a sample sitting on one is drawn as a grey cross labelled `tracker
parked` and kept out of the trail. Still drawn, because hiding it would hide a real failure of
the tracker; no longer drawn as a shuttle.

**The pulsing.** Clips 1, 3 and 5 were still on `event_source: tracknet`, where landings arrive
every 0.8s straight through dead time: 34, 65 and 50 of them outside any rally, each flashed for
12 frames. Those clips now use the flight detector, which puts one landing at the end of each
rally and none anywhere else. Landing counts fell from 51, 82 and 75 to 6, 5 and 4, and no clip
now has a single landing outside a detected rally.

**The orphans.** A chain the rally gates rejected still left its events behind, because the
timeline was assembled before the gates ran. Clip 2 flashed two landings for chains it had
decided were not rallies. Events whose rally did not survive are now pruned, which also removed
five false hits and took hit precision from 0.778 to **0.824**, F1 to **0.882**.

Clip 2 after the fix: hits 0.82 precision, 0.95 recall, F1 0.882; hitter 92.7%; rally boundaries
6 of 6; coverage 92.9%.

### Clip 3, labelled and scored

Clip 3 is now labelled and `outputs/test-video3/baseline.json` is its accuracy record. **The
labels are model-produced, not hand-made** -- written by reading frames of `work.mp4` against the
raw TrackNet shuttle CSV -- so they are weaker evidence than clip 2's, and `_provenance` in the
truth file says so. The same model that tuned this pipeline produced them, which means a
systematic mistake would read as agreement rather than as failure.

| criterion | target | clip 2 | clip 3 |
|---|---|---|---|
| both players tracked in active play | >= 90% | 92.9% | **98.3%** |
| hitter assignment on detected hits | >= 90% | 92.5% | **90.0%** (9 of 10) |
| rally boundaries approximately correct | >= 85% | 6 of 6 | **4 of 4** |
| hit detection | precision / recall | 0.81 / 0.95, F1 0.875 | **0.45 / 0.71, F1 0.556** |

Three criteria clear their targets on a second recording. Hitter assignment clears it on ten
scored hits, which is too few to mean much. **Hit detection does not transfer**: precision falls
from 0.81 to 0.45. Of the 12 false positives in 22 predictions, 3 are a second detection of a
contact already found (bursts that survived consolidation), 5 are extra hits inside real rallies
and 4 are an entire invented rally at frames 2017-2122. Of the 4 missed contacts, 2 sit in the
stretch where the shuttle track is lost altogether.

**The clip is far thinner than its draft suggested, and most of what the draft contained was
not there.** The seeded draft had 6 rallies and 32 candidate contacts. The clip actually holds
**4 rallies and 14 rally shots, all inside its first 30 seconds**; labelled play covers 723 of
5031 frames, 14.4%. The rest is an interval. Specifically:

- draft rallies 5 (2296-2460) and 6 (3949-4034) contain no play at all -- the court is empty and
  officials are standing about at the far end;
- draft rally 1 (9-477) is two rallies: the shuttle lands at about frame 155, both players stand
  still, and the near player picks it up at 300 and serves again at 375;
- draft rally 3 ends at 1702 but play runs to about 1790;
- the candidates at 1068 and 1152 are the tracker sitting on a shuttle lying on the floor, which
  the near player then walks over and picks up.

From frame 1790 to the end -- 53 of the clip's 85 seconds -- nobody plays. The players collect
shuttles and hand-throw them, the near player catching with both hands raised; then the court
empties entirely. No racket contacts are recorded there, so every hit the pipeline reports after
1790 is a real false positive, and the invented rally at 2017-2122 is the detector firing on
hand-thrown shuttles. Both pose slots stay filled throughout by stationary bystanders (the far
slot's box moves 1px between frames 2114 and 2182), which is why coverage must be measured over
labelled spans and not over the clip.

One contact is recorded with `in_rally: false`: at frame 315 the near player flicks the shuttle
up to himself before serving. The detector found it, and the in_rally mechanism counted it in
neither column, which is what it exists for.

**What this says about the pipeline.** Player tracking and rally boundaries transfer. Hit
detection was tuned on clip 2 and its precision is not a property of the detector but of that
clip: the threshold that gives 0.81 there gives 0.45 here. The largest single cause is that
nothing asks whether a rally is being played at all before reporting hits -- the same blind spot
that let a seated umpire pass as a tracked player.

### Where it stands

All four milestone-1 criteria pass on clip 2 with the flight-segment detector, against labels
that were re-measured from the video rather than taken on trust. Three of the four pass again on
clip 3 against model-produced labels. Hit detection does not: its precision falls from 0.81 to
0.45, and that is the honest headline.

**Player tracking and rally boundaries look like properties of the pipeline. Hit-detection
precision looks like a property of clip 2.** It was tuned there and does not survive the move.
Clip 1 is a poor clip -- occlusion alone costs it ten points of far-player coverage, so it cannot
pass the tracking criterion however good the code is -- and is kept as a regression clip, not an
acceptance one.

Clip 3 also turned out to be thin: 4 rallies and 14 rally shots in its first 30 seconds, against
clip 2's 6 rallies and 62 contacts, with 53 of its 85 seconds containing no play at all. It is
enough to falsify a claim, as it just did, and not enough to establish one.

The next thing worth doing is fixing what clip 3 exposed rather than labelling more footage: the
detector reports hits when nobody is playing, and invented a whole rally out of two people
hand-throwing shuttles during an interval. Clip 5 is the remaining unlabelled clip, and its
camera is steady once the trailing junk frames are discounted.

`badminton-poc report --video <clip>` renders any clip's `analysis.json` as a page, with every
figure carrying the caveat it rests on. `badminton-poc refresh-labels` re-seeds a ground-truth
file against a new run while keeping judgements already made.

### Six clips off YouTube, and the four that were never going to work

The one thing the milestone-1 numbers needed was a second recording, so six singles clips were
pulled off YouTube to find one. Four of them cannot be used, and none of the four fails in a way
that looks like failure from the pipeline's output — each one just reports no rallies, which is
also what a bug looks like. Two of them cost a hand-fitted court before the real reason showed up,
which is what `tools/screen_clip.py` now exists to prevent.

| clip | camera drift | what it is | verdict |
|---|---|---|---|
| yt-armori | 231 px, 18% of width | handheld, low, close behind the baseline | reject |
| yt-purplehall | 218 px, 23% of width | handheld, dark hall, panning between courts | reject |
| yt-bluehall | fixed | side-on and close; the far half is outside the frame | reject |
| yt-irish | fixed | the framed court is empty; the match is off the right edge | reject |
| yt-ccaa | fixed | single court, elevated behind the baseline, whole court in view | fitted |
| yt-melbourne | fixed | multi-court hall, wide lens, umpire chair inside the frame | fitted |
| yt-warehouse | 40 px step at frame ~3250 | multi-court hall viewed down the row | fitted, with a caveat |

**The camera has to be measurably fixed, not apparently fixed.** Phase-correlating sample frames
against the first one puts a number on it, and the number separates the clips cleanly: four sit
under 4 px across the whole clip and two drift by a fifth of the frame width. Warehouse is the
interesting one. It holds still for 3136 frames, shifts 40 px horizontally and 2.5 px vertically
somewhere before frame 3360, and then holds still again for the remaining 5600. A single court
polygon is wrong for one side or the other of that step, by 0.021 of frame width. The net line is
a `y` threshold and barely moves, so near/far assignment survives it; what does not survive is a
player near a sideline, who can be on court before the bump and off court after it.

**yt-irish is the one worth remembering.** The camera is fixed, the framing is a textbook
elevated view of a full court, and the pipeline found two players on 87% of frames. The court in
frame is empty for all 4500 frames: the match is being played on the next court along, mostly
outside the right edge, and the two people it tracked are those players.

Every check that existed when it was screened passed it. The slot-plausibility measurements
described below do not: over its detected rallies the near slot ranges 0.09 body heights
sideways and the far slot moves 0.21 body heights per second, against 0.48 to 0.90 and 1.5 to
4.9 for genuine tracks. Three warnings, and they are right. That does not rescue the screening
tool, which runs before the pose pass and so has no track to measure; it means the failure is
visible in `analysis.json` after a run rather than only to someone watching the video. It is
also a thin sample -- two detected rallies, 239 slot-frames -- so treat it as one clip agreeing,
not as a check that has been validated on wrong-court footage.

### The far player was too small for the input size, on every 1080p clip

Both fitted multi-court clips missed the far player far too often to pass the tracking criterion
-- 65% of frames on warehouse and 52% on melbourne, against a 90% target -- and the obvious
suspect was the court polygon, since that is what decides who counts as on court. It was not. On
melbourne frame 1415 the far player is plainly visible, mid-rally, and the pose pass returned five
detections, none of them at that position. Re-running the same model on the same frame at larger
input sizes:

| input size | detection confidence for the far player |
|---|---|
| 1920 (configured) | 0.149, below the 0.25 threshold |
| 2560 | 0.262 |
| 3200 | 0.741 |

The player's box is 8% of frame height. That is where `yolo11m-pose` at `imgsz: 1920` starts to
lose people, and it is exactly the geometry that a good clip produces: a camera far enough back
and high enough to see the whole court makes the far player small. The two original test clips do
not show it because their far player is larger in frame; they pass at 1920 with 93% and 89%
coverage, which is why this sat undetected until a clip with a smaller far player arrived.

Cost on an RTX 3070, measured: 52 ms/frame at 1920, 92 at 2560, 144 at 3200.

Which is worth doing only where size is the thing that is wrong, and on two of the three fitted
clips it is not. Warehouse ran again at 3200 and gained four points, 65.4% to 69.5%, for 2.7x the
runtime. `tools/screen_clip.py` reports the far player's height in pixels after the model's own
resize rather than as a share of the frame, because that is the number that separates the cases:
melbourne's far player is 103 px at 1920 and needs more, warehouse's is 194 px at 3200 and never
did.

### Warehouse: the camera was behind the baseline but not elevated

Warehouse kept its far player on 69% of rally frames whatever the input size, and the reason is
visible the moment you crop to the right place. Its camera sits down the row of a multi-court hall
at roughly playing height, which squeezes the whole far half into the band y 0.549 to 0.616 --
6.7% of frame height for 6.7 m of court. The near player is 35% of frame height and stands between
the camera and that band. On frame 3340, mid-rally, their legs cover it from x 0.45 to 0.53 and
the far player is simply behind them.

Counted over every rally frame with no far player: 1326 have no detection anywhere near where the
far player normally stands, and 351 have one that was rejected. It is occlusion, not detection
sensitivity and not the court polygon, and nothing in the pipeline can recover a player the camera
cannot see.

The clip also takes a one-off 40 px horizontal pan between frames 3136 and 3360. That looked like
a promising second cause -- `foot_outside_court` rejections jump from 170 to 3701 across the step
-- but shifting the polygon by the measured 40 px puts a far-side candidate back on court in only
18 of the 1202 affected frames. The extra rejections are bystanders wandering the hall later in
the clip, not the court moving out from under the polygon. The pan is real and worth knowing
about; it is not what costs the coverage.

So the recording-conditions list understates one line. **Elevated** is not a nice-to-have that
improves the view, it is what stops the near player from standing in front of the far one.

### None of the six is a second acceptance clip

| clip | why not |
|---|---|
| yt-armori | handheld, 18% of frame width |
| yt-purplehall | handheld, 23% of frame width |
| yt-bluehall | side-on; the far half is outside the frame |
| yt-irish | the framed court is empty |
| yt-warehouse | camera not elevated; the near player occludes the far half |
| yt-melbourne | far player 103 px at `imgsz: 1920`; untested at a larger size |
| yt-ccaa | closest. 4 rallies, 26 hits, near player 99%, far player 88% overall but 65% during rallies |

yt-ccaa is the one to pick up. Its camera is fixed to within 4 px, it is a single court with a
real singles match, and fitting its court took it from 0 rallies to 4 -- the screening rectangle
had been tracking the umpire on his chair, whose foot position never moved from (0.906, 0.508),
and every hit was being discarded as out of racket reach. What stands between it and the tracking
criterion is its far player dropping out for up to two thirds of a second at a time. Only 30% of
those missing frames sit in gaps of five frames or fewer, so interpolation would take 65% to about
76% and no further. It is 30 fps, and more input pixels make it worse rather than better: at 3200
the far player falls out of the detection list entirely while background spectators a third her
size come back at 0.75 confidence.

### Coverage counts filled frames, not the right person

Twice the pipeline has reported a passing coverage figure while a slot held someone who was not
playing. Under the screening rectangle, yt-ccaa's far slot sat on the umpire in his chair and
reported 94%; relaxing the detection threshold on the same clip later gave 90% during rallies
with the slot on officials standing behind the back boards. Both read as a pass, and a false pass
is worse than a fail because a fail gets investigated.

Three measurements separate a player from a bystander, taken over rally frames only and scaled by
the tracked person's own body height so that frame rate, resolution and distance from the camera
all divide out:

| measurement | what it is | flagged below |
|---|---|---|
| median detection confidence | how sure the pose model was | 0.45 |
| median movement | frame-to-frame travel, body heights per second | 0.25 |
| lateral range | how far the slot ranged sideways, in body heights | 0.8 |

Across every clip that reaches rally detection, measured over that clip's *detected* rallies,
which is what `analysis.json` reports:

| clip | near move / sway | far move / sway | verdict |
|---|---|---|---|
| test-video2 | 0.55 / 3.09 | 0.50 / 3.27 | no warnings |
| yt-ccaa | 0.57 / 1.50 | 0.90 / 2.54 | no warnings |
| yt-melbourne | 0.66 / 1.82 | 0.88 / 1.79 | no warnings |
| yt-warehouse | 0.72 / 3.86 | 0.74 / 3.69 | no warnings |
| yt-irish | 0.38 / 0.09 | 0.21 / 0.80 | three warnings |

Only test-video2 has hand-labelled rallies, and `baseline.json` measures it over those instead:
0.55 / 2.98 near and 0.48 / 4.88 far. Same clip, same run, different spans, so the two files do
not agree to the decimal and neither is wrong. The ranges quoted above and in the warnings --
0.48 to 0.90 for movement, 1.5 to 4.9 for lateral range -- span both sets, which is why their
ends do not all appear in the table.

The thresholds sit below the genuine range rather than halfway between the two, because a real
player having a quiet rally must not be flagged. Nothing here decides anything: the coverage
number is reported unchanged, with the measurements next to it, so a figure earned by tracking
furniture says so.

**What it cannot do.** It catches a slot on something that does not move like a player. A wrong
person who does move like one -- a player on an adjacent court close enough to fill the frame --
would pass it, and yt-irish is one clip agreeing rather than evidence it catches that class. The
lowest genuine confidence measured so far is yt-ccaa's far player at 0.64, against 0.24 for the
officials, so the confidence threshold has roughly one and a half times the margin the movement
one does.
