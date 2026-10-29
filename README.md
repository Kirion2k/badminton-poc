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
- camera behind a baseline and **elevated** -- not optional: at head height the near player hides the far one (see "The first club recording" below)
- nobody walking between the camera and the court
- 1080p, 60 fps genuine constant frame rate
- 1–5 minutes
- the original file, not a screen recording of one

Two of those are worth more than they look, because four downloaded clips died on them.

**Fixed means measurably fixed.** A clip that looks steady can still drift a fifth of the frame
width, and a tripod can be knocked once and hold still either side of the knock. One polygon
describes the whole video or it does not; `tools/screen_clip.py` measures the drift so the answer
is a number rather than an impression.

It measures the drift that *persists*, not the worst frame in the clip. Judging on the worst
frame, the tool rejected clips 3 and 5 for 14.9% and 23.0% of frame width when neither camera
moves more than 0.3px: both are phone screen recordings of YouTube, and both end with the iOS
Control Centre swiped in to stop the capture. It now discards the three largest samples before
deciding, which a stop gesture fits inside and a moving camera does not -- a knocked tripod
leaves every later sample displaced, not three. The outlier frames are still reported by number,
because a clip that is steady apart from a few frames has something wrong with those frames.

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
| hit detection scorable | precision/recall | **0.82 / 0.95**, F1 0.882 |

Two commits have moved those numbers since they were first recorded, and re-scoring every
intermediate commit against the same labels says exactly which. `Gate contacts on racket reach`
left three hits with no hitter rather than a guessed one, taking hitter accuracy from 92.7% to
92.5% over 53 scored hits instead of 55. `Recover contacts that barely deflect the shuttle` added
one prediction on this clip and it was a false positive, taking precision from 0.824 to 0.812 and
F1 from 0.882 to 0.875; recall did not move, so the contact it recovered was not one of this
clip's three missed hits. Both were judged worth it for reasons measured elsewhere.

`Merge the second hit a shuttlecock's drag knee produces` has since taken precision back to
**0.824** and F1 to **0.882**, by running hit consolidation on the flight path where it had never
run. It removed one false positive here and four on clip 3, and cost no recall on either.

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
| hit detection | precision / recall | 0.82 / 0.95, F1 0.882 | **0.56 / 0.71, F1 0.625** |

Three criteria clear their targets on a second recording. Hitter assignment clears it on ten
scored hits, which is too few to mean much. **Hit detection does not transfer**: precision is
0.56 against clip 2's 0.82.

It was 0.45 when this clip was first scored. Four of those twelve false positives were the same
contact reported twice, from the drag knee described below, and consolidating them took precision
to 0.556 with no loss of recall. What is left is 8 false positives in 18 predictions: 4 are an
entire invented rally at frames 2017-2122 and 4 are extra hits inside real rallies. Of the 4
missed contacts, 2 sit in the stretch where the shuttle track is lost altogether.

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

### Seven attempts at a play gate, all measured, none of them separates

Clip 3's detector invented a rally at frames 2017-2122 out of two people hand-throwing shuttles
during an interval. The obvious cure is a gate: suppress hits when no rally is in progress. One
already exists -- `detect_play_spans` keys on shuttle motion -- but it is disabled on the flight
path and would not have caught this, because a thrown shuttle moves.

Seven signals were measured, against clip 2 as well as clip 3, because clip 2 is the only clip
with hand labels and so the only place a threshold fitted on clip 3 can be falsified:

| signal | real rallies | the invented rally | why it fails |
|---|---|---|---|
| median shuttle speed | 4.85-6.29 px/frame | 5.46 | inside the range |
| player motion, slower slot | 0.322-0.742 bh/s | 0.235 | 1.4x margin, fitted to one example |
| shuttle p90 speed | 15.8-22.8 | 11.4 | shuttle collection scores 29.8, above every rally |
| wrist speed p90 | 2.37-8.71 bh/s | 4.13 | a real rally scores *lower* than the invented one |
| per-rally slot plausibility | 0-2 warnings | 2 warnings | flags two real rallies; the invented one has the highest far-slot sway of all |
| shuttle lateral range | 0.234-0.568 of width | 0.122 | separates clip 3; a clip 2 dead gap scores 0.478, above four real rallies |
| slot identity switches | up to 3 ids | 2 ids | clip 2's real rallies switch more |

**So no gate was added.** Every signal that looked decisive on clip 3 overlapped on clip 2, and
the lateral-range one would have dropped real rallies there. Fitting a threshold to a single
invented rally is how a number that means nothing gets into a scoring path.

What was added is `play_evidence`, attached to each detected rally in `analysis.json`. It reports
four measurements over the rally and names the ones falling outside `LABELLED_RALLY_RANGES` --
the range every hand-labelled rally in the project sits in, ten rallies across two clips. It
suppresses nothing, returns no verdict, and a test asserts it never grows one.

On clip 3 the invented rally falls outside on three of four; clip 2's six rallies fall outside on
none. Clip 3's real rally 3 falls outside on one -- usable shuttle fraction 0.808 against a bound
of 0.81 -- and that is left as it is rather than widened, because it is the honest demonstration
of the point: these bounds come from ten rallies and a real rally already sits on the edge of
them. Read the numbers, not the count.

**The gate was also the smaller half of the problem.** Of clip 3's twelve false positives at the
time, only 4 were the invented rally; 4 were the same contact reported twice and 4 were spurious
hits inside real rallies. Consolidating the duplicates has since been done and took precision from
0.45 to 0.556 on clip 3 and 0.812 to 0.824 on clip 2 -- more than a perfect play gate would have
been worth on either. The remaining 4 in-rally false positives are the next target.

### One contact, two hits: the shuttlecock's drag knee

A shuttlecock does not fly like a ball. It sheds its horizontal speed to drag within about a
third of a second, and there is a moment in every struck flight where the horizontal speed is
spent and the vertical speed has not yet built -- the shuttle is barely moving at all. For a few
frames it travels below `min_flight_speed_px_s` (150 px/s), the segmenter stops calling it
flight, and **one physical flight is cut into two segments**.

The join across that cut turns 60-80 degrees, because the path genuinely bends from horizontal
to vertical. `min_turn_degrees` is 60, chosen because real contacts sit at a median 150 degrees
and frames inside a flight at 9. The knee falls between the two and is read as a racket contact.
So a real contact is followed about 0.3s later by a second hit that never happened.

Clip 3 does it at frames 106/123, 810/831 and 1619/1636; clip 2 at 7054/7071. On clip 3's frames
110-145 it is plain in the raw track: struck at 109, the shuttle runs left at 47px/frame decaying
to 11 by frame 122, then from 123 moves 0.7px/frame sideways while falling 3.4px/frame.

`consolidate_events` exists to collapse exactly this, and had never been run on the flight path:
`analyze` took `events = raw_events` whenever a flight model was present, so
`min_hit_interval_seconds` applied to nothing the flight detector produced. Running it there
costs no recall on either clip and removes 4 false positives on clip 3 and 1 on clip 2:

| | clip 2 before | clip 2 after | clip 3 before | clip 3 after |
|---|---|---|---|---|
| precision | 0.812 | **0.824** | 0.455 | **0.556** |
| recall | 0.949 | 0.949 | 0.714 | 0.714 |
| F1 | 0.875 | **0.882** | 0.556 | **0.625** |

The claim consolidation rests on is about badminton rather than about this detector: two racket
contacts are never 0.3s apart. The closest pair in the project's hand labels is 0.350s (clip 2,
frames 7227 and 7248), above the window, and a test holds that pair against it.

Checking that interval turned up a labelling fault: clip 2 held **two byte-identical contact rows
at frame 2006**, 0.000s apart. A racket cannot strike the shuttle twice in one frame, and the
hitters either side read `far, far` with both rows and alternate correctly with one. It scored
correctly only by luck -- `match_contacts` claims contacts by frame number, so the two collapsed
to one entry and a prediction did reach that frame. With nothing matching it, both rows land in
the missed list and the file carries a false negative no detector change could ever clear.
The duplicate is removed (labelled contacts 62 to 61, rally shots 60 to 59, no accuracy figure
moved) and `evaluate_against_truth` now refuses a truth file with two contacts on one frame.

**It does not fix the cause.** The knee still splits flights; the second hit is merged away
afterwards. Two of the pairs -- clip 3's 810/831 at 0.355s and 881/899 at 0.304s -- fall outside
the window and survive as false positives. Fixing it at source means the segmenter not cutting a
flight where the shuttle slows, which is a change to what counts as flight, and that is worth
doing against a clip labelled for it rather than guessed at here. Two discriminators were tried
on the link itself and both overlap on clip 2: the speed ratio across the join (real contacts run
as low as 0.22, false ones as high as 3.18) and the single-frame velocity jump (a tracker
teleporting 576px scores 79.6, far above any real contact).

### Where it stands

All four milestone-1 criteria pass on clip 2 with the flight-segment detector, against labels
that were re-measured from the video rather than taken on trust. Three of the four pass again on
clip 3 against model-produced labels. Hit detection does not: its precision is 0.56 there against
0.82 on clip 2, and that is the honest headline.

**Player tracking and rally boundaries look like properties of the pipeline. Hit-detection
precision looks like a property of clip 2.** It was tuned there and does not survive the move.
*(Superseded: most of that gap was the detector firing at the top of a lift. See "The top of a
lift is not a racket" at the end of this file -- clip 3's precision is 0.9091 now.)*
Clip 1 is a poor clip -- occlusion alone costs it ten points of far-player coverage, so it cannot
pass the tracking criterion however good the code is -- and is kept as a regression clip, not an
acceptance one.

Clip 3 also turned out to be thin: 4 rallies and 14 rally shots in its first 30 seconds, against
clip 2's 6 rallies and 62 contacts, with 53 of its 85 seconds containing no play at all. It is
enough to falsify a claim, as it just did, and not enough to establish one.

The duplicates are fixed: one racket contact was producing two hits, and consolidating them took
clip 3's precision from 0.45 to 0.556 and clip 2's from 0.812 to 0.824 with no loss of recall. A
play gate was attempted before that and abandoned -- seven signals measured, none separates on
both labelled clips -- so each detected rally carries a `play_evidence` block instead, which
describes it without deciding anything.

What is left on clip 3 is 8 false positives: 4 an invented rally during an interval, 4 spurious
hits inside real rallies. Both need the hit detector to know something it currently cannot, and
both would be better attacked against a third labelled clip than against this one. Clip 5 is the
remaining unlabelled candidate, and its camera is steady once the trailing junk frames are
discounted.

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

### A second search, for club-hall footage (2026-09-16)

Milestone 2 found BST reading 0.47 on the one club-hall clip against 0.63 and 0.75 on two
tournament broadcasts, so the next recording has to be club footage. Four more clips were
screened. None qualifies.

| clip | what it is | why not |
|---|---|---|
| `vcESpUeecZk` (DK Way Copper MS final) | school sports hall, behind the baseline at head height | camera is operated: no stable window of 2 minutes anywhere in 32; 272 px sustained drift in the window pulled |
| `6meyO5TIeho` (Alberta Series MS final) | multi-court club hall, 8.8 stable minutes | low diagonal view; the far side of the court leaves the frame |
| `YeC9oHEzHM8` (Play Factory, Chennai) | wooden-floor club hall, stable for the whole 10 minutes | doubles: four players on court |
| `00a3X-xbBQk` (Kerala sub-junior U13 GS final) | academy tournament | longest stable window 64 seconds |

Worth keeping from it: DK Way's far player measured **218 px at 0.91 confidence**, the largest
seen on any public clip, and it was discarded only because the screening rectangle's net line
sits above where a far player's feet appear under a low camera. A low camera is not by itself
disqualifying for the far player's size. It is disqualifying for occlusion and, in every case so
far, it has come with a camera someone is holding.

The pattern across eleven screened clips: **club footage on YouTube is filmed the way club
players film** -- handheld or at head height, often doubles. The combination this pipeline needs
(tripod, elevated, whole court, singles) is what a tournament broadcast provides, which is why the
two clips that pass are both broadcasts. Recording a club match to the spec is likely faster than
finding one.

Screening a candidate without any GPU work: pull the whole video at 360p, phase-correlate a
sample every two seconds against the previous one, and report the longest run in which no step
exceeds 1% of frame width. That rejected three of these four in under a minute each.

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

### A line judge held the near slot for 2048 frames, and half of clip 5 disappeared

Clip 5 was picked up as the third clip to label, and it is not a marginal recording: a
tournament match, camera fixed behind the near baseline, both players plainly visible, far
player 126 px tall. The pipeline reported 95.0% near coverage, 89.4% far, no plausibility
warnings, and three rallies.

All three rallies sit in the first 56 seconds. From frame 2580 to the end -- 2048 frames, 44%
of the clip -- the near slot held a **seated line judge in a chair on the red surround** while
the real player stood on court in front of him. Nothing in the output said so.

| frames | what the near slot held |
|---|---|
| 0-43 | the judge, before he had been seen often enough to be judged static |
| 44-1270 | nobody: he is correctly rejected as `static_track` |
| 1271-2579 | the real player; the judge is eligible again but loses |
| 2580-4627 | the judge, for 2048 consecutive frames |

**Why nothing said so.** Slot plausibility is measured over *detected* rallies, and no rally is
detected in the span where the slot is wrong -- because the slot being wrong is what stops
rallies being detected. A failure that suppresses rally detection is invisible to a check that
only looks inside rallies. Coverage, meanwhile, counted the judge as a tracked player on every
one of those frames and reported near coverage of 100% during rallies.

The cost, measured: 32 of 41 hits dropped as unattributed, 40 of 52 events orphaned by the rally
gates, and no rallies at all after frame 2467 in a span holding 812 frames of moving shuttle --
39.6% of it, against 51.8% before the lock.

**Two defects hold the lock shut.**

*The foot point moves when the box does.* Mobility was measured from the foot point, which is
the bottom edge of the box, so it travels whenever the model loses the legs. Over frames
993-996 the judge's box collapsed from 125 to 86 px and his confidence from 0.78 to 0.42, so his
foot point climbed 43 px up his body while the box top moved 0.3 px, 481.7 to 481.6. That
cleared the 15.8 px threshold and mobility is permanent, so one flicker of a bad detection
bought a seated official lifetime eligibility. Mobility is now measured at the top of the box.
Heads do not fall off, and a player who really moves moves their head too.

That alone does not fix it: it moved the promotion from frame 993 to 1271, where a genuine
shift in his chair crossed the threshold instead. **A per-window motion test applied across
thousands of frames will eventually promote anything**, and promotion is for life.

*An incumbent never has to re-earn its slot.* Continuity (1.5) and proximity (0.8) are 2.3
points the occupant gets for free, against at most 1.0 for a stranger however well detected. At
frame 3600 the judge beat the real player -- track 4, confidence 0.938, proven mobile, standing
on court -- by 3.19 to 1.05, and beat him identically on every frame to the end of the clip.
Once a slot is wrong it stays wrong.

So the eligibility ladder gains a rung. `mobile` says a track moved once; `recent` says it is
still moving like a player, and a slot prefers a recent candidate to a stale one. A preference,
not a gate: with nobody recent on that side the stale candidate still fills the slot, and the
pick is recorded as `recently_mobile: false` next to `proven_mobile`.

**The horizon is 20 seconds, and both populations it separates were measured.**

| track | longest stretch reading as motionless |
|---|---|
| clip 2 near player | 7.38 s |
| clip 2 far player | 8.90 s |
| clip 3 near / far player | 3.36 s / 2.79 s |
| clip 5 near player | 3.05 s |
| clip 5's line judge | quiet 37 s when he took the slot, 98.5 s afterwards |

He reads as moving on 45 of 4628 frames, in two runs of 18 and 26 frames around a single
posture shift. Any horizon between roughly 10 and 35 seconds separates the two; 20 s is 2.2x
above the longest genuine stillness measured. In seconds rather than frames, because a frame
count judges a 60 fps clip and a 34 fps clip by different standards -- the same mistake an
absolute pixel threshold made across the two ends of the court.

**What changed.**

| | before | after |
|---|---|---|
| clip 5 rallies | 3 | 8 |
| clip 5 hits | 52 | 83 |
| clip 5 frames inside a detected rally | 1258 | 2030 |
| clip 5 near slot, frames 2580-4627 | judge 2048 | player 2025, judge 2 |
| clip 2 precision / recall / F1 / hitter / coverage | 0.8235 / 0.9492 / 0.8819 / 0.9245 / 0.9286 | unchanged |
| clip 3 `baseline.json` | | byte-identical |

The five recovered rallies fall at 2646-2681, 3008-3110, 3203-3325, 3506-3862 and 4400-4552,
which are the spans where the raw shuttle track was already moving before any of this was
changed. That is corroboration from a file the fix does not touch.

**What it does not establish.** Clip 5 still has no labels, so 8 rallies and 83 hits are counts,
not accuracy. The far slot also lost 81 frames to the first fix, of median box height 85 px
against 138 px for the frames it kept, sitting motionless at (0.577, 0.593) -- the line-judge
depth band this clip's config warns about, in one case holding the far slot for 36 frames inside
a detected rally. They look like officials rather than players, which makes the coverage figure
more honest rather than worse, but no label says so yet.

The blind spot in the plausibility check is also still there. It measures inside detected
rallies, so it still cannot see a slot failure that prevents rallies from being detected. What
changed is that this particular failure no longer happens, not that it would be reported if it
did.

### The top of a lift is not a racket, and it was a third of the false positives

Clip 3's hit precision of 0.56 was the honest headline of this project: the detector was tuned
on clip 2 and did not survive the move to a second recording. It survives now, and what was
wrong was not tuning.

A racket and gravity both reverse the shuttle. At the top of a lift or a clear the shuttle
rises, slows, stops and falls, and a detector reading curvature sees the same 180 degrees there
that it sees at the end of somebody's racket. **Energy separates them.** A racket adds it: the
shuttle leaves faster than it arrived, or leaves in a direction gravity could not have sent it.
Gravity removes it: at the top of the arc the shuttle is at its slowest, and gravity can only
turn a rising shuttle into a falling one, never the reverse.

So a join is discarded when *both* hold -- rising in, falling out, and no faster leaving than
arriving. Both are needed. A smash taken at the top of the bounce turns over exactly the same
way; it leaves three times quicker, and the speed condition is what keeps it.

| | before | after |
|---|---|---|
| clip 2 precision / recall / F1 | 0.8235 / 0.9492 / 0.8819 | **0.8852** / 0.9153 / **0.9000** |
| clip 3 precision / recall / F1 | 0.5556 / 0.7143 / 0.6250 | **0.9091** / 0.7143 / **0.8000** |
| clip 3 spurious rallies | 1 | **0** |
| clip 2 hitter accuracy | 0.9245 | 0.9245 |
| clip 3 hitter accuracy | 0.90 | 0.90 |
| rally boundaries | 6 of 6, 4 of 4 | unchanged |
| coverage | 0.9286, 0.9834 | unchanged |

On clip 3 it is free: ten joins discarded, all ten of them false positives, no real contact
touched and the invented rally at 2017-2122 gone with them. On clip 2 it discards ten joins to
remove eight false positives and costs two real contacts, which is a trade the F1 says yes to
and which should still be described rather than counted.

**What the two cost contacts are.** Truth 1743 is labelled with no hitter at all -- the labeller
could not tell who played it -- and it is a lift so high the shuttle leaves the top of the
frame, so the turnover in the track sits at the frame edge. Truth 7227 is a near-player lift
whose labelled frame sits at the top of the arc rather than at the racket, a few frames after
the strike. Both are cases where the recorded contact frame is at or near an apex, which is
exactly where this rule looks. **The class it can genuinely lose is a soft contact played at the
top of a trajectory**, and clip 5 shows one: the far player's net reply at 1851, verified by eye,
is discarded.

**Where it came from, and a second opinion.** It came out of labelling clip 5 by hand, because
`contact_sheet.py` draws the raw track either side of a candidate and an apex stops looking like
a contact the moment you can see its shape. Twenty-four candidates in clip 5's longest rally were
then adjudicated frame by frame, before this rule existed. It keeps 16 of the 17 contacts that
labelling confirmed and discards 4 of the 7 it rejected -- agreement between a physical rule and
a human reading of the video, which is not the same as agreement between a detector and itself.

**What it does not establish.** Two labelled clips and eighteen false positives is what this
rests on, and one of those clips carries model-produced labels. Clip 5 is still unlabelled, so
its drop from 83 predicted hits to 65 and from 8 rallies to 7 is a count, not an accuracy. The
three false positives that survive on clip 5's rally 3 -- 1828, 1891 and 2114, all rejected by
eye -- are not apexes and are not addressed by any of this.

### Clip 5, labelled in parts, and the first clip where rally boundaries fail

*The figures in this section were later re-measured against hand labels and are superseded. See
"What a hand labelling was worth" at the end of this file. The rally-boundary finding stands
unchanged; the hit and hitter figures do not.*

Clip 5 is a third labelled recording, and the first labelled by parts: its two longest rallies
have every contact adjudicated, and the rest of the clip has rallies but not contacts. That is
what `labelled_contact_spans` is for, and why it exists.

| | clip 2 | clip 3 | clip 5 |
|---|---|---|---|
| hit precision | 0.8852 | 0.9091 | 0.7692 |
| hit recall | 0.9153 | 0.7143 | 0.8571 |
| hit F1 | 0.9000 | 0.8000 | 0.8108 |
| hitter accuracy | 0.9245 | 0.90 | 0.8947 |
| rally boundaries | 6 of 6 | 4 of 4 | **4 of 7** |
| two-player coverage | 0.9286 | 0.9834 | 0.9016 |

Coverage passes on all three. Hitter assignment misses its 0.90 target on clip 5 by one
judgement in nineteen. **Rally boundaries fail for the first time**: 0.571 against a 0.85
target, and the three that fail do so for two different reasons.

**A break in the shuttle track reads as the end of a rally.** The first rally runs from the
serve at frame 213 to frame 915; the pipeline reports it starting at 513, nearly nine seconds
late. The track goes *invisible* twice inside that rally -- frames 306-339 and 476-509 -- and
the detector ends a rally at each break rather than linking across it. Nothing in the pipeline
distinguishes "the tracker lost the shuttle" from "the shuttle stopped moving", and on this clip
those two look identical.

Establishing that it is one rally took the scoreboard rather than the video: it reads 3-7 at
frames 200, 480, 900 and 960 and 3-8 by 1020, so exactly one point is played across the whole
stretch. Three separate runs of moving shuttle, one point. The other two failures are the same
shape in miniature -- rally 3's detected span is 2.5 s late and 2.6 s short.

**The nine false positives inside the labelled spans are all frames rejected by eye**: 1828,
1888, 2059, 2114, 2234, 2256, 2307, 2437 and 3726. Most sit at the top or the middle of a
flight with no racket within reach. The five missed contacts are 1851, 3613, 3757, 3788 and
3866; 1851 is the soft net reply the gravity-apex rule discards, which is the cost that rule was
already known to carry.

**What the labelling can and cannot support.** A contact is visible in the shape of the raw
track -- it arrives falling and leaves rising, and an apex does the opposite. *Who played it*
often is not: the far player is about 130 px of dark kit and spends much of both rallies at the
net directly behind the near player. So a hitter is recorded only where a racket is visibly at
the marker, or where an unbroken flight into a player whose identity is certain leaves no
alternative. Nine of the 35 contacts carry no hitter at all, and the scorer leaves those out of
the hitter denominator rather than guessing.

Alternation was used as an instrument, not as a source of labels. In singles the hitters of a
rally are a two-colouring, so one known hitter fixes every other by parity, and any second known
hitter that disagrees is proof that something is wrong -- a mislabelled hitter, or a contact
nobody recorded. It found three: a missing contact at 3588 between two apexes, one strike
recorded twice at 3769 and 3776, and a rejected candidate at 2061 whose removal restores parity
across 2031-2089. Both labelled spans alternate cleanly once those are fixed, which is the
evidence offered that the spans are exhaustive.

They remain model-produced labels, and one clip in three of this project's labelled set is now
hand-labelled. `tools/contact_sheet.py` is what made them possible at all: 126 candidates were
proposed by two instruments -- the flight detector and an independent turning-point read of the
raw CSV -- which agreed on only 52 of them.

### What a hand labelling was worth

Clip 5's two labelled rallies were labelled a second time, by hand, by the project owner,
watching the working video at a third speed with its frame numbers burned in and no pipeline
output drawn on it. The model labelling for the same two spans had been written first and was
not consulted. Comparing them measures the thing that every model-produced figure in this file
rests on.

| | model labels | hand labels |
|---|---|---|
| contacts found | 35 | 38 |
| contacts with a hitter | 26 | 38 |
| agreed with the other file | 29 | 29 |
| missed | 9 | -- |
| invented | 6 | -- |

**As a labeller the model scores 0.829 precision and 0.763 recall on contacts, and 0.864 hitter
accuracy over the 22 contacts it was willing to name.** It missed nine contacts, invented six,
and named the wrong player three times. Roughly one contact in five and one hitter call in seven
was wrong.

**The direction of the error matters more than its size.** Scored against the model's labels the pipeline
reads 0.769 precision and 0.811 F1; scored against the hand labels it reads 0.872 and 0.883.
**Seven of the nine frames the model had recorded as detector false positives were real contacts
that the model itself had missed** -- 1828, 1888, 2059, 2114, 2234, 2256 and 3726. The detector
was right and the labelling was wrong, and a labelling that is wrong in that direction makes a
working pipeline look broken.

| clip 5, same run | vs model labels | vs hand labels |
|---|---|---|
| hit precision / recall / F1 | 0.7692 / 0.8571 / 0.8108 | **0.8718 / 0.8947 / 0.8831** |
| hitter accuracy | 0.8947 over 19 scored | **0.9000 over 30 scored** |
| rally boundaries | 4 of 7 | 4 of 7 |
| coverage | 0.9016 | 0.9016 |

Rally boundaries and coverage do not move: neither depends on the contact list. The
rally-boundary failure and its cause -- a break in the shuttle track read as the end of a rally
-- are unaffected by any of this and stand as recorded above.

**Where the model went wrong is legible, and it is not where it expected.** In rally A it
invented nothing at all but missed six contacts. In rally B it invented six and missed three. The
two hitter calls it got wrong in rally A, at 1851 and 2089, were both produced by the same piece
of reasoning: an unbroken flight arriving at a player whose identity is certain implies the other
player struck it. That inference is sound only if the flight really is unbroken, and it was not
-- the model had missed contacts at 1835 and 1890, either side of 1851. **A missing contact does
not stay a missing contact; it propagates into a wrong hitter.** The parity argument that was
used to find missing contacts also invented one, at 3588, which the hand labels do not contain.

**What this does to the rest of the file.** Clip 3 is labelled entirely by the same method by the
same model, and its figures -- 0.9091 precision, 0.7143 recall -- should be read as carrying an
error bar of about this size until it too is labelled by hand. Its low recall is exactly the
shape a labelling with invented contacts produces. Clip 2's hand labels are unaffected.

The reverse also holds and is worth saying plainly: the hand labelling took well under an hour
and moved three of this project's numbers, one of them by ten points. Nothing else available to
this project buys that much.

### A gap the tracker spent blind is not the end of a rally

Clip 5's first rally runs from the serve at 213 to 912 and was reported as starting at 513,
nearly nine seconds late. The chain broke twice inside it, at 284-370 and again at 451-512, and
both breaks were the same rule: `max_link_gap_seconds`, at 1.8 s. The second one lost by a
single frame, 62 against a limit of 61.

**The limit was measuring wall-clock time, and a gap the tracker spent blind is not evidence
that play stopped -- it is evidence of nothing at all.** Counted over every gap-limit break in
the three labelled clips:

| | gaps | shuttle visible through the gap | `settled_between` fires |
|---|---|---|---|
| gap ends a real rally | 9 | 100% on all nine | 9 of 9 |
| gap sits inside a rally | 2 | 32% and 19% | 0 of 2 |

The two populations separate perfectly, and not on length: the largest gap inside a rally is 87
frames and the smallest gap that really ends one is 85, so no ceiling on duration could ever
have split them. What splits them is whether the tracker had anything to look at. The limit now
counts only the frames of a gap that carry a sample, and `settled_between` -- which was already
measured to separate the two kinds of gap on the development clip -- does the deciding.

Clips 2 and 3 cannot move under this change and do not: every gap in both is visible for all of
it. What it leaves exposed is written into a test rather than hidden -- two rallies separated by
a long stretch the tracker saw nothing of would now be joined, and there is no measurement in
these clips to set a ceiling from, so none is invented.

### The rally labels were wrong too, and the scoreboard settled it

Fixing the detector moved clip 5 from 4 of 7 rallies to 5 of 7, which was still short of target,
and the remaining failures turned out not to be the detector's. **The clip holds five rallies and
the first labelling claimed seven.**

The labels had been derived by taking every run of moving shuttle for a rally. Two of those runs
are players knocking the shuttle back to each other between points. The scoreboard says so
plainly -- it reads 3-7, 3-8, 3-9, 4-9, 4-10, 4-11 across the clip, five points and therefore
five rallies -- and it is visible in the frames: at 2745 the near player is preparing to serve,
and the score ticks 3-9 to 4-9 *between* frames 3210 and 3260, so the span labelled 3191-3336
sits after that point rather than during it.

Rally ends are now the frame the shuttle comes to rest, read off the raw track. Three automatic
rules for that were tried first and each failed differently, which is worth recording because
each looked reasonable: the first sustained standstill runs late, because a shuttle lying on the
court jitters by up to 31 px and defeats any tolerance tight enough to mean anything; a
last-sustained-travel rule runs late for the same reason and also cuts rallies short at the very
blind gaps the fix above exists for; and both are confounded by the tracker leaping onto its
parked point, which looks exactly like motion.

| clip 5 rally | labelled | predicted | start error | end error |
|---|---|---|---|---|
| 1 | 213-912 | 191-896 | -0.65 s | -0.47 s |
| 2 | 1674-2470 | 1674-2467 | 0.00 s | -0.09 s |
| 3 | 2998-3112 | 3008-3110 | +0.30 s | -0.06 s |
| 4 | 3509-3878 | 3506-3862 | -0.09 s | -0.47 s |
| 5 | 4395-4557 | 4400-4552 | +0.15 s | -0.15 s |

**Five of five recovered, every edge inside tolerance, median start error 0.00 s and median end
error -0.15 s.** Clip 2 stays at 6 of 6 and clip 3 at 4 of 4, both unchanged in every figure.

**What is left is two invented rallies**, at 2646-2681 and 3203-3325, and both are the same thing
the labels got wrong: players knocking the shuttle back between points. That is not a boundary
problem, it is the play gate, and seven attempts at one are recorded above as having failed to
separate on two clips. It now has a third clip to fail or succeed against, and two labelled
examples of exactly the stretch it has to reject.

### The eighth play-gate signal, and the bounds that had gone stale (2026-10-08)

Three spurious rallies remained: clip 3's 317-465 and clip 5's 2646-2681 and 3203-3325. Frames
pulled from all three show the same thing, players knocking the shuttle back across the net
between points, and in clip 5 the near player stands in the same spot in the same pose 35 frames
apart.

**An eighth signal was measured and it overlaps like the other seven.** The frames suggest one
nothing had tried: between-point knocking never drives the shuttle *deep*, where a rally has to
reach the backcourt. Measured as shuttle depth excursion over the span, real rallies bottom out
at 0.279 of frame height and clip 5's invented 3203-3325 scores 0.361, above two real rallies.
It joins the list.

| signal | real rallies | the invented ones | why it fails |
|---|---|---|---|
| shuttle depth excursion | 0.279-0.889 of height | 0.247, 0.285, 0.361 | clip 5's 3203-3325 sits above two real rallies |

Lateral range multiplied by depth does separate all seventeen detected rallies, at a margin of
1.28x on three examples. That is **below the 1.4x player-motion margin this project already
rejected** as fitted to one example, so it is recorded here and wired to nothing.

**`start_kind` cannot help either.** It reads `serve` on all sixteen non-truncated detected
rallies, including all three invented ones, because it means "preceded by a quiet stretch" and
knocking between points is preceded by one too. The name promises a discrimination the
measurement does not make.

**What was actually wrong was the bounds.** `LABELLED_RALLY_RANGES` defines itself as the range
every hand-labelled rally in the project falls in. It was taken over ten rallies on two clips,
clip 5 was hand-labelled afterwards, and nobody recomputed it. All four bounds were describing a
sample smaller than the one on disk:

| measurement | documented | the 14 labelled rallies |
|---|---|---|
| `slower_slot_movement_bh_s` | 0.318-0.742 | **0.093**-0.720 |
| `median_shuttle_speed_px` | 5.30-11.88 | 6.410-**19.600** |
| `usable_shuttle_fraction` | 0.81-1.00 | **0.738**-1.00 |
| `shuttle_lateral_range_frac` | 0.188-0.568 | **0.184**-0.568 |

The slower-slot floor was out by 3.4x, and that is the bound the seven attempts' player-motion
claim rested on. Correcting it takes the evidence block from flagging **2 of 3 invented rallies
and 4 of 14 real ones** to **3 of 3 and 3 of 14**. The bounds are computed from real labelled
rallies only, so the invented ones falling outside is an observation and not a fit.

**It still decides nothing.** Three real rallies are outside the corrected bounds, so a gate on
them would drop real play. `play_evidence` reports and does not suppress, and the test that says
so still passes.

Two of the synthetic tests had been reading a slot at 0.23 body heights/s as barely moving. It
is not: clip 5's real rally 3 has a slot at 0.093, a player who stands still through a genuine
point. The fixture was moved to 0.05 rather than the bound bent back to keep the test reading
the way it used to.

`configs/labelled-rally-measurements.json` now holds the rally-by-rally measurement behind every
bound, `tools/measure_labelled_rallies.py` regenerates it, and a test fails if a clip gains hand
labels without being folded in -- which is the thing that actually went wrong.

### Working the open-problems list: one fix, four diagnoses, two dead ends (2026-10-08)

**`start_kind` no longer claims to have seen a serve.** It read `serve` on all sixteen
non-truncated rallies across the three clips, including all three the detector invented, because
what it measures is silence before the first flight and knocking the shuttle back between points
is preceded by silence too. The value is now `after_quiet`, and `quiet_before_frames` carries the
number it was decided on.

This was never cosmetic. `after_quiet` **synthesises a contact** at the first flight's start
frame, on the reasoning that an unseen stroke launched it. That synthesis is worth keeping — on
clip 2 all five are real contacts — but it also manufactures a false positive on every invented
rally, and the old name hid that the evidence was silence rather than a stroke.

**Clip 3's hit precision is not an independent problem.** Its four false positives decompose
completely:

| false positive | cause |
|---|---|
| 317, 372, 436 | the invented rally at 317-465 — the play-gate problem |
| 1035 | rally 3 running past its labelled end (765-985 labelled, 768-1069 detected) |

Remove the invented rally and precision goes 0.667 to 0.889 with recall unchanged. There is
nothing to fix here that is not already the play gate or the clip-3 rally-3 boundary, and the
list should stop carrying it as a fifth item.

**The second parked point is not a threshold problem, and the entry saying it was is corrected.**
`docs/milestone-2-phases-b-to-j.md` recorded the back-wall mark as holding "59 frames across four
runs" and so falling under `fixed_point_min_runs`. Measured, that is not what happens:

| | drift across the run |
|---|---|
| the real fixed point (918, 222) | **0.0 px** — pixel-identical |
| 4170-4180 | 44.2 px |
| 7216-7228 | 62.4 px |
| 7247-7261 | 64.7 px |
| 7347-7360 | 54.7 px |

`rest_tolerance_px` is 8.0, so the tracker sliding down the wall never becomes a rest run at all
and never reaches the run count to fall under it. **Lowering `fixed_point_min_runs` to 1 finds
nothing.** Closing this means detecting a tracker drifting over a static feature with no shuttle
present, which is a different detector, not a smaller number. A test holds both halves with a
frozen control, so it cannot rot into passing for the wrong reason.

The video settles what the labels at 7227 and 7248 are. At both frames the tracker is on the wall
mark with no shuttle near it; in between, at 7233-7243, it is on the near player's racket at the
net. The surrounding labels are a near lift at 7196 and a far smash at 7286, which is a complete
and correctly alternating exchange with no room for a contact between them. The labeller marked
both `uncertain`. **They are not contacts**, and with the owner's approval both were removed.

The two rows, the reasoning and the frames are kept in `truth.json` under `_removed_contacts`, so
the deletion is auditable and exactly reversible. What it did to clip 2:

| | before | after |
|---|---|---|
| labelled contacts | 63 | 61 |
| hit recall | 0.9016 | **0.9322** |
| hit F1 | 0.9016 | **0.9167** |
| hit precision | 0.9016 | 0.9016 |
| hitter accuracy | 0.9811 | 0.9811 |
| impossible alternations | 2 | **0** |

Deleting labels improved a score, which is the move most worth being suspicious of, so what
happened is worth stating plainly: the two rows were false negatives **no detector could ever
have cleared**, because there was no contact at either frame to detect. Nothing else moved —
precision, hitter accuracy and every other clip are untouched. The manifest is re-frozen at
`milestone-1/1.2`.

All three clips now read zero impossible alternations, and that is the moment the check stops
distinguishing them, so the manifest now says so: clips 3 and 5 reached zero by being labelled
blind, clip 2 reached it by repair. A zero means *repaired* for clip 2, and its labelling is
still the least trustworthy of the three.

**Drive is a sample-size problem, not a tuning problem.** It is 0 of 4, and the shape of the
failure rules out the hopeful readings:

| true drive | BST called it | where drive ranked |
|---|---|---|
| clip 2 f6326 | net (0.27) | **#2** at 0.207 |
| clip 5 f1736 | drop (0.92) | #3 at 0.012 |
| clip 5 f2219 | smash (0.86) | #4 at 0.024 |
| clip 5 f3568 | other (0.66) | #7 at 0.021 |

Only one is a near miss. On the other three BST puts drive at about 0.02 and is confident in
something else, so this is not a threshold, a reweighting or a calibration away. The four
predictions of drive are all wrong too (smash, lift, net, net). With four examples, anything that
got them right would be fitted to four points. Drive needs labelled drives, which is the same
blocker as the club recording, and it should be tracked as that rather than as its own problem.

### A stroke taken above the frame, and two measurements that could not see it (2026-10-20)

Clip 3's rally 3 was carried as "ends 1.4s late". Looked at frame by frame it is two problems,
and the one underneath was a **missed smash**. The far player's clear at 884 climbs out of the
top of the frame and the tracker loses it at 902. It comes back into view at 966, falling, and
at 971-972 the near player smashes it: the step jumps from 12 px to 104 px and the heading
swings. The fit refuses those kinked frames, so the next segment starts at 976, after the
smash, and the join from the clear to the smash read as the top of one arc -- `gravity_apex`,
no contact. Clip 2 at 4312-4323 is the same thing inside a join already called a contact: the
far clear at 4238 left the frame, came back at 4312, and the near player's stroke at 4316 sat
in the refused frames, missed.

Two obvious measurements were tried first and **both are wrong**:

| measurement | why it fails |
|---|---|
| apex speed ratio, measured from the re-entry instead of the segment | a lift whose top is out of frame comes back faster than it left with no racket involved (clip 2 at 4641-4686: 13 px/frame out, 17-20 back). It flips a gravity join to a contact and the phantom hit then wins consolidation against the real one beside it -- clip 2 lost a true hit |
| fit residual in the refused frames | clip 5's re-entry at 355-371 stutters at about 10 px residual throughout with nothing near a racket; it would have planted a hit at 357, which the frames show is a one-sample teleport |

What does separate them is a **sudden jump in step against an approach the tracker saw**, the
same physics the apex test already rests on (a racket adds energy), measured at the kink instead
of across the gap:

| blind re-entry | largest step / median approach step | labelled contact there |
|---|---|---|
| clip 3, 966-976 | **8.4** | yes (960; the kink is at 971) |
| clip 2, 4312-4323 | **7.2** | yes (4316) |
| the other eight, all clips | 2.3 or less | no |

`FlightConfig.reentry_kink_speed_ratio = 3.0` records it on the link as `reentry_contact_frame`,
alongside the join's own `kind` and `hit_frame`, and the chain counts it as a hit. **Three is a
round number in the gap, and the evidence is two positives.**

| clip | hit P | hit R | hitter |
|---|---|---|---|
| test-video2 | 0.9016 -> **0.9032** | 0.9322 -> **0.9492** | 0.9811 -> 0.9815 |
| test-video3 | 0.6667 -> **0.6923** | 0.8000 -> **0.9000** | 0.8750 -> **0.8889** |
| test-video5 | unchanged | unchanged | unchanged |

Both recovered hits are attributed to the right player and no false positive was added on any
labelled frame. This is the first re-freeze caused by the detector rather than the labels, and
the manifest is at `milestone-1/1.3` with a `_detector_change_note` saying so.

What it does **not** fix: clip 3's rally 3 still ends at 1069 rather than about 1000. The smash
lands at the far player's feet at 999, inside the segment that started at 976, and the tracker
then follows the shuttle being picked up and handled -- 1029-1035 and 1045-1069 -- which links on
as a `contact_by_reach` at 1035, the one remaining false positive on that clip not caused by the
play gate. That is the post-point handling problem the play gate already names, seen from the
other end of a rally.

### The first club recording: the camera height costs both tracking and hitter assignment (2026-10-20)

`test-video6` is three minutes of the owner's club footage (`club-recording.MP4`, 120-300 s,
the first game), the recording the last two handoffs were blocked on. It is technically the
cleanest clip in the project -- genuine constant 59.94 fps, 1.6 px of camera drift, a far player
165 px tall, a shuttle track `track_quality` calls sound -- and it breaks one rule of the
recording spec: the camera is behind the baseline at about **head height**, not raised, and the
near baseline is below the bottom of the frame.

`tools/propose_court.py` refused it (seated officials are the far player's size at the far
player's depth), so the court was fitted by hand from a foot-position survey under a permissive
polygon; the measurements are in `configs/test-video6.json`'s notes. Unlabelled, the run says:

| | clip 6 | labelled clips |
|---|---|---|
| both players, every frame | **0.855** (near 0.975, far 0.880) | 0.90-0.98 |
| adjacent attributed hits given to the same player | **0.51** | 0.07, 0.22, 0.21 |
| rallies detected | 7 in 3 minutes | -- |

**Both failures have the same cause, and it is the camera.** At head height, a near player who
comes forward to the net stands at the same image height as the far player and covers them.

- **Coverage.** In 68% of the frames the far player is lost, the place they were last seen is
  more than half covered by the near player's box. They are not being missed; they are not
  visible. No code change recovers that honestly -- holding the last position would raise the
  figure without anyone being seen.
- **Hitter assignment.** It is decided by raw pixel distance from the shuttle to each player's
  nearest anchor. On a raised camera a near player at the net is still *below* the far player
  in frame, so the distance sorts them; here their heads are level and the near player's box is
  two and a half times larger, so the near player wins contacts that cannot be theirs -- rally 5
  reads near seven times out of eleven consecutive hits. Singles alternates, so 0.51 against
  0.07-0.22 is a measurement of this, not an impression, and it needs no labels.

**Tried and rejected: decoding each rally's hitters as one alternating sequence.** A singles
rally has only two possible hitter sequences, so choosing the one the per-hit evidence favours
overall looks like it should repair this. Measured against the hand labels it is worse on two of
three clips -- clip 2 0.946 -> 0.768, clip 5 0.794 -> 0.676, clip 3 0.889 -> 1.0 on nine hits --
because one missed or spurious hit flips the parity of everything after it. Doing it properly
means a sequence model that allows a hit to be spurious or missing, and that should be designed
against clip 6's labels, not ahead of them.

What this changes:

1. **"Elevated" is a requirement, not a preference.** It is now worded that way under Recording
   conditions. For a future recording, a tripod raised to 2.5-3 m behind the baseline is the
   cheapest fix in the project.
2. **This clip is still worth labelling**, because it is real club play and the first chance to
   test the domain-shift question for Milestone 2. `outputs/test-video6/label-part-A.mp4` and
   `-B.mp4` carry frame numbers and nothing else; `outputs/test-video6/labels.tsv` is the sheet.
3. **Unrelated, noted for later:** the shuttle tracker's fixed point is at x = 918 px here,
   (918, 270), and was at (918, 222) on clip 3 -- a different hall, camera and resolution
   pipeline. A real object would not land on the same column twice. It looks like a TrackNet
   default output when it finds nothing.

### The (918, 222) point is InpaintNet, and a third of "visible" shuttle samples were never seen (2026-10-29)

The fixed point the project has called a ceiling light since the first clip is not in any room.
It is the same normalised position, **(0.4765, 0.2482)**, on clips 2, 3, 5, 6 and yt-melbourne
at three different frame heights (884, 886, 1080): InpaintNet's output for a gap with nothing to
inpaint from. Upstream fills every gap between two detections with InpaintNet and writes the
result with `Visibility = 1`, and inference never writes the `Inpaint_Mask` column, so
`BallSample.inpainted` -- which the parser already supported -- was `False` on every sample of
every run.

The two kinds can be told apart exactly. A TrackNet detection is a bounding-box centre on the
512x288 heatmap scaled to the frame, so it lies on a grid of `width/1024` by `height/576`;
InpaintNet's output is a free float. On every clip, **every off-grid sample lies inside a gap
between on-grid ones**, which is upstream's `_inpaint_mask` rule, so the test agrees with the code
that produced the file.

| clip | visible samples | inpainted | of which the constant | rally-frame coverage, all / observed |
|---|---|---|---|---|
| test-video2 | 7198 | 3136 | 2553 | 0.951 / 0.911 |
| test-video3 | 4934 | 3680 | 3219 | 0.938 / 0.936 |
| test-video5 | 4210 | 1949 | 1481 | 0.927 / 0.847 |
| test-video6 | 10570 | 6021 | 4789 | 0.914 / 0.795 |

`load_ball_csv(path, frame_size)` now marks them, `shuttle_track_quality` reports
`frames_inpainted` and `observed_coverage` beside `coverage`, and each fixed point carries
`source: inpaintnet | tracknet` (yt-melbourne's (906, 798) is a real `tracknet` fixation). **No
decision reads the flag**: re-analysed from the same inputs, every event, rally and hitter on
clips 2, 3, 5 and 6 is identical, and the three frozen clips reproduce `milestone-1/1.3` exactly.

**Checked against upstream's real mask.** Upstream has no option to write the mask, but it is
a pure function of TrackNet's raw output, so `tools/export_inpaint_mask.py` re-runs upstream's
own classes once, takes `InferencePipeline._inpaint_mask`, and inpaints the way `events.py`
does -- no change to upstream or its model. The re-run reproduced all four committed shuttle
CSVs exactly, and the mask and the grid agree on **all 30,902 visible samples** (clips 2, 3, 5,
6; no sample on either side only). The parser keeps the grid: it needs no second inference and
nothing to differ from. The tool stays as the check to repeat if upstream or its checkpoints
change.

Using it to gate contacts was measured and is not justified: of the hits with at least three of
the seven surrounding samples inpainted, clips 2 and 3 have 5 true and 0 false, and clip 5 has
5 true and 6 false (matched within 0.25 s, not the scorer's closest-first pairing). InpaintNet
bridges short gaps inside real rallies as well as fabricating positions in long ones.

### Why a player slot is empty, and clip 6 made ready to label (2026-10-29)

`analysis.json` now carries `player_visibility`, from `src/badminton_poc/visibility.py`: for
every frame with an empty slot, the likely reason, read from that slot's last observed box. It
is a report. No slot is filled, no position is predicted, no decision reads it, and its
`both_observed_fraction` equals the existing `two_player_coverage` on every clip, which is the
check that the definition did not move.

| state | meaning |
|---|---|
| `candidate_not_selected` | someone was detected on that side and passed the court filter, but was not chosen |
| `occluded` | the slot's last box is at least half covered by the other player's box now |
| `at_frame_edge` | the last box touched the image border |
| `unknown` | none of the above |

During detected rallies, far player:

| clip | observed | occluded | not selected | unknown |
|---|---|---|---|---|
| test-video2 | 0.921 | 55 frames | 124 | 111 |
| test-video3 | 0.986 | 9 | 0 | 2 |
| test-video5 | 0.902 | 136 | 46 | 41 |
| test-video6 | 0.874 | **378** | 26 | **4** |

On clip 6, 93% of the far player's absence in rallies is occlusion, and the longest occluded stretches in the clip
(1513-1590, 9672-9700, 2487-2513) are the far player directly behind the near one -- checked by
eye with `tools/occlusion_sheet.py`. The 0.5 cover line is measured, not chosen; the
`OCCLUDED_COVER` docstring has the distributions, including the part that does not separate:
on the low cameras a far player is still detected behind 11-15% box cover.

What the states cannot say: `unknown` on clip 6 includes players walking off court between
points and a far player in plain view whom selection lost, and `at_frame_edge` is mostly a near
player still in shot with their feet cut off. They are likely reasons, not observations.

**Occlusion does not explain clip 6's hitter repeats.** See the last section of
`docs/hitter-assignment-investigation.md`: 14 of 21 same-player pairs have both players in
frame and the shuttle at the near player's wrist.

**Labelling.** `labels.tsv` gained a `u` line for a contact the video cannot confirm, scored as
neither hit nor miss and left out of the hitter figure (`truth.score_hit_detection` reports
`uncertain_contacts`); no existing truth file uses it, so no frozen figure moves. The sheet now
says that a missed swing gets no line, and `tools/ingest_labels.py` refuses to write a
"labelled" truth file with no contacts in it.
