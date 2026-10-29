# External badminton CV projects, read and measured (2026-10-29)

Four repositories were cloned and their implementations read, not their READMEs. What could be
run against our labelled clips was run; nothing was merged into the pipeline.

Labelled clips: test-video2, 3 and 5 (107 rally contacts). Every number below comes from the
project's own `score_hit_detection` / `score_hitter_assignment`, and the `ours` row reproduces
the frozen `milestone-1/1.3` figures exactly, which is the check that the harness is fair.
Reproduce with `tools/benchmark_courtside.py`.

## 1. COURTSIDE (jayclim/BadmintonAI, MIT) -- contact detection and attribution

**What it does** (`src/badminton/hits.py`): four passes over the TrackNetV3 track per rally
window. (1) Kink: velocity in over 4 frames against velocity out over 4; a hit when the speed
change exceeds 30 px/frame or a turn past 60 degrees, weighted by speed, exceeds 20; peaks 8
frames apart. (2) Gap: the same test across raw visibility gaps of 2-14 frames. (3) Serve: the
first 3-frame run at 6 px/frame. (4) Wrist: a wrist moving at 10 px/frame or more with the
shuttle within 60 px. Attribution is the nearest confident wrist over the hit frame +-1.

**Could replace:** `flight.segment_flight`'s contact decisions (1-4) and `assoc.assign_hit`
(attribution). Ported as `src/badminton_poc/experiments/courtside.py`. Its constants were
tuned on 720p/30 fps broadcast; the port rescales them to the same durations and frame-width
shares and re-tunes nothing.

**Contact detection, pooled over the three clips:**

| variant | windows | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|---|
| ours (frozen) | detected rallies | 99 | 15 | 8 | 0.868 | 0.925 | **0.896** |
| COURTSIDE | detected rallies | 98 | 18 | 9 | 0.845 | 0.916 | 0.879 |
| COURTSIDE, no wrist pass | detected rallies | 94 | 16 | 13 | 0.855 | 0.879 | 0.866 |
| ours + COURTSIDE wrist pass | detected rallies | 100 | 20 | 7 | 0.833 | 0.935 | 0.881 |
| ours (frozen) | labelled rallies +-20 | 99 | 11 | 8 | 0.900 | 0.925 | **0.912** |
| COURTSIDE | labelled rallies +-20 | 100 | 21 | 7 | 0.826 | 0.935 | 0.877 |

Per clip, detected rallies, F1: clip 2 0.926 vs 0.896, clip 3 0.783 vs 0.720, **clip 5 0.883
vs 0.904**. Clip 5 is the one clip where COURTSIDE wins, by fewer false positives (2 vs 5),
and the only one recorded near COURTSIDE's 30 fps (33.75). Three clips cannot say whether that
is frame rate or chance.

**Attribution, on our own hits** (so only the attribution differs):

| | scored | correct | wrong | declined |
|---|---|---|---|---|
| `assign_hit` (frozen) | 93 | 88 (0.946) | 5 | 6 |
| nearest wrist | 99 | 92 (0.929) | 7 | 0 |

Nearest wrist never declines, so it gets four more right and two more wrong. The label-free
alternation check sides with declining: consecutive same-player pairs within a rally rise from
0.07/0.22/0.21 to 0.20/0.33/0.35 on clips 2/3/5. On clip 6 (unlabelled) it is 0.51 for ours,
0.52 for our hits with nearest wrist, 0.41 for COURTSIDE end to end -- which also detects more
hits there, of unknown precision.

**Verdict on our clips.** Detector: **reject on clips 2, 3 and 5** -- lower F1 pooled and on two of three, but see section 5: those are the clips ours was built on, and out of sample the result reverses. Wrist pass:
**reject** -- +1 true hit for +5 false positives. Attribution: **reject** -- our gate is
already a nearest-wrist rule plus a refusal to guess, and removing the refusal costs accuracy.
**Continue investigating** one thing: the speed-change term (`|v_out - v_in|`), which our
turn-based joins do not have, on clip 6 once it is labelled, since clip 5's fewer false
positives are the only place COURTSIDE was better.

## 2. Badminton Hit Detection (kwyoke, no licence) -- temporal model

**What it does** (`domain_rnn.ipynb`, `hit_detection/process_pred_probs.ipynb`): a 2-layer,
32-unit GRU over 14-frame windows of shuttle position, both players' 17-point poses and the
four court corners mapped through a homography, predicting no hit / near hit / far hit; a
window is positive when a hit falls in its last 6 frames. Per-frame probabilities are decoded
by dynamic programming that rewards confident hits and enforces a minimum gap. Its rule
baseline is a cubic-spline second-derivative threshold, weaker than COURTSIDE's kink.

**Not run.** The repository has no licence, so neither its code nor its TensorFlow weights can
be reused, and training an equivalent GRU on our 107 contacts would fit three clips. Its own
README reports near/far confusion on wide amateur angles, which is clip 6's problem.

**What is worth keeping:** the joint "hit + who" output and the DP decoding are the design the
sequence model in `docs/hitter-assignment-investigation.md` already proposes, from different
evidence. Its **amateur singles and doubles annotations** (Google Drive, linked from the README,
licence unstated) are the more valuable artefact: independent hit labels on amateur footage,
which is what this project is shortest of. Ask the author before using them.

## 3. Good-Badminton (Apache-2.0) -- court calibration

**What it does** (`court/detector.py`): a white-line mask, Hough segments split into
horizontal and side lines, every pair of each intersected into a quadrilateral, each scored
against a projected badminton court line model; the best quad's corners give a homography.

**Run** on the middle frame of each clip (the scratch copy needed one change: newer OpenCV
returns `HoughLinesP` as N x 4). It found **no court on any clip**: 0, 0 and 0 side lines on
clips 2, 3 and 5, and on clip 6 the near baseline is below the frame, so no quad can close.

**Verdict: reject for automatic calibration.** For the mini-court work later, COURTSIDE's own
route is the practical one: four hand-clicked court corners per recording and
`cv2.getPerspectiveTransform`, checked against the service lines.

## 4. TrackNetV3 (qaz812345, MIT)

**Already in use**, through BadmintonTrackNet (MIT), with the same TrackNet and InpaintNet
checkpoints. Read for the inpainting: `test.py:generate_inpaint_mask` marks gaps between two
detections whose y is above 30 px; BadmintonTrackNet uses 5% of frame height. Neither writes the
mask at inference, which is the gap this project's `BallSample.inpainted` now closes. No change
proposed.

## 5. The first out-of-sample test: kwyoke's amateur singles labels

kwyoke's dataset (no licence; used locally, kept under the gitignored `data/input/kwyoke/` and
never committed) has 7 labelled amateur rallies from two matches, 55 single-frame hits with the
hitter, labelled by its authors. Both cameras are low, behind the near baseline -- the clip-6
situation. `tools/ingest_kwyoke.py` converts the labels (1 = near, 2 = far, checked by eye);
`configs/kw-china2.json` and `configs/kw-yewtee.json` are hand-fitted courts with detector
settings copied from test-video6 unchanged. Nothing was tuned on these clips.

**It found a bug first.** china2 is 24.8 fps, where the 0.12 s fit window rounds to 3 frames,
and a quadratic through 3 points is exact: the residual was 0.0 on every frame and no contact
could ever break a segment. The window is now at least 5 frames. Our own clips (34-60 fps)
already had 5-7, so every frozen figure is unchanged.

| match | detector | windows | TP | FP | FN | P | R | F1 | hitter |
|---|---|---|---|---|---|---|---|---|---|
| china2 (33 hits) | ours, before the fix | detected | 4 | 8 | 29 | 0.333 | 0.121 | 0.178 | 3/4 |
| | **ours, after** | detected | 9 | 21 | 24 | 0.300 | 0.273 | **0.286** | 8/8 |
| | COURTSIDE | our detected rallies | 23 | 12 | 10 | 0.657 | 0.697 | **0.676** | 19/23 |
| | COURTSIDE | labelled rallies | 30 | 15 | 3 | 0.667 | 0.909 | 0.769 | 26/30 |
| yewtee (22 hits) | ours | detected | 5 | 5 | 17 | 0.500 | 0.227 | 0.312 | 4/5 |
| | COURTSIDE | our detected rallies | 9 | 6 | 13 | 0.600 | 0.409 | 0.486 | 6/9 |
| | COURTSIDE | labelled rallies | 21 | 16 | 1 | 0.568 | 0.955 | 0.712 | 12/19 |

yewtee is outside the recording spec (camera at floor level): both players are found together on
0% of one rally's frames, so its hits are dropped for want of a hitter. china2 is the fair test.
There our hits are not offset by a constant; they scatter from -42 to +32 frames around the
labels with many spurious ones.

**This reverses section 1's verdict on the detector.** On clips 2, 3 and 5 ours wins, but ours
was built on those clips, so that comparison is in-sample. On the only data neither detector was
developed on, COURTSIDE's kink detector is far better: 0.676 against 0.286 F1 on china2 inside
the same rally windows. Two matches and 55 hits are not enough to replace the frozen detector,
and clip 6 -- once labelled -- is the third independent test. **Status: continue, as the leading
candidate.** Our rally windows also limit it: with labelled windows COURTSIDE reaches 0.909 recall
on china2, with ours 0.697, because our segmentation misses or truncates rallies on this footage.
