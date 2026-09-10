# Milestone 2: can a known contact be classified into a broad shot type?

Findings from the first pass, 10 September 2026. Development clip is `test-video2`, the only
fully labelled recording. Milestone 1 is treated as an upstream dependency and was not retuned.

**Short answer: no, not yet, and not close.** The published BST checkpoint runs on our footage
and beats a rule baseline, but reaches 40% accuracy against a 75% target. The gap is not the
frame rate, not the clipping and not the joint format, all of which were handled and measured.

Section 9 records a second pass that changed two of the conclusions below. Official inference
*was* reproduced, to within 0.001 of the published figure. And a framing mismatch nobody had
looked for was found and fixed, worth five points. Read section 9 before quoting section 5.

---

## 1. What was built

`src/badminton_poc/shots/`, consuming the Milestone 1 event anchor and never re-detecting a hit.

| module | what it does |
|---|---|
| `windows.py` | one shot sample per Milestone 1 hit, stored as frame references |
| `features.py` | pose, shuttle and position arrays with explicit resampling and masks |
| `court.py` | court-relative depth from the calibrated polygon, no homography |
| `taxonomy.py` | versioned mapping from BST's 25 classes to our 8 |
| `baseline.py` | interpretable rule classifier from Milestone 1 data alone |
| `bst_adapter.py` | runs the published checkpoint; never modifies the upstream checkout |
| `truth.py` | hand-label schema, seeded from the timeline and never from a prediction |
| `evaluate.py` | metrics, always beside class counts, split by near and far |
| `runner.py` | wiring, previews, contact sheets, error report, confusion image |

One command runs the lot:

```
badminton-poc shots --video data/videos/test-video2.mp4 --config configs/test-video2.json
```

96 new tests, 323 passing overall, none needing a GPU.

---

## 2. What BST actually requires, read from the source

Cloned to `../BST` and read rather than assumed. Two findings shaped everything.

**BST-0 needs no court homography.** Four of the five variants take a `pos` input in court
coordinates, which needs a camera-to-court homography we do not have and cannot honestly derive
from a hand-drawn six-vertex polygon. On the authors' own merged 25-class benchmark at sequence
length 100:

| variant | extra inputs | accuracy | macro F1 |
|---|---|---|---|
| BST-0 | shuttle only | 0.8284 | 0.8041 |
| BST-CG-AP | player positions + shuttle | 0.8322 | 0.8097 |

Four tenths of a point is a cheap price for not fabricating a homography. We use BST-0.

**The joints are COCO-17, the same as ours.** `Task(n_joints=17)` with `get_bone_pairs('coco')`
giving 19 bone pairs. Our YOLO11 pose output is COCO-17 in the same order, so no joint remapping
is needed and none was invented. Normalization is reproduced exactly from
`prepare_train_on_shuttleset.py`: shuttle divided by the video resolution; joints relative to the
bounding-box top-left over the box diagonal, then centre-aligned; zeros meaning missing.

**Player axis order matters and is easy to get backwards.** Upstream sorts Top before Bottom by
court depth, and its class names carry a `Top_` or `Bottom_` prefix, so the axis order and the
label space are coupled. Top is our far slot. Getting this wrong would silently swap every
prediction's side.

**The window is defined in seconds, not frames.** `between_2_hits_with_max_limits` runs from the
previous contact to the next contact plus 0.25s, capped at 1.5s either side. Upstream writes
those as `fps // 2`, `t // 2` and `fps * 3 // 2`, so the rule transfers to any frame rate
unchanged. We reproduce it exactly.

### The frame-rate trap, measured

ShuttleSet's videos are not 60 fps. Dividing `frame_num` by the annotated timestamp across the
repository's own 104 annotation files gives 46 matches at 25 fps and 58 at 30.

| | ShuttleSet | our clip 2 |
|---|---|---|
| frame rate | 25 or 30 | 60 |
| median window | 1.80 s | 1.80 s |
| samples in that window | 45 to 54 | 108 |

Upstream's `make_seq_len_same` only decimates above 100 samples, so raw 60 fps windows arrive at
roughly double the temporal density the weights were trained on, with the tail truncated. We
resample by time to a configurable rate before the model sees anything. **This was worth having**:

| sampling | accuracy | macro F1 | top-2 |
|---|---|---|---|
| 30 Hz (default) | **0.349** | **0.379** | **0.603** |
| 25 Hz | 0.286 | 0.354 | 0.556 |
| 60 Hz, no resampling | 0.302 | 0.266 | 0.524 |
| 15 Hz, over-decimated | 0.143 | 0.121 | 0.286 |

Resampling to 30 Hz buys 42% relative macro F1 over feeding raw 60 fps. The collapse at 15 Hz
confirms the model is genuinely sensitive to temporal density, so this was a real hazard and not
a theoretical one.

---

## 3. Reference inference: blocked on the first pass, resolved on the second

*Superseded by section 9: the reproduction succeeded. This section records the first attempt.*

The instruction was to reproduce one official inference example on the authors' own data before
integrating. On the first pass that was not done.

What was verified: the checkpoint
`bst_0_JnB_bone_between_2_hits_with_max_limits_seq_100_merged_2.pt` loads into `BST_0` with
`in_dim=72`, `n_class=25`, `seq_len=100` and reports `All keys matched successfully`, 86 tensors
and 1.83M parameters, matching the paper's stated size. A forward pass produces 25 logits and
runs 67 shots in 2.07 seconds on this machine.

What was not: running it against the authors' own preprocessed test set to reproduce a published
accuracy figure. Their `.npy` archives are on Google Drive and the link began returning
"Cannot retrieve the public link of the file... or have had many accesses" after the weights
folder download. Retries with backoff over the session did not clear it.

So the architecture and checkpoint are confirmed compatible, but **there is no independent
confirmation that our preprocessing reproduces theirs**. If our normalization is subtly wrong,
the results below would look exactly as they do. That check should be the first thing done when
the Drive link frees up.

**Environment.** Python 3.11.16, torch 2.14.0+cu130, CUDA available, ultralytics 8.4.143.
Upstream asks for torch 2.4.0+cu121; the newer runtime loaded and ran the weights without
complaint. `positional_encodings` and `torchinfo` had to be installed for the upstream import.

---

## 4. Ground truth

67 shot samples from Milestone 1's 68 hits, one skipped as unattributed. Labelled in-session
from three-frame strips cut out of `annotated.mp4`, at the contact, +0.3s, and the end of the
window.

**This labelling is the second weak point.** It was made by the same party that built the
pipeline, from stills rather than played video, and is not an independent expert judgement.
25 of 67 are flagged uncertain. Every figure below should be read with that in front of it.

| class | count |
|---|---|
| lift | 19 |
| net | 13 |
| clear | 12 |
| drop | 5 |
| serve | 5 |
| drive | 4 |
| other | 4 |
| smash | **1** |

Plus 4 contacts marked not real, excluded from scoring, leaving 63.

Two things fall out of the counts before any model is scored.

**There is one smash in the entire clip.** The acceptance criterion of smash F1 ≥ 0.80 cannot be
meaningfully evaluated on a single example: the figure can only be 0 or 1, and it is 0 for both
models. This clip is amateur rally play, heavy on lifts and net exchanges. Testing smash
detection needs a clip with smashes in it.

**Lift alone is 30% of the shots.** Always-predict-lift scores 0.30 accuracy, which is a
sharper reference point than either model's headline number.

### What labelling exposed about Milestone 1

Labelling 67 contacts by eye surfaced things the rally-level metrics could not.

- **9 shots have a suspect hitter.** The worst is the end of rally 4, where six consecutive
  contacts are all credited to the far player. Two players cannot hit in succession. The cause
  is visible in the frames: it is a net exchange with both players overlapping in the image.
- **4 hits are not real contacts.** Three are the shuttle at the apex of its arc with neither
  player near it; one is a second hit credited to the same player 0.28s after the first.
- Both are consistent with Milestone 1's measured 0.82 hit precision. They are recorded in the
  labels rather than fixed, because Milestone 1 is frozen for this work.

---

## 5. Results

63 labelled shots. Targets were fixed before any result was seen.

| | BST-0 | rule baseline | target |
|---|---|---|---|
| accuracy | **0.349** | 0.286 | 0.75 |
| macro F1 | **0.379** | 0.198 | 0.65 |
| top-2 accuracy | **0.603** | 0.286 | 0.90 |
| abstention (predicted `other`) | 0.206 | 0.143 | — |
| near player | 0.407 | 0.222 | — |
| far player | 0.306 | 0.333 | — |

Every acceptance criterion fails except two: clear, lift and drop do stay discriminable under
BST, and far-player performance is not catastrophic relative to near.

### Per class, BST-0

| class | support | predicted | precision | recall | F1 |
|---|---|---|---|---|---|
| serve | 5 | 4 | 0.75 | 0.60 | 0.67 |
| smash | 1 | 12 | 0.00 | 0.00 | 0.00 |
| drop | 5 | 4 | 0.50 | 0.40 | 0.44 |
| clear | 12 | 9 | 0.56 | 0.42 | 0.48 |
| lift | 19 | 9 | 0.67 | 0.32 | 0.43 |
| drive | 4 | 2 | 1.00 | 0.50 | 0.67 |
| net | 13 | 10 | 0.40 | 0.31 | 0.35 |
| other | 4 | 13 | 0.00 | 0.00 | 0.00 |

The single most telling row is smash: **one real smash, twelve predicted.**

### Most confused pairs

| labelled | predicted | count |
|---|---|---|
| net | other | 6 |
| lift | smash | 5 |
| lift | clear | 4 |
| net | lift | 3 |
| other | smash | 3 |

---

## 6. Where the limitation actually is

Sliced by evidence already on disk:

| slice | n | BST accuracy |
|---|---|---|
| labels I was confident about | 42 | 0.452 |
| labels flagged uncertain | 21 | 0.143 |
| hitter pose present on ≥95% of the window | 9 | 0.667 |
| hitter pose below 95% | 54 | 0.296 |
| shuttle present on ≥95% | 51 | 0.314 |
| shuttle below 95% | 12 | 0.500 |

**It is not the shuttle track.** Accuracy is no better where the shuttle is nearly complete.

**It is not clipping or frame rate.** Both were derived from upstream's own code and measured;
the resampling ablation shows the handling helps rather than hurts.

**Pose completeness correlates**, but on nine shots, which is too few to lean on.

**Label noise is large.** A three-fold accuracy gap between my confident and uncertain labels
says as much about the labelling as about the model.

**The dominant effect is distribution mismatch.** BST predicts smash twelve times where there is
one, and lift nine times where there are nineteen. ShuttleSet is elite tournament singles, where
smashes are frequent and lifts are a smaller share of play. This clip is amateur play built on
lifts and net exchanges. The model brings a prior that does not fit the footage, and no amount of
preprocessing care fixes that.

**The taxonomy is also leaking.** `other` has 4 real examples and 13 predictions, and net-to-other
is the single largest confusion. Two upstream classes were deliberately mapped to `other` (push
and net kill) because calling them would be unsupported. That decision is defensible but it is
now visibly costing accuracy, and it is one line to change and re-score.

---

## 7. Answers to the eight questions

1. **Can official BST inference be reproduced?** Partly. The checkpoint loads with all keys
   matched and runs. Reproducing a published number on the authors' own test set is blocked by
   Google Drive rate limiting and remains outstanding.
2. **Can it consume our amateur footage?** Yes, cleanly. COCO-17 joints match, the window rule is
   frame-rate independent, and 66 of 68 windows have ≥80% pose coverage.
3. **Broad-class accuracy?** 0.349, macro F1 0.379, top-2 0.603, against targets of 0.75, 0.65
   and 0.90.
4. **Which pairs are most confused?** net→other, lift→smash, lift→clear.
5. **Main limitation?** Distribution mismatch first, label quality second, taxonomy third. Not
   pose, not shuttle tracking, not clipping.
6. **Does the baseline beat BST?** No. BST is ahead on every headline figure, most decisively on
   top-2 (0.603 against 0.286). But both trail always-predict-lift on raw accuracy (0.30), which
   is the number that should keep everyone honest.
7. **Do we need fine-tuning?** Probably, but not yet. Two cheaper things come first.
8. **Highest-value next experiment?** Label a second clip that actually contains smashes, and
   have someone other than the pipeline's author do it. Everything above rests on 63 shots from
   one recording, a quarter of them uncertain, with one smash. A second labelling would tell us
   whether 35% is the model's ceiling on amateur footage or an artefact of this sample.

---

## 8. What to do next, in order

1. **Unblock the reference check.** Retry the Drive archive and reproduce one published number.
   Until that is done, a preprocessing bug cannot be ruled out and every figure here is
   provisional in a second, deeper sense.
2. **Independent labelling of a clip with smashes.** Clip 3 or 5. The label guide and the CLI
   preview command already exist for it.
3. **Re-score with push and net kill mapped to their contested alternatives.** One edit to
   `taxonomy.py`, already versioned and tested, and it directly addresses the largest confusion.
4. Only then consider fine-tuning, and only with a clear picture of how much of the gap is
   distribution and how much is labelling.

---

## 9. Second pass

Four things were done after the first write-up. Two changed its conclusions.

### Official inference reproduced, exactly

The authors' preprocessed archive downloaded on a later retry: 894MB of RAR5 holding per-shot
`.npy` arrays foldered by true class. The system's `7z` is p7zip 16.02, which predates RAR5's
newer compression and extracted 10,497 silently empty files; libarchive 3.6, already installed,
reads it correctly. Worth knowing, because the failure was silent.

Running BST-0 over their 3,499-shot test split, across all seven merged sequence-100 checkpoints:

| | reproduced | published |
|---|---|---|
| accuracy | **0.8294** mean, 0.8345 best | 0.8284 |
| macro F1 | **0.8050** mean | 0.8041 |

Within a thousandth. So the model, the weights and our torch 2.14 runtime are all confirmed
sound, and the 40% on our footage is not a broken model path. Runtime is 1.1s for 3,499 shots.

### Our preprocessing checked against theirs, which found a real bug

Having their arrays makes a direct comparison possible for the first time:

| | ShuttleSet | our clip 2 |
|---|---|---|
| window samples, median | 58 | 54 |
| joints non-zero range | −0.62 to +0.72, std 0.189 | −0.53 to +0.59, std 0.182 |
| joints missing | 7.7% | 14.0% |
| shuttle missing | 10.1% | 5.8% |
| shuttle x range | 0.011 to 0.974 | 0.187 to 0.755 |
| shuttle y range | 0.003 to 0.917 | 0.005 to 0.693 |

Window length and joint normalization match, which validates the resampling and the bounding-box
maths. Our pose is worse and our shuttle track is better, both expected.

The last two rows are the finding. **Upstream normalizes the shuttle by the video resolution,
which silently assumes the court fills the frame.** ShuttleSet is broadcast footage, so it does.
A fixed camera set back from an amateur court leaves the court in the middle of the picture, so
the same shot lands at a compressed input value the weights never saw. Rescaling the configured
shuttle-airspace polygon onto the unit square restores the expected framing, uses only the court
calibration and no labels, and is worth:

| shuttle framing | accuracy | macro F1 |
|---|---|---|
| by video resolution, as upstream | 0.349 | 0.379 |
| **by the configured airspace, as now** | **0.397** | **0.414** |

This is now the default. It is a preprocessing bug we introduced by following upstream too
literally, and it would have gone unnoticed without their arrays to compare against.

### What is left, decomposed

With framing fixed, each further step relaxes a condition that is not available at inference
time, so these are bounds rather than achievable scores.

| condition | accuracy | gain |
|---|---|---|
| as measured | 0.397 | — |
| + oracle class prior | 0.460 | +0.064 |
| + only labels I was confident about | 0.571 | +0.111 |
| + only shots whose hitter looks right | 0.595 | +0.023 |
| target | 0.750 | **+0.155 unexplained** |

**This overturns the first pass's ranking.** Label quality, not distribution mismatch, is the
largest identified factor. The ShuttleSet prior really is skewed against us -- smash is 11.6% of
their strokes and 1.6% of ours, lift 15.4% against 30.2% -- and correcting it is worth six
points, but only with an oracle. Dividing by the source prior alone, the only version deployable
on an unlabelled clip, is worth nothing at all.

Even with a perfect prior, perfect labels and perfect hitters, BST reaches 0.595. **That 0.155
remainder is the honest case for fine-tuning**, and it is the first evidence in this project that
actually supports it.

### Things that did not help, recorded so nobody retries them

| tried | result |
|---|---|
| the shorter `middle_in_a_sec` window with the sequence-30 head | 0.191, far worse |
| the unmerged 35-class head at sequence 100 | 0.365 accuracy but 0.492 top-2 |
| ensembling all seven checkpoints | 0.365, no better than their mean of 0.361 |
| dividing by the source prior alone | no change |
| mapping push to drive | +0.016 accuracy, −0.025 macro F1 |
| mapping net kill to smash | +0.032 accuracy, +0.023 macro F1 |

Single-checkpoint spread is 0.318 to 0.397, so any difference under about four points is noise.
The taxonomy decisions move the result by about three points, which demotes the taxonomy from
third place in section 6 to a rounding error. The two contested mappings are left as committed:
adopting the better-scoring variant after seeing the score would be fitting to the test set.

### Revised answers

Only the ones that changed.

1. **Can official BST inference be reproduced?** **Yes**, to within 0.001 of published.
5. **Main limitation?** Label quality first, then distribution mismatch, then an unexplained
   0.155 that is genuine model failure on amateur footage. Not pose, not shuttle tracking, not
   clipping, and not the taxonomy.
7. **Do we need fine-tuning?** The 0.155 remainder says yes eventually, but the 0.111 attributable
   to labelling says do the labelling first. Fine-tuning on labels this noisy would bake the noise
   in.
8. **Highest-value next experiment?** Unchanged and now better supported: an independent labelling
   of a clip that contains smashes.

---

## 10. Cross-validated on clip 5, labelled independently

The project owner labelled all 53 shots of clip 5 from the marked preview clips. This is the
first labelling in the project not made by the party that built the pipeline, on a different
venue, a different standard of play and a different frame rate.

| | clip 2 (my labels, 60 fps) | clip 5 (owner's labels, 33.75 fps) |
|---|---|---|
| accuracy | 0.397 | **0.377** |
| macro F1 | 0.414 | 0.272 |
| top-2 | 0.587 | 0.585 |
| near / far | 0.481 / 0.333 | 0.458 / 0.310 |

**Accuracy cross-validates.** Two clips, two labellers, two frame rates, two venues, and the
figure moves two points. The roughly 38% result is real and not an artefact of my labelling.

Macro F1 falls because clip 5's tail classes are tiny: one drop, two drives, three serves, each
scoring zero. That is a property of the sample, not a change in the model.

### Smash is finally testable, and I was wrong about it

I had told the owner that no clip we hold contains enough smashes to test the criterion, based on
counting fast descending contacts with the rule baseline. **That was wrong.** Clip 5 contains ten
smashes by the owner's count; my heuristic found two, because the heuristic is poor, not because
the smashes were absent. The lesson is that a weak baseline is not a survey instrument.

| class | support | precision | recall | F1 |
|---|---|---|---|---|
| clear | 9 | 0.75 | 0.67 | **0.71** |
| smash | 10 | 0.67 | 0.40 | **0.50** |
| lift | 14 | 0.63 | 0.36 | 0.45 |
| net | 11 | 0.50 | 0.36 | 0.42 |
| other | 3 | 0.06 | 0.33 | 0.10 |
| serve | 3 | 0.00 | 0.00 | 0.00 |

Smash F1 of 0.50 misses the 0.80 target but the model does find smashes, at two-thirds precision.
Clear at 0.71 is the strongest class in the project so far.

**The `other` class is now the clearest single defect.** Eighteen predictions against three real
examples, absorbing six lifts and six net shots. It is where the abstention goes and where the
accuracy dies.

### The contested taxonomy mappings, settled on a held-out clip

The decision to send push and net kill to `other` was made before clip 5 existed, so testing the
alternative here is not fitting to the test set.

| | clip 2 | clip 5 |
|---|---|---|
| as committed | 0.397 | 0.377 |
| push→drive, net kill→smash | 0.429 | 0.377 |

It helps the clip it was tuned against and does nothing on the held-out one. **Left as committed.**

### Two things the labelling exposed that are not about shot type

**The taxonomy has no way to say "backhand".** The owner used it five times. Twice they added the
shot type in brackets, so those were taken at their word as clears. Three remain unresolved and
are parked as `other` with `uncertain=true` and an open question recorded in the file. Backhand
describes which side of the body played the stroke, not where the shuttle went, and this taxonomy
is defined by destination and flight shape. A backhand can be a clear, a drive, a net shot or a
lift. This is a real gap and the decision is the owner's, not mine.

**Clip 5's rally boundaries are wrong.** The pipeline calls shots 3, 14 and 40 the first of a
rally. The owner labelled serves at 14, 15 and 40, and called shot 3 a jump smash. So two of the
four detected rallies start mid-point, and one real serve is missing from the boundaries. Eight
further hits on this clip could not be attributed to a player, against one on clip 2, and rally 4
carries a run of five consecutive contacts credited to the same player. Milestone 1 passed on
clip 2 and visibly does not transfer to clip 5 at the same quality. That is the cross-validation
Milestone 1 never had, arriving as a side effect of Milestone 2.

---

## 11. Is BST the wrong tool, or just the wrong head?

Pretrained BST reaches about 38% on our footage and no amount of preprocessing moved it. That
number cannot say whether the representation is unable to carry shot type on amateur footage, or
whether it carries it fine and only the output layer is aimed at the wrong distribution. The two
lead to opposite decisions, so we asked directly.

Freeze the backbone, take the 300-dimensional vector it hands to its own classifier, fit a
multinomial logistic regression on labels, and vary only *what is allowed into training*.

| what the probe was trained on | clip 2 | clip 5 |
|---|---|---|
| other rallies **and the same rally** as the test shot | 0.759 | 0.773 |
| other rallies of the same clip only | **0.476** | **0.491** |
| the other clip entirely | 0.317 | 0.340 |
| zero-shot pretrained BST, for reference | 0.397 | 0.377 |
| always answer the commonest class | 0.302 | 0.264 |

**The first row is an illusion and it is worth dwelling on.** Random five-fold splitting puts
shot N in training and shot N+1 in test, and consecutive shots share a rally, an exchange and a
pair of body positions. It reports 76%, comfortably past the 75% target, and it is measuring
memorisation of context. Splitting by rally instead drops it to 48%. Had we not checked, this
project would have declared the milestone met.

Reading the honest rows:

**The representation does carry adaptable signal, but only locally.** Within a clip and across
rallies a probe reaches 0.48 against zero-shot's 0.40, so roughly eight points are available from
fitting to a venue.

**None of it survives a change of clip.** Trained on one clip and tested on the other, the probe
reaches 0.33, which is *worse than zero-shot* and barely past always-guessing-lift. Fitting a
head on 116 shots from two recordings produces something that does not generalise to a third.

**So fine-tuning is not the next step**, which reverses what section 9's unexplained remainder
seemed to suggest. Training a head on the data we have makes cross-clip performance worse, not
better. A control probe on BST's raw input arrays, with no transformer in between, scores 0.23
and 0.33, so the transformer is not the weak link either.

### What this actually says the bottleneck is

The gating quantity is **the number of distinct recordings, not the number of labelled shots**.
Two clips means "leave one clip out" is training on exactly one venue, one pair of players and one
camera position, which is the hardest possible version of the test. The generalisation ladder --
0.76 within a rally, 0.48 within a clip, 0.33 across clips -- is the signature of a model
learning setting rather than stroke.

The experiment that would settle it is to label eight to ten clips and plot leave-one-clip-out
accuracy against the number of training clips. If it climbs, this is a data-collection problem
with a known price. If it stays flat near 0.35, skeleton-plus-shuttle features do not determine
shot type on amateur footage at this camera distance, and the answer is a different input, not
more of the same.

That is a much more specific and much cheaper question than "should we fine-tune", and it is the
one to answer next.
