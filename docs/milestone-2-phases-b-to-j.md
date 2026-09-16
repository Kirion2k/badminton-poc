# Milestone 2, Phases B-J: pretrained BST against hand-labelled contacts

Milestone 1 is frozen at tag `milestone-1-frozen`, manifest `milestone-1/1.1`, and nothing here
retunes it. Every number below comes from `outputs/bst_reference/phase_h.json` and
`shot_dataset.json`.

> **Superseded on 2026-09-16 by Phase K.** Phases B-J were measured on two recordings. A third
> was labelled, and accuracy fell from 0.7273 to **0.5895**. The per-phase findings below stand
> as what those two clips showed; the headline does not. **Read Phase K at the end before
> quoting any figure from this document.** Every two-recording number is left in place rather
> than rewritten, because the gap between them and Phase K is the finding.

**Two-recording headline, now superseded: 0.7273 accuracy and 0.6569 macro F1 over 44
hand-labelled contacts, against 0.3409 for a hand-written geometry baseline on the same shots.**
Two of the four exploratory targets are met, two are missed narrowly, and the dataset is two
recordings deep.

---

## Phase B: samples from hand-labelled contacts

The adapter already existed and already matched the contract Phase A re-derived. What it lacked
was the right source: `build_samples` read `analysis.json`, which is Milestone 1's own hits.
Scoring a classifier on those asks it to name strokes that never happened.

`build_samples_from_truth` reads the hand-labelled contacts, and `load_clip` defaults to it.
Every sample carries clip, rally, contact frame and timestamp, hitter, the window, the pose
sequence with its mask, the shuttle sequence with its mask, player position, source fps and the
original resolution, plus the raw bounding boxes so any normalisation can be re-derived. Windows
are the upstream `between_2_hits_with_max_limits` rule restated in seconds, so they transfer
across our 33.75 fps and 59.94 fps clips.

Three kinds of contact are left out, each recorded with its reason: a contact marked not a rally
shot, a contact outside `labelled_contact_spans`, and a contact the labeller could not attribute.
A contact left out still counts as a neighbour for the windows either side, because it happened.

## Phase C: the joint mapping is the identity, verified

BST wants COCO 17 with 19 bone pairs. Ultralytics emits COCO 17. Rather than assume they agree,
the median normalised y of each joint index was measured over every sample in all three clips:

| group | indices | median normalised y |
|---|---|---|
| head | 0-4 | −0.362 |
| shoulders | 5, 6 | −0.256 |
| elbows | 7, 8 | −0.136 |
| wrists | 9, 10 | −0.053 |
| hips | 11, 12 | −0.014 |
| knees | 13, 14 | +0.153 |
| ankles | 15, 16 | +0.334 |

Every index sits where COCO says it should. **No unmappable joints, no synthesized joints, no
renumbering.** Normalisation follows upstream exactly: bounding-box relative, origin at the
top-left, divided by the box diagonal, centre-aligned.

## Phase D: masks preserved, with one gap stated

The shuttle mask is true only where the tracker saw the shuttle *and* it was not sitting on a
parked fixed point, which is stricter than upstream's. Missing frames are zero-filled with the
mask false and are never interpolated.

**The gap:** our TrackNetV3 run emits no inpainted points at all — zero across all three clips —
so the three-state distinction the brief asks for (observed / reconstructed / missing) collapses
to two. There is no reconstructed state to preserve because nothing reconstructs. `BallSample`
carries the `inpainted` flag regardless, so the day an InpaintNet pass is added it has somewhere
to go.

## Phase E: the taxonomy mapping

Version `broad-v2`, an explicit table, and all 25 upstream classes accounted for with none
unmapped. `推球` (push) and `撲球` (net kill) go to `other` rather than being forced: a push sits
between drive and net, a net kill between smash and net, and folding a net kill into smash would
quietly inflate the one class with its own acceptance threshold. Both carry their contested
alternative in the table. Every prediction record keeps the original upstream class, all 25
original scores and the top-3 alongside the broad reading; nothing overwrites the original.

## Phase G: the dataset

| clip | shots built | with a shot type | rallies | excluded |
|---|---|---|---|---|
| test-video2 | 58 | **0** | 6 | 3 |
| test-video3 | 10 | 10 | 3 | 0 |
| test-video5 | 38 | 38 | 2 | 0 |

**48 labelled shots across two recordings.** By class: lift 12, net 12, smash 8, clear 7, serve
4, drive 3, drop 2. By hitter: near 25, far 23. Always-predict-largest scores 0.25.

Class imbalance is not hidden: **serve (4), drive (3) and drop (2) have fewer than five examples
each**, and any per-class figure for them is one or two shots wide. Clip 2's 58 hand-verified
contacts carry no shot type and contribute nothing until they are labelled.

## Phase H: pretrained BST, no training

Checkpoint `bst_0_JnB_bone_between_2_hits_with_max_limits_seq_100_merged_2.pt`, architecture
`BST_0`, mapping `broad-v2`. Serve is scored separately: it is the first contact of a rally and
derivable from rally structure, so counting it would flatter the model for something the pipeline
already knows.

| | value | target | |
|---|---|---|---|
| accuracy | 0.7273 | ≥ 0.75 | missed by one shot in forty-four |
| balanced accuracy | 0.6389 | — | |
| macro F1 | **0.6569** | ≥ 0.65 | **met** |
| top-2 accuracy | 0.8864 | ≥ 0.90 | missed by one shot |

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| clear | 1.000 | 1.000 | **1.000** | 7 |
| smash | 0.889 | 1.000 | **0.941** | 8 |
| net | 0.833 | 0.833 | 0.833 | 12 |
| lift | 1.000 | 0.500 | 0.667 | 12 |
| drop | 0.500 | 0.500 | 0.500 | 2 |
| drive | 0.000 | 0.000 | 0.000 | 3 |

**Smash F1 0.941 against a 0.80 target**, on eight examples. Clear is perfect on seven.

Breakdowns: clip 3 0.625, clip 5 0.750. Far player 0.8095, near player 0.6522 — a 16-point gap
on 21 and 23 shots. Serve alone 0.75 of 4.

One unplanned check: BST's class names carry a side prefix, so each prediction is also an opinion
about who hit. **It agrees with Milestone 1's hitter on 0.9792 of shots**, which is independent
corroboration of Milestone 1's hitter assignment from a model that never saw our labels.

### Separability, which the brief asked about directly

- **clear vs drop** — separable. Zero confusions either way.
- **clear vs lift** — separable. Zero confusions either way.
- **drop vs net** — one confusion in two drop examples, and it is the aggregation artefact below.
- **drive vs everything** — not separable. 0 of 3, scattered to drop, smash and other.

## Phase I: where the errors come from

Twelve non-serve errors. Shuttle coverage is 1.00 on every one and hitter joint coverage runs
0.79 to 0.96, so **no error is attributable to missing pose or missing shuttle**, and the
contacts are hand-verified so none is a Milestone 1 timestamp error.

| cause | n | evidence |
|---|---|---|
| broad-score aggregation artefact | 2 | the top-1 upstream class was already correct |
| upstream `推球` push, deliberately unmapped | 4 | 3 lifts and 1 drive predicted as push |
| genuine classifier disagreement | 6 | upstream names a different stroke with confidence |

**The aggregation artefact is worth stating plainly.** Broad scores are summed over the upstream
classes that map into them, and `other` and `net` each collect three upstream classes while
smash, lift, clear, drive and drop collect one each. Summing therefore favours the multi-member
classes structurally. At frame 1698 the top-1 upstream class is `Top_挑球` — lift, the correct
answer — and the summed reading returns `other`. At 2115 the top-1 is `Top_切球`, drop, and the
summed reading returns `net`.

Reading the top-1 upstream class instead gives **0.7727 against 0.7273**, two shots better. That
is reported as a diagnostic and **not** adopted: the summed rule was chosen before any result was
seen, and swapping it now would be tuning the mapping on the only data we have.

The four push cases are the other half of the same layer. Our labeller called them lifts; BST
called them `推球`, which we deliberately send to `other`. Mapping push to lift would repair
four errors and would also be a decision made after seeing which way it helps.

**So half the errors sit in the mapping layer rather than in the model's reading of the stroke.**

## Phase J: is BST better than badminton geometry?

The rule baseline is hand-written geometry over the same features — first contact is a serve,
fast and steeply downward is a smash, upward and deep is a lift or clear, slow in the forecourt
is a net shot, flat and fast is a drive — with no learning anywhere.

| on the same 44 shots | accuracy | balanced accuracy | macro F1 |
|---|---|---|---|
| pretrained BST | **0.7273** | 0.6389 | 0.6569 |
| rule baseline | 0.3409 | 0.2579 | 0.2378 |
| always predict the largest class | 0.25 | — | — |

**BST beats the geometry baseline by 38.6 points and the majority class by 47.7.** The answer to
the phase question is yes, decisively, and it is the difference that makes the headline number
mean something.

---

## What this does and does not establish

**It establishes that the earlier 0.38-0.40 was not the model's ceiling.** That figure came from
model-produced labels over detector-produced contacts. With hand-verified contacts and hand-given
shot types the same checkpoint reads 0.7273. The change is in the ground truth, not the model.

**Two recordings.** Clip 3 contributes 10 shots and clip 5 contributes 38. The prior probe study
found the gating quantity is the number of distinct recordings, not the number of shots, and two
is the hardest possible version of that test. Per-clip accuracy differs by 12.5 points already.

**Small per-class supports.** Drive has 3 examples and drop 2. Their F1 figures are one or two
shots wide and should not be quoted without the support beside them.

**The near/far gap is unexplained.** 0.8095 far against 0.6522 near, on 21 and 23 shots. It could
be perspective, it could be that the near player in clip 5 plays more of the classes that fail,
or it could be noise at this sample size. Nothing here separates those.

---

# Phase K: the third recording, 2026-09-16

Clip 2's 58 hand-verified contacts got shot types from the project owner, from previews carrying
no model output. That takes Milestone 2 from two recordings to three, which the prior probe study
named as the gating quantity.

**Headline: accuracy falls from 0.7273 to 0.5895, and macro F1 from 0.6569 to 0.5395.** The
two-recording figure did not survive contact with a third recording.

| | two recordings | three recordings |
|---|---|---|
| shots scored | 44 | **95** |
| accuracy | 0.7273 | **0.5895** |
| balanced accuracy | 0.6389 | 0.5223 |
| macro F1 | 0.6569 | 0.5395 |
| top-2 accuracy | 0.8864 | 0.8211 |
| geometry baseline | 0.3409 | 0.3579 |
| BST − baseline | +0.3864 | **+0.2316** |

BST still beats hand-written geometry by 23 points, so Phase J's answer survives. Nothing else
about the Phase H headline does.

| clip | accuracy | labelling |
|---|---|---|
| test-video5 | 0.750 | blind |
| test-video3 | 0.625 | blind |
| **test-video2** | **0.4706** | migrated, partly corrected |

| class | precision | recall | F1 | support | F1 on two recordings |
|---|---|---|---|---|---|
| net | 0.739 | 0.809 | 0.773 | 21 | 0.833 |
| clear | 0.706 | 0.800 | 0.750 | 15 | **1.000** |
| smash | 0.643 | 0.750 | 0.692 | 12 | **0.941** |
| lift | 0.933 | 0.467 | 0.622 | 30 | 0.667 |
| drop | 0.571 | 0.308 | 0.400 | 13 | 0.500 |
| drive | 0.000 | 0.000 | 0.000 | 4 | 0.000 |

**The two classes that carried the Phase H headline are the two that fell furthest.** Clear went
1.000 → 0.750 and smash 0.941 → 0.692. On clip 2 alone smash is 1 of 4 and clear 5 of 8, against
8 of 8 and 7 of 7 on clips 3 and 5. A perfect score on seven examples was seven examples.

Two classes did improve, both because they got examples rather than because the model changed:
drop went from 2 supports to 13, and drive from 3 to 4 — still the only class under five, and
still 0 of 4.

## Why clip 2 is worse, and what it is not

**It is not missing inputs.** Median shuttle coverage is 1.000 on all three clips and clip 2's
median hitter-joint coverage is 0.865, the *highest* of the three. Phase I's finding that no
error traces to missing pose or shuttle holds on the larger set.

**It is not class mix.** Reweighting clip 2's per-class accuracies to the class mix of clips 3
and 5 moves it from 0.471 to 0.491. Composition accounts for two points of a twenty-six point
gap.

**It is not the weak labels, and this one took work to rule out.** Clip 2's labels are the worst
of the three (below), so the obvious reading is that the model is being marked against a bad
answer key. BST's class names carry a side prefix, so every prediction is also an independent
opinion about who hit. Restricting to the shots where BST's own opinion corroborates our hitter:

| | accuracy | shots |
|---|---|---|
| clip 2, hitter corroborated by BST | 0.489 | 45 |
| clips 3+5, hitter corroborated by BST | 0.744 | 43 |

The gap survives the restriction almost untouched. **Clip 2 is genuinely harder for this model.**

It is the same 1920-wide framing at 59.99 fps, closest of the three to clip 3's 59.18, while the
best-scoring clip 5 is the 33.75 fps outlier. So it is not frame rate either. What is left is
something about this recording — court angle, player style, lighting — that this write-up does
not identify, and three recordings is not enough to say which.

## The label-quality finding, which is separate and arrived first

While labelling strokes the project owner disputed the **hitter** on four contacts, unprompted.
Checking them exposed a defect in how clip 2 was labelled in the first place.

Clip 2's contacts were migrated from candidate verdicts whose `true_hitter` equals
`detected_hitter` on **60 of 61** real hits. That is the signature of a pass that read the
detector's answer and confirmed it — the same failure the 15 September session found and fixed on
clips 3 and 5 by relabelling them blind. Clip 2 never had that treatment.

A new check makes this visible without consulting any model:
`badminton_poc.truth.find_impossible_alternations`. A singles rally strictly alternates, so two
adjacent contacts credited to one player is a labelling that cannot be right — a wrong hitter, a
contact that never happened, or a real contact nobody recorded. It says where to look, not which.

| clip | labelling | impossible alternations |
|---|---|---|
| test-video3 | blind | **0** |
| test-video5 | blind | **0** |
| test-video2 | migrated | **13** |

The check separates the two labelling methods perfectly, on a rule taken from the sport rather
than from any threshold. It now runs in every evaluation under `label_consistency` and is carried
per clip in the freeze manifest, so a clip whose labels were confirmed rather than given cannot
pass unremarked because its accuracy reads well.

**Four corrections were applied; nine violations' worth were not.** The four the owner raised each
also sat in an impossible run, so two independent reasons agreed, and 13 violations fell to 6.
The remaining six are left alone deliberately: the rule says one of the two contacts in a pair is
wrong without saying which, and a long gap (1.8-1.9s on two of them) is as consistent with a
contact nobody recorded as with a wrong hitter. Inventing a contact from memory would be worse
than declaring the gap.

Three of the four corrections moved the label towards what the *current* pipeline already says —
the migration had carried an older detector's answer that Milestone 1's September fixes had since
corrected. The fourth, frame 5060, goes the other way and exposes a live pipeline error the stale
label was hiding. Clip 2's hitter accuracy therefore moved 0.9245 → 0.9623, and **that is the
labelling changing, not the detector improving.** The manifest is re-frozen at `milestone-1/1.1`.

One scoring defect was fixed in passing: `uncertain` was being scored as a class. The labelling
sheet invites `uncertain` in place of a guess, so scoring it punishes the model for failing to
reproduce an answer no human would give. Clips 3 and 5 carry none, which is why it had never
mattered; clip 2 carries two. They are now excluded and counted separately. This is worth 1.2
points of the headline (0.5773 → 0.5895) and is stated so the gain is not mistaken for anything
else.

## What this establishes

**A two-recording result was not predictive of a third.** That is the concrete version of the
probe study's warning, and it is now measured rather than anticipated: 0.7273 → 0.5895, with the
two strongest classes falling hardest.

**BST still beats geometry by 23 points**, on three recordings and 95 shots. That is the durable
part.

**The answer key itself needed auditing, and one clip failed the audit.** Clip 2's labels were
detector-confirmed, not independently given. Six impossible alternations remain in it, and its
0.4706 should be read with that beside it — though the BST-corroboration split above says the
labels are not what makes it low.

## What is still not established

- **Three recordings is three recordings.** The drop from two to three says the curve is not flat.
  It does not say where it settles.
- **Nothing explains the clip-2 gap.** Inputs, class mix and label quality are each ruled out
  above. No positive cause is offered.
- **drive remains 0 of 4** across every configuration tried, and is still the only class under
  five examples.
- **Six impossible alternations in clip 2 are unresolved.** Each needs a human to look at the
  preview and say whether the hitter is wrong or a contact is missing.
- **Clip 2 has never been labelled blind.** Correcting four contacts is not the same as relabelling
  it the way clips 3 and 5 were, and the four that surfaced did so only because the owner happened
  to notice while doing something else.

---

## Phase K addendum: four of the six alternations resolved, and a second parked point found

The six impossible alternations were put to the project owner as a labelling task. They asked
whether it could be done from evidence already on disk. Four of the six could.

**Important: these four are model-derived, not hand-given.** Every contact touched carries
`hitter_source=model-derived-2026-09-16`, the revision entry says so, and neither added contact
was given a shot type, because naming the stroke is the owner's call and Milestone 2 skips a
contact that has none. **Clip 2 is now a mixed labelling** and that has to be stated wherever it
is scored. Clips 3 and 5 remain entirely hand-given.

| pair | verdict | evidence |
|---|---|---|
| 1801→1877, 1877→1931 | **1877 is far, not near** | at that frame the near player stands with the racket down and the far player is mid-strike |
| 3980→4095 | **contact added at 4017, near** | shuttle descends to y=540 and reverses with dy −72, which gravity cannot do; frame shows the near player lunging |
| 4238→4348 | **contact added at 4316, near** | near player's wrist peaks at 4315-4317 then falls; frame shows them airborne in full overhead extension |
| 7196→7227, 7227→7248 | **unresolved, and why is the finding** | below |

The run of three resolved to a single error, as its shape suggested: one wrong contact in the
middle of three, not two independent mistakes.

### The 7227/7248 pair exposed a second parked point

TrackNet rests on a mark on the back wall at about (400, 175) in four stretches of 11 to 15
frames: 4170-4180, 7216-7228, 7247-7261 and 7347-7360. **The labelled contacts at 7227 and 7248
sit inside two of them.** What reads as a sharp direction reversal at those frames is the tracker
jumping off a stationary object, not a racket.

The fixed-point detector did not catch it. It found the large parked point at (918, 222), which
holds 2340 frames across 14 runs — 35% of every visible frame in the clip — and stopped there.
This one holds 59 frames across four runs and falls under the threshold. **A parked point does
not have to be long-lived to put a false contact in the labels; it only has to be parked while
something is being decided.**

Three things agree that 7227 and 7248 are artefacts: the tracker is parked at both, the far
player is untracked from 7228 to 7248, and the current pipeline emits no hit at either frame. The
owner independently marked both `uncertain` when asked for a stroke — they could not name one
because there was no stroke to name.

**Neither label was changed anyway.** "The current detector does not produce this contact" is
precisely the circular reasoning that put clip 2 in this state, and it cannot be the grounds for
deleting a label. It is recorded in `tracking_failures` and left for a human. Two impossible
alternations remain, both in this run.

### What it changed, and what it did not

| | before | after |
|---|---|---|
| impossible alternations | 6 | **2** |
| hit precision / recall | 0.9016 / 0.8852 | 0.9016 / **0.9016** |
| hitter accuracy | 0.9623 | **0.9811** |
| BST agreement with our hitter | 0.9231 | **0.9519** |
| **BST accuracy, three recordings** | **0.5895** | **0.5895** |
| clip 2 accuracy | 0.4706 | **0.4706** |

The Milestone 1 figures moved because the labelling changed, not because the detector did. 4316
was a genuine missed hit the labels were hiding, and 4014 was being scored as a false positive
against a contact that was really there.

**The Milestone 2 headline did not move at all.** Clip 2 still reads 0.4706 after four label
defects in it were repaired. That is the strongest evidence yet that its low score is a property
of the recording rather than of its answer key, which is what the BST-corroboration split already
suggested and what this now confirms from the other direction.
