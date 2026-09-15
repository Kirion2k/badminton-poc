# Milestone 2, Phases B-J: pretrained BST against hand-labelled contacts

Milestone 1 is frozen at tag `milestone-1-frozen`, manifest `milestone-1/1.1`, and nothing here
retunes it. Every number below comes from `outputs/bst_reference/phase_h.json` and
`shot_dataset.json`.

**Headline: 0.7273 accuracy and 0.6569 macro F1 over 44 hand-labelled contacts, against 0.3409
for a hand-written geometry baseline on the same shots.** Two of the four exploratory targets are
met, two are missed narrowly, and the dataset is two recordings deep.

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
