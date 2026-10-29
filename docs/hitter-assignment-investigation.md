# Why the hit stream repeats the same player, and what is not the cause

2026-09-11. Investigation only. No pipeline change was made, deliberately: five candidate fixes
were measured and every one of them was worse than the code already in the repository.

## The instrument

In singles a player cannot hit twice in a row, so every consecutive same-player contact in our
output is an error. That check needs no ground truth, so it runs on every clip we have and every
clip we ever add. Measured on the four recordings (the fifth input file is byte-identical to
`test-video3`):

| clip | hits | unattributed | same-player repeats |
|---|---|---|---|
| test-video | 32 | 4 | 50% |
| test-video2 | 68 | 1 | 23% |
| test-video3 | 22 | 3 | 43% |
| test-video5 | 61 | 8 | 24% |

`test-video2` is the clip Milestone 1 was tuned against and it is the best of the four.

## What was measured and rejected

Each rule decides one hit at a time and never reads its neighbours' answers, so the repeat rate
stays an independent score rather than something the rule enforces. Percentages are repeats over
all four clips, lower is better. The current pipeline scores 29%.

| candidate | result | why it failed |
|---|---|---|
| raw pixel proximity (what the code does) | 42% | the honest per-hit rate; see below |
| distance ÷ player box height | 54% | fixes the far bias, but where the far box is unstable it amplifies the noise |
| shuttle recedes from the hitter | 69% wrong-hitter rate | the shuttle rises after almost every contact, so vertical image motion measures height gained, not court direction |
| side of a smoothed trajectory run | 46% | run structure is clean between contacts and ambiguous at them, which is the only frame that matters |
| larger racket swing wins | 50% | both players are often mid-motion; a lunge to retrieve reads like a stroke |

The pipeline's 29% is better than the 42% its own rule scores because two later passes flatter
it: ambiguous hits are dropped, removing the hardest cases from the denominator, and
`enforce_alternation` fills unassigned hits by *assuming* alternation. The honest per-hit
accuracy of the current assignment is the 42% row.

## Two things that look like causes and are not

**The far slot's identity churn.** The far player is held by 7 track ids on `test-video2` and 14
on `test-video5`, against 1 and 2 for the near player. This is real, but it is one genuine player
being repeatedly re-identified, confirmed by eye at frames 5016-5120 and by the tracks tiling the
video end to end. `assoc.py` never reads a track id, so churn cannot produce a wrong hitter. It
matters only to the mobility proof and the identity-transfer gate.

**Contact density.** An earlier version of this document claimed the detector fires far too
often on three of the four clips, from contact rates of 2.27 and 2.45 per second of play against
`test-video2`'s 1.12. That was wrong, and the error was mine: the clips do not share a frame
rate. The working videos run at 28.0, 59.99, 59.18 and 33.75 fps respectively, and every timing
figure had been computed at 60. Corrected, the rates are close to 1.1-1.3 per second everywhere
and nothing separates the clips. No timing argument survives, and none is needed -- the hand
labels below measure the same thing directly and without assumptions.

## The measurement that was missing, and what it said

The repeat rate cannot distinguish a misassigned hitter from a missed contact from a spurious
one, and all three are present. So the project owner labelled every contact in two whole rallies,
57 of them, naming the player or saying no contact happened. The pipeline's own answer was not
shown during labelling. The labels are in `data/labels/hitter_truth.json`.

| | |
|---|---|
| detected contacts that never happened | 14 of 57, **25%** |
| hitter correct, on the real contacts | 35 of 41, **85%** |

**This reverses the diagnosis.** Hitter assignment was the healthiest part of the component all
along, and the repeat rate had been pointing at it because a spurious contact produces a repeat
whatever player you attach to it. Every one of the five rules above was aimed at the wrong thing,
which is why none of them helped: re-scored against the labels they land at 83%, 83% and 76%
against the shipped code's 85%, so the assignment really was at its ceiling for this approach.

## The fix

A contact happens at the end of a racket, so the shuttle must be within about arm plus racket of
somebody. At a real contact the shuttle sits a median 75 px from the nearest player anchor; at a
spurious one, 238 px. The gate that should already have caught this is
`max_assignment_distance_frac`, 0.35 of the frame diagonal, which is 740 px and never fires.

The threshold has to be in the player's own body heights rather than pixels, because the far
player is half the near player's size in frame and the same real reach is half as many pixels at
the far end. Fitting on one clip and testing on the other gives 0.75 and 1.08 body heights;
`max_contact_distance_body_heights` is the 0.9 midpoint, which is also roughly what arm plus
racket measures against a person's height. Fitted in pixels the same procedure gives 118 and 185
and does not transfer between the clips.

One thing had to change with it. `enforce_alternation` fills an unassigned hit from its
neighbours, and it was handing a hitter straight back to the contacts the new gate had just
rejected -- four of the seven on `test-video5`. Alternation is for contacts we could not call,
not for ones nobody could reach, so the outcomes in `NOT_FILLABLE_BY_ALTERNATION` are now left
alone.

### Result

Against the hand labels, both clips:

| | before | after |
|---|---|---|
| real contacts kept | 41 of 43 | 40 of 43 |
| spurious contacts kept | 11 of 14 | 3 of 14 |
| precision | 79% | **93%** |
| hitter correct on what it keeps | 85% | **88%** |

Same-player repeats fall on all four recordings, which uses no labels: 50 to 44%, 23 to 12%,
43 to 33%, 24 to 22%. They stay high because the other half of the fault is still there -- the
detector also *misses* contacts, and a missed contact shows up as two genuine same-player
contacts in a row. The labels prove those exist and do not say how many.

## The other half: contacts that were never detected

The same labels measure recall for free. Two genuine same-player contacts in a row means a real
contact was never found, and there are six such gaps in the two rallies: recall is **at most
88%**, which made it the weaker half once precision reached 93%.

Opening all six:

| gap | what was actually there |
|---|---|
| `test-video2` 3932-4014 | a real contact *inside* one unbroken 74-frame flight |
| `test-video2` 4234-4345 | a 55-frame shuttle dropout; nothing to segment |
| `test-video5` 1828-1888, 1980-2029, 2217-2256, 2329-2382 | joins classified as tracker dropout |

The last group is the pattern worth having. `min_turn_degrees` asks whether the shuttle changed
direction enough to have been hit, and a drive, a flat push or a block barely changes it at all,
so the test threw them away as tracker blinks -- at turns of 42.8, 56.7 and 57.7 degrees against
the 60 degree threshold. Across both clips, 15 of the 31 joins called continuations sit within a
racket's reach of a player.

Angle and position answer different questions, so both are asked now. A large turn is a contact
wherever it happened; a join at the end of somebody's racket is a contact however small the turn.
`promote_continuation_within_body_heights` is the second test, sharing units and threshold with
the gate above. `segment_flight` takes a `racket_reach_at` callback rather than learning about
players, so it stays a function of the shuttle track alone.

The first gap, a contact hidden inside a single flight, is a different fault and is **not fixed**.
A quadratic fitted to x(t) *is* a parabola, so a clean there-and-back in x fits it perfectly and
the residual test cannot see it. Exactly one segment across both clips reverses in x, so a
monotonicity test would find it with no false positives. It is one contact and was not worth the
change today.

## Where this leaves the component

| over the 57 hand labels | before | after |
|---|---|---|
| real contacts found | 41 of 43 | 39 of 43 |
| contacts that never happened, kept | 11 of 14 | **3 of 14** |
| precision | 79% | **93%** |
| hitter correct | 85% | **90%** |
| known-missing contacts recovered | 0 of 6 | 2 of 6 |

**A regression to state plainly.** On `test-video` the promotion rule takes the hit count from 20
to 39 and pushes same-player repeats from 44% to 52%. That clip's shuttle track is the broken one
-- 13% of steps over 150 px, 89 over 400 px in a single frame -- and its flight segments have a
median length of 4 frames for *ordinary* contacts too, against 16 and 8 on the other clips. Its
segmentation is built on an unusable track, so neither its before nor its after number means
much, but the rule does make it worse and no labels exist there to say by how much.

## What is still open

Three of fourteen spurious contacts survive the gate, and four of the six known-missing contacts
are still missing. Everything here rests on two rallies from two recordings, so the next move is
labels on a third and fourth clip, not more tuning against these.

Two smaller things found on the way. The `analyze` subcommand never passed the flight
configuration through, so that whole section of a config file was silently ignored and three
different `contact_frame_rule` settings produced byte-identical output; it is now passed, and the
flight config is recorded in `analysis.json` alongside the rest, which is what would have made
the bug visible. And `contact_frame_rule: "exit"` timestamps a contact at the last frame before
the tracker loses it, which is early by the whole dropout: the contacts labelled "nobody hit it"
have a median gap of 15.5 frames against 6 for the real ones. Switching to `midpoint` moves the
median reach from 0.45 to 0.36 body heights on one clip and 0.59 to 0.42 on the other, but it
rescues none of the spurious contacts -- every one that lands within reach at the midpoint was
already within reach at the exit frame -- so it is left at its default pending a reason to move
it.

## What perspective does to this problem

The camera sits low and close to one baseline, so the far half of the court is compressed into a
thin strip above the net. The far player stands about 130 px tall against the near player's 280.
A metre of court near the camera is many more pixels than a metre at the back, so comparing the
two players' distances in pixels is not comparing distance at all. Dividing by box height is the
crude correction and it is too noisy to help. The principled version is a court homography, which
gives both players' positions in metres from their feet on the floor, and which Milestone 2 needs
anyway for BST's court coordinates. It does not directly place the shuttle, which is airborne at
contact, and `assoc.py` already documents that trap.

## Also found

`test-video`'s shuttle track is worse than the others', though not in the way first claimed here.
That claim -- 13% of steps over 150 px, 89 over 400 px -- was measured on the raw TrackNet CSV,
which is not what the pipeline reads: `mask_offcourt_shuttle` drops detections outside the court
airspace first, and most of those jumps never reach the detector. Measured on the track actually
used, and against a speed limit scaled by the player's own size, 0.3% of its steps are physically
impossible.

What is genuinely wrong with it is thinner and slower: the shuttle is missing on 25% of its rally
frames against 2-5% elsewhere, and its 99th percentile step is 78% of the physical maximum where
the other three sit between 12% and 21%. A shuttle is near top speed for a few frames of a smash
and nowhere else, so that tail is the tracker jumping. `shuttle_track_quality` in `analysis.json`
now reports all of this per clip. Nothing branches on the verdict; it exists so that a clip like
this announces itself instead of quietly producing 38 confident contacts.

## Clip 6, before labels: what goes wrong, and what a sequence model would need (2026-10-29)

Clip 6 (club, camera at head height) has no hand labels yet, so nothing here is an accuracy
figure. What can be measured without labels is the singles invariant: within a rally, two
consecutive attributed hits should not go to the same player.

| clip | consecutive same-player pairs | of which near via `closest_anchor`, both players observed | far player occluded at the hit |
|---|---|---|---|
| test-video2 | 3 / 51 | 0 | 0 of 62 hits |
| test-video3 | 4 / 12 | 2 | 0 of 13 |
| test-video5 | 10 / 42 | 7 | 5 of 70 |
| test-video6 | **21 / 40** | **14** | 8 of 57 |

(From a fresh `analyze` run; the far-player state is `visibility.classify`.)

**Occlusion is not what drives clip 6's repeats.** Only 2 of the 21 come from `only_near_detected`.
Fourteen are the near player winning `closest_anchor` with both players in frame, and in all 14
the shuttle is genuinely at the near player's wrist in the image: 0.04-0.43 of the near player's
body height away, against 0.48-2.03 of the far player's. Normalising distance by body height
flips **none** of them, so that obvious fix is out before it is tried. Each repeat is one of:

1. a real near hit, with the hit before it misattributed or spurious;
2. a far contact whose position projects onto the near player's body, because a head-height
   camera puts both players on the same image rows;
3. a spurious contact detected as the shuttle passes the near player's racket.

The labels decide between these. Until then, any change to `assign_hit` would be tuned blind.

### Inputs a sequence model would need

Per detected hit, all already in `analysis.json` or one join away:

- both players' best-anchor distances and body heights (`assignment_evidence`);
- each slot's visibility state at the hit (`player_visibility`, `visibility.classify`);
- the shuttle's direction of travel after the contact, from the flight segment that starts at
  it: **towards** the camera means far hit it, away means near did. This is the one strong cue
  pixel distance does not use, and on a head-height camera the depth axis is still readable
  from image y and apparent speed;
- whether the shuttle samples around the hit are observed or inpainted
  (`BallSample.inpainted`): a direction read from InpaintNet's fill is weaker evidence;
- the hit's own detection confidence and kind (`rule`, flight join kind).

### Proposed model, not built

A two-state HMM (near, far) over each rally's detected hits, with two extra transitions: a
**missed hit** (the same player appears twice because the other's hit was not detected) and a
**spurious hit** (an emission that belongs to no stroke and is skipped). Per-hit emissions come
from the features above, calibrated on clips 2, 3 and 5. Decoding is Viterbi per rally. This is
what forced alternation was not: one missed or spurious hit costs one decision, not every
decision after it.

### How it would be judged

- Hitter accuracy on matched hits, per clip, against the frozen `milestone-1/1.3` figures on 2,
  3 and 5: no clip may regress.
- Clip 6 is held out completely: emission weights fitted on 2, 3 and 5 only, scored once on 6.
- Contacts the labeller marked `?` or `u` are excluded, as the scorer already does.
- The miss/spurious transition probabilities come from the scored hit-detection recall and
  precision on 2, 3 and 5, not from tuning on any clip's hitter figure.
