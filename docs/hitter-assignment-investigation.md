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

### What is still open

Three of fourteen spurious contacts survive the gate, and the missed-contact half is unmeasured.
Both need labels on a third and fourth recording before anything else is tuned, since every
number here rests on two rallies.

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

`test-video`'s shuttle track is broken in a way the other clips' are not: 13% of steps exceed
150 px and 89 exceed 400 px within a single frame. Its 50% repeat rate has that separate cause,
and it should not be pooled with the others when scoring this component.
