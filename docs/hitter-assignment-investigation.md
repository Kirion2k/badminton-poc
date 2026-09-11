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

**Phantom contacts, as first framed.** Contact density is 1.12 per second of play on
`test-video2` against 2.27 on `test-video5` and 2.45 on `test-video`, and 46-49% of gaps on those
two clips are under 0.35 s. But the shuttle shows a genuine 114 degree median reversal at those
short-gap contacts, and the project owner named a plausible shot in all 53 preview windows on
`test-video5`. Most of those contacts are real events. What a run of six contacts 0.25 s apart
actually is remains unexplained.

## The measurement that is missing

The repeat rate cannot distinguish a misassigned hitter from a missed contact from a spurious
one, and all three are present. Separating them requires knowing which contacts are real and who
played them, and no such label exists: `test-video5` was labelled for shot type only, and the
`hitter_looks_wrong` flags on `test-video2` were set by the pipeline's own author and are not
safe to diagnose from. Several of them look like alternation was assumed and the labels bent to
fit, which would make any measurement against them circular.

Naming the hitter is a two-way choice per contact and far quicker to label than shot type. Sixty
contacts across two clips would give the first honest score for this component.

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
