# Badminton POC: session handoff, 8 October 2026

For someone (human or model) picking this up with no other context.

## The project in two lines

A local proof of concept that turns one fixed-camera singles badminton video into a timeline of
players, shuttle, contacts and rallies (Milestone 1), then classifies each stroke with the
pretrained BST model (Milestone 2, exploratory).

## What this session did

It worked the open-problems list from the 16 September handoff. **The list went from five items
to two.** One was fixed, one was closed, two turned out not to be problems in their own right,
and two are genuinely blocked. Nothing was tuned to reach a number.

## The list, as it now stands

| # | 16 September | 8 October |
|---|---|---|
| 1 | No club-hall recording | **Blocked on the owner.** Unchanged. |
| 2 | Two alternations + a missed fixed point | **Closed.** Labels removed; all three clips at 0. |
| 3 | Three spurious rallies (the play gate) | **Open.** Eight signals measured, none separates. |
| 4 | Clip 3 hit precision 0.667 | **Dissolved** into #3 plus one boundary error. |
| 5 | Drive 0 of 4 | **Dissolved** into #1. It needs labelled drives. |

What is actually left is **the play gate, and more footage**.

## What changed in the code

**`LABELLED_RALLY_RANGES` was stale and all four bounds were wrong.** The constant defines itself
as the range every hand-labelled rally falls in. It was taken over ten rallies on two clips; clip
5 was hand-labelled on 15 September and nobody recomputed it, so it described a smaller sample
than the one on disk for 23 days. The slower-slot floor was out by 3.4x -- 0.318 against a real
rally at 0.093 -- and that is the bound the seven play-gate attempts' player-motion argument
rested on. Corrected, the evidence block goes from flagging 2 of 3 invented rallies and 4 of 14
real ones to **3 of 3 and 3 of 14**. It still suppresses nothing.

`configs/labelled-rally-measurements.json` now holds the rally-by-rally measurement behind every
bound, `tools/measure_labelled_rallies.py` regenerates it, and a test fails if a clip gains hand
labels without being folded in -- which is the thing that actually went wrong.

**`start_kind` no longer claims to have seen a serve.** It read `serve` on all sixteen
non-truncated rallies including all three invented ones, because it measures silence before the
first flight and knocking the shuttle back between points is preceded by silence too. It is now
`after_quiet`, and `quiet_before_frames` carries the number it was decided on. This was never
cosmetic: `after_quiet` **synthesises a contact** at the first flight's start frame. Keeping that
is right -- on clip 2 all five are real -- but it also manufactures a false positive on every
invented rally, which the old name hid.

**Clip 2's contacts at 7227 and 7248 were removed**, with the owner's approval, from the video
frames. At both the tracker sits on a mark on the back wall with no shuttle near it; between
them, at 7233-7243, it jumps to the near player's racket and back. The surrounding labels are a
near lift at 7196 and a far smash at 7286, a complete alternating exchange with no room between.
The labeller wrote "?" at both. The rows and the reasoning are in `truth.json` under
`_removed_contacts`, exactly reversible. That the detector emits no hit there was deliberately
**not** part of the grounds -- confirming the detector is how clip 2's labelling went wrong.

## Current numbers (Milestone 1, frozen at `milestone-1/1.2`)

| clip | hit P | hit R | hitter | rallies | spurious | coverage | alternations |
|---|---|---|---|---|---|---|---|
| test-video2 | 0.902 | **0.932** | 0.981 | 6/6 | 0 | 0.929 | **0** |
| test-video3 | 0.667 | 0.800 | 0.875 | 2/3 | 1 | 0.979 | 0 |
| test-video5 | 0.872 | 0.895 | 0.900 | 5/5 | 2 | 0.904 | 0 |

Clip 2's recall and F1 moved because two phantom labels were removed; **deleting labels improved
a score, which is the move most worth distrusting**, so plainly: they were false negatives no
detector could ever have cleared, there being no contact at either frame. Precision, hitter
accuracy and the other two clips are untouched.

**All three clips now read zero alternations, so the check no longer distinguishes them.** Clips
3 and 5 got there by being labelled blind; clip 2 got there by repair. The manifest's
`_label_quality_note` says so, and clip 2's labelling is still the least trustworthy of the three.

Milestone 2 is unchanged: BST at 0.5895 on 95 scored shots. The dataset is 106 shots rather than
108, the two removed contacts having been its only `uncertain` rows. The **scored set is the same
104**, verified by comparing `phase_h.json`'s prediction rows against the rebuilt dataset, so no
accuracy figure moved and the model was not re-run.

## Two things the project believed that are not true

**The second parked point is not a threshold problem.** `docs/milestone-2-phases-b-to-j.md`
recorded the back-wall mark as holding "59 frames across four runs" and so falling under
`fixed_point_min_runs`. Measured, the real fixed point at (918, 222) is *pixel-identical* across
its runs -- 0.0 px drift -- while the four wall stretches drift 44.2, 62.4, 64.7 and 54.7 px,
because the tracker slides along the wall. `rest_tolerance_px` is 8.0, so none of them ever
becomes a rest run and none reaches the run count to fall under it. **Lowering
`fixed_point_min_runs` to 1 finds nothing.** Closing this means detecting a tracker that *drifts*
over a static feature with no shuttle present, which is a different detector. The doc carries a
correction and `tests/test_flight.py` holds both halves, with a frozen control.

**Clip 3's precision was never its own problem.** Its four false positives are 317, 372 and 436
-- the invented rally -- and 1035, which is rally 3 running past its labelled end (765-985
labelled, 768-1069 detected). Remove the invented rally and precision is 0.889 with recall
unchanged.

## The play gate, after eight attempts

The frames show all three invented rallies are players knocking the shuttle back across the net
between points. An eighth signal was measured this session and overlaps like the seven before it:

| signal | real rallies | the invented ones | why it fails |
|---|---|---|---|
| shuttle depth excursion | 0.279-0.889 of height | 0.247, 0.285, 0.361 | clip 5's 3203-3325 beats two real rallies |

Lateral range multiplied by depth does separate all seventeen detected rallies, at **1.28x on
three examples** -- below the 1.4x this project already rejected as fitted to one example. It is
recorded and wired to nothing.

**The architectural read, offered as a hypothesis:** every one of the eight assumes a between-
point knock is distinguishable from a rally by how the span physically looks. It may simply not
be -- a soft net exchange inside a real rally *is* the same physical activity. The labeller
resolved these spans by the scoreboard ticking, and scoreboard OCR is a stated non-goal until
Milestone 1 passes. If that read is right, the gate is not reachable from the shuttle track and
the pose model alone, and the honest move is to keep reporting rather than gating.

## Drive, and why it is not a tuning problem

| true drive | BST called it | where drive ranked |
|---|---|---|
| clip 2 f6326 | net (0.27) | **#2** at 0.207 |
| clip 5 f1736 | drop (0.92) | #3 at 0.012 |
| clip 5 f2219 | smash (0.86) | #4 at 0.024 |
| clip 5 f3568 | other (0.66) | #7 at 0.021 |

Only one is a near miss. On the other three BST puts drive at about 0.02 and is confident in
something else, so no threshold, reweighting or calibration reaches it. The four predictions of
drive are all wrong too. With four examples, anything that got them right would be fitted to four
points.

## Recommended next step

Unchanged from 16 September, and now the blocker for two list items rather than one: **the owner
records a club singles match to spec** -- fixed tripod, raised behind one baseline, whole court,
singles, 1080p+, 5-10 minutes of rallies. It tests the domain-shift hypothesis (around 0.47
confirms it, around 0.7 means clip 2 was just hard) and it is the only route to more labelled
drives.

## Open problems

1. No second club-hall recording. Blocks the domain question and the drive class.
2. The play gate: three spurious rallies, eight signals measured, none separates. See the
   architectural read above before attempting a ninth.
3. Clip 3's rally 3 ends 1.4s late (765-985 labelled, 768-1069 detected), which is the one false
   positive on that clip not caused by the play gate.
4. Clip 5's hitter attribution declines 4 of 34 matched hits at the net.
5. The fixed-point detector cannot see a tracker that drifts over a static feature.

## Where to read more

- `README.md`, the last two sections: this session in full, with the measurements.
- `configs/milestone-1-frozen.json`: frozen metrics at `milestone-1/1.2`.
- `configs/labelled-rally-measurements.json`: the rallies behind every evidence bound.
- `outputs/test-video2/truth.json`, `_removed_contacts`: the deletion and its grounds.
- `docs/handoff-2026-09-16-third-recording.md`: the previous session.
