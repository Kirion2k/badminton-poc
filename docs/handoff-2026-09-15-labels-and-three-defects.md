# Badminton POC: session handoff, 15 September 2026

Context document for another model or engineer picking this up cold. Every figure here is copied
from an artefact on disk, not from memory. Where something is unmeasured or uncertain it says so.

Repository state at the end of this session: branch `master`, clean tree, head `942a634`,
428 tests passing (`.venv/bin/python -m pytest`).

---

## 1. What this project is

A local technical proof of concept, deliberately not a product. It converts one fixed-camera
singles badminton video into a machine-readable timeline of players, shuttle events and rallies.
The pipeline is five stages, each writing its intermediate output to `outputs/<clip>/`:

```
source video
  preprocess   ffmpeg: bake rotation, uniform fps, one frame per source frame
  players      Ultralytics pose + ByteTrack, court-polygon ROI, near/far identity slots
  shuttle      upstream BadmintonTrackNet (TrackNetV3 + InpaintNet) via subprocess
  flight       shuttle track -> flight segments -> hits, landings, rallies
  analyze      hit -> player association, rally assembly, coverage metrics
  render       annotated diagnostic video
```

Shuttle inference and player inference never touch each other; they meet only in `analyze`, which
reads files from disk. `flight` takes only the shuttle CSV and the frame rate, so it needs no GPU
and a change to it can be scored against every clip in seconds.

The acceptance criteria, from `CLAUDE.md`:

- both players tracked on >= 90% of active-play frames
- shuttle trajectory coherent through most rallies
- hit detection manually scored for precision/recall
- hitter assignment >= 90% on clean detected hits
- rally boundaries >= 85% approximately correct

---

## 2. Bottom line: where it sits now

All five criteria pass on all three labelled clips.

| clip | labels | hit P | hit R | hit F1 | hitter | rallies | spurious | coverage |
|---|---|---|---|---|---|---|---|---|
| test-video2 | human | 0.8852 | 0.9153 | 0.9000 | 0.9245 (n=53) | 6 of 6 | 0 | 0.9286 |
| test-video3 | **model** | 0.9091 | 0.7143 | 0.8000 | 0.9000 (n=10) | 4 of 4 | 0 | 0.9834 |
| test-video5 | human | 0.8718 | 0.8947 | 0.8831 | 0.9000 (n=30) | 5 of 5 | **2** | 0.9044 |

Clip inventory:

| clip | frames | fps | duration | labelled contacts | labelled rallies |
|---|---|---|---|---|---|
| test-video2 | 7376 | 59.99 | 123 s | 61 | 6 |
| test-video3 | 5031 | 59.18 | 85 s | 15 | 4 |
| test-video5 | 4660 | 33.75 | 138 s | 38 (two spans) | 5 |

At the start of this session clip 3's hit precision was 0.5556 and was described in the README as
"the honest headline" failure of the project: the detector had been tuned on clip 2 and did not
survive a second recording. It is 0.9091 now, and what was wrong was not tuning.

---

## 3. What happened this session, in order

Eight commits, oldest first.

| commit | what it did |
|---|---|
| `ce4b2f5` | Stop a seated line judge holding the near slot for 2048 frames |
| `4a3b47a` | Render candidate contacts as a sheet, so labelling leaves evidence |
| `6428c94` | Let a clip be labelled in parts without lying about the parts that are not |
| `64d7264` | Stop reading the top of a lift as a racket contact |
| `f7e7a81` | Label clip 5's two longest rallies, and find rally boundaries failing |
| `da2fe59` | Give a human something to label with |
| `42e5a8a` | Replace clip 5's model labels with hand labels, and measure what that was worth |
| `942a634` | Stop a blind gap in the shuttle track from ending a rally |

19 files changed, 1602 insertions, 618 deletions.

The session began as an attempt to label clip 5 as a third clip. Labelling it surfaced two
pipeline defects and one defect in the labelling method itself, and each of those was chased to a
root cause before being fixed.

---

## 4. Defect one: a bystander could hold a player slot permanently

**Symptom.** Clip 5 reported 95.0% near-player coverage, no plausibility warnings and three
rallies. All three rallies sat in the first 56 seconds. From frame 2580 to the end of the clip --
2048 frames, 44% of the recording -- the near slot held a **seated line judge in a chair on the
red surround** while the real player stood on court in front of him.

**Why nothing reported it.** Slot plausibility is measured over *detected* rallies, and no rally
is detected in the span where the slot is wrong, because the slot being wrong is what stops
rallies being detected. A failure that suppresses rally detection is invisible to a check that
only looks inside rallies.

**Measured cost.** 32 of 41 hits dropped as unattributed, 40 of 52 events orphaned by rally
gates, and no rallies at all after frame 2467 in a span holding 812 frames of moving shuttle
(39.6% of it, against 51.8% before the lock).

**Root cause, part one: the foot point moves when the box does.** Mobility was measured from the
foot point, which is the bottom edge of the detection box, so it travels whenever the model loses
the legs. Over frames 993-996 the judge's box collapsed from 125 to 86 px and his confidence from
0.78 to 0.42, so his foot point climbed 43 px up his body while the box top moved 0.3 px, from
481.7 to 481.6. That cleared the 15.8 px mobility threshold, and mobility is granted permanently.

| frame | box top y | foot y | height | conf |
|---|---|---|---|---|
| 992 | 481.7 | 606.8 | 125.1 | 0.78 |
| 993 | 481.9 | 586.8 | 104.9 | 0.71 |
| 995 | 481.7 | 571.2 | 89.5 | 0.55 |
| 996 | 481.6 | 568.0 | 86.4 | 0.42 |

Fix: mobility is now measured at the top of the box (`Candidate.head`). Heads do not fall off, and
a player who really moves moves their head too.

**That alone did not fix it.** It moved the promotion from frame 993 to 1271, where a genuine
shift in his chair crossed the threshold instead. A per-window motion test applied across
thousands of frames will eventually promote anything, and promotion is for life.

**Root cause, part two: an incumbent never re-earns its slot.** Continuity (1.5) and proximity
(0.8) are 2.3 points the occupant of a slot gets for free, against at most 1.0 for a stranger
however well detected. At frame 3600 the judge beat the real player -- track 4, confidence 0.938,
proven mobile, standing on court -- by 3.19 to 1.05, and beat him identically on every frame to
the end of the clip.

Fix: the eligibility ladder gained a rung. `mobile` says a track moved once; `recent` says it is
still moving like a player, and a slot prefers a recent candidate to a stale one. It is a
preference, not a gate: with nobody recent on that side the stale candidate still fills the slot,
and the pick is recorded as `recently_mobile: false` beside `proven_mobile`.

**The 20-second horizon was measured from both populations, not chosen.**

| track | longest stretch reading as motionless |
|---|---|
| clip 2 near player | 7.38 s |
| clip 2 far player | 8.90 s |
| clip 3 near / far player | 3.36 s / 2.79 s |
| clip 5 near player | 3.05 s |
| clip 5's line judge | quiet 37 s when he took the slot, 98.5 s afterwards |

He reads as moving on 45 of 4628 frames, in two runs of 18 and 26 frames around a single posture
shift. Any horizon between roughly 10 and 35 seconds separates the two; 20 s is 2.2x above the
longest genuine stillness measured. The knob is `mobility_yield_seconds`, in seconds rather than
frames, because a frame count judges a 60 fps clip and a 34 fps clip by different standards.

**Result.** Clip 5 went from 3 rallies to 8 and from 52 hits to 83; the near slot held the real
player on 2025 of those 2048 frames (the judge got 2). Clip 2 was unchanged on every accuracy
figure -- one plausibility decimal moved, far lateral range 4.883 to 4.887 -- and clip 3's
`baseline.json` was byte-identical.

**Files:** `src/badminton_poc/players.py`, `src/badminton_poc/config.py`,
`src/badminton_poc/pipeline.py`, `src/badminton_poc/cli.py` (the players stage now needs `fps`),
`tests/test_players.py`.

---

## 5. Defect two: the top of a lift was being read as a racket contact

**Physical claim.** A racket and gravity both reverse the shuttle, and a detector reading
curvature sees the same 180 degrees at the top of a lift that it sees at the end of somebody's
racket. Energy separates them. A racket adds it: the shuttle leaves faster than it arrived, or
leaves in a direction gravity could not have sent it. Gravity removes it: at the top of the arc
the shuttle is at its slowest, and gravity can only turn a rising shuttle into a falling one,
never the reverse.

**Rule.** A segment join is discarded when *both* hold: rising in, falling out, and no faster
leaving than arriving. Both conditions are needed -- a smash taken at the top of the bounce turns
over exactly the same way and leaves roughly three times quicker, and the speed condition is what
keeps it. Knobs: `gravity_apex_max_speed_ratio` (1.0) and `gravity_apex_window_seconds` (0.135,
in seconds because it was validated at 8 frames on two 60 fps clips).

**Measured effect.**

| | before | after |
|---|---|---|
| clip 2 precision / recall / F1 | 0.8235 / 0.9492 / 0.8819 | 0.8852 / 0.9153 / 0.9000 |
| clip 3 precision / recall / F1 | 0.5556 / 0.7143 / 0.6250 | 0.9091 / 0.7143 / 0.8000 |
| clip 3 spurious rallies | 1 | 0 |
| clip 2 / clip 3 hitter accuracy | 0.9245 / 0.90 | unchanged |
| rally boundaries, coverage | -- | unchanged |

On clip 3 the rule is free: ten joins discarded, all ten of them false positives, and the invented
rally at frames 2017-2122 goes with them. On clip 2 it discards ten joins to remove eight false
positives and costs two real contacts.

**The two contacts it costs are worth knowing.** Truth 1743 is labelled with no hitter at all --
the labeller could not tell who played it -- and it is a lift so high the shuttle leaves the top
of the frame, so the turnover in the track sits at the frame edge. Truth 7227 is a near-player
lift whose labelled frame sits at the top of the arc rather than at the racket. **The class this
rule can genuinely lose is a soft contact played at the top of a trajectory**, and clip 5 shows
one: the far player's net reply at 1851.

**Provenance of the rule.** It came out of labelling clip 5 by hand, because `contact_sheet.py`
draws the raw track either side of a candidate and an apex stops looking like a contact once you
can see its shape. Twenty-four candidates in clip 5's longest rally had been adjudicated frame by
frame *before* the rule existed; it keeps 16 of the 17 that labelling confirmed and discards 4 of
the 7 it rejected.

**Files:** `src/badminton_poc/flight.py` (`_gravity_apex`), `src/badminton_poc/config.py`,
`tests/test_flight.py`.

---

## 6. Defect three: a gap the tracker spent blind ended a rally

**Symptom.** Clip 5's first rally runs from the serve at frame 213 to 912 and was reported as
starting at 513, nearly nine seconds late. The chain broke twice inside it, at 284-370 and again
at 451-512. Both breaks were the same rule, `max_link_gap_seconds` at 1.8 s (61 frames); the
second lost by a single frame, 62 against 61.

**Root cause.** The limit measured wall-clock time. A gap the tracker spent blind is not evidence
that play stopped -- it is evidence of nothing at all. Counted over every gap-limit break in the
three labelled clips:

| | gaps | shuttle visible through the gap | `settled_between` fires |
|---|---|---|---|
| gap ends a real rally | 9 | 100% on all nine | 9 of 9 |
| gap sits inside a rally | 2 | 32% and 19% | 0 of 2 |

The two populations separate perfectly, and **not on length**: the largest gap inside a rally is
87 frames and the smallest gap that really ends one is 85, so no ceiling on duration could ever
have split them.

**Fix.** The limit counts only the frames of a gap that carry a sample. `settled_between` -- which
was already measured on the development clip to separate the two kinds of gap -- does the
deciding. Clips 2 and 3 provably cannot move under this change, because every gap in both is
visible for all of it, and they did not.

**Exposure this leaves, stated rather than hidden.** Two rallies separated by a long stretch the
tracker saw nothing of would now be joined. There is no measurement in these clips to set a
ceiling from, so none was invented; the exposure is written into
`test_a_gap_with_no_samples_at_all_does_not_break_the_chain`.

**Files:** `src/badminton_poc/flight.py`, `src/badminton_poc/config.py`, `tests/test_flight.py`.

---

## 7. The labelling programme, and the most important result in this session

### 7.1 What was built

Labelling had no tooling. Three tools now exist:

- **`tools/contact_sheet.py`** tiles the frames around each candidate contact, cropped to where
  the raw shuttle CSV puts the shuttle, and draws the raw track either side of the frame -- orange
  before, cyan after. A contact arrives falling and leaves rising; an apex does the opposite, and
  the difference is visible at a glance. Only the shuttle marker and its raw track are drawn:
  player boxes, hit verdicts and rally spans are deliberately omitted, because a labeller who can
  see what the pipeline decided is not an independent witness.
- **`tools/label_clip.py`** renders a span of the working video with its frame numbers burned in
  and nothing else on it, at a third speed by default, because a racket contact at 34 fps is two
  or three frames long.
- **`tools/ingest_labels.py`** converts one line per contact (`c <frame> <near|far|?>`) into a
  truth file. The conversion is deliberately dumb: it will not consult the detector, will not fill
  in a hitter left as `?`, and will not move a frame number to where something was detected. It
  refuses a file with two contacts on one frame, or a contact outside the spans it was told were
  labelled.

The truth format gained **`labelled_contact_spans`**: the spans in which every racket contact has
been recorded. Predictions outside them are counted in `predictions_outside_labelled_spans` and
scored as neither hit nor miss, because the contact list there is incomplete by declaration. An
empty list still means the whole clip. A contact labelled outside the declared spans is refused
rather than scored. Rallies stay scored over the whole clip: this scopes contacts, not rallies.

### 7.2 The model labelled clip 5, then the project owner labelled it blind

The model adjudicated 126 candidate frames across two spans, proposed by two instruments so that
agreement between them meant something: the project's flight detector (83 hits) and an independent
turning-point read of the raw CSV (95 candidates). They agreed on only 52.

The project owner then labelled the same two spans by hand, watching the burned-in video, without
seeing the model's answers.

| | model labels | hand labels |
|---|---|---|
| contacts found | 35 | 38 |
| contacts with a hitter | 26 | 38 |
| agreed with the other file | 29 | 29 |
| missed | 9 | -- |
| invented | 6 | -- |
| strict alternation across both spans | only after 3 corrections | yes, first time |

**As a labeller the model scores 0.829 precision and 0.763 recall on contacts, and 0.864 hitter
accuracy over the 22 contacts it was willing to name.** Roughly one contact in five and one hitter
call in seven was wrong.

### 7.3 The direction of the error matters more than its size

**Seven of the nine frames the model had recorded as detector false positives were real contacts
that the model itself had missed** -- 1828, 1888, 2059, 2114, 2234, 2256 and 3726.

| clip 5, identical pipeline run | scored vs model labels | scored vs hand labels |
|---|---|---|
| hit precision / recall / F1 | 0.7692 / 0.8571 / 0.8108 | 0.8718 / 0.8947 / 0.8831 |
| hitter accuracy | 0.8947 over 19 scored | 0.9000 over 30 scored |
| rally boundaries | 4 of 7 | 4 of 7 |
| coverage | 0.9016 | 0.9016 |

A labelling that is wrong in that direction makes a working pipeline look broken. Rally boundaries
and coverage do not move, because neither depends on the contact list.

### 7.4 How the model's hitter errors were produced

Both wrong hitter calls in rally A came from one inference: *an unbroken flight arriving at a
player whose identity is certain implies the other player struck it*. That is sound only if the
flight really is unbroken, and it was not -- the model had missed contacts at 1835 and 1890,
either side of 1851. **A missing contact does not stay a missing contact; it propagates into a
wrong hitter.** The same parity reasoning, used to hunt for missing contacts, invented one at 3588
that the hand labels do not contain.

### 7.5 The rally labels were wrong too, and the scoreboard settled it

After the boundary fix clip 5 read 5 of 7 rallies, and the remaining failures turned out not to be
the detector's. **The clip holds five rallies; the first labelling claimed seven.** Those labels
had been derived by taking every run of moving shuttle for a rally, and two of the runs are the
players knocking the shuttle back to each other between points.

The scoreboard is the instrument that settles it: it reads 3-7, 3-8, 3-9, 4-9, 4-10, 4-11 across
the clip -- five points, five rallies. It is corroborated in the frames: at 2745 the near player is
preparing to serve, and the score ticks 3-9 to 4-9 *between* frames 3210 and 3260, so the span
labelled 3191-3336 sits after that point ended rather than during it. The umpire's entry lags the
end of a point by about 120 frames, measured against rally 2, whose last hand-labelled contact is
at 2449 and whose score change appears between 2570 and 2600.

Rally ends are now the frame the shuttle comes to rest, read off the raw track. **Three automatic
rules for that were tried first and each failed differently**, which is worth recording because
each looked reasonable:

1. *First sustained standstill* runs late: a shuttle lying on the court jitters by up to 31 px
   under this tracker, which defeats any tolerance tight enough to mean anything.
2. *Last sustained travel* runs late for the same reason, and also cuts rallies short at the very
   blind gaps that defect three exists for.
3. Both are confounded by the tracker leaping onto its parked point at (915, 220), which looks
   exactly like motion.

| clip 5 rally | labelled | predicted | start error | end error |
|---|---|---|---|---|
| 1 | 213-912 | 191-896 | -0.65 s | -0.47 s |
| 2 | 1674-2470 | 1674-2467 | 0.00 s | -0.09 s |
| 3 | 2998-3112 | 3008-3110 | +0.30 s | -0.06 s |
| 4 | 3509-3878 | 3506-3862 | -0.09 s | -0.47 s |
| 5 | 4395-4557 | 4400-4552 | +0.15 s | -0.15 s |

Five of five recovered, every edge inside the 1.0 s tolerance, median start error 0.00 s and
median end error -0.15 s.

---

## 8. Open problems, ranked

1. **The play gate: two invented rallies remain on clip 5**, at 2646-2681 and 3203-3325. Both are
   players knocking the shuttle back between points -- the same thing the first labelling got
   wrong. This is not a boundary problem. The README records seven attempts at a play gate, all
   measured, none of which separated on two clips. It now has a third clip and, for the first
   time, two labelled examples of exactly the stretch it has to reject.
2. **Clip 3 is still model-labelled.** Its 0.9091 precision and 0.7143 recall carry an error bar
   of about the size measured in section 7.2 until it is labelled by hand. Its low recall is
   exactly the shape a labelling with invented contacts produces. It is the cheapest clip to
   label: 4 rallies and about 14 contacts, all inside the first 30 seconds of an 85-second clip.
3. **Hit recall**, 0.7143 on clip 3 and 0.8947 on clip 5. The gravity-apex rule knowingly costs
   soft contacts played at the top of a trajectory.
4. **Hitter attribution at the net.** The pipeline declined to assign a hitter on 4 of clip 5's 34
   matched hits. It is genuinely hard on this camera: the far player is about 130 px of dark kit
   and spends much of both labelled rallies at the net directly behind the near player.
5. **No acceptance-grade clip from a public source.** Seven YouTube clips were screened in an
   earlier session and none qualified -- four rejected outright, three fitted but failing on
   camera height, far-player size or occlusion.

---

## 9. What is not established

- **Three clips is three clips.** Two now carry hand labels; one does not.
- **The slot-plausibility blind spot survives.** It measures inside detected rallies, so it still
  cannot see a slot failure that prevents rallies from being detected. What changed is that this
  particular failure no longer happens, not that it would be reported if it did.
- **A wrong person who moves like a player would still pass** the plausibility check. The one clip
  where it caught a wrong-court track (`yt-irish`) is one clip agreeing, not a validated check.
- **The blind-gap fix has no ceiling** (section 6).
- **The gravity-apex rule rests on eighteen false positives across two clips**, one of which was
  model-labelled at the time.
- **Input representativeness is untested.** Nothing here establishes that the pipeline works on
  footage resembling whatever it is eventually meant to run on. `tools/screen_clip.py` answers
  that per clip in a couple of minutes and is the cheapest high-value check available.

---

## 10. Method notes for whoever continues

These cost real time this session and are worth inheriting.

- **A labelling can be wrong in a direction that flatters or damns the pipeline**, and the damning
  direction is the dangerous one, because it sends you hunting for a bug that is not there. Check
  a sample of claimed false positives against the video before believing them.
- **Alternation is an instrument, not a source of labels.** In singles the hitters of a rally are
  a two-colouring, so one known hitter fixes every other by parity; a second known hitter that
  disagrees proves something is wrong nearby. Used to *find* errors it caught three. Used to
  *fill in* a hitter it produced two wrong attributions and one invented contact.
- **Prefer `hitter: null` to a guess.** The scorer drops null-hitter contacts from the hitter
  denominator, so an honest refusal costs nothing and a guess corrupts the figure.
- **Do not derive labels from the same noisy signal the detector consumes.** Three automatic rules
  for rally ends failed before reading them off the track by hand.
- **Check whether an artefact on disk was produced by the current code** before measuring against
  it. Clip 5's `analysis.json` was six commits stale at the start of this session, and its
  `metadata.json` still embeds the config from a superseded preprocess run.
- **Verify a "regression" before treating it as one.** The head-anchor fix removed 81 far-slot
  frames on clip 5, which read as a coverage loss; the removed frames had median box height 85 px
  against 138 px for the frames kept and were sitting motionless in the line-judge depth band.
  They were officials, not players.

---

## 11. Reproduction

```bash
# tests, no GPU needed
.venv/bin/python -m pytest                       # 428 passing

# re-score a labelled clip without re-running any model
.venv/bin/badminton-poc run --video data/input/test-video5.mp4 \
    --config configs/test-video5.json --stages analyze
.venv/bin/badminton-poc baseline --video data/input/test-video5.mp4 \
    --config configs/test-video5.json

# re-run the pose pass (GPU; about 4 minutes for clip 5 on an RTX 3070)
.venv/bin/badminton-poc run --video data/input/test-video5.mp4 \
    --config configs/test-video5.json --stages players,analyze

# labelling a new span
.venv/bin/python tools/label_clip.py --clip test-video5 --span 1674 2462 --out label-A.mp4
.venv/bin/python tools/ingest_labels.py --clip test-video5 \
    --labels outputs/test-video5/labels.tsv --span 1674 2462 --span 3501 3884
```

**Files that are not regenerable** and are kept in git: `outputs/*/truth.json`,
`outputs/*/labels.tsv`, `outputs/*/baseline.json`, `outputs/*/ground_truth.json`,
`outputs/*/evaluation.json`. Everything else under `outputs/` is derived.

**Where the detail lives.** `README.md` is the project's narrative record and is long on purpose;
every finding in it carries the measurement behind it. The sections added this session are "A line
judge held the near slot for 2048 frames", "The top of a lift is not a racket", "Clip 5, labelled
in parts", "What a hand labelling was worth", "A gap the tracker spent blind is not the end of a
rally" and "The rally labels were wrong too".

The model's superseded labels for clip 5 are at git `f7e7a81:outputs/test-video5/truth.json`, kept
because the comparison in section 7 is the calibration for every model-labelled figure.
