# Badminton POC: session handoff, 10 September 2026

Context document for another model or engineer picking this up cold. It covers what the project
is, what state it was in, what was changed and why, what the numbers are, and what those numbers
do not establish. Figures are copied from artefacts on disk, not from memory.

---

## 1. What the project is

A local, offline proof of concept that turns one fixed-camera singles badminton video into a
machine-readable timeline of players, shuttle events and rallies. It is explicitly **not** the
production app: no frontend, no auth, no cloud, no LLM coach, no doubles, no shot classification
until the base vision pipeline is validated on a real clip.

Stages, each writing files to disk:

```
source video
  preprocess   ffmpeg: bake rotation, uniform fps, one frame per source frame
  players      Ultralytics YOLO11m-pose + ByteTrack, court-polygon ROI, near/far identity slots
  shuttle      upstream BadmintonTrackNet (TrackNetV3 + InpaintNet) run as a subprocess
  flight       shuttle track -> flight segments -> hits, landings, rallies   [NEW this session]
  analyze      hit -> player association, rally assembly, coverage metrics
  render       annotated diagnostic video
```

Player inference and shuttle inference never touch each other. They meet only in `analyze`,
which reads files from disk.

**Milestone 1 acceptance criteria**, from `CLAUDE.md`, for one clean singles test clip:

- both players tracked on >= 90% of active-play frames
- shuttle trajectory coherent through most rallies
- hit detection manually scorable for precision/recall
- hitter assignment >= 90% on clean detected hits
- rally boundaries >= 85% approximately correct

There are four test clips. Only clip 2 (`test-video2`, 7376 frames, 123s, 60 fps, 1920x884) is
fully labelled. Clip 1 is a poor recording kept as a regression clip; occlusion alone costs it
ten points of far-player coverage, so it cannot pass the tracking criterion however good the code
is. Clips 3 and 5 have truth drafts sitting at `labelled: false`.

---

## 2. State at the start of this session

Two of the criteria passed. **Rally boundaries were failing at 4 of 6** against a target of 85%.

The previous sessions had measured four separate routes to better rally boundaries and recorded
that all four failed:

1. a play gate on shuttle motion — recovered 4 of 6, same as doing nothing, and introduced two
   new failures
2. scoreboard OCR — legible on only one of three clips
3. a swing gate on hits, using peak wrist speed from the pose model — improved hit precision from
   0.62 to 0.84 but made rally boundaries *worse*, 4 of 6 down to 2 of 6, because removing hits
   opens gaps longer than the rally break inside real rallies and splits them
4. ending a rally at its last hit instead of at a landing — 4 of 6, unchanged

The recorded diagnosis was that rally ends run long because false-positive hits during dead time
get pulled into the rally before its terminating event is looked for, and that **rally boundaries
cannot be fixed downstream of the upstream event detector**. The stated next step was to replace
that detector.

The upstream detector is `tracknet/inference/events.py` in the BadmintonTrackNet checkout. It
applies finite-difference rules to the shuttle track (`speed_local_max`, `acceleration_peak`,
`vx_reversal`, `vy_reversal`) and attaches a fixed confidence per rule. On clip 2 it emits **777
hit candidates and 665 landing candidates against 60 real rally shots**. Its "confidence" is not a
learned score; a threshold on it selects which rules to trust, not how certain a detection is.

---

## 3. The investigation

Four probes on clip 2's shuttle track, before writing anything.

**The shuttle track itself is good.** 7198 of 7376 frames carry a visible shuttle, 97.6%. So the
problem was never missing data.

**Smoothness separates hits cleanly.** Fit a quadratic to the shuttle path over a centred
seven-frame window and take the RMS residual:

| where | median residual |
|---|---|
| in a rally, away from any hit | 0.88 px |
| at a real labelled hit | 5.14 px |
| dead time, tracker frozen | 0.00 px |

A shuttle in flight is almost perfectly smooth. A racket contact is where that smoothness breaks.

**Direction change separates them too.** Angle between the velocity over the four frames before a
frame and the four after: median 150 degrees at real hits (10th percentile 101), against a median
of 9 degrees across all in-rally frames (90th percentile 90). A twenty-line prototype using only
this already matched the tuned production pipeline, 0.61 precision and 0.93 recall against 0.62
and 0.90.

**The false positives are mostly not inside rallies.** Of the prototype's 36 false positives, only
9 fell inside a rally span. The other 27 were in dead time, and they clustered: eleven fell in a
single 400-frame stretch between rallies where the tracker was following something erratic.

**Dead time is bimodal.** Either the tracker is frozen on a fixed object, median inter-frame speed
0.00 px, or it is jumping between blobs. Neither looks like flight. A static-lock test over 15
frames caught 2309 of 3646 dead frames while wrongly catching only 64 of 3730 play frames.

That pointed at a detector built on segmentation rather than per-frame rules.

---

## 4. The new detector, `src/badminton_poc/flight.py`

Takes only the shuttle CSV and the frame rate. It never reads the pose model or upstream's events
file, which makes it testable against a hand-written track and re-runnable on every clip in
seconds with no GPU.

### The model

1. **Measure every frame.** RMS residual of a quadratic fit, and median local speed, over a
   centred window. A frame without a full window of visible samples is left unmeasured rather
   than measured on a partial one.
2. **Find fixed points.** Positions the tracker freezes on in several separate stretches.
3. **Cut flight segments.** Maximal runs of frames that are smooth enough, fast enough, long
   enough, and not sitting on a fixed point.
4. **Classify the joins.** Two adjacent segments close enough in time are linked. A large turn
   makes the link a **contact**; a small turn makes it a **continuation**, one flight the tracker
   briefly lost. A standstill anywhere inside the gap breaks the link entirely.
5. **Chain them.** A rally is a maximal chain of segments joined by links. Its start is the first
   segment's start, its end is the last segment's end. A hit is emitted per contact link, plus a
   serve at the chain's start when a quiet period precedes it, plus one landing at the end.

### The four things that had to be got right

**Not every join is a contact.** Treating a low-turn join as a rally break split two rallies in
half on clip 2. Merging them is what made the chains hold together.

**Gaps must be generous, and something other than length must close them.** Three intra-rally
dropouts on clip 2 run 0.8s, 1.53s and 1.55s, so a tight gap limit splits rallies. What separates
a long intra-rally dropout from the gap between two rallies is not its length but whether the
shuttle track came to a standstill inside it. Measured on clip 2 this separates them exactly: of
roughly 75 gaps, the only one containing a sustained fixed-point stretch is between rallies, and
every intra-rally gap contains none, losing the shuttle to invisibility instead.

**A frozen position is usually furniture.** TrackNetV3 must emit exactly one shuttle per frame
whether or not one is in play, so between points it settles on a ceiling light. On clip 2 that
position is pixel-identical across **14 separate stretches totalling 2340 frames**, while the one
genuine shuttle-at-rest appears **once for 85 frames**. Recurrence tells them apart and needs no
court geometry. It is the same fixture in all four clips, at roughly (918, 222), accounting for
860 to 2991 frames each.

**Where the contact is placed decides who gets blamed for it.** The shuttle leaves the racket
fast, so a hit reported a few frames late already sits nearer the opponent. Under a late rule all
six wrong hitter assignments were near-player shots credited to the far player. Measured
end to end on clip 2:

| contact placed at | precision | recall | F1 | hitter |
|---|---|---|---|---|
| the worst-fitting frame in the gap | 0.736 | 0.898 | 0.809 | 0.885 |
| the middle of the gap | 0.750 | 0.915 | 0.824 | 0.926 |
| **the last frame of the incoming flight** | **0.778** | **0.949** | **0.855** | **0.927** |

`exit` wins on every measure despite reporting contacts a median three frames (50ms) early,
because it is the last sample where the shuttle is unambiguously still on its way *to* the racket.

### Parameters

All durations in seconds and all distances in pixels at a reference width, because the clips agree
on neither axis: frame rates run 28 to 60 fps, so a window fixed in frames measures a different
amount of time on each, and a speed in px/frame means half as much at 60 fps as at 30. Distances
scale by actual frame width at run time.

```
reference_width           = 1920
fit_window_seconds        = 0.12
max_fit_residual_px       = 2.5     # vs 0.88 median in flight, 5.14 at a hit
min_flight_speed_px_s     = 150.0
min_flight_seconds        = 0.10
max_link_gap_seconds      = 1.8     # intra-rally dropouts reach 1.55s
min_turn_degrees          = 60.0    # contacts 150 median, in-flight 9 median
rest_window_seconds       = 0.20
rest_tolerance_px         = 8.0
min_rest_seconds          = 0.25
fixed_point_min_runs      = 3
fixed_point_tolerance_px  = 12.0
min_serve_quiet_seconds   = 1.0
min_hits_per_rally        = 2
contact_frame_rule        = "exit"
```

### Two gates in `analyze`

`require_hitter` discards hits no player could have played. It barely fires on clip 2 (3 hits),
because during dead time the players are on court and a spurious hit near them still gets
attributed. Left off by default.

`require_both_players` discards a chain in which only one player ever made contact. A rally is an
exchange; a run of contacts by one player is a handback, a knock-up, or somebody bouncing a
shuttle between points. On clip 2 the two chains that are not rallies read `NNN` and `FF` while
all six real rallies carry contacts from both sides. It asks only that both players appear, not
that they alternate, because at 93% assignment accuracy a real rally routinely shows runs of one
name. **This is on for all four clips.**

### Related fix in `assoc.py`

A hit landing in a one- or two-frame dropout of one player left the only remaining candidate to
win by default, and which frame a detector lands on is close to chance. Three of six wrong
assignments were hits where the far player was simply missing, with the same player present a few
frames either side. Nobody crosses a court in 60ms, so the nearest frame holding both players is
now borrowed, with the offset recorded. **This helped the old path too**, taking its hitter
accuracy from 92.3% to 94.1%.

---

## 5. The labels were wrong, and were corrected

This is the part most worth scrutinising, because correcting your own yardstick is exactly how
results get faked. What follows is what was done and what was checked.

**How it surfaced.** Rally 2 was labelled 773-1098, and the shuttle track showed the tracker
frozen on one pixel from frame 1005 to 1080. Frames were pulled out of `outputs/test-video2/work.mp4`
with ffmpeg and read directly. At frame 933 the far player is lunging low at the net with the
point live. By 941 both players are upright and walking. The labelled end was 2.7 seconds into
dead time.

**What was then done.** All six ends were re-measured the same way at 8-frame granularity, each
bracketed by one frame with the point visibly live and one with both players upright and
disengaged. Five of six had been labelled 20 to 40 frames late and rally 2 by 160. That is a
convention drifting rather than a mistake: a rally ends when the shuttle lands, not when the
players finish walking.

| rally | was | now | evidence |
|---|---|---|---|
| 1 | 83-236 | **0**-198 | clip opens mid-rally, far player already mid-stroke at frame 0; play stops between 193 and 201 |
| 2 | 773-1098 | 773-**938** | live at 933, both upright by 941 |
| 3 | 1502-2767 | 1502-**2740** | shuttle visible falling 2721-2737, lands about 2740 |
| 4 | 3821-5217 | **3810**-**5175** | near player airborne at the net at 5163, turned away by 5179 |
| 5 | 6185-6439 | **6171**-**6378** | near player lunging at 6364, both upright by 6376 |
| 6 | 7024-7355 | **7013**-**7375** | still live at 7355; clip ends at 7375 and cuts the rally off |

Rally starts were separately aligned to the labelled serve contact, which the labeller had already
recorded, moving three starts 11 to 14 frames earlier.

**Why this is not self-serving, checked four ways:**

1. It was applied to all six ends uniformly, before any of them was scored.
2. It moves two ends earlier than the new detector reports and four later.
3. Aligning starts to the serve contact moves them *earlier*, against a detector whose starts
   already ran late.
4. It broke a rest-based rally-end rule that the uncorrected labels had passed, forcing a simpler
   rule. And scored on the corrected labels the **old pipeline drops from 4 of 6 to 1 of 6**,
   so the correction penalises the incumbent far more than it favours the replacement.

Two rallies turn out to be truncated by the recording. The schema now records that: a clip that
starts mid-point has a rally whose start cannot be observed, and scoring a detector against a start
that is not in the video is meaningless.

The full revision record is in `outputs/test-video2/truth.json` under `label_revisions`, with a
per-rally note giving the evidence.

---

## 6. Results

Clip 2, corrected labels, `event_source: flight`, from `outputs/test-video2/baseline.json`.

| criterion | target | result |
|---|---|---|
| both players tracked in active play | >= 90% | **92.9%** (3288 of 3541 frames) |
| hitter assignment on detected hits | >= 90% | **92.7%** (51 of 55 scored) |
| rally boundaries approximately correct | >= 85% | **6 of 6**, 0 split, 0 missed, 0 spurious |
| hit detection scorable | precision/recall | **0.824 / 0.949**, F1 0.882 |
| shuttle trajectory coherent through rallies | qualitative | 84 flight segments; each rally is one chain |

Hit detail: 56 true positives, 12 false positives, 3 false negatives, 68 predicted against 60
labelled rally shots.

Per-rally boundary errors, seconds, positive meaning late:

| rally | start error | end error |
|---|---|---|
| 1 | +0.267 | +0.483 |
| 2 | +0.067 | +0.900 |
| 3 | +0.133 | +0.017 |
| 4 | +0.200 | +0.533 |
| 5 | +0.067 | +0.817 |
| 6 | +0.033 | −0.050 |

Median start error +0.10s, median end error +0.51s, tolerance 1.0s.

**Head to head on identical labels and the same clip:**

| | upstream (`tracknet`) | flight |
|---|---|---|
| hits emitted | 86 | 68 |
| precision | 0.616 | **0.824** |
| recall | 0.898 | **0.949** |
| F1 | 0.731 | **0.882** |
| hitter accuracy | **0.941** | 0.927 |
| rally boundaries | 1 of 6 | **6 of 6** |
| spurious rallies | 1 | **0** |

Upstream still wins hitter accuracy by 1.4 points, on a smaller and differently-placed set of
hits. Both paths remain runnable; `event_source` in the clip config selects between them, so this
is a measurement that can be re-run rather than a claim.

**Chain structure per rally**, showing the segmentation matching the play:

| rally | span | contacts detected | flight segments | start |
|---|---|---|---|---|
| 1 | 16-227 | 3 | 5 | truncated |
| 2 | 777-992 | 4 | 5 | serve |
| 3 | 1510-2741 | 21 | 23 | serve |
| 4 | 3822-5207 | 27 | 27 | serve |
| 5 | 6175-6427 | 5 | 6 | serve |
| 6 | 7015-7372 | 8 | 8 | serve |

---

## 7. The annotated video fix

A separate, later fix after watching the output. The shuttle marker sat frozen mid-frame between
rallies with `LANDING` pulsing over it. Three distinct causes, none of them the shuttle.

**The marker.** The renderer drew a parked tracker like any other detection: a bright filled dot,
mid-court, apparently hovering. `analysis.json` now carries `shuttle_fixed_points` whichever
detector produced its events, and a sample sitting on one is drawn as a grey cross labelled
`tracker parked` and kept out of the trail. Still drawn, because hiding it would hide a real
failure of the tracker; no longer drawn as a shuttle.

**The pulsing.** Clips 1, 3 and 5 were still on `event_source: tracknet`, where landings arrive
every 0.8s straight through dead time: 34, 65 and 50 outside any rally, each flashed for twelve
frames. Those clips now use the flight detector.

| clip | landings before | after | outside any rally |
|---|---|---|---|
| 1 | 51 | 6 | 0 |
| 2 | 8 | 6 | 0 |
| 3 | 82 | 5 | 0 |
| 5 | 75 | 4 | 0 |

**Orphan events.** A chain rejected by the rally gates still left its events behind, because the
timeline was assembled before the gates ran. Clip 2 flashed two landings for chains it had already
decided were not rallies. Events whose rally did not survive are now pruned by
`analyze.prune_to_rallies`, which also removed five false hits and took hit precision from 0.778
to 0.824 and F1 from 0.855 to 0.882.

Verified on extracted frames from clips 2 and 3, not assumed. Through clip 3's longest dead
stretch, 376 frames read as parked and are greyed while 135 still show a live marker, because
there the tracker really is following something that moves. Those stay visible deliberately.

---

## 8. What this does NOT establish

Read this section before quoting any number above.

**Everything rests on one recording.** The flight detector's parameters were measured on clip 2,
and clip 2's labels were corrected in the same session by the same process. There is no held-out
data. The correction is defensible on four independent checks (section 5) but it is still the same
clip on both sides of the equation.

**Six rallies is a coarse yardstick.** With six labelled rallies, one rally is 17 percentage
points. The 85% target is effectively "get all six", and 6 of 6 could become 5 of 6 on a single
borderline call. Rally 2's end error is +0.90s against a 1.0s tolerance, so it is one marginal
judgement away from failing.

**Three clips run the detector on argument, not evidence.** Clips 1, 3 and 5 were switched to
`event_source: flight` because it is measurably better on clip 2. Nothing has been measured on
them. Setting `event_source: tracknet` in their configs reverts that.

**What the detector does on the unlabelled clips**, for whoever picks this up:

| clip | draft rallies | detected | note |
|---|---|---|---|
| 1 | no truth file | 6 | poor clip, regression only |
| 3 | 6 | 5 | |
| 5 | 8 | 4 | one detected rally runs 23.5s; either a long point or two merged |

Clip 5's 23.5-second rally is the single most suspicious output and only the video can settle it.
Clip 5 also runs at 33.75 fps, half of clip 2, so it is the best test of whether the
seconds-based parameters really are frame-rate independent.

**Known residual bias.** Rally ends run a median 0.51s late because the last flight segment
includes the shuttle's descent after the final shot, plus some tracker noise after the point. The
cause is understood and the effect is well inside tolerance, but it is systematic and it is the
obvious next thing to shave.

**Coverage says a slot was filled, not that it held the right person.** The 92.9% figure counts
frames where both near and far slots had somebody in them. A slot following a bystander reads as
100%. An earlier session found exactly that failure and fixed it; the mechanism is documented in
the README under the far-player section.

---

## 9. Files, commands, artefacts

**New**
- `src/badminton_poc/flight.py` — the detector, 587 lines, no numpy, no model dependency
- `tests/test_flight.py` — 24 tests, all against hand-written tracks
- `tests/test_render.py` — 5 tests for shuttle marker classification

**Changed**
- `src/badminton_poc/config.py` — `FlightConfig`; `event_source`, `require_hitter`,
  `require_both_players` on `AnalysisConfig`
- `src/badminton_poc/analyze.py` — the flight branch, `prune_to_rallies`, `shuttle_fixed_points`
- `src/badminton_poc/rallies.py` — `rallies_from_flight`
- `src/badminton_poc/assoc.py` — borrow the nearest frame holding both players
- `src/badminton_poc/render.py` — `shuttle_state`, parked-marker drawing
- `configs/*.json` — all four clips on `event_source: flight`
- `outputs/test-video2/truth.json` — corrected labels plus `label_revisions`
- `README.md` — the findings journal

**Test suite:** 227 passing, 29 of them new.

**Commits:** `e7e777f` (detector plus milestone), `b65f87e` (render fix).

**To reproduce the headline numbers:**

```bash
badminton-poc run --video data/videos/test-video2.mp4 --config configs/test-video2.json --stages analyze
badminton-poc baseline --video data/videos/test-video2.mp4 --config configs/test-video2.json
```

Flip `analysis.event_source` to `tracknet` in the config and repeat for the comparison. The
shuttle and player stages do not need re-running; their outputs are already on disk under
`outputs/<clip>/`.

Everything the detector decided is in `analysis.json` under `flight`: every segment with its span,
peak speed and median residual; every link with its gap, turn angle and kind; every chain; and
counts of what was rejected and why. A rally boundary can be traced back to the frames that
produced it without re-running anything.

---

## 10. Is milestone 1 done?

**Yes, as the criteria are written.** They specify one clean singles test clip, and all five pass
on clip 2 with margin on four of them.

**No, in the sense that matters for building on it.** One clip, corrected in the same session as
the detector was tuned, is not validation. The README has said throughout that a single clip
cannot validate anything, and that has not changed just because the numbers now pass.

**The single highest-value next action is labelling clip 3.** It is the smallest unlabelled clip
at 85 seconds, its draft holds 32 candidate contacts and 6 rally spans, and it was seeded from
upstream's detector rather than the new one, so it is an independent starting point rather than
the thing being validated agreeing with itself.

Two things matter while labelling it:

1. A rally ends when the shuttle lands, not when the players stop walking. That is exactly the
   drift that made clip 2's labels wrong, and it went unnoticed until the frames were checked.
   Rally starts sit at the serve contact.
2. Check whether the clip opens or closes mid-point, and mark the span to the clip edge if so.
   Two of clip 2's six rallies are truncated that way.

Then set `labelled: true` and run `badminton-poc baseline`. The scorer refuses to run while any
contact is still `reviewed: false`, so it cannot silently score the detector against its own
opinion.

After that, in order: shave the +0.51s systematic bias on rally ends; decide whether
`require_hitter` earns its place on a second clip; and only then revisit the non-goals list in
`CLAUDE.md`.
