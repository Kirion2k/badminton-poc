# Badminton POC: session handoff, 16 September 2026

For someone (human or model) picking this up with no other context.

## The project in two lines

A local proof of concept that turns one fixed-camera singles badminton video into a timeline of
players, shuttle, contacts and rallies (Milestone 1), then classifies each stroke with the
pretrained BST model (Milestone 2, exploratory).

## Where it stood at the start of the session

- Milestone 1 frozen, on three clips (test-video2, 3, 5).
- Milestone 2: pretrained BST scored **0.7273 accuracy** on 44 hand-labelled shots from **two**
  recordings (clips 3 and 5). Clip 2's 58 contacts had no stroke labels yet.

## What happened, in order

1. **The project owner labelled clip 2's 58 strokes.** Milestone 2 went from two recordings to
   three (106 shots built, 95 scored after excluding serves and `uncertain`).

2. **BST accuracy fell from 0.7273 to 0.5895** (macro F1 0.6569 → 0.5395). By clip: clip 5
   0.750, clip 3 0.625, **clip 2 0.4706**. Clear (1.000 → 0.750) and smash (0.941 → 0.692), the
   classes that carried the old headline, fell furthest. BST still beats a hand-written geometry
   baseline by 23 points.

3. **Clip 2's labels turned out to be weak.** Its hitter labels matched the old detector's output
   on 60 of 61 contacts, which means someone confirmed the detector's answer rather than labelling
   independently. Clips 3 and 5 had already been relabelled blind; clip 2 had not.

4. **A new check was added: impossible alternations.** In singles the two players must alternate,
   so two consecutive contacts credited to the same player cannot both be right. It uses no model.
   Clips 3 and 5 score 0; clip 2 scored 13. The check now runs in every evaluation and is recorded
   in the Milestone 1 freeze manifest (now `milestone-1/1.1`).

5. **Clip 2's labels were repaired in two rounds.**
   - The owner flagged four wrong hitters while labelling strokes (870, 2689, 4510, 5060). Applied.
     13 → 6 violations.
   - Four more were resolved by Claude from the shuttle track, pose keypoints and video frames:
     1877 changed to far, and missing near contacts added at 4017 and 4316. 6 → 2 violations.
     **These are model-derived**, marked `hitter_source=model-derived-2026-09-16`, and given no
     stroke label. Clip 2 is now a mixed human/model labelling.
   - The last two (7227, 7248) were left alone. The shuttle tracker is parked on a mark on the back
     wall at both frames, an undetected second fixed point, so they are probably not real contacts,
     but that was not treated as proof.

6. **The repairs did not change BST's score at all.** Clip 2 stayed at 0.4706. So its low score is
   not caused by its labels. Also ruled out: missing inputs (clip 2's pose coverage is the best of
   the three), class mix (reweighting moves it 2 points), and the class-mapping layer (using BST's
   raw top-1 class gains only 2 points on clip 2).

7. **Likely cause: domain shift.** Clips 3 and 5 are professional tournament broadcasts, the same
   kind of footage BST was trained on (ShuttleSet). Clip 2 is a club hall with a lower, closer
   camera. The three scores are ordered by how much each clip resembles the training data. This is
   a hypothesis resting on one club clip.

8. **BST's confidence is useful on club footage.** It is uncertain there rather than confidently
   wrong (median confidence 0.57 vs 0.84 on clip 5). Answering only when confidence ≥ 0.7 gives
   0.85 accuracy on 25% of clip 2's shots; ≥ 0.8 gives 0.90 on 20%. The threshold was chosen after
   seeing the data, on 13 shots, so it is provisional and not yet in the code.

9. **A search for a second club-hall clip failed.** Four YouTube candidates screened, none usable:
   one camera operated by hand, one low diagonal view, one doubles, one stable for only 64 seconds.
   Eleven public clips screened in total across sessions; only tournament broadcasts pass.

10. **Smaller fixes:** `uncertain` stroke labels are no longer scored as a class; the owner's
    `shot_labels.tsv` is now tracked in git (it was being ignored); a tool
    `tools/preview_alternations.py` cuts review clips for alternation problems.

## Current numbers (Milestone 1, frozen manifest)

| clip | hit precision / recall | hitter accuracy | rallies found | spurious rallies | both players tracked |
|---|---|---|---|---|---|
| test-video2 | 0.902 / 0.902 | 0.981 | 6/6 | 0 | 0.929 |
| test-video3 | 0.667 / 0.800 | 0.875 | 2/3 | 1 | 0.979 |
| test-video5 | 0.872 / 0.895 | 0.900 | 5/5 | 2 | 0.904 |

## Decisions and recommendations

- **Keep BST for now; do not swap models yet.** The main public labelled stroke data is
  professional broadcast footage, so another pretrained model would likely hit the same wall. What
  fixes domain shift is club-footage training data, and there are only 106 labelled shots in total.
- **Next step: the owner records a club singles match to spec** (fixed tripod, raised behind one
  baseline, whole court, singles, 1080p+, 5-10 minutes of rallies). Scoring BST on it tests the
  domain hypothesis: around 0.47 confirms it, around 0.7 means clip 2 was just a hard clip.
- **Then:** consider BST with a confidence cut-off, validated on the new clip, and keep
  accumulating club-footage labels until fine-tuning is realistic (a few hundred shots).

## Open problems

1. No second club-hall recording (blocks the domain question).
2. Two unresolved alternations on clip 2 (7227, 7248), plus a second parked-shuttle point the
   fixed-point detector misses.
3. Three spurious rallies (the play gate problem, unsolved after seven attempts).
4. Clip 3's hit precision, 0.667.
5. Drive is 0 of 4 in every configuration.

## Where to read more

- `docs/milestone-2-phases-b-to-j.md`, Phase K and its addendum: full Milestone 2 results.
- `configs/milestone-1-frozen.json`: frozen Milestone 1 metrics, including impossible alternations.
- `README.md`, "A second search, for club-hall footage": the screening results.
- Commits `45f0779` through `b50f022`.
