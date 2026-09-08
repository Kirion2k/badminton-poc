# Badminton POC project instructions

## Goal
Build and validate a local technical proof of concept that converts one fixed-camera singles badminton video into a machine-readable timeline of players, shuttle events, and rallies.

This is NOT the production app. Do not build a frontend, auth, cloud infrastructure, an LLM coach, doubles support, or broad shot classification until the base vision pipeline is validated on a real clip.

## Current intended pipeline
1. Input: 1-5 minute fixed-camera singles MP4, full court visible.
2. Shuttle: integrate BadmintonTrackNet / TrackNetV3 to output per-frame shuttle coordinates and hit/landing/out-of-frame candidates.
3. Players: use an Ultralytics pose model to track the two court players and retain boxes, IDs, and keypoints.
4. Merge: associate hit events with near/far player and group events into rallies.
5. Outputs: raw machine-readable files, analysis.json, and annotated.mp4.
6. Evaluation: manually compare a small real clip to ground truth and report metrics/failure modes.

## Engineering principles
- Inspect existing code before rewriting it.
- Prefer simple, inspectable heuristics for the first POC.
- Keep model inference separate from analysis logic.
- Never hide failures. Persist intermediate outputs and confidence/debug information.
- Add small tests for parsing and event/rally logic.
- Do not assume upstream CLI/file formats. Inspect the current BadmintonTrackNet repository and adapt our integration to its actual outputs.
- Use Python 3.11 unless a dependency requires otherwise.
- Detect CUDA availability and use it when available. Keep CPU fallback possible.
- Do not commit model weights or user videos.

## POC acceptance criteria
For one clean singles test clip:
- both players tracked on >= 90% of active-play frames,
- shuttle trajectory is coherent through most rallies,
- hit detection can be manually scored for precision/recall,
- hitter assignment target >= 90% on clean detected hits,
- rally boundary target >= 85% approximately correct.

If a threshold is missed, diagnose the failure before adding new features.

## Non-goals until milestone 1 passes
- smash/drop/clear classification
- winner/error inference
- tactical LLM coaching
- scoreboard OCR
- doubles
- mobile/web UI
- cloud deployment
