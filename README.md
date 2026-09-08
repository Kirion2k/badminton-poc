# Badminton video-analysis technical POC

This is intentionally **not** the app. It is a local experiment to answer one question:

> Can a fixed phone recording of a singles badminton match be converted into a usable machine-readable rally timeline with player tracks and shuttle events?

## POC output

Given one fixed-camera singles MP4, the experiment should produce:

- TrackNet shuttle coordinates (`*_ball.csv`)
- TrackNet hit/landing candidates (`*_events.json`)
- near/far player boxes + pose keypoints (`players.jsonl`)
- merged rally/event timeline (`analysis.json`)
- combined annotated video (`annotated.mp4`)

The current code does **not** classify smash/drop/clear yet. That is deliberately milestone 2.

## Recording conditions

For the first test, use a simple input:

- singles
- landscape
- fixed phone/tripod
- whole court visible
- camera behind one baseline if possible
- 1080p preferred
- 60 fps preferred, 30 fps is still worth testing
- 1-5 minutes is enough initially

## 1. Create the environment

Python 3.11 is recommended.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

## 2. Install BadmintonTrackNet

Clone the MIT-licensed BadmintonTrackNet repository next to this repo and install it in the same virtual environment:

```bash
cd ..
git clone https://github.com/ZSHYC/BadmintonTrackNet.git
cd BadmintonTrackNet
pip install -e .
```

Download the official TrackNetV3 checkpoints referenced by that repository and place these files in `BadmintonTrackNet/ckpts/`:

- `TrackNet_best.pt`
- `InpaintNet_best.pt`

Only load model checkpoints from a source you trust.

## 3. Run shuttle + event detection

From the BadmintonTrackNet repo:

```bash
python -m tracknet.inference.events \
  --video_file /absolute/path/to/match.mp4 \
  --tracknet_file ckpts/TrackNet_best.pt \
  --inpaintnet_file ckpts/InpaintNet_best.pt \
  --save_dir /absolute/path/to/badminton-poc/outputs/tracknet \
  --output_video
```

The upstream tool should create a `*_ball.csv`, `*_events.json`, and an event-overlay video.

## 4. Track the two players

Back in this repo:

```bash
badminton-poc players \
  --video /absolute/path/to/match.mp4 \
  --output outputs/players.jsonl
```

The first run downloads the Ultralytics pose model automatically.

If spectators or people on nearby courts get detected, restrict inference to a rough normalized court rectangle:

```bash
badminton-poc players \
  --video /absolute/path/to/match.mp4 \
  --output outputs/players.jsonl \
  --roi 0.15,0.05,0.85,0.98
```

The ROI values are `x1,y1,x2,y2`, each between 0 and 1.

## 5. Merge the signals

Find the video's FPS with ffprobe, VLC, or the video metadata. Then run:

```bash
badminton-poc analyze \
  --players outputs/players.jsonl \
  --ball outputs/tracknet/YOUR_VIDEO_ball.csv \
  --events outputs/tracknet/YOUR_VIDEO_events.json \
  --fps 60 \
  --output outputs/analysis.json
```

This assigns each detected hit to the near or far player using shuttle-to-wrist proximity, then groups hits and terminal events into rallies.

## 6. Produce one combined diagnostic video

```bash
badminton-poc render \
  --video /absolute/path/to/match.mp4 \
  --players outputs/players.jsonl \
  --ball outputs/tracknet/YOUR_VIDEO_ball.csv \
  --analysis outputs/analysis.json \
  --output outputs/annotated.mp4
```

## What we measure

For the first clip, manually inspect the output and record:

1. **Two-player detection coverage**
   - target: >= 90% of active-play frames
2. **Shuttle tracking coverage**
   - target: trajectory usable through most rallies, not necessarily every frame
3. **Hit detection precision/recall**
   - manually count true hits vs detected hits
4. **Hitter assignment accuracy**
   - target: >= 90% on clean hits is a good sign
5. **Rally segmentation accuracy**
   - target: >= 85% of rallies have approximately correct boundaries

The JSON already reports two-player frame coverage and shuttle-visible frame coverage. The other metrics should be manually labelled for this first POC rather than pretending we have ground truth.

## Decision rule

**Continue** if the shuttle path is mostly coherent, the two main players are stable, and most rallies/hits can be recovered with light heuristics.

**Do not abandon the idea** just because a generic pretrained model misses difficult frames. If failures are concentrated around lighting, occlusion, small far-side players, or unusual angles, those are exactly the cases that custom data/fine-tuning can improve.

**Re-think the recording requirements** if the same pipeline repeatedly fails even on clean, fixed, full-court footage.

## Next milestone

Only after this works do we add:

- manual court homography
- court-space player movement
- landing positions
- broad shot classes: serve, clear/lift, drop, smash, drive, net
- winner/error inference
