"""Re-run upstream once and write the InpaintNet mask it actually used, beside its output.

Upstream inference never writes ``Inpaint_Mask``, so ``parsing.load_ball_csv`` recovers it from
TrackNet's decode grid. This is the check on that: it calls upstream's own classes, runs
TrackNet once, takes ``InferencePipeline._inpaint_mask`` of the raw trajectory, then inpaints it
the way ``events.py`` does. On 2026-10-29 the inpainted output reproduced the committed
``work_ball.csv`` of clips 2, 3, 5 and 6 exactly, and the mask agreed with the grid on all 30,902
visible samples. Re-run it if upstream or its checkpoints change.

Same settings as the pipeline (weight, batch 8, max_sample_num 300). Takes 6-16 min per clip on
a GPU. Run from the BadmintonTrackNet checkout:

    cd ../BadmintonTrackNet && PYTHONPATH=$PWD python <this file> \\
        --video ../badminton-poc-claude-ready/outputs/test-video3/work.mp4 --out mask.csv
"""
import argparse
import csv

from tracknet.inference.tracknet import _load_pipeline, build_parser

own = argparse.ArgumentParser()
own.add_argument("--video", required=True)
own.add_argument("--out", required=True)
a = own.parse_args()

args = build_parser().parse_args([
    "--video_file", a.video, "--tracknet_file", "ckpts/TrackNet_best.pt",
    "--inpaintnet_file", "ckpts/InpaintNet_best.pt", "--save_dir", "/dev/null",
    "--device", "auto", "--batch_size", "8", "--eval_mode", "weight", "--max_sample_num", "300",
])
pipeline = _load_pipeline(args)
inpaintnet, pipeline.inpaintnet = pipeline.inpaintnet, None
raw = pipeline.predict_video(a.video, eval_mode="weight", batch_size=8, max_sample_num=300)
pipeline.inpaintnet = inpaintnet
mask = pipeline._inpaint_mask(raw)
full = pipeline.inpaint_trajectory(raw, eval_mode="weight", batch_size=8)

with open(a.out, "w", newline="") as handle:
    w = csv.writer(handle)
    w.writerow(["Frame", "Visibility", "X", "Y", "Inpaint_Mask", "Raw_Visibility", "Raw_X", "Raw_Y"])
    for i, frame in enumerate(full.frame_ids):
        w.writerow([int(frame), int(full.visibility[i]), full.x[i], full.y[i], int(mask[i]),
                    int(raw.visibility[i]), raw.x[i], raw.y[i]])
