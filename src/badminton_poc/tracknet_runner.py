"""Subprocess boundary to the upstream BadmintonTrackNet repository.

We shell out rather than importing upstream code so that:

* upstream stays an unmodified checkout we can update with ``git pull``;
* if it ever needs an incompatible dependency set, pointing ``shuttle.python`` at a second
  interpreter is a one-line config change instead of a rewrite;
* the exact command we ran is recorded next to its outputs, so a result can be reproduced
  by hand.

The CLI contract below was read out of ``tracknet/inference/tracknet.py::build_parser`` and
``tracknet/inference/events.py`` in the checkout, not from documentation.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ShuttleConfig


class TrackNetError(RuntimeError):
    pass


class MissingCheckpointError(TrackNetError):
    """The upstream model weights are not present. These must be fetched by a human from the
    official source; we never download opaque pickles automatically."""


CHECKPOINT_SOURCE = (
    "https://drive.google.com/file/d/1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA/view?usp=sharing"
)


@dataclass(frozen=True)
class TrackNetOutputs:
    ball_csv: Path
    events_json: Path
    events_csv: Path | None
    overlay_video: Path | None
    command: list[str]
    duration_s: float
    stdout: str
    stderr: str


def resolve_repo(cfg: ShuttleConfig, base: Path) -> Path:
    repo = Path(cfg.repo)
    if not repo.is_absolute():
        repo = (base / repo).resolve()
    if not repo.exists():
        raise TrackNetError(
            f"BadmintonTrackNet checkout not found at {repo}.\n"
            "Clone it next to this repository:\n"
            "  git clone https://github.com/ZSHYC/BadmintonTrackNet.git"
        )
    if not (repo / "tracknet" / "inference" / "events.py").exists():
        raise TrackNetError(
            f"{repo} does not look like BadmintonTrackNet "
            "(tracknet/inference/events.py is missing)."
        )
    return repo


def checkpoint_paths(cfg: ShuttleConfig, repo: Path) -> tuple[Path, Path]:
    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (repo / path)

    return resolve(cfg.tracknet_ckpt), resolve(cfg.inpaintnet_ckpt)


def check_checkpoints(cfg: ShuttleConfig, repo: Path) -> None:
    tracknet, inpaintnet = checkpoint_paths(cfg, repo)
    missing = [p for p in (tracknet, inpaintnet) if not p.exists()]
    if missing:
        listing = "\n".join(f"  - {p}" for p in missing)
        raise MissingCheckpointError(
            "Missing TrackNetV3 checkpoint(s):\n"
            f"{listing}\n\n"
            "Download TrackNetV3_ckpts.zip from the official TrackNetV3 project:\n"
            f"  {CHECKPOINT_SOURCE}\n"
            "(linked from https://github.com/qaz812345/TrackNetV3#inference)\n"
            f"then unzip it and place TrackNet_best.pt and InpaintNet_best.pt in {repo / 'ckpts'}/\n\n"
            "These are pickled PyTorch checkpoints; only load them from that official source."
        )


def run_shuttle_inference(
    video: str | Path,
    save_dir: str | Path,
    cfg: ShuttleConfig,
    *,
    base: Path,
    log_path: str | Path | None = None,
) -> TrackNetOutputs:
    repo = resolve_repo(cfg, base)
    check_checkpoints(cfg, repo)
    tracknet_ckpt, inpaintnet_ckpt = checkpoint_paths(cfg, repo)

    video = Path(video).resolve()
    save_dir = Path(save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    interpreter = cfg.python or sys.executable
    if shutil.which(interpreter) is None and not Path(interpreter).exists():
        raise TrackNetError(f"configured shuttle interpreter does not exist: {interpreter}")

    command = [
        interpreter, "-m", "tracknet.inference.events",
        "--video_file", str(video),
        "--tracknet_file", str(tracknet_ckpt),
        "--inpaintnet_file", str(inpaintnet_ckpt),
        "--save_dir", str(save_dir),
        "--device", cfg.device,
        "--batch_size", str(cfg.batch_size),
        "--eval_mode", cfg.eval_mode,
        "--max_sample_num", str(cfg.max_sample_num),
    ]
    if cfg.output_video:
        command.append("--output_video")

    estimate = estimate_median_memory_bytes(cfg.max_sample_num, video)
    if estimate > _available_memory_bytes() * 0.7:
        raise TrackNetError(
            "upstream would need about "
            f"{estimate / 1e9:.1f} GB to build its background median from "
            f"{cfg.max_sample_num} frames of this video, but only "
            f"{_available_memory_bytes() / 1e9:.1f} GB is available. It holds every sample in "
            "memory at full resolution, so it would be killed by the OOM killer with an empty "
            "stderr. Lower shuttle.max_sample_num in the config."
        )

    started = time.time()
    process = subprocess.run(
        command, cwd=str(repo), capture_output=True, text=True,
        env={**_inherited_env(), "PYTHONPATH": str(repo)},
    )
    duration = time.time() - started

    if log_path is not None:
        log = Path(log_path)
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(
            json.dumps(
                {
                    "command": command,
                    "cwd": str(repo),
                    "returncode": process.returncode,
                    "duration_s": duration,
                    "stdout": process.stdout,
                    "stderr": process.stderr,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    if process.returncode != 0:
        raise TrackNetError(
            f"upstream shuttle inference failed (exit {process.returncode}).\n"
            f"command: {' '.join(command)}\n"
            f"cwd: {repo}\n"
            f"--- stderr ---\n{process.stderr.strip()[-4000:]}"
        )

    stem = video.stem
    ball_csv = save_dir / f"{stem}_ball.csv"
    events_json = save_dir / f"{stem}_events.json"
    if not ball_csv.exists() or not events_json.exists():
        produced = sorted(p.name for p in save_dir.iterdir())
        raise TrackNetError(
            "upstream reported success but did not write the expected artefacts "
            f"({ball_csv.name}, {events_json.name}). Directory contains: {produced}"
        )
    events_csv = save_dir / f"{stem}_events.csv"
    overlay = save_dir / f"{stem}.mp4"
    return TrackNetOutputs(
        ball_csv=ball_csv,
        events_json=events_json,
        events_csv=events_csv if events_csv.exists() else None,
        overlay_video=overlay if overlay.exists() else None,
        command=command,
        duration_s=duration,
        stdout=process.stdout,
        stderr=process.stderr,
    )


def estimate_median_memory_bytes(max_sample_num: int, video: Path) -> int:
    """Peak memory upstream needs for its background median.

    It keeps ``max_sample_num`` decoded frames in a list and then medians them, and the median
    partitions a copy, so budget roughly twice the raw sample volume.
    """
    import cv2

    capture = cv2.VideoCapture(str(video))
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    samples = min(max_sample_num, max(frames, 1))
    return 2 * samples * width * height * 3


def _available_memory_bytes() -> int:
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 8 * 1024**3  # conservative guess when /proc is unavailable


def _inherited_env() -> dict[str, str]:
    import os

    env = dict(os.environ)
    env.pop("PYTHONHOME", None)
    return env
