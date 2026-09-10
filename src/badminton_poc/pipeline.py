"""End-to-end orchestration: video -> players + shuttle -> analysis -> annotated video."""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from . import videoprep
from .analyze import analyze
from .config import PipelineConfig
from .evaluation import write_template
from .players import extract_players
from .render import render
from .tracknet_runner import run_shuttle_inference

STAGES = ("preprocess", "players", "shuttle", "analyze", "render")


class Paths:
    def __init__(self, output_root: Path, stem: str) -> None:
        self.root = output_root / stem
        self.work_video = self.root / "work.mp4"
        self.metadata = self.root / "metadata.json"
        self.players = self.root / "players.jsonl"
        self.tracknet_dir = self.root / "tracknet"
        self.analysis = self.root / "analysis.json"
        self.annotated = self.root / "annotated.mp4"
        self.ground_truth = self.root / "ground_truth.json"
        self.truth = self.root / "truth.json"
        self.baseline = self.root / "baseline.json"
        self.logs = self.root / "logs"
        self.court_preview = self.root / "court_roi.png"

    def mkdirs(self) -> None:
        for directory in (self.root, self.tracknet_dir, self.logs):
            directory.mkdir(parents=True, exist_ok=True)


def run(
    video: str | Path,
    config: PipelineConfig,
    *,
    output_root: str | Path = "outputs",
    base: Path,
    stages: tuple[str, ...] = STAGES,
    start_frame: int = 0,
    end_frame: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    source = Path(video).resolve()
    paths = Paths(Path(output_root), source.stem)
    paths.mkdirs()
    report: dict[str, Any] = {"stages": {}, "output_dir": str(paths.root)}

    metadata = _load_json(paths.metadata) or {}

    if "preprocess" in stages:
        started = time.time()
        info = videoprep.probe(source)
        working = videoprep.normalize(
            source,
            paths.work_video,
            info=info,
            target_width=config.target_width,
            target_fps=config.target_fps,
            start_frame=start_frame,
            end_frame=end_frame,
            overwrite=force,
        )
        metadata = {
            "source": info.to_dict(),
            "working": working,
            "source_frame_times_s": videoprep.source_frame_times(info),
            "config": config.to_dict(),
            "warnings": _warnings(info),
        }
        paths.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        _preview_court(paths, config)
        report["stages"]["preprocess"] = {**working, "seconds": round(time.time() - started, 1)}

    working = metadata.get("working")
    if not working:
        raise RuntimeError(
            f"{paths.metadata} has no working-video record. Run the preprocess stage first."
        )
    fps = float(working["fps"])
    width, height = int(working["width"]), int(working["height"])

    if "players" in stages:
        started = time.time()
        stats = extract_players(paths.work_video, paths.players, config.court, config.players)
        stats["seconds"] = round(time.time() - started, 1)
        report["stages"]["players"] = stats

    if "shuttle" in stages:
        outputs = run_shuttle_inference(
            paths.work_video,
            paths.tracknet_dir,
            config.shuttle,
            base=base,
            log_path=paths.logs / "tracknet.json",
        )
        report["stages"]["shuttle"] = {
            "ball_csv": str(outputs.ball_csv),
            "events_json": str(outputs.events_json),
            "events_csv": str(outputs.events_csv) if outputs.events_csv else None,
            "seconds": round(outputs.duration_s, 1),
            "command": outputs.command,
        }

    ball_csv = paths.tracknet_dir / f"{paths.work_video.stem}_ball.csv"
    events_json = paths.tracknet_dir / f"{paths.work_video.stem}_events.json"

    if "analyze" in stages:
        started = time.time()
        result = analyze(
            paths.players, ball_csv, events_json, paths.analysis,
            fps=fps, court=config.court, cfg=config.analysis, flight_cfg=config.flight,
            width=width, height=height,
            metadata={
                "source_video": str(source),
                "working_video": str(paths.work_video),
                "source_frame_offset": working.get("source_frame_offset", 0),
                "frame_count": working.get("frame_count"),
            },
        )
        write_template(paths.ground_truth, result, fps)
        report["stages"]["analyze"] = {
            "output": str(paths.analysis),
            "summary": result["summary"],
            "seconds": round(time.time() - started, 1),
        }

    if "render" in stages:
        started = time.time()
        stats = render(
            paths.work_video, paths.players, ball_csv, paths.analysis, paths.annotated,
            court=config.court,
        )
        stats["seconds"] = round(time.time() - started, 1)
        report["stages"]["render"] = stats

    (paths.logs / "run.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _preview_court(paths: Paths, config: PipelineConfig) -> None:
    from .calibrate import render_calibration

    render_calibration(paths.work_video, paths.court_preview, frame_index=0,
                       court=config.court, grid=False)


def _warnings(info) -> list[str]:
    warnings: list[str] = []
    if getattr(info, "count_warning", None):
        warnings.append(info.count_warning)
    if info.rotation:
        warnings.append(
            f"source carries {info.rotation} degrees of display rotation; baked into the working "
            "video so OpenCV and ffmpeg cannot disagree about orientation"
        )
    if info.is_vfr:
        warnings.append(
            f"source is variable-frame-rate (r_frame_rate reports {info.r_frame_rate:.2f} fps but "
            f"{info.frame_count} frames span {info.duration_s:.2f}s = "
            f"{videoprep.effective_fps(info):.2f} fps); the working video is resampled onto a "
            "uniform grid and true source timestamps are kept in source_frame_times_s"
        )
    return warnings


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
