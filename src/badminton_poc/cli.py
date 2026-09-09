"""Command-line interface.

``badminton-poc run --video data/input/test01.mp4`` is the one command that does everything;
the individual stages are exposed separately because during a POC you re-run one piece far more
often than the whole thing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import PipelineConfig
from .pipeline import STAGES, Paths, run

REPO_BASE = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="badminton-poc",
        description="Badminton video-analysis technical POC",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", default=None, help="pipeline config JSON (see configs/)")
        p.add_argument("--output-root", default="outputs")

    p_run = sub.add_parser("run", help="run the whole pipeline")
    p_run.add_argument("--video", required=True)
    common(p_run)
    p_run.add_argument("--stages", default=",".join(STAGES),
                       help=f"comma-separated subset of: {','.join(STAGES)}")
    p_run.add_argument("--start-frame", type=int, default=0)
    p_run.add_argument("--end-frame", type=int, default=None,
                       help="exclusive end frame; use to drop unusable footage at the end")
    p_run.add_argument("--force", action="store_true", help="re-encode the working video")

    p_probe = sub.add_parser("probe", help="report source video properties and problems")
    p_probe.add_argument("--video", required=True)

    p_cal = sub.add_parser("calibrate", help="render a frame with the court ROI and a coordinate grid")
    p_cal.add_argument("--video", required=True)
    p_cal.add_argument("--output", required=True)
    p_cal.add_argument("--frame", type=int, default=0)
    p_cal.add_argument("--config", default=None)
    p_cal.add_argument("--no-grid", action="store_true")

    p_players = sub.add_parser("players", help="pose-track the two court players")
    p_players.add_argument("--video", required=True)
    p_players.add_argument("--output", required=True)
    p_players.add_argument("--config", default=None)

    p_shuttle = sub.add_parser("shuttle", help="run upstream TrackNetV3 shuttle + event inference")
    p_shuttle.add_argument("--video", required=True)
    p_shuttle.add_argument("--save-dir", required=True)
    p_shuttle.add_argument("--config", default=None)

    p_analyze = sub.add_parser("analyze", help="merge players + shuttle into analysis.json")
    p_analyze.add_argument("--players", required=True)
    p_analyze.add_argument("--ball", required=True)
    p_analyze.add_argument("--events", required=True)
    p_analyze.add_argument("--output", required=True)
    p_analyze.add_argument("--fps", type=float, required=True)
    p_analyze.add_argument("--width", type=int, required=True)
    p_analyze.add_argument("--height", type=int, required=True)
    p_analyze.add_argument("--config", default=None)

    p_render = sub.add_parser("render", help="render the annotated diagnostic video")
    p_render.add_argument("--video", required=True)
    p_render.add_argument("--players", required=True)
    p_render.add_argument("--ball", required=True)
    p_render.add_argument("--analysis", required=True)
    p_render.add_argument("--output", required=True)
    p_render.add_argument("--config", default=None)

    p_eval = sub.add_parser("evaluate", help="score analysis.json against a labelled ground_truth.json")
    p_eval.add_argument("--video", required=True, help="source video, used to locate outputs/<stem>/")
    common(p_eval)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = PipelineConfig.load(getattr(args, "config", None))

    if args.command == "run":
        stages = tuple(s.strip() for s in args.stages.split(",") if s.strip())
        unknown = [s for s in stages if s not in STAGES]
        if unknown:
            raise SystemExit(f"unknown stage(s) {unknown}; valid stages are {list(STAGES)}")
        report = run(
            args.video, config, output_root=args.output_root, base=REPO_BASE,
            stages=stages, start_frame=args.start_frame, end_frame=args.end_frame,
            force=args.force,
        )
        print(json.dumps(report, indent=2, default=str))
        return 0

    if args.command == "probe":
        from . import videoprep

        info = videoprep.probe(args.video)
        payload = info.to_dict()
        payload["effective_fps"] = round(videoprep.effective_fps(info), 4)
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "calibrate":
        from .calibrate import render_calibration

        path = render_calibration(args.video, args.output, frame_index=args.frame,
                                  court=config.court, grid=not args.no_grid)
        print(path)
        return 0

    if args.command == "players":
        from .players import extract_players

        print(json.dumps(extract_players(args.video, args.output, config.court, config.players), indent=2))
        return 0

    if args.command == "shuttle":
        from .tracknet_runner import run_shuttle_inference

        outputs = run_shuttle_inference(args.video, args.save_dir, config.shuttle, base=REPO_BASE)
        print(json.dumps({"ball_csv": str(outputs.ball_csv),
                          "events_json": str(outputs.events_json),
                          "seconds": round(outputs.duration_s, 1)}, indent=2))
        return 0

    if args.command == "analyze":
        from .analyze import analyze

        result = analyze(args.players, args.ball, args.events, args.output,
                         fps=args.fps, court=config.court, cfg=config.analysis,
                         width=args.width, height=args.height)
        print(json.dumps(result["summary"], indent=2))
        return 0

    if args.command == "render":
        from .render import render as render_video

        print(json.dumps(render_video(args.video, args.players, args.ball, args.analysis,
                                      args.output, court=config.court), indent=2))
        return 0

    if args.command == "evaluate":
        from .evaluation import NotLabelledError, evaluate
        from .parsing import load_jsonl

        paths = Paths(Path(args.output_root), Path(args.video).stem)
        analysis = json.loads(paths.analysis.read_text(encoding="utf-8"))
        ground_truth = json.loads(paths.ground_truth.read_text(encoding="utf-8"))
        players = load_jsonl(paths.players)
        try:
            result = evaluate(analysis, ground_truth, players)
        except NotLabelledError as exc:
            print(f"cannot evaluate: {exc}", file=sys.stderr)
            return 2
        out = paths.root / "evaluation.json"
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 0

    raise SystemExit(f"unhandled command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
