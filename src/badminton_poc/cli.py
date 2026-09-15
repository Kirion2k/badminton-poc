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

    p_truth = sub.add_parser(
        "truth", help="create or migrate the detector-independent truth file for a clip")
    p_truth.add_argument("--video", required=True, help="source video, used to locate outputs/<stem>/")
    p_truth.add_argument("--migrate", action="store_true",
                         help="seed it from an existing labelled ground_truth.json")
    p_truth.add_argument("--draft", action="store_true",
                         help="seed contacts and rallies from the current run, all unreviewed")
    common(p_truth)

    p_baseline = sub.add_parser(
        "baseline", help="score the current run against truth.json and save the report")
    p_baseline.add_argument("--video", required=True, help="source video, used to locate outputs/<stem>/")
    p_baseline.add_argument("--match-tolerance", type=float, default=0.25,
                            help="seconds either side of a labelled contact that a prediction may match")
    p_baseline.add_argument("--rally-tolerance", type=float, default=1.0,
                            help="seconds either side of a labelled rally edge counted as recovered")
    common(p_baseline)

    p_refresh = sub.add_parser(
        "refresh-labels",
        help="re-seed ground_truth.json against the current run, keeping judgements already made")
    p_refresh.add_argument("--video", required=True, help="source video, used to locate outputs/<stem>/")
    common(p_refresh)

    p_shots = sub.add_parser(
        "shots", help="Milestone 2: build shot samples, classify them, label and score")
    p_shots.add_argument("--video", required=True,
                         help="source video, used to locate outputs/<stem>/")
    p_shots.add_argument(
        "--stages", default="build,predict,seed,score,report",
        help="comma-separated: build, predict, seed, score, report, previews")
    p_shots.add_argument("--model", default="bst",
                         help="which model's errors to report on (bst or rules)")
    p_shots.add_argument("--target-hz", type=float, default=None,
                         help="resample each window to this rate before the model sees it")
    p_shots.add_argument("--no-bst", action="store_true",
                         help="skip the pretrained model and run only the rule baseline")
    common(p_shots)

    p_report = sub.add_parser(
        "report", help="render analysis.json as a readable summary page")
    p_report.add_argument("--video", required=True, help="source video, used to locate outputs/<stem>/")
    p_report.add_argument("--html", help="write the page here (default outputs/<stem>/report.html)")
    p_report.add_argument("--title", help="page title (default '<stem> Rally Report')")
    common(p_report)

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
        from .videoprep import effective_fps, probe

        # The standalone stage has no metadata.json to read the working frame rate out of, and
        # mobility_yield_seconds is in seconds, so ask the video itself rather than assume.
        fps = config.target_fps or effective_fps(probe(args.video))
        print(json.dumps(
            extract_players(args.video, args.output, config.court, config.players, fps=fps),
            indent=2,
        ))
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
                         flight_cfg=config.flight,
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

    if args.command == "truth":
        from .truth import empty_truth, migrate_legacy_ground_truth

        paths = Paths(Path(args.output_root), Path(args.video).stem)
        clip = Path(args.video).stem
        if paths.truth.exists():
            print(f"{paths.truth} already exists; not overwriting labelling work", file=sys.stderr)
            return 2
        if args.migrate:
            legacy = json.loads(paths.ground_truth.read_text(encoding="utf-8"))
            truth = migrate_legacy_ground_truth(legacy, clip=clip)
            note = (f"migrated {truth['migrated_from']['contacts_kept']} contacts, discarded "
                    f"{truth['migrated_from']['discarded_candidate_verdicts']} candidate verdicts")
        elif args.draft:
            from .truth import draft_truth_from_analysis

            analysis = json.loads(paths.analysis.read_text(encoding="utf-8"))
            truth = draft_truth_from_analysis(analysis, clip=clip)
            note = (f"drafted {len(truth['contacts'])} contacts and {len(truth['rallies'])} "
                    "rallies from the current run, all unreviewed - review every one, add what "
                    "was missed, then set reviewed=true and labelled=true")
        else:
            analysis = json.loads(paths.analysis.read_text(encoding="utf-8"))
            truth = empty_truth(clip, float(analysis.get("config", {}).get("fps") or 0.0))
            note = "empty; label contacts and rallies by watching annotated.mp4"
        paths.truth.write_text(json.dumps(truth, indent=2), encoding="utf-8")
        print(f"{paths.truth}\n  {note}")
        return 0

    if args.command == "shots":
        from .shots.features import FeatureConfig
        from .shots.runner import (
            confusion_image, error_report, load_clip, make_previews, run_predictions,
            score_clip, seed_truth, write_predictions, write_samples,
        )
        from .shots.truth import NotLabelledError

        stages = tuple(s.strip() for s in args.stages.split(",") if s.strip())
        clip = Path(args.video).stem
        feature_cfg = FeatureConfig(target_hz=args.target_hz) if args.target_hz else None
        clip_shots = load_clip(clip, output_root=Path(args.output_root), config=config,
                               feature_cfg=feature_cfg)
        print(f"{len(clip_shots.samples)} shot samples, {len(clip_shots.skipped)} hits skipped")
        if "build" in stages:
            print("samples:", write_samples(clip_shots))
        if "predict" in stages:
            records = run_predictions(clip_shots, include_bst=not args.no_bst)
            print("predictions:", write_predictions(clip_shots, records))
        if "seed" in stages:
            print("ground truth:", seed_truth(clip_shots))
        if "previews" in stages:
            made = make_previews(clip_shots)
            print(f"previews: {len(made)} clips in {clip_shots.paths.previews}")
        if "score" in stages:
            try:
                result = score_clip(clip_shots)
            except NotLabelledError as exc:
                print(f"cannot score yet: {exc}", file=sys.stderr)
                return 2
            for name, scores in result["models"].items():
                print(f"  {name}: accuracy {scores['accuracy']} macro F1 {scores['macro_f1']} "
                      f"top2 {scores['top2_accuracy']}")
            print("evaluation:", clip_shots.paths.evaluation)
        if "report" in stages and clip_shots.paths.evaluation.exists():
            print("errors:", error_report(clip_shots, args.model))
            print("confusion:", confusion_image(clip_shots, args.model))
        return 0

    if args.command == "baseline":
        from .parsing import load_jsonl
        from .truth import evaluate_against_truth

        paths = Paths(Path(args.output_root), Path(args.video).stem)
        analysis = json.loads(paths.analysis.read_text(encoding="utf-8"))
        truth = json.loads(paths.truth.read_text(encoding="utf-8"))
        players = load_jsonl(paths.players)
        try:
            report = evaluate_against_truth(
                analysis, truth, players,
                match_tolerance_seconds=args.match_tolerance,
                rally_tolerance_seconds=args.rally_tolerance)
        except ValueError as exc:
            print(f"cannot produce a baseline: {exc}", file=sys.stderr)
            return 2
        paths.baseline.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "refresh-labels":
        from .evaluation import refresh_template

        paths = Paths(Path(args.output_root), Path(args.video).stem)
        analysis = json.loads(paths.analysis.read_text(encoding="utf-8"))
        existing = json.loads(paths.ground_truth.read_text(encoding="utf-8"))
        fps = float(analysis.get("config", {}).get("fps") or existing.get("fps") or 0.0)
        merged = refresh_template(existing, analysis, fps)
        backup = paths.ground_truth.with_suffix(".json.bak")
        backup.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        paths.ground_truth.write_text(json.dumps(merged, indent=2), encoding="utf-8")
        info = merged["refresh"]
        print(f"{paths.ground_truth}  (previous copy at {backup})")
        print(f"  carried over {info['carried_over']} judgements, "
              f"{info['new_to_judge']} new hits to judge, "
              f"{info['dropped_no_longer_detected']} dropped")
        return 0

    if args.command == "report":
        from .report import build_report, render_html

        paths = Paths(Path(args.output_root), Path(args.video).stem)
        analysis = json.loads(paths.analysis.read_text(encoding="utf-8"))
        report = build_report(analysis)
        destination = Path(args.html) if args.html else paths.root / "report.html"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(render_html(report, title=args.title), encoding="utf-8")
        (paths.root / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
        warn = [c for c in report["caveats"] if c["level"] == "warn"]
        print(f"{destination}")
        print(f"  {report['headline']['rallies'].get('detected', 0)} rallies, "
              f"{report['headline']['hits'].get('detected', 0)} hits, "
              f"{len(warn)} caveats to read before quoting any of it")
        return 0

    raise SystemExit(f"unhandled command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
