from __future__ import annotations

import argparse
import json



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Badminton analysis technical POC")
    sub = parser.add_subparsers(dest="command", required=True)

    p_players = sub.add_parser("players", help="Run generic player pose tracking")
    p_players.add_argument("--video", required=True)
    p_players.add_argument("--output", required=True)
    p_players.add_argument("--model", default="yolo26n-pose.pt")
    p_players.add_argument("--roi", default=None, help="Normalized x1,y1,x2,y2 court ROI")
    p_players.add_argument("--conf", type=float, default=0.25)

    p_analyze = sub.add_parser("analyze", help="Merge player tracks with TrackNet shuttle/events")
    p_analyze.add_argument("--players", required=True)
    p_analyze.add_argument("--ball", required=True)
    p_analyze.add_argument("--events", required=True)
    p_analyze.add_argument("--fps", type=float, required=True)
    p_analyze.add_argument("--output", required=True)

    p_render = sub.add_parser("render", help="Render combined player + shuttle + event overlay")
    p_render.add_argument("--video", required=True)
    p_render.add_argument("--players", required=True)
    p_render.add_argument("--ball", required=True)
    p_render.add_argument("--analysis", required=True)
    p_render.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "players":
        from .players import extract_players
        extract_players(args.video, args.output, model_name=args.model, roi=args.roi, conf=args.conf)
    elif args.command == "analyze":
        from .analyze import analyze
        result = analyze(args.players, args.ball, args.events, args.output, args.fps)
        print(json.dumps(result["summary"], indent=2))
    elif args.command == "render":
        from .render import render
        render(args.video, args.players, args.ball, args.analysis, args.output)


if __name__ == "__main__":
    main()
