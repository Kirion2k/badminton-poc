"""Human-readable summary of an analysis.json.

``analysis.json`` is the record; this is the view of it. It derives nothing the pipeline did
not already measure except shot tempo, which is just the gaps between hit timestamps.

The caveats are the point of this module. A rally count and a shot count read as facts, and on
this pipeline they are not: they are what the detector produced on a clip nobody has labelled.
Every number that rests on something the reader should know about carries that something with
it, so the summary cannot be quoted without its caveat attached.
"""
from __future__ import annotations

from html import escape as html_escape
from pathlib import Path
from typing import Any

# The POC acceptance thresholds from CLAUDE.md, repeated here so the report can say whether a
# number met the bar it was written against rather than leaving the reader to remember it.
COVERAGE_TARGET = 0.90
HITTER_TARGET = 0.90
RALLY_BOUNDARY_TARGET = 0.85

_CONSOLIDATION_RATIO = 1.5
_OFFCOURT_SHARE = 0.10
# Suppression keeps the strongest candidate and absorbs everything within one minimum-interval
# window on each side of it, so twice that window is the widest a healthy merge can be. Beyond
# it, merging has started chaining separate shots together again.
_MERGE_SPAN_WINDOWS = 2.0
_FALLBACK_MIN_HIT_INTERVAL_S = 0.30


def hit_intervals(hits: list[dict[str, Any]]) -> list[float]:
    """Seconds between consecutive hits. One hit is not a rhythm, so it yields nothing."""
    times = [float(h["time_s"]) for h in hits if h.get("time_s") is not None]
    return [round(b - a, 3) for a, b in zip(times, times[1:])]


def _rally_rows(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for rally in analysis.get("rallies", []):
        intervals = hit_intervals(rally.get("hits", []))
        rows.append({
            "rally": rally.get("rally"),
            "start_s": rally.get("start_s"),
            "end_s": rally.get("end_s"),
            "duration_s": rally.get("duration_s"),
            "hit_count": rally.get("hit_count", 0),
            "near_hits": rally.get("near_hits", 0),
            "far_hits": rally.get("far_hits", 0),
            "unassigned_hits": rally.get("unassigned_hits", 0),
            "terminating_event": rally.get("terminating_event"),
            "termination_reason": rally.get("termination_reason"),
            "seconds_per_shot": round(sum(intervals) / len(intervals), 3) if intervals else None,
            "intervals_s": intervals,
            "hitters": [h.get("hitter") for h in rally.get("hits", [])],
        })
    return rows


def consolidation_stats(analysis: dict[str, Any]) -> dict[str, Any]:
    """How wide the merged hit bursts were, in seconds.

    Consolidation chains: candidates each within the merge window of the next collapse
    transitively, so a dense enough stream merges an entire rally into one hit no matter how
    long it runs. The span is the number that shows it; the count of merges does not.
    """
    config = analysis.get("config", {})
    fps = float(config.get("fps") or 0.0)
    interval = float(
        (config.get("analysis") or {}).get("min_hit_interval_seconds")
        or _FALLBACK_MIN_HIT_INTERVAL_S
    )
    limit = _MERGE_SPAN_WINDOWS * interval
    # analysis.json concatenates the hit and landing audits into one list, and landings use a
    # window more than twice as wide. Measured together they make healthy output look chained.
    merges = [m for m in (analysis.get("hit_consolidation") or [])
              if m.get("event_type", "hit") == "hit"]
    spans = []
    for merge in merges:
        frames = merge.get("merged_frames") or []
        if len(frames) > 1 and fps > 0:
            spans.append((max(frames) - min(frames)) / fps)
    spans.sort()
    return {
        "merges": len(merges),
        "candidates_merged": sum(len(m.get("merged_frames") or []) for m in merges),
        "longest_merge_s": round(spans[-1], 3) if spans else None,
        "median_merge_s": round(spans[len(spans) // 2], 3) if spans else None,
        "healthy_span_limit_s": round(limit, 3),
        "merges_over_the_limit": sum(1 for s in spans if s > limit),
    }


def filter_stats(analysis: dict[str, Any]) -> dict[str, Any]:
    """What the confidence threshold discarded, and which upstream rules produced it."""
    dropped = analysis.get("low_confidence_hits_dropped") or []
    rules: dict[str, int] = {}
    for entry in dropped:
        rule = entry.get("rule") or "unknown"
        rules[rule] = rules.get(rule, 0) + 1
    threshold = (analysis.get("config", {}).get("analysis") or {}).get("min_hit_confidence") or 0.0
    return {"dropped": len(dropped), "threshold": threshold, "rules": rules}


def _caveats(analysis: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    summary = analysis.get("summary", {})
    coverage = summary.get("two_player_coverage", {})
    rally_cov = coverage.get("during_detected_rallies", {})
    shuttle = summary.get("shuttle_coverage", {})
    hits = summary.get("hits", {})
    out: list[dict[str, str]] = []

    out.append({"level": "info", "text": (
        "These are pipeline outputs, not accuracy. Nothing here has been compared against a "
        "hand label, so precision, recall, hitter accuracy and rally-boundary accuracy are "
        "unknown. Fill in ground_truth.json and run 'badminton-poc evaluate' to measure them."
    )})

    raw = hits.get("raw_detector_candidates")
    detected = hits.get("detected")
    filtered = filter_stats(analysis)

    if filtered["dropped"]:
        worst = max(filtered["rules"].items(), key=lambda kv: kv[1])
        out.append({"level": "warn", "text": (
            f"A confidence threshold of {filtered['threshold']} discarded "
            f"{filtered['dropped']} of {raw} hit candidates before anything was merged, "
            f"{worst[1]} of them from the {worst[0]} rule alone. Upstream scores by rule "
            "identity and how many rules agree, not by certainty, so this selects which rules "
            "to trust. It was chosen against unlabelled output and is not validated. Every "
            "discarded candidate is in analysis.json under low_confidence_hits_dropped."
        )})

    surviving = (raw - filtered["dropped"]) if raw else None
    if surviving and detected and surviving > detected * _CONSOLIDATION_RATIO:
        out.append({"level": "warn", "text": (
            f"{surviving} candidates survived to consolidation and became {detected} hits, so "
            f"{detected} is a merged count rather than a count of independent detections. "
            "Every merge is listed in analysis.json under hit_consolidation."
        )})

    during = rally_cov.get("both_players_fraction")
    if during is not None and during < COVERAGE_TARGET:
        out.append({"level": "warn", "text": (
            f"Both players were detected on {during:.1%} of frames during detected rallies, "
            f"below the {COVERAGE_TARGET:.0%} acceptance target "
            f"(near {rally_cov.get('near_detected_fraction', 0):.1%}, "
            f"far {rally_cov.get('far_detected_fraction', 0):.1%}). A hit on a frame where a "
            "player is missing cannot be assigned on evidence."
        )})

    offcourt = shuttle.get("offcourt_fraction")
    if offcourt and offcourt >= _OFFCOURT_SHARE:
        out.append({"level": "warn", "text": (
            f"{offcourt:.1%} of shuttle samples were rejected as a neighbouring court's. "
            "TrackNetV3 emits exactly one shuttle per frame and cannot tell the courts apart; "
            "these were excluded by the configured airspace polygon."
        )})

    visible = shuttle.get("visible_fraction")
    if visible is not None and visible < 0.80:
        out.append({"level": "info", "text": (
            f"The shuttle was visible and on our court on {visible:.1%} of frames. Gaps are "
            "hits and landings the detector had no trajectory to reason about."
        )})

    unassigned = hits.get("unassigned")
    if unassigned:
        out.append({"level": "warn", "text": (
            f"{unassigned} of {detected} hits could not be attributed to a player and are "
            "reported as unassigned rather than guessed. Each one records why in "
            "analysis.json under assignment_reason."
        )})

    merge = consolidation_stats(analysis)
    longest = merge["longest_merge_s"]
    limit = merge["healthy_span_limit_s"]
    if longest is not None and limit and longest > limit:
        out.append({"level": "warn", "text": (
            f"Hit consolidation merged candidates across {longest:.1f} seconds in its widest "
            f"case, and {merge['merges_over_the_limit']} of {merge['merges']} merges span "
            f"longer than {limit:.1f}s. Suppression should absorb candidates only within one "
            "interval either side of what it keeps, so a wider merge means separate shots are "
            "being chained together and the shot count below is an undercount."
        )})

    if rows and all(r["hit_count"] <= 2 for r in rows):
        out.append({"level": "warn", "text": (
            "Every rally has two hits or fewer, which is not what a real rally looks like. "
            "Rally segmentation is probably splitting on noise."
        )})

    return out


def build_report(analysis: dict[str, Any]) -> dict[str, Any]:
    """Everything the summary page renders, derived from one analysis.json."""
    summary = analysis.get("summary", {})
    metadata = analysis.get("metadata", {})
    config = analysis.get("config", {})
    coverage = summary.get("two_player_coverage", {})
    hits = summary.get("hits", {})
    rows = _rally_rows(analysis)

    all_intervals = [v for row in rows for v in row["intervals_s"]]
    tempo = {
        "shots_measured": len(all_intervals),
        "mean_seconds_per_shot": round(sum(all_intervals) / len(all_intervals), 3)
        if all_intervals else None,
        "fastest_s": min(all_intervals) if all_intervals else None,
        "slowest_s": max(all_intervals) if all_intervals else None,
    }

    fps = config.get("fps") or 0.0
    frames = summary.get("frames_analysed") or metadata.get("frame_count") or 0
    return {
        "clip": {
            "source_video": metadata.get("source_video"),
            "frames": frames,
            "fps": round(fps, 3) if fps else None,
            "duration_s": round(frames / fps, 2) if fps else None,
            "source_frame_offset": metadata.get("source_frame_offset", 0),
        },
        "headline": {
            "rallies": summary.get("rallies", {}),
            "hits": hits,
            "coverage": coverage,
            "shuttle": summary.get("shuttle_coverage", {}),
            "events_by_type": summary.get("events_by_type", {}),
        },
        "tempo": tempo,
        "consolidation": consolidation_stats(analysis),
        "filtered": filter_stats(analysis),
        "rallies": rows,
        "caveats": _caveats(analysis, rows),
    }


# --- HTML rendering -------------------------------------------------------------------
# Emitted as a fragment (title, style, content) with no <html>/<head>/<body> wrapper: a
# browser renders the file as-is, and the Artifact publisher supplies the skeleton. One code
# path serves both.

_FONTS = ("https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600"
          "&family=Public+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap")

_CSS = """
:root {
  --paper: #f4f6f3;
  --surface: #ffffff;
  --ink: #12211d;
  --muted: #63736c;
  --rule: #d7ded9;
  --near: #0d6e63;
  --far: #b4531f;
  --warn: #8a5a09;
  --warn-bg: #fdf6e6;
  --net: #9aa8a2;
}
:root:not([data-theme="light"]) {
  color-scheme: light dark;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --paper: #0d1714;
    --surface: #142320;
    --ink: #e8efeb;
    --muted: #93a49c;
    --rule: #26362f;
    --near: #4dbfae;
    --far: #e5924f;
    --warn: #e0b458;
    --warn-bg: #22301f;
    --net: #47584f;
  }
}
:root[data-theme="dark"] {
  --paper: #0d1714;
  --surface: #142320;
  --ink: #e8efeb;
  --muted: #93a49c;
  --rule: #26362f;
  --near: #4dbfae;
  --far: #e5924f;
  --warn: #e0b458;
  --warn-bg: #22301f;
  --net: #47584f;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--paper);
  color: var(--ink);
  font-family: "Public Sans", ui-sans-serif, system-ui, -apple-system, sans-serif;
  font-size: 16px;
  line-height: 1.6;
}
.wrap { max-width: 1080px; margin: 0 auto; padding: 48px 24px 96px; }
.stack { display: flex; flex-direction: column; gap: 44px; }
h1, h2, h3 { font-family: Fraunces, ui-serif, Georgia, serif; font-weight: 600;
             text-wrap: balance; margin: 0; line-height: 1.15; }
h1 { font-size: clamp(30px, 4.2vw, 46px); letter-spacing: -0.015em; }
h2 { font-size: 23px; letter-spacing: -0.01em; }
h3 { font-size: 16px; font-family: "Public Sans", sans-serif; font-weight: 600; }
p { margin: 0; max-width: 68ch; }
.eyebrow { font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 11px;
           letter-spacing: 0.14em; text-transform: uppercase; color: var(--muted); }
.mono { font-family: "IBM Plex Mono", ui-monospace, SFMono-Regular, monospace;
        font-variant-numeric: tabular-nums; }
.masthead { display: flex; flex-direction: column; gap: 10px;
            border-bottom: 2px solid var(--ink); padding-bottom: 22px; }
.masthead .sub { color: var(--muted); font-size: 14px; }
.banner { display: flex; gap: 14px; align-items: flex-start; background: var(--warn-bg);
          border-left: 3px solid var(--warn); padding: 16px 20px; }
.banner .mark { font-family: Fraunces, serif; font-size: 20px; color: var(--warn);
                line-height: 1.2; }
.banner p { font-size: 14.5px; }
.band { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1px; background: var(--rule); border: 1px solid var(--rule); }
.cell { background: var(--surface); padding: 14px 16px; display: flex;
        flex-direction: column; gap: 3px; }
.cell .k { font-size: 12px; color: var(--muted); }
.cell .v { font-family: "IBM Plex Mono", monospace; font-variant-numeric: tabular-nums;
           font-size: 21px; font-weight: 500; }
.cell .n { font-size: 11.5px; color: var(--muted); }
.split { display: inline-flex; gap: 10px; align-items: baseline; }
.near { color: var(--near); }
.far { color: var(--far); }
.legend { display: flex; flex-wrap: wrap; gap: 18px; font-size: 12.5px; color: var(--muted);
          align-items: center; }
.legend b { font-weight: 500; }
.swatch { display: inline-block; width: 9px; height: 9px; border-radius: 50%;
          margin-right: 6px; vertical-align: middle; }
.rallies { display: flex; flex-direction: column; gap: 2px; }
.rally { display: grid; grid-template-columns: 62px 1fr 132px; gap: 16px; align-items: center;
         padding: 9px 4px; border-bottom: 1px solid var(--rule); }
.rally .id { font-family: "IBM Plex Mono", monospace; font-size: 12px; color: var(--muted);
             font-variant-numeric: tabular-nums; }
.rally .meta { font-family: "IBM Plex Mono", monospace; font-size: 12px; text-align: right;
               color: var(--muted); font-variant-numeric: tabular-nums; }
.rally svg { display: block; width: 100%; height: 54px; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13.5px; min-width: 620px; }
th, td { text-align: right; padding: 7px 10px; border-bottom: 1px solid var(--rule);
         font-family: "IBM Plex Mono", monospace; font-variant-numeric: tabular-nums; }
th { font-size: 11px; letter-spacing: 0.07em; text-transform: uppercase; color: var(--muted);
     font-weight: 500; border-bottom: 1px solid var(--ink); }
th:first-child, td:first-child { text-align: left; }
tbody tr:hover { background: var(--surface); }
.caveats { display: flex; flex-direction: column; gap: 14px; }
.caveat { display: grid; grid-template-columns: 74px 1fr; gap: 16px; align-items: start;
          padding-top: 14px; border-top: 1px solid var(--rule); }
.caveat .tag { font-family: "IBM Plex Mono", monospace; font-size: 10.5px;
               letter-spacing: 0.1em; text-transform: uppercase; padding-top: 3px; }
.caveat .tag.warn { color: var(--warn); }
.caveat .tag.info { color: var(--muted); }
.caveat p { font-size: 14.5px; }
.notmeasured { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
               gap: 20px; }
.notmeasured div { border-top: 1px solid var(--rule); padding-top: 12px; }
.notmeasured .h { font-weight: 600; font-size: 14px; margin-bottom: 3px; }
.notmeasured .b { font-size: 13.5px; color: var(--muted); }
footer { color: var(--muted); font-size: 12.5px; border-top: 1px solid var(--rule);
         padding-top: 18px; }
@media (max-width: 620px) {
  .rally { grid-template-columns: 44px 1fr; }
  .rally .meta { grid-column: 2; text-align: left; }
}
"""

def _esc(value: Any) -> str:
    return html_escape("" if value is None else str(value))


def _pct(value: Any) -> str:
    return "—" if value is None else f"{float(value) * 100:.1f}%"


def _secs(value: Any) -> str:
    return "—" if value is None else f"{float(value):.2f}s"


def _rally_svg(row: dict[str, Any]) -> str:
    """A rally drawn the way it was played: the shuttle crossing the net, over time.

    x is time within the rally, the horizontal rule is the net, and a hit sits on the side of
    the player who made it. Alternation shows as a zig-zag, so two consecutive hits by the same
    player -- which in singles means a miss, a let, or a bad assignment -- reads as a kink.
    Unassigned hits sit on the net line itself, which is exactly what the pipeline knows about
    them: something happened, on neither side of the evidence.
    """
    width, height, mid = 1000.0, 54.0, 27.0
    y = {"near": 42.0, "far": 12.0, None: mid}
    hitters = row["hitters"]
    intervals = row["intervals_s"]
    if not hitters:
        return (f'<svg viewBox="0 0 {width:.0f} {height:.0f}" preserveAspectRatio="none" '
                f'role="img" aria-label="rally with no hits">'
                f'<line x1="0" y1="{mid}" x2="{width}" y2="{mid}" stroke="var(--net)" '
                f'stroke-width="1" stroke-dasharray="3 4" /></svg>')

    # Place hits by elapsed time so the spacing is tempo, not just shot order.
    elapsed, total = [0.0], sum(intervals)
    for gap in intervals:
        elapsed.append(elapsed[-1] + gap)
    span = total if total > 0 else 1.0
    xs = [26.0 + (t / span) * (width - 52.0) for t in elapsed]

    parts = [f'<svg viewBox="0 0 {width:.0f} {height:.0f}" preserveAspectRatio="none" '
             f'role="img" aria-label="{len(hitters)} hits, '
             f'{row["near_hits"]} near and {row["far_hits"]} far">']
    parts.append(f'<line x1="0" y1="{mid}" x2="{width}" y2="{mid}" stroke="var(--net)" '
                 f'stroke-width="1" stroke-dasharray="3 4" />')
    points = " ".join(f"{x:.1f},{y.get(h, mid):.1f}" for x, h in zip(xs, hitters))
    parts.append(f'<polyline points="{points}" fill="none" stroke="var(--net)" '
                 f'stroke-width="1.4" stroke-linejoin="round" />')
    for x, hitter in zip(xs, hitters):
        colour = {"near": "var(--near)", "far": "var(--far)"}.get(hitter)
        if colour is None:
            parts.append(f'<rect x="{x - 3.2:.1f}" y="{mid - 3.2:.1f}" width="6.4" height="6.4" '
                         f'transform="rotate(45 {x:.1f} {mid:.1f})" fill="var(--paper)" '
                         f'stroke="var(--net)" stroke-width="1.4" />')
        else:
            parts.append(f'<circle cx="{x:.1f}" cy="{y[hitter]:.1f}" r="4" fill="{colour}" />')
    parts.append("</svg>")
    return "".join(parts)


def render_html(report: dict[str, Any], *, title: str | None = None) -> str:
    clip = report["clip"]
    head = report["headline"]
    cov = head["coverage"]
    during = cov.get("during_detected_rallies", {})
    hits = head["hits"]
    rallies = head["rallies"]
    shuttle = head["shuttle"]
    tempo = report["tempo"]
    rows = report["rallies"]

    stem = Path(str(clip.get("source_video") or "clip")).stem
    page_title = title or f"{stem} Rally Report"

    lead = next((c for c in report["caveats"] if "not accuracy" in c["text"].lower()), None)
    rest = [c for c in report["caveats"] if c is not lead]

    cells = [
        ("Rallies", str(rallies.get("detected", 0)),
         f'{rallies.get("average_hits", 0)} hits average · longest {rallies.get("longest_hits", 0)}'),
        ("Shots", str(hits.get("detected", 0)),
         f'{hits.get("near", 0)} near · {hits.get("far", 0)} far · '
         f'{hits.get("unassigned", 0)} unassigned'),
        ("Seconds per shot", _secs(tempo.get("mean_seconds_per_shot")),
         f'over {tempo.get("shots_measured", 0)} intervals'),
        ("Both players in frame", _pct(during.get("both_players_fraction")),
         f'during rallies · near {_pct(during.get("near_detected_fraction"))} · '
         f'far {_pct(during.get("far_detected_fraction"))}'),
        ("Shuttle tracked", _pct(shuttle.get("visible_fraction")),
         f'{_pct(shuttle.get("offcourt_fraction"))} rejected as another court’s'),
        ("Clip", f'{_secs(clip.get("duration_s"))}',
         f'{clip.get("frames", 0)} frames at {clip.get("fps")} fps'),
    ]

    out = [f"<title>{_esc(page_title)}</title>",
           f'<link rel="stylesheet" href="{_FONTS}">',
           f"<style>{_CSS}</style>",
           '<div class="wrap"><div class="stack">']

    out.append(
        '<header class="masthead">'
        f'<div class="eyebrow">Badminton POC · analysis.json</div>'
        f"<h1>{_esc(stem)}</h1>"
        f'<div class="sub mono">{clip.get("frames", 0)} frames · '
        f'{_secs(clip.get("duration_s"))} · {clip.get("fps")} fps · '
        f'source frame offset {clip.get("source_frame_offset", 0)}</div>'
        "</header>")

    if lead:
        out.append('<div class="banner"><div class="mark">!</div>'
                   f"<p>{_esc(lead['text'])}</p></div>")

    out.append('<section class="band">' + "".join(
        f'<div class="cell"><span class="k">{_esc(k)}</span>'
        f'<span class="v">{_esc(v)}</span><span class="n">{_esc(n)}</span></div>'
        for k, v, n in cells) + "</section>")

    out.append(
        '<section class="stack" style="gap:16px">'
        "<h2>Rally timeline</h2>"
        "<p>Each strip is one rally, drawn as the shuttle crossing the net. Time runs left to "
        "right within the rally, so the spacing between marks is the real tempo. A hit sits on "
        "the side of the player who played it, which makes a normal exchange a zig-zag — two "
        "marks running on the same side mean a miss, a let, or an assignment the pipeline got "
        "wrong.</p>"
        '<div class="legend">'
        '<span><i class="swatch" style="background:var(--near)"></i><b>Near player</b></span>'
        '<span><i class="swatch" style="background:var(--far)"></i><b>Far player</b></span>'
        '<span><i class="swatch" style="background:transparent;border:1.4px solid var(--net);'
        'border-radius:1px;transform:rotate(45deg)"></i><b>Unassigned</b> — on the net line, '
        "because that is what we know</span></div>"
        '<div class="rallies">' + "".join(
            f'<div class="rally"><span class="id">#{_esc(r["rally"])}</span>'
            f"<div>{_rally_svg(r)}</div>"
            f'<span class="meta">{r["hit_count"]} shots · {_secs(r["duration_s"])}<br>'
            f'{_esc(r["terminating_event"] or "no terminal event")}</span></div>'
            for r in rows) +
        "</div></section>")

    out.append(
        '<section class="stack" style="gap:16px"><h2>Rally detail</h2><div class="scroll">'
        "<table><thead><tr><th>Rally</th><th>Start</th><th>Duration</th><th>Shots</th>"
        "<th>Near</th><th>Far</th><th>Unassigned</th><th>s / shot</th><th>Ended on</th>"
        "</tr></thead><tbody>" + "".join(
            f'<tr><td>#{_esc(r["rally"])}</td><td>{_secs(r["start_s"])}</td>'
            f'<td>{_secs(r["duration_s"])}</td><td>{r["hit_count"]}</td>'
            f'<td class="near">{r["near_hits"]}</td><td class="far">{r["far_hits"]}</td>'
            f'<td>{r["unassigned_hits"] or "—"}</td>'
            f'<td>{_secs(r["seconds_per_shot"])}</td>'
            f'<td>{_esc(r["terminating_event"] or "—")}</td></tr>'
            for r in rows) +
        "</tbody></table></div></section>")

    if rest:
        out.append(
            '<section class="stack" style="gap:16px"><h2>What these numbers rest on</h2>'
            '<div class="caveats">' + "".join(
                f'<div class="caveat"><span class="tag {c["level"]}">{c["level"]}</span>'
                f"<p>{_esc(c['text'])}</p></div>" for c in rest) +
            "</div></section>")

    out.append(
        '<section class="stack" style="gap:16px"><h2>Not measured here</h2>'
        "<p>Deliberately out of scope until the base vision pipeline is validated. Classifying "
        "a shot detected in the wrong place is worse than not classifying it.</p>"
        '<div class="notmeasured">'
        '<div><div class="h">Shot type</div><div class="b">No smash, drop, clear or lift. '
        "Needs a validated hit position and a shuttle height estimate first.</div></div>"
        '<div><div class="h">Winner and error</div><div class="b">Who won the rally is not '
        "inferred. Rally ends are detected, not adjudicated.</div></div>"
        '<div><div class="h">Score</div><div class="b">No scoreboard OCR, so rallies are '
        "numbered in order of appearance and carry no match context.</div></div>"
        '<div><div class="h">Shuttle speed</div><div class="b">Derivable from the stored '
        "trajectory, but not reported: it would need the court homography to mean anything in "
        "real units.</div></div>"
        "</div></section>")

    out.append(
        "<footer>Generated by <span class=\"mono\">badminton-poc report</span> from "
        f"<span class=\"mono\">{_esc(clip.get('source_video'))}</span>. "
        "Every figure traces to analysis.json; every rejected detection and merged burst is "
        "recorded there with its reason.</footer>")

    out.append("</div></div>")
    return "\n".join(out)
