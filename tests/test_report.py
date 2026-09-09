"""Report derivation tests. Pure functions over an analysis.json dict, so no video, no model."""
from __future__ import annotations

import pytest

from badminton_poc.report import build_report, hit_intervals


def _analysis(**over):
    base = {
        "metadata": {"source_video": "clip.mp4", "frame_count": 600},
        "config": {"fps": 60.0},
        "summary": {
            "frames_analysed": 600,
            "two_player_coverage": {
                "both_players_fraction": 0.90, "near_detected_fraction": 0.95,
                "far_detected_fraction": 0.94, "no_player_frames": 2,
                "during_detected_rallies": {
                    "frames": 400, "both_players_fraction": 0.93,
                    "near_detected_fraction": 0.97, "far_detected_fraction": 0.95},
            },
            "shuttle_coverage": {"visible_fraction": 0.80, "offcourt_fraction": 0.02},
            "hits": {"detected": 4, "near": 2, "far": 2, "unassigned": 0,
                     "raw_detector_candidates": 5, "assignment_reasons": {}},
            "rallies": {"detected": 1, "average_hits": 4.0, "longest_hits": 4},
            "events_by_type": {"hit": 4},
        },
        "rallies": [{
            "rally": 1, "start_frame": 0, "end_frame": 180, "start_s": 0.0, "end_s": 3.0,
            "duration_s": 3.0, "hit_count": 4, "near_hits": 2, "far_hits": 2,
            "unassigned_hits": 0, "terminating_event": "landing",
            "hits": [{"time_s": 0.0, "hitter": "near"}, {"time_s": 1.0, "hitter": "far"},
                     {"time_s": 2.0, "hitter": "near"}, {"time_s": 3.0, "hitter": "far"}],
        }],
        "events": [],
    }
    base["summary"].update(over.pop("summary", {}))
    base.update(over)
    return base


def test_hit_intervals_are_the_gaps_between_consecutive_hits():
    assert hit_intervals([{"time_s": 0.0}, {"time_s": 1.5}, {"time_s": 2.0}]) == [1.5, 0.5]


def test_a_single_hit_rally_has_no_interval_to_measure():
    assert hit_intervals([{"time_s": 4.0}]) == []
    assert hit_intervals([]) == []


def test_rally_rows_carry_tempo_in_seconds_between_shots():
    report = build_report(_analysis())
    row = report["rallies"][0]
    assert row["hit_count"] == 4
    assert row["seconds_per_shot"] == pytest.approx(1.0)


def test_every_report_says_these_are_not_accuracy_numbers():
    """The one caveat that is never conditional: analysis.json reports what the pipeline
    produced, and nothing here has been compared against a hand label."""
    report = build_report(_analysis())
    assert any("not accuracy" in c["text"].lower() for c in report["caveats"])


def test_heavy_burst_consolidation_is_called_out():
    """153 raw candidates for 58 kept hits meant the detector was over-firing on a
    variable-frame-rate source. A reader must not see 58 and assume 58 were detected."""
    report = build_report(_analysis(summary={
        "hits": {"detected": 58, "near": 33, "far": 13, "unassigned": 12,
                 "raw_detector_candidates": 153, "assignment_reasons": {}}}))
    hit = [c for c in report["caveats"] if "candidate" in c["text"]]
    assert hit and "153" in hit[0]["text"] and "58" in hit[0]["text"]


def test_no_consolidation_caveat_when_the_detector_behaved():
    report = build_report(_analysis())
    assert not [c for c in report["caveats"] if "candidate" in c["text"]]


def test_coverage_below_the_acceptance_target_is_flagged():
    report = build_report(_analysis(summary={"two_player_coverage": {
        "both_players_fraction": 0.75, "near_detected_fraction": 0.95,
        "far_detected_fraction": 0.86, "no_player_frames": 2,
        "during_detected_rallies": {"frames": 400, "both_players_fraction": 0.80,
                                    "near_detected_fraction": 0.95,
                                    "far_detected_fraction": 0.86}}}))
    flagged = [c for c in report["caveats"] if "90%" in c["text"]]
    assert flagged and flagged[0]["level"] == "warn"


def test_a_neighbouring_courts_shuttle_is_flagged_when_it_is_a_large_share():
    report = build_report(_analysis(summary={
        "shuttle_coverage": {"visible_fraction": 0.65, "offcourt_fraction": 0.30}}))
    assert [c for c in report["caveats"] if "neighbouring" in c["text"]]


def test_unassigned_hits_are_reported_rather_than_quietly_dropped():
    report = build_report(_analysis(summary={
        "hits": {"detected": 58, "near": 33, "far": 13, "unassigned": 12,
                 "raw_detector_candidates": 58, "assignment_reasons": {}}}))
    assert report["headline"]["hits"]["unassigned"] == 12
    assert [c for c in report["caveats"] if "unassigned" in c["text"]]


from badminton_poc.report import render_html


def test_html_leads_with_the_not_accuracy_caveat():
    """The caveat cannot be scrolled past: a rally count reads as a fact and this one is not."""
    html = render_html(build_report(_analysis()))
    assert "not accuracy" in html.lower()
    assert html.index("not accuracy") < html.index("Rally timeline")


def test_html_escapes_content_from_the_analysis_file():
    """Clip paths and termination reasons come from a file on disk, not from us."""
    a = _analysis()
    a["metadata"]["source_video"] = '<script>alert("x")</script>'
    html = render_html(build_report(a))
    assert "<script>alert" not in html
    assert "&lt;script&gt;" in html


def test_html_renders_a_rally_with_no_hits_without_crashing():
    a = _analysis()
    a["rallies"] = [{"rally": 1, "start_s": 0.0, "end_s": 1.0, "duration_s": 1.0,
                     "hit_count": 0, "near_hits": 0, "far_hits": 0, "unassigned_hits": 0,
                     "hits": []}]
    html = render_html(build_report(a))
    assert "Rally timeline" in html


def test_html_defines_every_colour_token_before_any_theme_block():
    """A token defined only inside a prefers-color-scheme block never applies to viewers on
    the default 'system' setting, which renders one theme's text on the other's ground."""
    html = render_html(build_report(_analysis()))
    root = html.index(":root {")
    dark = html.index("prefers-color-scheme")
    for token in ("--ink", "--paper", "--near", "--far", "--muted"):
        assert token in html[root:dark], f"{token} is not defined in the bare :root block"
