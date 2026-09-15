"""Tests for shot-window creation.

The window rule is BST's, restated in seconds. These tests pin it at two frame rates, because
the whole point of restating it in seconds is that the same shot yields the same duration at
25 fps and at 60.
"""
from __future__ import annotations

import pytest

from badminton_poc.shots.windows import ShotSample, ShotWindowConfig, build_samples, window_for

CFG = ShotWindowConfig()


def test_window_runs_between_the_neighbouring_hits():
    start, end = window_for(600, 540, 660, fps=60.0, cfg=CFG)
    assert start == 540                 # the previous contact
    assert end == 660 + 15              # the next contact plus 0.25s


def test_window_falls_back_to_half_a_second_when_a_neighbour_is_missing():
    start, end = window_for(600, None, None, fps=60.0, cfg=CFG)
    assert (start, end) == (600 - 30, 600 + 30)


def test_a_long_gap_is_capped_at_the_limit_either_side():
    start, end = window_for(600, 100, 1200, fps=60.0, cfg=CFG)
    assert start == 600 - 90            # 1.5s before contact
    assert end == 600 + 90 + 15         # 1.5s after, plus the 0.25s extension


def test_the_same_shot_spans_the_same_duration_at_any_frame_rate():
    """The rule is written in seconds precisely so that this holds."""
    at60 = window_for(600, 540, 660, fps=60.0, cfg=CFG)
    at25 = window_for(250, 225, 275, fps=25.0, cfg=CFG)
    seconds60 = (at60[1] - at60[0] + 1) / 60.0
    seconds25 = (at25[1] - at25[0] + 1) / 25.0
    assert seconds60 == pytest.approx(seconds25, abs=0.05)


def test_the_window_is_clipped_to_the_available_frames():
    start, end = window_for(10, None, None, fps=60.0, cfg=CFG, first_frame=0, last_frame=25)
    assert (start, end) == (0, 25)


def test_a_window_never_ends_before_it_starts():
    start, end = window_for(5, None, None, fps=60.0, cfg=CFG, first_frame=100, last_frame=200)
    assert start <= end


def test_fps_must_be_positive():
    with pytest.raises(ValueError):
        window_for(10, None, None, fps=0.0, cfg=CFG)


# --------------------------------------------------------------------------------------------
# building samples from an analysis document


def _analysis(hits, fps=60.0):
    return {
        "config": {"fps": fps},
        "metadata": {"width": 1920, "height": 884, "frame_count": 5000},
        "rallies": [{
            "rally": 1, "start_frame": 100, "end_frame": 900, "start_kind": "serve",
            "hits": hits,
        }],
    }


def test_every_attributed_hit_becomes_one_sample():
    analysis = _analysis([
        {"frame": 200, "hitter": "near"},
        {"frame": 260, "hitter": "far"},
        {"frame": 330, "hitter": "near"},
    ])
    samples, skipped = build_samples(analysis, clip="c", width=1920, height=884)
    assert skipped == []
    assert [s.shot_id for s in samples] == [0, 1, 2]
    assert [s.hitter for s in samples] == ["near", "far", "near"]
    assert samples[1].prev_hit_frame == 200 and samples[1].next_hit_frame == 330
    assert samples[0].is_first_in_rally and not samples[0].is_last_in_rally
    assert samples[2].is_last_in_rally


def test_an_unattributed_hit_is_skipped_with_a_reason_not_guessed():
    analysis = _analysis([
        {"frame": 200, "hitter": "near"},
        {"frame": 260, "hitter": None},
    ])
    samples, skipped = build_samples(analysis, clip="c", width=1920, height=884)
    assert [s.hit_frame for s in samples] == [200]
    assert len(skipped) == 1 and skipped[0]["frame"] == 260
    assert "attribute" in skipped[0]["reason"]


def test_unattributed_hits_can_be_kept_when_asked():
    analysis = _analysis([{"frame": 200, "hitter": None}])
    samples, skipped = build_samples(analysis, clip="c", width=1920, height=884,
                                     cfg=ShotWindowConfig(require_hitter=False))
    assert len(samples) == 1 and samples[0].hitter is None and skipped == []


def test_a_skipped_hit_does_not_break_the_neighbour_chain():
    """Windows reach to the neighbouring *contact*, whether or not we kept it as a sample."""
    analysis = _analysis([
        {"frame": 200, "hitter": "near"},
        {"frame": 260, "hitter": None},
        {"frame": 330, "hitter": "far"},
    ])
    samples, _ = build_samples(analysis, clip="c", width=1920, height=884)
    assert samples[-1].prev_hit_frame == 260


def test_sample_round_trips_through_a_dict():
    analysis = _analysis([{"frame": 200, "hitter": "near"}, {"frame": 260, "hitter": "far"}])
    samples, _ = build_samples(analysis, clip="c", width=1920, height=884)
    again = ShotSample.from_dict(samples[0].to_dict())
    assert again == samples[0]


def test_seconds_and_frames_agree_with_the_frame_rate():
    analysis = _analysis([{"frame": 200, "hitter": "near"}, {"frame": 260, "hitter": "far"}])
    samples, _ = build_samples(analysis, clip="c", width=1920, height=884)
    s = samples[0]
    assert s.frames == s.end_frame - s.start_frame + 1
    assert s.seconds == pytest.approx(s.frames / 60.0)


def test_a_missing_frame_size_is_refused_rather_than_defaulted():
    """A silent default would turn every normalized coordinate downstream into raw pixels."""
    analysis = _analysis([{"frame": 200, "hitter": "near"}])
    with pytest.raises(ValueError):
        build_samples(analysis, clip="c", width=0, height=884)
    with pytest.raises(ValueError):
        build_samples(analysis, clip="c", width=1920, height=0)


def test_an_analysis_without_a_frame_rate_is_refused():
    with pytest.raises(ValueError):
        build_samples({"config": {}, "rallies": []}, clip="c", width=1920, height=884)


def test_windows_are_clipped_to_the_end_of_the_clip():
    analysis = _analysis([{"frame": 4990, "hitter": "near"}])
    analysis["rallies"][0]["hits"] = [{"frame": 4990, "hitter": "near"}]
    samples, _ = build_samples(analysis, clip="c", width=1920, height=884,
                               cfg=ShotWindowConfig(require_hitter=False))
    assert samples[0].end_frame <= 4999


# --- samples built from hand labels rather than from detector output ---
# Milestone 2 evaluates the classifier, not Milestone 1, so a shot sample has to come from a
# contact a human confirmed. Building them from analysis.json would put the detector's false
# positives into the dataset and score the classifier on strokes that never happened.


def _truth(contacts, rallies=None, spans=None):
    out = {
        "clip": "t", "fps": 60.0, "labelled": True,
        "contacts": contacts,
        "rallies": rallies if rallies is not None else [
            {"rally": 1, "start_frame": 0, "end_frame": 1000}],
    }
    if spans is not None:
        out["labelled_contact_spans"] = spans
    return out


def test_samples_come_from_the_labelled_contacts_and_keep_their_shot_type():
    from badminton_poc.shots.windows import build_samples_from_truth
    truth = _truth([
        {"frame": 100, "hitter": "near", "shot_type": "serve"},
        {"frame": 160, "hitter": "far", "shot_type": "lift"},
        {"frame": 230, "hitter": "near", "shot_type": "smash"},
    ])
    samples, skipped = build_samples_from_truth(truth, clip="t", width=1920, height=1080)
    assert [s.hit_frame for s in samples] == [100, 160, 230]
    assert [s.shot_type for s in samples] == ["serve", "lift", "smash"]
    assert [s.index_in_rally for s in samples] == [0, 1, 2]
    assert samples[1].prev_hit_frame == 100 and samples[1].next_hit_frame == 230
    assert samples[0].is_first_in_rally and samples[2].is_last_in_rally
    assert skipped == []


def test_a_contact_that_is_not_a_rally_shot_is_left_out():
    """Handing the shuttle back is a real contact and not a stroke to classify."""
    from badminton_poc.shots.windows import build_samples_from_truth
    truth = _truth([
        {"frame": 100, "hitter": "near", "shot_type": "serve"},
        {"frame": 160, "hitter": "far", "shot_type": "lift"},
        {"frame": 900, "hitter": "near", "in_rally": False},
    ])
    samples, skipped = build_samples_from_truth(truth, clip="t", width=1920, height=1080)
    assert [s.hit_frame for s in samples] == [100, 160]
    assert [s["frame"] for s in skipped] == [900]
    assert "not a rally shot" in skipped[0]["reason"]


def test_a_contact_outside_the_labelled_spans_is_left_out():
    """Clip 5 is labelled in parts. Outside those spans the contact list is incomplete by
    declaration, so a window there would carry neighbours that were never checked."""
    from badminton_poc.shots.windows import build_samples_from_truth
    truth = _truth(
        [{"frame": 100, "hitter": "near", "shot_type": "serve"},
         {"frame": 160, "hitter": "far", "shot_type": "lift"}],
        spans=[{"start_frame": 0, "end_frame": 120}],
    )
    samples, skipped = build_samples_from_truth(truth, clip="t", width=1920, height=1080)
    assert [s.hit_frame for s in samples] == [100]
    assert [s["frame"] for s in skipped] == [160]
    assert "labelled_contact_spans" in skipped[0]["reason"]


def test_a_contact_with_no_hitter_is_left_out_and_says_so():
    from badminton_poc.shots.windows import build_samples_from_truth
    truth = _truth([
        {"frame": 100, "hitter": "near", "shot_type": "serve"},
        {"frame": 160, "hitter": None, "shot_type": "net"},
        {"frame": 230, "hitter": "near", "shot_type": "smash"},
    ])
    samples, skipped = build_samples_from_truth(truth, clip="t", width=1920, height=1080)
    assert [s.hit_frame for s in samples] == [100, 230]
    assert [s["frame"] for s in skipped] == [160]
    assert "could not tell which player" in skipped[0]["reason"]
    # the neighbour chain still runs through the dropped contact, because it happened
    assert samples[1].prev_hit_frame == 160


def test_contacts_are_grouped_by_the_rally_they_fall_in():
    from badminton_poc.shots.windows import build_samples_from_truth
    truth = _truth(
        [{"frame": 100, "hitter": "near", "shot_type": "serve"},
         {"frame": 160, "hitter": "far", "shot_type": "lift"},
         {"frame": 700, "hitter": "far", "shot_type": "serve"},
         {"frame": 760, "hitter": "near", "shot_type": "net"}],
        rallies=[{"rally": 1, "start_frame": 0, "end_frame": 300},
                 {"rally": 2, "start_frame": 650, "end_frame": 900}],
    )
    samples, _ = build_samples_from_truth(truth, clip="t", width=1920, height=1080)
    assert [s.rally for s in samples] == [1, 1, 2, 2]
    assert [s.index_in_rally for s in samples] == [0, 1, 0, 1]
    # the window of the last shot of rally 1 must not reach into rally 2
    assert samples[1].next_hit_frame is None
