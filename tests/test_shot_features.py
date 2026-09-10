"""Tests for the shot feature adapter.

The normalization rules here are copied from the BST repository, so several tests assert the
exact arithmetic rather than a property. If upstream changes, these should fail loudly.
"""
from __future__ import annotations

import numpy as np
import pytest

from badminton_poc.parsing import BallSample, BallTrack
from badminton_poc.shots.features import (
    FeatureConfig,
    N_JOINTS,
    PLAYER_ORDER,
    build_features,
    normalize_joints,
    normalize_shuttle,
    resample_times,
)
from badminton_poc.shots.windows import ShotSample

FPS = 60.0


def _sample(start=100, end=200, hit=150, hitter="near", fps=FPS):
    return ShotSample(
        clip="c", shot_id=0, rally=1, index_in_rally=1, hit_frame=hit,
        hit_time_s=hit / fps, hitter=hitter, prev_hit_frame=start, next_hit_frame=end,
        start_frame=start, end_frame=end, fps=fps, width=1920, height=884,
        is_first_in_rally=False, is_last_in_rally=False,
        rally_start_frame=0, rally_end_frame=300, rally_start_kind="serve",
    )


def _player(x0=100.0, y0=100.0, size=200.0, conf=0.9):
    keypoints = [[x0 + 10 * j, y0 + 5 * j, conf] for j in range(N_JOINTS)]
    return {
        "bbox": [x0, y0, x0 + size, y0 + size],
        "foot": [x0 + size / 2, y0 + size],
        "conf": 0.9,
        "track_id": 1,
        "keypoints": keypoints,
    }


def _rows(frames, near=True, far=True):
    out = {}
    for f in frames:
        row = {"frame": f}
        if near:
            row["near"] = _player(900.0, 500.0)
        if far:
            row["far"] = _player(400.0, 300.0, size=90.0)
        out[f] = row
    return out


def _ball(frames, x=800.0, y=400.0, visible=True):
    return BallTrack([BallSample(frame=f, x=x, y=y, visible=visible) for f in frames])


# --------------------------------------------------------------------------------------------
# temporal resampling


def test_resampling_halves_a_60fps_window_to_30hz():
    cfg = FeatureConfig(target_hz=30.0)
    frames, times = resample_times(100, 219, 150, 60.0, cfg)
    assert len(frames) == 60                     # 2.0s at 30 Hz
    assert frames[0] == 100 and frames[1] == 102  # every other frame
    assert times[0] == pytest.approx((100 - 150) / 60.0)


def test_the_same_duration_gives_the_same_sample_count_at_any_source_rate():
    """This is the whole point: 2 seconds of play is 60 samples whether shot at 25 or 60."""
    cfg = FeatureConfig(target_hz=30.0)
    at60, _ = resample_times(0, 119, 60, 60.0, cfg)
    at25, _ = resample_times(0, 49, 25, 25.0, cfg)
    assert len(at60) == len(at25) == 60


def test_a_source_slower_than_the_target_is_not_invented_into_extra_samples():
    """At 25 fps toward a 30 Hz target we repeat frames rather than interpolate poses."""
    cfg = FeatureConfig(target_hz=30.0)
    frames, _ = resample_times(0, 24, 12, 25.0, cfg)
    assert len(frames) == 30
    assert len(set(frames.tolist())) < 30          # some frames are reused, none invented
    assert frames.max() <= 24


def test_resampling_never_exceeds_the_model_sequence_length():
    cfg = FeatureConfig(target_hz=30.0, max_samples=100)
    frames, _ = resample_times(0, 5999, 3000, 60.0, cfg)
    assert len(frames) == 100


def test_resampling_never_runs_past_the_window():
    cfg = FeatureConfig(target_hz=30.0)
    frames, _ = resample_times(100, 200, 150, 60.0, cfg)
    assert frames.min() >= 100 and frames.max() <= 200


def test_times_are_relative_to_the_contact():
    cfg = FeatureConfig(target_hz=30.0)
    frames, times = resample_times(100, 200, 150, 60.0, cfg)
    assert times[np.argmin(np.abs(frames - 150))] == pytest.approx(0.0, abs=1 / 60.0)
    assert times[0] < 0 < times[-1]


def test_bad_rates_and_windows_are_refused():
    with pytest.raises(ValueError):
        resample_times(0, 10, 5, 0.0, FeatureConfig())
    with pytest.raises(ValueError):
        resample_times(0, 10, 5, 60.0, FeatureConfig(target_hz=0.0))
    with pytest.raises(ValueError):
        resample_times(10, 0, 5, 60.0, FeatureConfig())


# --------------------------------------------------------------------------------------------
# normalization, matching upstream exactly


def test_joint_normalization_matches_the_upstream_formula():
    bbox = np.array([100.0, 100.0, 300.0, 300.0])
    points = np.zeros((N_JOINTS, 2))
    points[0] = (100.0, 100.0)       # the box's top-left corner
    points[1] = (300.0, 300.0)       # the bottom-right corner
    out = normalize_joints(points, bbox, center_align=True)
    diagonal = np.hypot(200.0, 200.0)
    half = 100.0 / diagonal
    assert out[0] == pytest.approx([-half, -half])
    assert out[1] == pytest.approx([half, half])


def test_centre_alignment_can_be_turned_off():
    bbox = np.array([100.0, 100.0, 300.0, 300.0])
    points = np.zeros((N_JOINTS, 2))
    points[0] = (300.0, 300.0)
    out = normalize_joints(points, bbox, center_align=False)
    diagonal = np.hypot(200.0, 200.0)
    assert out[0] == pytest.approx([200.0 / diagonal, 200.0 / diagonal])


def test_a_zero_joint_stays_zero_because_that_is_how_upstream_marks_it_missing():
    bbox = np.array([100.0, 100.0, 300.0, 300.0])
    points = np.zeros((N_JOINTS, 2))
    out = normalize_joints(points, bbox, center_align=True)
    assert np.all(out == 0.0)


def test_a_degenerate_box_does_not_divide_by_zero():
    out = normalize_joints(np.ones((N_JOINTS, 2)), np.array([5.0, 5.0, 5.0, 5.0]),
                           center_align=True)
    assert np.all(out == 0.0)


def test_shuttle_normalization_is_division_by_the_resolution():
    assert normalize_shuttle(960.0, 442.0, 1920, 884) == pytest.approx((0.5, 0.5))
    with pytest.raises(ValueError):
        normalize_shuttle(1.0, 1.0, 0, 884)


# --------------------------------------------------------------------------------------------
# assembling a window


def test_the_player_axis_puts_the_far_player_first():
    """BST orders Top before Bottom and its classes carry that prefix, so this must not flip."""
    assert PLAYER_ORDER == ("far", "near")
    frames = list(range(90, 220))
    feats = build_features(_sample(hitter="near"), _rows(frames), _ball(frames))
    assert feats.hitter_index == 1
    feats_far = build_features(_sample(hitter="far"), _rows(frames), _ball(frames))
    assert feats_far.hitter_index == 0


def test_shapes_are_what_the_model_expects():
    frames = list(range(90, 220))
    feats = build_features(_sample(), _rows(frames), _ball(frames))
    t = feats.n_samples
    assert feats.joints.shape == (t, 2, N_JOINTS, 2)
    assert feats.joints_mask.shape == (t, 2, N_JOINTS)
    assert feats.shuttle.shape == (t, 2)
    assert feats.position.shape == (t, 2, 2)
    assert feats.bbox.shape == (t, 2, 4)


def test_a_missing_player_is_zero_and_masked_not_carried_over():
    frames = list(range(90, 220))
    rows = _rows(frames)
    for f in range(140, 160):
        rows[f].pop("far")
    feats = build_features(_sample(), rows, _ball(frames))
    gap = [i for i, f in enumerate(feats.frames) if 140 <= f < 160]
    assert gap
    assert not feats.joints_mask[gap, 0, :].any()
    assert np.all(feats.joints[gap, 0] == 0.0)
    assert feats.joints_mask[gap, 1, :].all()      # the other player is unaffected


def test_a_low_confidence_joint_counts_as_missing():
    frames = list(range(90, 220))
    rows = _rows(frames)
    for f in frames:
        rows[f]["near"]["keypoints"][9] = [500.0, 500.0, 0.05]
    feats = build_features(_sample(), rows, _ball(frames))
    assert not feats.joints_mask[:, 1, 9].any()
    assert np.all(feats.joints[:, 1, 9] == 0.0)
    assert feats.joints_mask[:, 1, 8].all()


def test_an_invisible_shuttle_is_zero_and_masked():
    frames = list(range(90, 220))
    ball = BallTrack([
        BallSample(frame=f, x=800.0, y=400.0, visible=not (150 <= f < 170)) for f in frames
    ])
    feats = build_features(_sample(), _rows(frames), ball)
    hidden = [i for i, f in enumerate(feats.frames) if 150 <= f < 170]
    assert not feats.shuttle_mask[hidden].any()
    assert np.all(feats.shuttle[hidden] == 0.0)


def test_a_shuttle_parked_on_a_fixed_point_is_not_a_shuttle():
    frames = list(range(90, 220))
    ball = _ball(frames, x=918.0, y=222.0)
    feats = build_features(_sample(), _rows(frames), ball,
                           fixed_points=[{"x": 918.0, "y": 222.0}])
    assert not feats.shuttle_mask.any()


def test_coverage_reports_what_was_actually_observed():
    frames = list(range(90, 220))
    rows = _rows(frames)
    for f in frames:
        rows[f].pop("far")
    feats = build_features(_sample(hitter="near"), rows, _ball(frames))
    cov = feats.coverage()
    assert cov["hitter_joints"] == pytest.approx(1.0)
    assert cov["opponent_joints"] == pytest.approx(0.0)
    assert cov["shuttle"] == pytest.approx(1.0)


def test_provenance_records_every_normalization_choice():
    frames = list(range(90, 220))
    feats = build_features(_sample(), _rows(frames), _ball(frames))
    p = feats.provenance
    assert p["target_hz"] == 30.0 and p["source_fps"] == 60.0
    assert p["decimation_ratio"] == 2.0
    assert p["joint_format"] == "coco17"
    assert p["player_axis"] == ["far", "near"]
    assert "court" in p["position_normalization"]      # says plainly that it is not court coords
    assert p["samples"] < p["samples_if_not_resampled"]


def test_positions_are_image_normalized_feet():
    frames = list(range(90, 220))
    feats = build_features(_sample(), _rows(frames), _ball(frames))
    assert feats.position_mask.all()
    assert 0.0 <= feats.position.min() and feats.position.max() <= 1.0


# --------------------------------------------------------------------------------------------
# shuttle framing


def test_shuttle_framing_defaults_to_the_airspace_box():
    from badminton_poc.shots.features import airspace_box
    assert FeatureConfig().shuttle_frame == "airspace"
    assert airspace_box(((0.13, 0.0), (0.9, 0.0), (0.9, 1.0), (0.13, 1.0))) == (0.13, 0.9, 0.0, 1.0)
    assert airspace_box(None) is None
    assert airspace_box(((0.5, 0.5), (0.5, 0.5))) is None


def test_the_airspace_box_stretches_that_region_onto_the_unit_square():
    """Upstream divides by the frame, which assumes the court fills it. Ours does not."""
    box = (0.2, 0.8, 0.0, 1.0)
    assert normalize_shuttle(0.2 * 1920, 0.0, 1920, 884, box) == pytest.approx((0.0, 0.0))
    assert normalize_shuttle(0.8 * 1920, 884, 1920, 884, box) == pytest.approx((1.0, 1.0))
    assert normalize_shuttle(0.5 * 1920, 442, 1920, 884, box) == pytest.approx((0.5, 0.5))


def test_a_shuttle_outside_the_airspace_box_is_clamped_not_extrapolated():
    box = (0.2, 0.8, 0.0, 1.0)
    x, _ = normalize_shuttle(0.05 * 1920, 442, 1920, 884, box)
    assert x == 0.0


def test_a_degenerate_framing_box_is_refused():
    with pytest.raises(ValueError):
        normalize_shuttle(100.0, 100.0, 1920, 884, (0.5, 0.5, 0.0, 1.0))


def test_the_video_framing_reproduces_upstream_exactly():
    frames = list(range(90, 220))
    feats = build_features(_sample(), _rows(frames), _ball(frames, x=960.0, y=442.0),
                           cfg=FeatureConfig(shuttle_frame="video"))
    assert feats.shuttle[0] == pytest.approx([0.5, 0.5])
    assert "divided by video" in feats.provenance["shuttle_normalization"]


def test_an_unknown_framing_choice_is_refused():
    frames = list(range(90, 220))
    with pytest.raises(ValueError):
        build_features(_sample(), _rows(frames), _ball(frames),
                       cfg=FeatureConfig(shuttle_frame="whatever"))


def test_framing_falls_back_to_the_video_when_no_polygon_is_configured():
    frames = list(range(90, 220))
    feats = build_features(_sample(), _rows(frames), _ball(frames, x=960.0, y=442.0),
                           shuttle_polygon=None)
    assert feats.shuttle[0] == pytest.approx([0.5, 0.5])
