from __future__ import annotations

import pytest

from badminton_poc.config import ConfigError, CourtROI, PipelineConfig, point_in_polygon

SQUARE = ((0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8))


def test_point_in_polygon():
    assert point_in_polygon(0.5, 0.5, SQUARE)
    assert not point_in_polygon(0.1, 0.5, SQUARE)
    assert not point_in_polygon(0.5, 0.9, SQUARE)


def test_trapezoid_excludes_neighbouring_court():
    """The real reason we use a polygon: a bounding rectangle would include both courts."""
    court = CourtROI(polygon=((0.42, 0.45), (0.80, 0.42), (0.95, 0.88), (0.17, 0.99)))
    assert court.contains_normalized(0.58, 0.72)     # near player
    assert court.contains_normalized(0.67, 0.52)     # far player
    assert not court.contains_normalized(0.32, 0.47)  # neighbouring court
    assert not court.contains_normalized(0.80, 0.42)  # umpire chair, above the court
    assert not court.contains_normalized(0.96, 0.97)  # picture-in-picture thumbnail


def test_sloping_net_line_splits_sides():
    court = CourtROI(polygon=SQUARE, net_line=((0.435, 0.52), (0.79, 0.60)))
    assert court.net_y_at(0.435) == pytest.approx(0.52)
    assert court.net_y_at(0.79) == pytest.approx(0.60)
    assert court.side_of(0.575, 0.72) == "near"
    assert court.side_of(0.675, 0.515) == "far"
    # A horizontal split at the mid-court net height would get this one wrong.
    assert court.side_of(0.95, 0.63) == "far"


def test_net_y_fallbacks():
    assert CourtROI(polygon=SQUARE, net_y=0.4).net_y_at(0.9) == pytest.approx(0.4)
    assert CourtROI(polygon=SQUARE).net_y_at(0.9) == pytest.approx(0.5)


def test_rect_string_backwards_compatible():
    court = CourtROI.from_rect_string("0.1,0.2,0.9,0.8")
    assert court.polygon == ((0.1, 0.2), (0.9, 0.2), (0.9, 0.8), (0.1, 0.8))


@pytest.mark.parametrize("bad", ["0.1,0.2,0.9", "0.9,0.2,0.1,0.8", "0,0,2,1"])
def test_rect_string_rejects_bad_input(bad):
    with pytest.raises(ConfigError):
        CourtROI.from_rect_string(bad)


def test_polygon_validation():
    with pytest.raises(ConfigError, match="at least 3 points"):
        CourtROI(polygon=((0.1, 0.1), (0.2, 0.2)))
    with pytest.raises(ConfigError, match="normalized"):
        CourtROI(polygon=((0.1, 0.1), (1.5, 0.2), (0.3, 0.3)))
    with pytest.raises(ConfigError, match="distinct"):
        CourtROI(polygon=SQUARE, net_line=((0.1, 0.1), (0.1, 0.1)))


def test_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "c.json"
    path.write_text('{"court": {"polygon": [[0,0],[1,0],[1,1]]}, "bogus": 1}', encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown config keys"):
        PipelineConfig.load(path)


def test_config_rejects_unknown_section_keys(tmp_path):
    path = tmp_path / "c.json"
    path.write_text('{"players": {"nope": 1}}', encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown PlayerConfig keys"):
        PipelineConfig.load(path)


def test_config_roundtrip(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(
        '{"court": {"polygon": [[0.4,0.4],[0.8,0.4],[0.9,0.9],[0.2,0.9]],'
        ' "net_line": [[0.4,0.5],[0.8,0.6]]}, "analysis": {"rally_gap_seconds": 3.0}}',
        encoding="utf-8",
    )
    config = PipelineConfig.load(path)
    assert config.analysis.rally_gap_seconds == 3.0
    assert config.court.net_line == ((0.4, 0.5), (0.8, 0.6))
    assert config.to_dict()["court"]["net_line"] == [[0.4, 0.5], [0.8, 0.6]]


def test_calibrated_test_clip_roi_classifies_known_points():
    """Regression test for the ROI shipped in configs/test-video.json.

    Every coordinate here was read out of an actual failing frame during calibration:
    the umpire's elevated chair put their apparent foot point above the net line and so on
    the 'far' side, and the near player at the baseline sits lower in frame than a naive
    court rectangle allows for.
    """
    from pathlib import Path

    from badminton_poc.config import PipelineConfig

    config_path = Path(__file__).resolve().parents[1] / "configs" / "test-video.json"
    court = PipelineConfig.load(config_path).court

    inside = {
        "near player mid-court": (0.512, 0.874),
        "near player at baseline": (0.544, 0.949),
        "far player": (0.675, 0.515),
    }
    outside = {
        "umpire on an elevated chair": (0.796, 0.438),
        "neighbouring court player": (0.293, 0.562),
        "neighbouring court player 2": (0.377, 0.517),
        "seated foreground spectator": (0.296, 0.998),
        "youtube picture-in-picture thumbnail": (0.960, 0.970),
    }
    for name, (x, y) in inside.items():
        assert court.contains_normalized(x, y), f"{name} should be on court"
    for name, (x, y) in outside.items():
        assert not court.contains_normalized(x, y), f"{name} should be off court"

    assert court.side_of(*inside["near player mid-court"]) == "near"
    assert court.side_of(*inside["far player"]) == "far"


def test_shuttle_airspace_polygon_separates_the_courts():
    """TrackNetV3 outputs one shuttle per frame; with three courts in view it locks onto the
    wrong one. Every coordinate here was checked against the actual video frame."""
    from pathlib import Path

    from badminton_poc.config import PipelineConfig

    config_path = Path(__file__).resolve().parents[1] / "configs" / "test-video.json"
    court = PipelineConfig.load(config_path).court

    for x, y in [(0.282, 0.453), (0.219, 0.367), (0.268, 0.415), (0.200, 0.526)]:
        assert not court.shuttle_contains_normalized(x, y)
    for x, y in [(0.442, 0.418), (0.682, 0.552), (0.638, 0.127)]:
        assert court.shuttle_contains_normalized(x, y)


def test_shuttle_roi_defaults_to_accepting_everything():
    """A single-court clip must need no extra configuration."""
    court = CourtROI(polygon=SQUARE)
    assert court.shuttle_contains_normalized(0.01, 0.99)
