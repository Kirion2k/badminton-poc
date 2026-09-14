"""Does the plausibility check tell a player from the furniture the pipeline mistook for one?

The two failures encoded here both happened. yt-ccaa's far slot sat on a seated umpire under a
permissive court polygon and reported 94% coverage; the same clip with a relaxed detection
threshold reported 90% during rallies while the slot held officials standing behind the boards.
"""
from __future__ import annotations

from badminton_poc.analyze import tracked_player_plausibility

FPS = 30.0
RALLIES = [{"start_frame": 0, "end_frame": 199}]
HEIGHT = 100.0


def rows(positions, conf):
    """One row per frame, a box of fixed height centred on each position.

    The near slot gets its own moving, confident track, so that a test about the far slot is not
    quietly also a test about a stationary near one.
    """
    out = []
    for frame, (x, y) in enumerate(positions):
        nx, ny = positions[frame % len(positions)]
        out.append({
            "frame": frame,
            "near": {"bbox": [nx + 380.0, ny + 300.0, nx + 460.0, ny + 500.0], "conf": 0.9},
            "far": {"bbox": [x - 20.0, y - HEIGHT, x + 20.0, y], "conf": conf},
        })
    return out


def a_player():
    """Moves about 0.6 body heights a second and ranges a couple of body heights sideways."""
    return [(500.0 + 100.0 * (i % 40) / 40.0 * (1 if (i // 40) % 2 == 0 else -1), 400.0 + (i % 7))
            for i in range(200)]


def test_a_moving_confident_track_raises_nothing():
    result = tracked_player_plausibility(rows(a_player(), 0.85), RALLIES, FPS)
    assert result["warnings"] == []
    assert result["far"]["median_detection_confidence"] == 0.85


def test_a_seated_umpire_is_called_out():
    """The foot point never moves, which is how the real one behaved."""
    result = tracked_player_plausibility(rows([(852.0, 532.0)] * 200, 0.8), RALLIES, FPS)
    assert any("stationary" in w or "fixed object" in w for w in result["warnings"])
    assert result["far"]["lateral_range_body_heights"] == 0.0


def test_low_confidence_background_people_are_called_out():
    """Officials behind the boards: they do drift, but the detector was never sure of them."""
    result = tracked_player_plausibility(rows(a_player(), 0.24), RALLIES, FPS)
    assert any("median detection confidence" in w for w in result["warnings"])


def test_movement_is_measured_per_second_not_per_frame():
    """The same motion at double the frame rate is the same player, so it must not change."""
    slow = tracked_player_plausibility(rows(a_player(), 0.85), RALLIES, 30.0)
    # Same path walked at the same real speed, sampled twice as often: each step is half as long
    # and there are twice as many, so body heights per second must come out the same.
    path = a_player()
    fine = []
    for (ax, ay), (bx, by) in zip(path, path[1:]):
        fine.append((ax, ay))
        fine.append(((ax + bx) / 2, (ay + by) / 2))
    fast = tracked_player_plausibility(rows(fine[:200], 0.85), RALLIES, 60.0)
    a = slow["far"]["median_movement_body_heights_per_s"]
    b = fast["far"]["median_movement_body_heights_per_s"]
    assert abs(a - b) < 0.15 * max(a, b)


def test_a_gap_in_the_slot_is_not_counted_as_movement():
    """A slot lost for a while and refound elsewhere has not travelled at superhuman speed."""
    r = rows(a_player(), 0.85)
    for row in r[50:150]:
        row["far"] = None
    result = tracked_player_plausibility(r, RALLIES, FPS)
    assert result["far"]["median_movement_body_heights_per_s"] < 2.0


def test_no_rallies_says_so_rather_than_passing_silently():
    assert "note" in tracked_player_plausibility(rows(a_player(), 0.85), [], FPS)
    assert tracked_player_plausibility(rows(a_player(), 0.85), [], FPS).get("warnings") is None


def test_a_coverage_pass_is_withheld_when_a_slot_looks_wrong():
    """The failure mode this exists for: coverage clears 90% by tracking the wrong person."""
    from badminton_poc.evaluation import _targets

    clean = _targets(0.95, 0.95, 0.9, [])
    assert clean["two_player_coverage_active_play >= 0.90"] == "PASS"
    assert "tracked_player_warnings" not in clean

    suspect = _targets(0.95, 0.95, 0.9, ["the far slot ranged 0.00 body heights sideways"])
    assert suspect["two_player_coverage_active_play >= 0.90"].startswith("NOT TRUSTED")
    assert suspect["tracked_player_warnings"]


def test_the_other_two_criteria_are_left_alone():
    """They are scored against hand labels, so a bad slot cannot inflate them."""
    from badminton_poc.evaluation import _targets

    suspect = _targets(0.95, 0.95, 0.9, ["the far slot looks stationary"])
    assert suspect["hitter_assignment_accuracy >= 0.90"] == "PASS"
    assert suspect["rally_boundary_accuracy >= 0.85"] == "PASS"


def test_a_genuine_coverage_failure_still_reads_as_fail():
    """A warning must not upgrade a FAIL into something vaguer."""
    from badminton_poc.evaluation import _targets

    assert _targets(0.40, 0.95, 0.9, ["the far slot looks stationary"])[
        "two_player_coverage_active_play >= 0.90"] == "FAIL"


# --- the baseline path ----------------------------------------------------------------
#
# truth.py, not evaluation.py, is what scores a run today: `badminton-poc baseline` reads the
# hand labels in truth.json and writes baseline.json. A guard that only qualifies the legacy
# evaluate path leaves the live one reporting coverage with nothing attached to it.


def truth_rows(positions, conf):
    """The same rows as above, plus the track ids score_player_tracking needs."""
    out = []
    for row in rows(positions, conf):
        for slot, track_id in (("near", 1), ("far", 2)):
            if row[slot]:
                row[slot]["track_id"] = track_id
        out.append(row)
    return out


LABELLED = [{"start_frame": 0, "end_frame": 199}]


def test_the_baseline_reports_plausibility_beside_the_coverage_figure():
    from badminton_poc.truth import score_player_tracking

    result = score_player_tracking(truth_rows(a_player(), 0.85), LABELLED, fps=FPS)
    assert result["both_players_fraction"] == 1.0
    assert result["coverage_trusted"] is True
    assert result["plausibility"]["far"]["median_detection_confidence"] == 0.85


def test_the_baseline_withholds_trust_when_a_slot_sits_still():
    """A seated umpire fills the far slot on every frame, so coverage reads 100%."""
    from badminton_poc.truth import score_player_tracking

    result = score_player_tracking(truth_rows([(852.0, 532.0)] * 200, 0.8), LABELLED, fps=FPS)
    assert result["both_players_fraction"] == 1.0, "the coverage figure itself is not altered"
    assert result["coverage_trusted"] is False
    assert result["plausibility"]["warnings"]


def test_plausibility_is_measured_over_the_labelled_spans_not_the_whole_clip():
    """Standing still between points is not a failure; standing still during one is."""
    from badminton_poc.truth import score_player_tracking

    positions = a_player()[:100] + [(600.0, 400.0)] * 100
    result = score_player_tracking(truth_rows(positions, 0.85),
                                   [{"start_frame": 0, "end_frame": 99}], fps=FPS)
    assert result["plausibility"]["warnings"] == []
    assert result["coverage_trusted"] is True


def test_no_labelled_rallies_leaves_trust_unstated_rather_than_true():
    """With nothing to measure over, claiming the slots are fine would be an invention."""
    from badminton_poc.truth import score_player_tracking

    result = score_player_tracking(truth_rows(a_player(), 0.85), [], fps=FPS)
    assert result["both_players_fraction"] is None
    assert result["coverage_trusted"] is None


def test_the_note_says_which_rally_spans_it_measured_over():
    """analyze measures over detected rallies, the baseline over hand-labelled ones. A note that
    claims the wrong one turns a file kept for its honesty into a misleading record."""
    from badminton_poc.truth import score_player_tracking

    detected = tracked_player_plausibility(rows(a_player(), 0.85), RALLIES, FPS)
    assert "detected" in detected["note"] and "labelled" not in detected["note"]

    scored = score_player_tracking(truth_rows(a_player(), 0.85), LABELLED, fps=FPS)
    assert "labelled" in scored["plausibility"]["note"]


def test_too_few_frames_to_judge_leaves_trust_unstated():
    """Six frames of a slot say nothing about it, and silence must not read as approval."""
    from badminton_poc.truth import score_player_tracking

    result = score_player_tracking(truth_rows(a_player(), 0.85),
                                   [{"start_frame": 0, "end_frame": 5}], fps=FPS)
    assert result["plausibility"]["far"]["note"] == "too few frames to judge"
    assert result["coverage_trusted"] is None
