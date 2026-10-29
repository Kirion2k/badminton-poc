"""Parser tests, including the malformed inputs we expect to hit in practice."""
from __future__ import annotations

import json

import pytest

from badminton_poc.parsing import (
    BallSample, BallTrack, ParseError, load_ball_csv, load_events_json, load_jsonl,
)


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_load_ball_csv_upstream_schema(tmp_path):
    path = write(tmp_path, "v_ball.csv", "Frame,Visibility,X,Y\n0,1,100,200\n1,0,0,0\n2,1,110.5,205.25\n")
    track = load_ball_csv(path)
    assert len(track) == 3
    assert track.get(0) == BallSample(0, 100.0, 200.0, True)
    assert track.get(1).visible is False
    assert track.get(2).x == pytest.approx(110.5)
    assert track.visible_count() == 2


def test_load_ball_csv_accepts_inpaint_mask_column(tmp_path):
    path = write(tmp_path, "b.csv", "Frame,Visibility,X,Y,Inpaint_Mask\n0,1,5,6,0\n1,1,7,8,1\n")
    track = load_ball_csv(path)
    assert track.get(0).inpainted is False
    assert track.get(1).inpainted is True


def test_load_ball_csv_marks_samples_off_the_tracknet_decode_grid_as_inpainted(tmp_path):
    """Rows taken from clip 6 (1920x1080): two TrackNet detections, then InpaintNet's constant
    gap fill and one of its interpolations. Upstream writes all four with Visibility 1."""
    path = write(tmp_path, "b.csv",
                 "Frame,Visibility,X,Y\n"
                 "0,1,736.875,455.625\n"
                 "1,1,729.375,451.87501430511475\n"
                 "2,1,914.9338483810425,268.08826871216297\n"
                 "3,1,731.4021,449.0517\n"
                 "4,0,0,0\n")
    track = load_ball_csv(path, (1920, 1080))
    assert [track.get(f).inpainted for f in range(5)] == [False, False, True, True, False]
    assert all(track.get(f).visible for f in range(4))
    # Without the frame size nothing can be inferred, so nothing is claimed.
    assert not any(s.inpainted for s in load_ball_csv(path))


def test_load_ball_csv_mask_column_wins_over_the_grid(tmp_path):
    path = write(tmp_path, "b.csv", "Frame,Visibility,X,Y,Inpaint_Mask\n0,1,914.9338,268.0882,0\n")
    assert load_ball_csv(path, (1920, 1080)).get(0).inpainted is False


def test_load_ball_csv_honours_real_frame_ids_with_gaps(tmp_path):
    """Upstream documents Frame as a real identifier that may start late and contain gaps."""
    path = write(tmp_path, "b.csv", "Frame,Visibility,X,Y\n100,1,1,2\n103,1,3,4\n")
    track = load_ball_csv(path)
    assert track.frames == [100, 103]
    assert track.get(101) is None


def test_load_ball_csv_infers_visibility_from_zero_coordinates(tmp_path):
    path = write(tmp_path, "b.csv", "Frame,X,Y\n0,0,0\n1,4,5\n")
    track = load_ball_csv(path)
    assert track.get(0).visible is False
    assert track.get(1).visible is True


def test_load_ball_csv_rejects_missing_columns(tmp_path):
    path = write(tmp_path, "b.csv", "Frame,Visibility\n0,1\n")
    with pytest.raises(ParseError, match="missing required TrackNet column"):
        load_ball_csv(path)


def test_load_ball_csv_rejects_non_numeric(tmp_path):
    path = write(tmp_path, "b.csv", "Frame,Visibility,X,Y\n0,1,abc,2\n")
    with pytest.raises(ParseError, match="non-numeric"):
        load_ball_csv(path)


def test_load_ball_csv_rejects_header_only(tmp_path):
    with pytest.raises(ParseError, match="no rows"):
        load_ball_csv(write(tmp_path, "b.csv", "Frame,Visibility,X,Y\n"))


def test_load_ball_csv_rejects_missing_file(tmp_path):
    with pytest.raises(ParseError, match="not found"):
        load_ball_csv(tmp_path / "nope.csv")


def test_ball_track_rejects_duplicate_frames():
    with pytest.raises(ParseError, match="duplicate Frame"):
        BallTrack([BallSample(1, 0, 0, True), BallSample(1, 1, 1, True)])


def test_nearest_visible_prefers_current_then_earlier():
    track = BallTrack([
        BallSample(8, 1, 1, True), BallSample(9, 2, 2, False),
        BallSample(10, 3, 3, False), BallSample(11, 4, 4, True),
    ])
    assert track.nearest_visible(8, 3).frame == 8
    # frame 10 is invisible; frame 9 also invisible; 11 is +1 and 8 is -2, so 11 wins on distance
    assert track.nearest_visible(10, 3).frame == 11
    # tie at distance 1 -> the earlier frame wins (moment before contact)
    assert track.nearest_visible(9, 3).frame == 8
    assert track.nearest_visible(10, 0) is None


def test_absence_spans():
    track = BallTrack([BallSample(i, 0, 0, i not in {3, 4, 5, 9, 10}) for i in range(12)])
    assert track.absence_spans(3) == [(3, 5)]
    assert track.absence_spans(2) == [(3, 5), (9, 10)]


def test_absence_spans_rejects_bad_length():
    with pytest.raises(ValueError):
        BallTrack([BallSample(0, 0, 0, True)]).absence_spans(0)


def test_load_events_json_upstream_schema(tmp_path):
    payload = [
        {"frame": 10, "x": 1.0, "y": 2.0, "event_type": "hit", "rule": "vy_reversal",
         "confidence": 0.85, "all_rules": ["vy_reversal", "y_local_max"]},
        {"frame": 4, "x": 3.0, "y": 4.0, "event_type": "landing", "rule": "speed_step",
         "confidence": 0.9, "all_rules": ["speed_step"]},
    ]
    path = write(tmp_path, "e.json", json.dumps(payload))
    events = load_events_json(path)
    assert [e.frame for e in events] == [4, 10]
    assert events[1].event_type == "hit"
    assert events[1].all_rules == ("vy_reversal", "y_local_max")
    assert events[0].confidence == 0.9


def test_load_events_json_empty_list_is_valid(tmp_path):
    assert load_events_json(write(tmp_path, "e.json", "[]")) == []


def test_load_events_json_unwraps_object(tmp_path):
    path = write(tmp_path, "e.json", json.dumps({"events": [{"frame": 1, "event_type": "hit"}]}))
    assert len(load_events_json(path)) == 1


def test_load_events_json_rejects_bad_shapes(tmp_path):
    with pytest.raises(ParseError, match="not valid JSON"):
        load_events_json(write(tmp_path, "a.json", "{not json"))
    with pytest.raises(ParseError, match="must contain a JSON list"):
        load_events_json(write(tmp_path, "b.json", json.dumps({"nope": 1})))
    with pytest.raises(ParseError, match="expected an object"):
        load_events_json(write(tmp_path, "c.json", json.dumps([1, 2])))
    with pytest.raises(ParseError, match="no 'frame' key"):
        load_events_json(write(tmp_path, "d.json", json.dumps([{"event_type": "hit"}])))
    with pytest.raises(ParseError, match="no 'event_type' key"):
        load_events_json(write(tmp_path, "e.json", json.dumps([{"frame": 1}])))
    with pytest.raises(ParseError, match="non-integer frame"):
        load_events_json(write(tmp_path, "f.json", json.dumps([{"frame": "x", "event_type": "hit"}])))


def test_load_jsonl(tmp_path):
    path = write(tmp_path, "p.jsonl", '{"frame": 0}\n\n{"frame": 1}\n')
    assert load_jsonl(path) == [{"frame": 0}, {"frame": 1}]


def test_load_jsonl_reports_bad_line(tmp_path):
    with pytest.raises(ParseError, match=r"p\.jsonl:2"):
        load_jsonl(write(tmp_path, "p.jsonl", '{"frame": 0}\nnot json\n'))
