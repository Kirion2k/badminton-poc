"""Source-probing tests. Pure functions over already-collected ffprobe numbers, so the suite
still needs no video file, no ffmpeg and no GPU."""
from __future__ import annotations

from badminton_poc.videoprep import reconcile_frame_count


def test_frame_count_prefers_decodable_frames_over_the_container_claim():
    """The second test clip advertises 7377 frames and decodes 7376: its last packet never
    becomes a picture. Trusting the container made normalization fail its own frame-alignment
    check and refuse to run, which is the right failure but the wrong diagnosis."""
    count, warning = reconcile_frame_count(packets=7377, decoded=7376)
    assert count == 7376
    assert warning is not None
    assert "7377" in warning and "7376" in warning


def test_frame_count_is_silent_when_the_container_agrees():
    count, warning = reconcile_frame_count(packets=3026, decoded=3026)
    assert count == 3026
    assert warning is None


def test_a_container_that_undercounts_is_still_believed_over_its_own_claim():
    """Whichever way they disagree, the decodable count is the one every downstream frame
    index refers to."""
    count, warning = reconcile_frame_count(packets=100, decoded=104)
    assert count == 104
    assert warning is not None


def test_an_unavailable_decode_count_falls_back_to_packets():
    """Some containers refuse to report nb_read_frames. Falling back is better than crashing,
    but it must say so."""
    count, warning = reconcile_frame_count(packets=500, decoded=None)
    assert count == 500
    assert warning is not None and "could not" in warning.lower()


from pathlib import Path

from badminton_poc.videoprep import VideoInfo, _working_dict


def _info(frame_count: int = 2000) -> VideoInfo:
    return VideoInfo(
        path="work.mp4", codec="h264", stored_width=1920, stored_height=884,
        display_width=1920, display_height=884, rotation=0, r_frame_rate=60.0,
        avg_frame_rate=60.0, duration_s=frame_count / 60.0, frame_count=frame_count,
        packet_count=frame_count, count_warning=None, pix_fmt="yuv420p", bit_rate=None,
        is_vfr=False, pts_deltas_histogram={},
    )


def test_working_video_records_the_trim_offset_so_frames_map_back_to_the_source():
    """metadata.json promises that working frame N is source frame N + source_frame_offset.
    The offset was never written by either path, so a clip trimmed with --start-frame reported
    0 and every event time silently pointed at the wrong instant in the original recording."""
    fresh = _working_dict(Path("work.mp4"), _info(), 60.0, start_frame=500, reused=False)
    assert fresh["source_frame_offset"] == 500


def test_reusing_an_existing_working_video_reports_the_same_offset():
    """The reuse path passed start_frame positionally into a signature that had no room for
    it, so re-running any stage on an already-normalized clip raised TypeError."""
    reused = _working_dict(Path("work.mp4"), _info(), 60.0, start_frame=500, reused=True)
    assert reused["source_frame_offset"] == 500
    assert reused["reused_existing"] is True
