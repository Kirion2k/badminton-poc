"""Source video probing and normalization.

Why this step exists at all: the first real clip we were handed is a phone recording that
is stored portrait with a 90-degree display-matrix rotation, and is genuinely
variable-frame-rate (a mix of 1/60, 2/60 and 3/60 second gaps averaging 28 fps) while
advertising ``r_frame_rate = 60``. OpenCV reports the advertised rate, so every downstream
consumer -- including upstream TrackNet's ``VideoReader`` and its ``BounceDetector`` kinematic
thresholds -- would silently work in the wrong time base, and rotation handling differs between
OpenCV builds and ffmpeg.

Normalizing once removes all of that ambiguity: the working video has rotation baked into the
pixels, a uniform frame rate, and exactly one frame per source frame, so frame index N in every
artefact refers to the same picture. The true source timestamps are preserved separately in
``metadata.json`` so no timing information is lost.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


class VideoError(RuntimeError):
    pass


@dataclass(frozen=True)
class VideoInfo:
    path: str
    codec: str
    stored_width: int
    stored_height: int
    display_width: int
    display_height: int
    rotation: int
    r_frame_rate: float
    avg_frame_rate: float
    duration_s: float
    frame_count: int
    """Frames that actually decode -- what every frame index downstream refers to."""
    packet_count: int
    """Video packets the container claims. Not always the same thing; see reconcile_frame_count."""
    count_warning: str | None
    pix_fmt: str
    bit_rate: int | None
    is_vfr: bool
    pts_deltas_histogram: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise VideoError(f"{name} not found on PATH. Install ffmpeg (provides ffmpeg and ffprobe).")
    return path


def _run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise VideoError(
            f"command failed ({result.returncode}): {' '.join(cmd)}\n{result.stderr.strip()}"
        )
    return result.stdout


def _parse_rate(text: str | None) -> float:
    if not text or text in {"0/0", "N/A"}:
        return 0.0
    if "/" in text:
        num, den = text.split("/", 1)
        den_value = float(den)
        return float(num) / den_value if den_value else 0.0
    return float(text)


def probe(path: str | Path) -> VideoInfo:
    """Probe a video with ffprobe, including a full packet-timestamp scan.

    The packet scan is what tells us whether the file is really constant-rate; it costs a
    couple of seconds and prevents a whole class of silent timing bugs.
    """
    _require_tool("ffprobe")
    path = Path(path)
    if not path.exists():
        raise VideoError(f"video not found: {path}")

    raw = _run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_streams", "-show_format", "-of", "json", str(path),
    ])
    payload = json.loads(raw)
    streams = payload.get("streams") or []
    if not streams:
        raise VideoError(f"no video stream found in {path}")
    stream = streams[0]

    rotation = 0
    for side in stream.get("side_data_list", []) or []:
        if "rotation" in side:
            rotation = int(side["rotation"]) % 360
    if rotation == 0:
        tag_rotate = (stream.get("tags") or {}).get("rotate")
        if tag_rotate is not None:
            # Legacy tag uses the opposite sign convention to the display matrix.
            rotation = (-int(tag_rotate)) % 360

    stored_w = int(stream["width"])
    stored_h = int(stream["height"])
    swapped = rotation % 180 == 90
    display_w, display_h = (stored_h, stored_w) if swapped else (stored_w, stored_h)

    pts = _packet_pts(path)
    packet_count = len(pts)
    if packet_count < 2:
        raise VideoError(f"{path} has {packet_count} video packets; nothing to analyse")
    frame_count, count_warning = reconcile_frame_count(
        packets=packet_count, decoded=_decoded_frame_count(path)
    )

    duration = float(stream.get("duration") or payload.get("format", {}).get("duration") or 0.0)
    if duration <= 0:
        duration = pts[-1] - pts[0]

    deltas: dict[str, int] = {}
    for i in range(packet_count - 1):
        key = f"{pts[i + 1] - pts[i]:.5f}"
        deltas[key] = deltas.get(key, 0) + 1
    # Anything with more than one distinct inter-frame gap (beyond float noise) is VFR.
    is_vfr = len([k for k, v in deltas.items() if v > 1]) > 1

    return VideoInfo(
        path=str(path),
        codec=str(stream.get("codec_name", "unknown")),
        stored_width=stored_w,
        stored_height=stored_h,
        display_width=display_w,
        display_height=display_h,
        rotation=rotation,
        r_frame_rate=_parse_rate(stream.get("r_frame_rate")),
        avg_frame_rate=_parse_rate(stream.get("avg_frame_rate")),
        duration_s=duration,
        frame_count=frame_count,
        packet_count=packet_count,
        count_warning=count_warning,
        pix_fmt=str(stream.get("pix_fmt", "unknown")),
        bit_rate=int(stream["bit_rate"]) if stream.get("bit_rate") else None,
        is_vfr=is_vfr,
        pts_deltas_histogram=dict(sorted(deltas.items(), key=lambda kv: -kv[1])),
    )


def reconcile_frame_count(*, packets: int, decoded: int | None) -> tuple[int, str | None]:
    """Decide how many frames a source really has, and say so when the file disagrees.

    A container's packet count is a claim, not a measurement: the second test clip advertises
    7377 and decodes 7376, because its last packet never becomes a picture. Every frame index
    downstream -- the trim range, players.jsonl, the shuttle CSV -- refers to decoded frames,
    so the decoded count is the one to believe. Trusting the packet count made normalization
    fail its own frame-alignment check, which is the right failure with the wrong diagnosis.
    """
    if decoded is None:
        return packets, (
            f"could not count decodable frames; falling back to the container's claim of "
            f"{packets} video packets. If normalization reports a frame-count mismatch, this "
            "is why."
        )
    if decoded != packets:
        return decoded, (
            f"container advertises {packets} video packets but only {decoded} decode to a "
            f"picture; using {decoded}, which is what every frame index downstream refers to"
        )
    return decoded, None


def _decoded_frame_count(path: Path) -> int | None:
    """Frames that actually decode. Costs a decode pass, which is why the result is carried
    on VideoInfo rather than recomputed."""
    try:
        raw = _run([
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
            "-show_entries", "stream=nb_read_frames",
            "-of", "default=nokey=1:noprint_wrappers=1", str(path),
        ])
    except VideoError:
        return None
    text = raw.strip()
    return int(text) if text.isdigit() else None


def _packet_pts(path: Path) -> list[float]:
    raw = _run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "packet=pts_time", "-of", "csv=p=0", str(path),
    ])
    values = []
    for line in raw.splitlines():
        line = line.strip().rstrip(",")
        if line and line != "N/A":
            values.append(float(line))
    return sorted(values)


def effective_fps(info: VideoInfo) -> float:
    """Frames actually delivered per second of wall clock.

    Deliberately not ``r_frame_rate``: on a VFR phone recording that is the timebase, not
    the rate, and using it makes every derived timestamp wrong by the ratio between them.
    """
    span = info.duration_s
    if span <= 0:
        raise VideoError("cannot determine effective fps: duration is zero")
    return (info.frame_count - 1) / span if info.frame_count > 1 else 0.0


def normalize(
    source: str | Path,
    destination: str | Path,
    *,
    info: VideoInfo | None = None,
    target_width: int = 1920,
    target_fps: float | None = None,
    start_frame: int = 0,
    end_frame: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Produce the working video: rotation baked in, uniform fps, one frame per source frame.

    ``start_frame``/``end_frame`` (end exclusive) trim the source. Trimming is sometimes
    unavoidable -- our first test clip ends with 96 frames of phone UI -- so the offset is
    recorded in the result and in metadata.json: working frame N is source frame N + start_frame.

    Returns a dict describing the working video, including the verified frame count.
    """
    _require_tool("ffmpeg")
    source = Path(source)
    destination = Path(destination)
    info = info or probe(source)

    fps = float(target_fps) if target_fps else effective_fps(info)
    if fps <= 0:
        raise VideoError("target fps must be positive")

    width = min(int(target_width), info.display_width)
    # Even dimensions are required by yuv420p; -2 keeps the aspect ratio and rounds to even.
    scale = f"scale={width}:-2" if width != info.display_width else "scale=iw:ih"

    last = info.frame_count if end_frame is None else int(end_frame)
    if not (0 <= start_frame < last <= info.frame_count):
        raise VideoError(
            f"invalid trim range [{start_frame}, {last}) for a {info.frame_count}-frame source"
        )
    expected_frames = last - start_frame
    trim = "" if (start_frame == 0 and last == info.frame_count) else \
        f"trim=start_frame={start_frame}:end_frame={last},"

    if destination.exists() and not overwrite:
        existing = probe(destination)
        if existing.frame_count == expected_frames:
            return _working_dict(destination, existing, fps, start_frame=start_frame, reused=True)

    destination.parent.mkdir(parents=True, exist_ok=True)
    # passthrough keeps exactly one output frame per input frame, so frame indices stay
    # aligned with the source; setpts then places them on a uniform grid.
    # -fps_mode arrived in ffmpeg 5.0; older builds spell the same thing -vsync.
    passthrough = ["-fps_mode", "passthrough"] if _supports_fps_mode() else ["-vsync", "0"]
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(source),
        "-an",
        *passthrough,
        "-vf", f"{scale},{trim}setpts=N/({fps:.6f}*TB)",
        "-r", f"{fps:.6f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(destination),
    ]
    _run(cmd)

    result = probe(destination)
    if result.frame_count != expected_frames:
        raise VideoError(
            "normalization produced the wrong frame count "
            f"(expected {expected_frames}, got {result.frame_count}); frame indices would no "
            "longer line up with the source. Check the ffmpeg trim/-vsync/-r arguments."
        )
    if result.rotation % 360 != 0:
        raise VideoError(
            f"working video still carries rotation metadata ({result.rotation}); "
            "downstream OpenCV/ffmpeg consumers would disagree about orientation."
        )
    return _working_dict(destination, result, fps, start_frame=start_frame, reused=False)


def _supports_fps_mode() -> bool:
    """ffmpeg >= 5.0 exposes -fps_mode; 4.x only has the older -vsync spelling."""
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-h", "full"], capture_output=True, text=True
    )
    return "-fps_mode" in (result.stdout or "")


def _working_dict(
    path: Path, info: VideoInfo, fps: float, *, start_frame: int = 0, reused: bool
) -> dict[str, Any]:
    return {
        "path": str(path),
        # metadata.json's contract: working frame N is source frame N + this.
        "source_frame_offset": start_frame,
        "width": info.display_width,
        "height": info.display_height,
        "fps": fps,
        "frame_count": info.frame_count,
        "codec": info.codec,
        "duration_s": info.frame_count / fps,
        "reused_existing": reused,
    }


def source_frame_times(info: VideoInfo, path: str | Path | None = None) -> list[float]:
    """True presentation timestamp of each source frame, in order.

    Kept in metadata so a reported event time can always be traced back to the real instant
    in the original recording, even though the working video is on a uniform grid.
    """
    return _packet_pts(Path(path or info.path))
