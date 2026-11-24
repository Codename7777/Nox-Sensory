"""Nox fork: inline multimodal content parts for OpenAI-compatible chat relays.

The ``multimodal_inline`` wire mode sends everything inside one chat request:
images as ``image_url`` data URIs, audio as ``input_audio`` base64, and video
as ordered, time-labelled still frames (sampled with FFmpeg) plus the video's
audio track as ``input_audio``.  Relays that front Gemini's OpenAI-compatible
endpoint accept exactly this shape.
"""

from __future__ import annotations

import asyncio
import base64
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from cove_sensory_mcp.errors import ErrorCode, SensoryError
from cove_sensory_mcp.models import Modality

from .base import MediaKind, ProviderRequest

DEFAULT_FRAME_INTERVAL_SECONDS = 5.0
DEFAULT_MAX_FRAMES = 24
DEFAULT_FRAME_WIDTH = 640
MIN_FRAMES = 4
_FRAME_CONCURRENCY = 4
_FFMPEG_TIMEOUT_SECONDS = 120.0
_AUDIO_FORMATS = {
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/wave": "wav",
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/flac": "flac",
}

_FRAME_FAILURE_MESSAGE = "Video frames could not be prepared for the provider."
_TOO_LARGE_MESSAGE = "The prepared media exceeds the provider size limit."


class FfmpegRunner(Protocol):
    async def run_ffmpeg(
        self, args: list[str], timeout: float, cancel_event: object | None = None
    ) -> object: ...

    async def run_ffprobe(self, args: list[str], timeout: float = 15) -> object: ...


@dataclass(frozen=True, slots=True)
class FrameSampling:
    interval_seconds: float = DEFAULT_FRAME_INTERVAL_SECONDS
    max_frames: int = DEFAULT_MAX_FRAMES
    width: int = DEFAULT_FRAME_WIDTH


@dataclass(slots=True)
class InlineParts:
    """Ordered content parts plus the raw byte total they carry."""

    parts: list[dict[str, object]] = field(default_factory=list)
    raw_bytes: int = 0

    def clear(self) -> None:
        self.parts.clear()
        self.raw_bytes = 0


def frame_times(duration: float | None, sampling: FrameSampling) -> list[float]:
    """Return ordered excerpt-relative frame times at the centre of equal slices."""
    if duration is None or not math.isfinite(duration) or duration <= 0:
        return [0.0]
    count = math.ceil(duration / sampling.interval_seconds)
    count = max(min(MIN_FRAMES, sampling.max_frames), min(count, sampling.max_frames))
    step = duration / count
    return [round((index + 0.5) * step, 3) for index in range(count)]


def audio_format(mime_type: str) -> str:
    return _AUDIO_FORMATS.get(mime_type.lower(), mime_type.partition("/")[2])


def _image_part(data: bytes, mime_type: str) -> dict[str, object]:
    encoded = base64.b64encode(data).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
    }


def _audio_part(data: bytes, mime_type: str) -> dict[str, object]:
    return {
        "type": "input_audio",
        "input_audio": {
            "data": base64.b64encode(data).decode("ascii"),
            "format": audio_format(mime_type),
        },
    }


def _returncode(result: object) -> int:
    code = getattr(result, "returncode", 1)
    return code if isinstance(code, int) else 1


class _Budget:
    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max_bytes
        self.used = 0

    def take(self, size: int) -> None:
        self.used += size
        if self.used > self.max_bytes:
            raise SensoryError(ErrorCode.MEDIA_TOO_LARGE, _TOO_LARGE_MESSAGE)


def _read_file(path: Path, budget: _Budget) -> bytes:
    try:
        size = path.stat().st_size
    except OSError:
        raise SensoryError(
            ErrorCode.PROVIDER_UNAVAILABLE,
            "The provider is temporarily unavailable.",
            retryable=True,
        ) from None
    if size <= 0:
        raise SensoryError(ErrorCode.UNSUPPORTED_MEDIA_TYPE, _FRAME_FAILURE_MESSAGE)
    budget.take(size)
    return path.read_bytes()


async def _probe_duration(runtime: FfmpegRunner, path: Path) -> float | None:
    result = await runtime.run_ffprobe(
        [
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    )
    try:
        value = float(str(getattr(result, "stdout", "")).strip())
    except ValueError:
        return None
    return value if math.isfinite(value) and value > 0 else None


async def _extract_frame(
    runtime: FfmpegRunner, video: Path, at: float, width: int, output: Path
) -> bool:
    result = await runtime.run_ffmpeg(
        [
            "-v",
            "error",
            "-ss",
            f"{at:.3f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-vf",
            f"scale={width}:-2",
            "-q:v",
            "4",
            "-map_metadata",
            "-1",
            "-y",
            str(output),
        ],
        timeout=_FFMPEG_TIMEOUT_SECONDS,
    )
    return _returncode(result) == 0 and output.is_file() and output.stat().st_size > 0


async def _extract_audio(runtime: FfmpegRunner, video: Path, output: Path) -> bool:
    result = await runtime.run_ffmpeg(
        [
            "-v",
            "error",
            "-i",
            str(video),
            "-vn",
            "-map",
            "0:a:0?",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "libmp3lame",
            "-b:a",
            "48k",
            "-map_metadata",
            "-1",
            "-y",
            str(output),
        ],
        timeout=_FFMPEG_TIMEOUT_SECONDS,
    )
    return _returncode(result) == 0 and output.is_file() and output.stat().st_size > 0


def _fmt(seconds: float) -> str:
    return f"{seconds:.1f}"


def _video_intro(
    request: ProviderRequest,
    duration: float | None,
    times: list[float],
    has_audio: bool | None,
) -> str:
    length = f"{_fmt(duration)} seconds long" if duration else "of unknown length"
    lines = [f"VIDEO INPUT: the video excerpt is {length}."]
    if request.start_seconds is not None or request.end_seconds is not None:
        start = request.start_seconds or 0.0
        lines.append(
            f"The excerpt begins at {_fmt(start)} s of the source video; "
            "source time = excerpt time + that offset."
        )
    if Modality.VIDEO_VISUAL in request.requested_modalities:
        lines.append(
            f"Its picture is given as {len(times)} still frames in chronological order, "
            "each preceded by a label with its time in seconds from the start of the "
            "excerpt. Infer motion and cuts only from differences between frames, and say "
            "so when something between frames cannot be seen."
        )
    if Modality.VIDEO_AUDIO in request.requested_modalities:
        if has_audio:
            lines.append(
                "The excerpt's complete audio track follows the frames as one audio input."
            )
        else:
            lines.append("This excerpt has no audio track; report the audio as absent.")
    lines.append(
        "Every segment and transcript time must be excerpt-relative"
        + (f", between 0 and {_fmt(duration)}." if duration else ".")
    )
    return " ".join(lines)


async def build_inline_parts(
    request: ProviderRequest,
    *,
    runtime_factory: object,
    sampling: FrameSampling,
    max_bytes: int,
) -> InlineParts:
    """Build ordered content parts for one compatible request."""
    budget = _Budget(max_bytes)
    built = InlineParts()
    media = request.media
    if media.media_kind is MediaKind.IMAGE:
        built.parts.append(_image_part(_read_file(media.path, budget), media.mime_type))
        built.raw_bytes = budget.used
        return built
    if media.media_kind is MediaKind.AUDIO:
        built.parts.append(_audio_part(_read_file(media.path, budget), media.mime_type))
        built.raw_bytes = budget.used
        return built

    runtime: FfmpegRunner = runtime_factory()  # type: ignore[operator]
    want_visual = Modality.VIDEO_VISUAL in request.requested_modalities
    want_audio = Modality.VIDEO_AUDIO in request.requested_modalities
    duration = media.duration_seconds or await _probe_duration(runtime, media.path)
    times = frame_times(duration, sampling) if want_visual else []

    with tempfile.TemporaryDirectory(prefix="cove-frames-") as scratch:
        root = Path(scratch)
        os.chmod(root, 0o700)
        frame_paths = [root / f"frame_{index:03d}.jpg" for index in range(len(times))]
        gate = asyncio.Semaphore(_FRAME_CONCURRENCY)

        async def one(at: float, output: Path) -> bool:
            async with gate:
                return await _extract_frame(runtime, media.path, at, sampling.width, output)

        frame_ok = (
            await asyncio.gather(*(one(at, out) for at, out in zip(times, frame_paths)))
            if times
            else []
        )
        if want_visual and not any(frame_ok):
            raise SensoryError(ErrorCode.UNSUPPORTED_MEDIA_TYPE, _FRAME_FAILURE_MESSAGE)
        audio_path = root / "audio.mp3"
        has_audio = await _extract_audio(runtime, media.path, audio_path) if want_audio else None
        kept = [(at, path) for at, path, ok in zip(times, frame_paths, frame_ok) if ok]

        built.parts.append(
            {
                "type": "text",
                "text": _video_intro(request, duration, [at for at, _ in kept], has_audio),
            }
        )
        for index, (at, path) in enumerate(kept, start=1):
            built.parts.append(
                {"type": "text", "text": f"[frame {index}/{len(kept)} at t={_fmt(at)}s]"}
            )
            built.parts.append(_image_part(_read_file(path, budget), "image/jpeg"))
        if has_audio:
            built.parts.append({"type": "text", "text": "[audio track of the excerpt]"})
            built.parts.append(_audio_part(_read_file(audio_path, budget), "audio/mpeg"))
    built.raw_bytes = budget.used
    return built
