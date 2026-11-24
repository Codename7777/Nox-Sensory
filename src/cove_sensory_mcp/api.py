"""Nox fork: call the sensory layer from scripts without going through MCP.

    from cove_sensory_mcp.api import sense
    report = sense("video", "/path/clip.mp4", question="...", start_seconds=0, end_seconds=60)
    print(report["text"])          # same text the MCP tool returns
    report["observations"] ...     # full structured report (same as MCP structured_content)

Raises ``cove_sensory_mcp.errors.SensoryError`` on failure.  Uses the same
config (~/.config/cove-sensory-mcp/config.yaml), routes, allowed media roots
and credentials as the MCP server; the credential env var named by the
provider's ``api_key_env`` must be set in the calling environment.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
from pathlib import Path
from typing import Any, Literal

from .reports.render import render_sensory_text
from .services import AppServices
from .tools.inputs import (
    SenseAudioInput,
    SenseImageInput,
    SenseMusicInput,
    SenseVideoInput,
)

Kind = Literal["image", "video", "audio", "music"]
_INPUTS = {
    "image": SenseImageInput,
    "video": SenseVideoInput,
    "audio": SenseAudioInput,
    "music": SenseMusicInput,
}


def default_services() -> AppServices:
    from .cli import _build_services

    return _build_services()


async def asense(
    kind: Kind,
    source: str,
    *,
    services: AppServices | None = None,
    **options: Any,
) -> dict[str, Any]:
    """Sense one local file; ``options`` are the MCP tool's arguments.

    Common: question, detail ("auto"/"quick"/"detailed"), language ("zh-CN").
    video/audio/music: start_seconds, end_seconds.  video: include_audio.
    audio: include_transcript.  music: include_lyrics_transcript.
    """
    try:
        model = _INPUTS[kind]
    except KeyError:
        raise ValueError(f"kind must be one of {sorted(_INPUTS)}") from None
    payload = model.model_validate({"source": str(source), **options})
    report = await (services or default_services()).sense(kind, payload)
    result = report.model_dump(mode="json")
    result["text"] = render_sensory_text(report)
    return result


def sense(kind: Kind, source: str, **options: Any) -> dict[str, Any]:
    """Synchronous wrapper around :func:`asense`."""
    return asyncio.run(asense(kind, source, **options))


# ---------------------------------------------------------------------------
# Nox fork: staging and a raw "ask" for home scripts.

STAGING_ROOT = Path("~/nox-state/sensory/staging").expanduser()


def stage(path: str | os.PathLike[str], services: AppServices | None = None) -> Path:
    """Return a path the sensory layer may read for ``path``.

    Files already under an allowed media root are returned as-is.  Anything
    else is hard-linked (or, across filesystems, copied with its mtime) into
    a stable per-file folder under the staging root, which is itself an
    allowed root, so repeated calls reuse the same path and cached uploads.
    """
    real = Path(path).expanduser().resolve(strict=True)
    if not real.is_file():
        raise FileNotFoundError(str(real))
    config = (services or default_services()).config_store.load()
    for root in config.allowed_media_roots:
        root_path = Path(root).expanduser().resolve()
        if real == root_path or root_path in real.parents:
            return real
    folder = STAGING_ROOT / hashlib.sha1(str(real).encode()).hexdigest()[:16]
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    target = folder / real.name
    source_stat = real.stat()
    if target.exists():
        target_stat = target.stat()
        if (target_stat.st_dev, target_stat.st_ino) == (
            source_stat.st_dev,
            source_stat.st_ino,
        ) or (
            target_stat.st_size == source_stat.st_size
            and target_stat.st_mtime_ns == source_stat.st_mtime_ns
        ):
            return target
        target.unlink()
    try:
        os.link(real, target)
    except OSError:
        shutil.copy2(real, target)
    return target


def ask(
    path: str | os.PathLike[str],
    prompt: str,
    *,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
    provider: str = "gemini-native",
    model: str | None = None,
    temperature: float | None = 0.2,
    json_output: bool = False,
    timeout_seconds: float = 600,
) -> str:
    """Send one media file plus a free-form prompt to native Gemini; return its text.

    Unlike :func:`sense`, there is no observation contract: the caller owns the
    prompt and parses the reply.  Video/audio files are uploaded once through
    the File API and cached (see ``file_cache_path``), and ``start_seconds`` /
    ``end_seconds`` select a window with videoMetadata offsets, so asking about
    many windows of one file never re-uploads it.  Images go inline.
    """
    import mimetypes
    import time

    from google import genai
    from google.genai import types

    from .providers.gemini_file_cache import DEFAULT_CACHE_PATH, GeminiFileCache

    config = default_services().config_store.load()
    provider_config = config.providers[provider]
    if provider_config.adapter != "gemini":
        raise ValueError("ask() needs a native gemini provider")
    key = os.environ.get(provider_config.api_key_env or "", "") or "nox-relay-placeholder"
    client = genai.Client(
        api_key=key,
        http_options=types.HttpOptions(
            base_url=provider_config.base_url, timeout=int(timeout_seconds * 1000)
        ),
    )
    media = Path(path).expanduser().resolve(strict=True)
    mime = mimetypes.guess_type(media.name)[0] or "application/octet-stream"
    if mime.startswith("image/"):
        part = types.Part.from_bytes(data=media.read_bytes(), mime_type=mime)
    else:
        cache = GeminiFileCache(
            provider_config.adapter_options.file_cache_path or DEFAULT_CACHE_PATH
        )
        hit = cache.get(media)
        remote = None
        if hit is not None:
            try:
                remote = client.files.get(name=hit.name)
                if getattr(remote.state, "name", "") != "ACTIVE":
                    remote = None
            except Exception:  # noqa: BLE001 - expired or deleted remotely
                remote = None
            if remote is None:
                cache.forget(media)
        if remote is None:
            with media.open("rb") as stream:
                remote = client.files.upload(
                    file=stream, config=types.UploadFileConfig(mime_type=mime)
                )
            deadline = time.monotonic() + timeout_seconds
            while getattr(remote.state, "name", "") == "PROCESSING":
                if time.monotonic() > deadline:
                    raise TimeoutError("Gemini file processing timed out")
                time.sleep(1)
                remote = client.files.get(name=remote.name)
            if getattr(remote.state, "name", "") != "ACTIVE":
                raise RuntimeError("Gemini could not process the file")
            cache.put(media, name=remote.name, uri=remote.uri, mime_type=remote.mime_type or mime)
        metadata = None
        if start_seconds is not None or end_seconds is not None:
            metadata = types.VideoMetadata(
                start_offset=f"{start_seconds or 0:.3f}s",
                end_offset=f"{end_seconds:.3f}s" if end_seconds is not None else None,
            )
        part = types.Part(
            file_data=types.FileData(file_uri=remote.uri, mime_type=remote.mime_type or mime),
            video_metadata=metadata,
        )
    response = client.models.generate_content(
        model=model or provider_config.model,
        contents=[types.Content(role="user", parts=[part, types.Part.from_text(text=prompt)])],
        config=types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="application/json" if json_output else None,
        ),
    )
    text = response.text
    if not text:
        raise RuntimeError("Gemini returned no text")
    return text
