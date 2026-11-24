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
