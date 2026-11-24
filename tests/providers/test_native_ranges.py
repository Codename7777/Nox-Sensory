"""Nox fork: file cache, clip ranges (videoMetadata), joint fallbacks, staging."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest

from cove_sensory_mcp.config.schema import AdapterOptions, AppConfig, ProviderConfig
from cove_sensory_mcp.config.secrets import MemorySecretStore
from cove_sensory_mcp.models import DetailLevel, Modality, ProviderRef, RouteConfig
from cove_sensory_mcp.providers.base import MediaKind, PreparedMedia, ProviderRequest
from cove_sensory_mcp.providers.gemini import (
    GeminiGeneration,
    GeminiProvider,
    GeminiRemoteFile,
    GeminiUploadedMedia,
)
from cove_sensory_mcp.providers.gemini_file_cache import GeminiFileCache
from cove_sensory_mcp.providers.router import ProviderRouter

JOINT = frozenset({Modality.VIDEO_VISUAL, Modality.VIDEO_AUDIO})


def _obs(modality: Modality, start: float, end: float) -> dict[str, object]:
    return {
        "modality": modality.value,
        "summary": "s",
        "segments": [{"start_seconds": start, "end_seconds": end, "text": "x"}],
        "transcript": [],
        "warnings": [],
        "confidence": "high",
    }


class FakeClient:
    def __init__(self, text: str) -> None:
        self.text = text
        self.uploads: list[Path] = []
        self.deleted: list[str] = []
        self.contents: list[Any] = []

    async def upload_file(self, *, path: Path, mime_type: str) -> GeminiRemoteFile:
        self.uploads.append(path)
        return GeminiRemoteFile(name=f"files/{len(self.uploads)}", uri="u", mime_type=mime_type)

    async def wait_until_active(self, file: GeminiRemoteFile) -> GeminiRemoteFile:
        return file

    async def generate_content(self, **kwargs: Any) -> GeminiGeneration:
        self.contents.append(kwargs["contents"])
        return GeminiGeneration(text=self.text)

    async def delete_file(self, *, name: str) -> None:
        self.deleted.append(name)

    async def aclose(self) -> None:
        return None


def _provider(client: FakeClient, cache_path: Path) -> GeminiProvider:
    store = MemorySecretStore()
    store.set("gem", "placeholder-credential")
    config = ProviderConfig(
        adapter="gemini",
        model="m",
        credential_ref="gem",
        declared_capabilities={m: True for m in Modality},
        adapter_options=AdapterOptions(
            inline_video_audio=True, inline_max_bytes=64, file_cache_path=str(cache_path)
        ),
    )
    return GeminiProvider(
        provider_id="native",
        config=config,
        secret_store=store,
        client_factory=lambda **_: client,  # type: ignore[arg-type,return-value]
    )


@pytest.mark.asyncio
async def test_clip_range_uploads_once_uses_offsets_and_shifts_times(tmp_path: Path) -> None:
    film = tmp_path / "film.mp4"
    film.write_bytes(b"v" * 1000)
    text = json.dumps({"observations": [_obs(Modality.VIDEO_VISUAL, 22, 30), _obs(Modality.VIDEO_AUDIO, 18, 25)]})
    client = FakeClient(text)
    provider = _provider(client, tmp_path / "files.json")
    request = ProviderRequest(
        PreparedMedia(film, "video/mp4", MediaKind.VIDEO, 20.0, clip_start=20.0, clip_end=40.0, reusable=True),
        JOINT, "", DetailLevel.QUICK, "en", 20.0, 40.0,
    )
    first = await provider.sense(request)
    await provider.sense(request)
    assert client.uploads == [film]                     # second call reuses the cache
    assert client.deleted == []                         # cached uploads are kept
    media = client.contents[0][0]
    assert media == GeminiUploadedMedia(uri="u", mime_type="video/mp4", start_offset=20.0, end_offset=40.0)
    visual = first.observations[Modality.VIDEO_VISUAL].segments[0]
    assert (visual.start_seconds, visual.end_seconds) == (2.0, 10.0)   # source time -> excerpt time
    audio = first.observations[Modality.VIDEO_AUDIO].segments[0]
    assert (audio.start_seconds, audio.end_seconds) == (0.0, 5.0)
    assert provider.supports_joint(JOINT) and provider.supports_clip_ranges


def test_file_cache_keys_on_mtime_and_expires(tmp_path: Path) -> None:
    clock = [datetime(2026, 11, 24, tzinfo=UTC)]
    cache = GeminiFileCache(tmp_path / "files.json", now=lambda: clock[0])
    media = tmp_path / "a.mp4"
    media.write_bytes(b"1")
    cache.put(media, name="files/a", uri="u", mime_type="video/mp4")
    assert cache.get(media) is not None
    os.utime(media, ns=(1, 1))
    assert cache.get(media) is None                     # changed file, new key
    cache.put(media, name="files/b", uri="u", mime_type="video/mp4")
    clock[0] += timedelta(hours=47)
    assert cache.get(media) is None                     # expired


def test_joint_fallbacks_need_authorization_and_joint_verification() -> None:
    def provider(joint: bool) -> ProviderConfig:
        return ProviderConfig(
            adapter="gemini", model="m", credential_ref="r",
            declared_capabilities={m: True for m in JOINT},
            verified_capabilities={m: True for m in JOINT},
            verified_joint_capabilities=[JOINT] if joint else [],
        )
    route = RouteConfig(primary="a", fallbacks=[ProviderRef(provider="b", authorized=True),
                                                 ProviderRef(provider="c", authorized=True)])
    config = AppConfig(
        providers={"a": provider(True), "b": provider(True), "c": provider(False)},
        routes={"video_visual": route, "video_audio": route},
    )
    router = ProviderRouter(config)
    assert [c.provider_id for c in router.joint_fallbacks(JOINT)] == ["b"]
    assert cast(Any, router.joint_candidate(JOINT)).provider_id == "a"


def test_stage_links_outside_files_into_staging(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import cove_sensory_mcp.api as api

    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "elsewhere" / "clip.mp4"
    outside.parent.mkdir()
    outside.write_bytes(b"x")
    monkeypatch.setattr(api, "STAGING_ROOT", tmp_path / "staging")

    class Store:
        def load(self) -> AppConfig:
            return AppConfig(allowed_media_roots=[str(allowed)])

    services = cast(Any, type("S", (), {"config_store": Store()})())
    inside = allowed / "a.jpg"
    inside.write_bytes(b"y")
    assert api.stage(inside, services) == inside.resolve()
    staged = api.stage(outside, services)
    assert staged.parent.parent == tmp_path / "staging" and staged.read_bytes() == b"x"
    assert api.stage(outside, services) == staged        # stable path, reused
