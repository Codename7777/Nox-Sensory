"""Nox fork: multimodal_inline wire mode (images, frame-sampled video, joint video)."""

from __future__ import annotations

import base64
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from cove_sensory_mcp.config.schema import (
    AdapterOptions,
    AppConfig,
    ProviderConfig,
)
from cove_sensory_mcp.config.secrets import MemorySecretStore
from cove_sensory_mcp.config.store import ConfigStore
from cove_sensory_mcp.models import DetailLevel, Modality
from cove_sensory_mcp.providers.base import (
    MediaKind,
    PreparedMedia,
    ProviderCallResult,
    ProviderRequest,
)
from cove_sensory_mcp.providers.inline_media import FrameSampling, frame_times
from cove_sensory_mcp.providers.openai_compatible import OpenAICompatibleProvider
from cove_sensory_mcp.providers.registry import ProviderRegistry
from cove_sensory_mcp.reports.schemas import ObservationEnvelope
from cove_sensory_mcp.verification.assets import SelfTestAssetStore
from cove_sensory_mcp.verification.verifier import CapabilityVerifier

ASSETS = Path(__file__).resolve().parents[2] / "src/cove_sensory_mcp/assets/self_test"
JOINT = frozenset({Modality.VIDEO_VISUAL, Modality.VIDEO_AUDIO})
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg missing")


def _observation(modality: Modality) -> dict[str, object]:
    return {
        "modality": modality.value,
        "summary": "A red ball moves left to right.",
        "segments": [{"start_seconds": 0, "end_seconds": 1, "text": "ball"}],
        "transcript": [],
        "warnings": [],
        "confidence": "high",
    }


def _reply(modalities: frozenset[Modality]) -> dict[str, object]:
    text = json.dumps(
        {"observations": [_observation(m) for m in Modality if m in modalities]}
    )
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


class _Body(httpx.AsyncByteStream):
    def __init__(self, body: bytes) -> None:
        self.body = body

    async def __aiter__(self):  # type: ignore[override]
        yield self.body


def _config(**options: Any) -> ProviderConfig:
    return ProviderConfig(
        adapter="openai-compatible",
        base_url="http://127.0.0.1:8901/sensory",
        model="relay-model",
        credential_ref="relay-credential",
        declared_capabilities={modality: True for modality in Modality},
        adapter_options=AdapterOptions(
            endpoint_path="/chat/completions",
            media_part_mode="multimodal_inline",
            **options,
        ),
    )


def _provider(
    captured: list[dict[str, Any]], config: ProviderConfig | None = None
) -> OpenAICompatibleProvider:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured.append(body)
        prompt = body["messages"][0]["content"][-1]["text"]
        wanted = frozenset(
            m for m in Modality if f'"{m.value}"' in prompt.split("\n")[2]
        )
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=_Body(json.dumps(_reply(wanted)).encode()),
        )

    store = MemorySecretStore()
    store.set("relay-credential", "relay-placeholder")
    return OpenAICompatibleProvider(
        provider_id="relay",
        config=config or _config(),
        secret_store=store,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def _request(
    path: Path,
    kind: MediaKind,
    mime: str,
    modalities: frozenset[Modality],
    duration: float | None = None,
) -> ProviderRequest:
    return ProviderRequest(
        media=PreparedMedia(path, mime, kind, duration),
        requested_modalities=modalities,
        question="",
        detail=DetailLevel.QUICK,
        language="en",
    )


def test_frame_times_cover_the_excerpt_evenly() -> None:
    assert frame_times(60, FrameSampling()) == [
        2.5, 7.5, 12.5, 17.5, 22.5, 27.5, 32.5, 37.5, 42.5, 47.5, 52.5, 57.5
    ]
    long = frame_times(600, FrameSampling())
    assert len(long) == 24 and long[0] == 12.5 and long[-1] == 587.5
    assert len(frame_times(2, FrameSampling())) == 4
    assert frame_times(None, FrameSampling()) == [0.0]
    assert len(frame_times(30, FrameSampling(interval_seconds=1, max_frames=10))) == 10


def test_inline_mode_rejects_unknown_frame_options() -> None:
    with pytest.raises(ValueError):
        AdapterOptions(video_max_frames=0)
    with pytest.raises(ValueError):
        AdapterOptions(video_frame_width=10)


@pytest.mark.asyncio
async def test_image_is_sent_as_image_url_data_uri() -> None:
    captured: list[dict[str, Any]] = []
    image = ASSETS / "shape.png"
    result = await _provider(captured).sense(
        _request(image, MediaKind.IMAGE, "image/png", frozenset({Modality.IMAGE}))
    )
    content = captured[0]["messages"][0]["content"]
    assert content[0]["type"] == "image_url"
    url = content[0]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.partition(",")[2]) == image.read_bytes()
    assert set(result.observations) == {Modality.IMAGE}


@pytest.mark.asyncio
async def test_audio_mp3_uses_mp3_input_audio_format(tmp_path: Path) -> None:
    captured: list[dict[str, Any]] = []
    clip = tmp_path / "a.mp3"
    clip.write_bytes(b"ID3fake")
    await _provider(captured).sense(
        _request(clip, MediaKind.AUDIO, "audio/mpeg", frozenset({Modality.AUDIO}))
    )
    part = captured[0]["messages"][0]["content"][0]
    assert part["type"] == "input_audio"
    assert part["input_audio"]["format"] == "mp3"


@needs_ffmpeg
@pytest.mark.asyncio
async def test_joint_video_sends_timed_frames_and_audio_in_one_request() -> None:
    captured: list[dict[str, Any]] = []
    provider = _provider(captured)
    assert provider.supports_joint(JOINT)
    result = await provider.sense(
        _request(ASSETS / "motion.mp4", MediaKind.VIDEO, "video/mp4", JOINT, 2.0)
    )
    assert len(captured) == 1
    content = captured[0]["messages"][0]["content"]
    kinds = [part["type"] for part in content]
    assert kinds.count("image_url") == 4
    assert kinds.count("input_audio") == 1
    assert kinds.index("input_audio") > max(
        i for i, kind in enumerate(kinds) if kind == "image_url"
    )
    assert content[0]["text"].startswith("VIDEO INPUT")
    labels = [p["text"] for p in content if p["type"] == "text" and p["text"].startswith("[frame")]
    assert labels[0] == "[frame 1/4 at t=0.2s]" and labels[-1] == "[frame 4/4 at t=1.8s]"
    for part in content:
        if part["type"] == "image_url":
            assert part["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert content[kinds.index("input_audio")]["input_audio"]["format"] == "mp3"
    assert set(result.observations) == JOINT


@needs_ffmpeg
@pytest.mark.asyncio
async def test_visual_only_video_sends_no_audio_and_respects_frame_options() -> None:
    captured: list[dict[str, Any]] = []
    config = _config(video_max_frames=2, video_frame_width=160)
    await _provider(captured, config).sense(
        _request(
            ASSETS / "motion.mp4",
            MediaKind.VIDEO,
            "video/mp4",
            frozenset({Modality.VIDEO_VISUAL}),
            2.0,
        )
    )
    kinds = [part["type"] for part in captured[0]["messages"][0]["content"]]
    assert kinds.count("image_url") == 2
    assert "input_audio" not in kinds


@needs_ffmpeg
@pytest.mark.asyncio
async def test_video_audio_from_video_file_extracts_the_track() -> None:
    captured: list[dict[str, Any]] = []
    await _provider(captured).sense(
        _request(
            ASSETS / "motion.mp4",
            MediaKind.VIDEO,
            "video/mp4",
            frozenset({Modality.VIDEO_AUDIO}),
            2.0,
        )
    )
    kinds = [part["type"] for part in captured[0]["messages"][0]["content"]]
    assert "image_url" not in kinds
    assert kinds.count("input_audio") == 1


@needs_ffmpeg
@pytest.mark.asyncio
async def test_frames_respect_the_size_limit() -> None:
    from cove_sensory_mcp.errors import ErrorCode, SensoryError

    captured: list[dict[str, Any]] = []
    with pytest.raises(SensoryError) as caught:
        await _provider(captured, _config(inline_max_bytes=100)).sense(
            _request(ASSETS / "motion.mp4", MediaKind.VIDEO, "video/mp4", JOINT, 2.0)
        )
    assert caught.value.code is ErrorCode.MEDIA_TOO_LARGE
    assert captured == []


def test_other_modes_do_not_claim_joint_support() -> None:
    config = ProviderConfig(
        adapter="openai-compatible",
        base_url="http://127.0.0.1:8901/sensory",
        model="relay-model",
        credential_ref="relay-credential",
        declared_capabilities={Modality.AUDIO: True},
        adapter_options=AdapterOptions(
            endpoint_path="/chat/completions", media_part_mode="input_audio_base64"
        ),
    )
    provider = OpenAICompatibleProvider(
        provider_id="relay", config=config, secret_store=MemorySecretStore()
    )
    assert not provider.supports_joint(JOINT)


class _JointFake:
    def __init__(self, *, joint_ok: bool) -> None:
        self.joint_ok = joint_ok
        self.calls: list[frozenset[Modality]] = []

    def supports_joint(self, modalities: frozenset[Modality]) -> bool:
        return modalities == JOINT

    async def sense(self, request: ProviderRequest) -> ProviderCallResult:
        self.calls.append(request.requested_modalities)
        wanted = request.requested_modalities
        if len(wanted) > 1 and not self.joint_ok:
            wanted = frozenset({Modality.VIDEO_VISUAL})
        return ProviderCallResult(
            observations={
                m: ObservationEnvelope.model_validate(_observation(m)) for m in wanted
            },
            provider_id="relay",
            model="relay-model",
            remote_file_deleted=None,
        )


@pytest.mark.parametrize("joint_ok", [True, False])
@pytest.mark.asyncio
async def test_verifier_records_joint_video_only_when_the_joint_call_passes(
    tmp_path: Path, joint_ok: bool
) -> None:
    store = ConfigStore(tmp_path / "config.yaml", jobs_dir=tmp_path / "jobs")
    store.save(AppConfig(providers={"relay": _config()}))
    fake = _JointFake(joint_ok=joint_ok)
    verifier = CapabilityVerifier(
        config_store=store,
        registry=ProviderRegistry({"relay": fake}),
        assets=SelfTestAssetStore.packaged(),
        now=lambda: datetime(2026, 11, 24, tzinfo=UTC),
    )
    results = await verifier.verify(
        "relay", [Modality.IMAGE, Modality.VIDEO_VISUAL, Modality.VIDEO_AUDIO]
    )
    assert all(result.verified for result in results)
    assert fake.calls[-1] == JOINT
    saved = store.load().providers["relay"]
    assert (JOINT in saved.verified_joint_capabilities) is joint_ok


@needs_ffmpeg
@pytest.mark.asyncio
async def test_frame_fallback_reads_a_clip_of_an_untrimmed_source() -> None:
    captured: list[dict[str, Any]] = []
    request = ProviderRequest(
        media=PreparedMedia(
            ASSETS / "motion.mp4", "video/mp4", MediaKind.VIDEO, 1.0,
            clip_start=0.5, clip_end=1.5, reusable=True,
        ),
        requested_modalities=JOINT,
        question="",
        detail=DetailLevel.QUICK,
        language="en",
        start_seconds=0.5,
        end_seconds=1.5,
    )
    provider = _provider(captured)
    assert provider.supports_clip_ranges
    await provider.sense(request)
    content = captured[0]["messages"][0]["content"]
    assert "begins at 0.5 s of the source" in content[0]["text"]
    labels = [p["text"] for p in content if p["type"] == "text" and p["text"].startswith("[frame")]
    assert labels[0] == "[frame 1/4 at t=0.1s]"          # excerpt-relative labels
    assert [p["type"] for p in content].count("input_audio") == 1
