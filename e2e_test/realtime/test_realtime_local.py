"""E2E test: Realtime WebSocket proxy through the HTTP router to a LOCAL worker.

Companion to ``test_realtime_ws.py`` (which drives a cloud OpenAI gateway). This
exercises the **HTTP router** path: the gateway proxies a realtime WebSocket
session to a locally-hosted vLLM worker serving ``Qwen/Qwen3-ASR-1.7B`` over
``ws /v1/realtime`` (streaming speech-to-text). The worker is registered with the
``realtime`` label so the HTTP router's capability gate selects it.

Protocol (vLLM realtime transcription):
    session.created -> session.update{model} ->
    input_audio_buffer.append/commit -> transcription.delta/done

Prerequisites:
- A GPU that can serve Qwen3-ASR with the realtime architecture.
- ``websockets`` pip package.

Usage:
    pytest e2e_test/realtime/test_realtime_local.py -v
    scripts/e2e-test.sh realtime
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
import wave
from pathlib import Path

import pytest
import websockets
import websockets.exceptions  # submodule isn't auto-exposed on the top-level module

logger = logging.getLogger(__name__)

MODEL = "Qwen/Qwen3-ASR-1.7B"
AUDIO_WAV = Path(__file__).parent / "fixtures" / "mary_had_lamb_16k.wav"
RECV_TIMEOUT = 60  # local ASR: model warmup + streaming latency
AUDIO_CHUNK = 4096  # bytes of PCM16 per input_audio_buffer.append


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _wait_for_model(gw, model: str, timeout: float = 180.0) -> None:
    """Block until the gateway reports ``model`` in GET /v1/models."""
    deadline = time.perf_counter() + timeout
    seen: list[str] = []
    while time.perf_counter() < deadline:
        seen = [m.get("id") for m in gw.list_models()]
        if model in seen:
            return
        time.sleep(2.0)
    raise TimeoutError(f"model {model} not registered within {timeout}s; saw {seen}")


@pytest.fixture(scope="module")
def realtime_gateway():
    """Launch a local vLLM Qwen3-ASR realtime worker + an IGW gateway, wired
    with the ``realtime`` label so the HTTP router routes realtime to it."""
    from infra import ConnectionMode, Gateway, start_workers, stop_workers

    workers = start_workers(MODEL, engine="vllm", mode=ConnectionMode.HTTP, count=1)
    gw = Gateway()
    try:
        gw.start(igw_mode=True)
        ok, info = gw.add_worker(
            workers[0].base_url, labels={"realtime": "true"}, ready_timeout=180.0
        )
        assert ok, f"failed to register realtime worker: {info}"
        _wait_for_model(gw, MODEL)
        yield gw
    finally:
        gw.shutdown()
        stop_workers(workers)


@pytest.fixture()
def ws_url(realtime_gateway):
    return f"{realtime_gateway.ws_base_url}/v1/realtime?model={MODEL}"


@pytest.fixture()
def ws_headers():
    # The HTTP-router realtime handler requires a bearer token to forward
    # upstream; the local worker has no api_key and ignores it, so any token works.
    return {"Authorization": "Bearer local-e2e"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_pcm16(path: Path) -> bytes:
    """Read a 16 kHz mono PCM16 WAV into raw little-endian bytes."""
    with wave.open(str(path), "rb") as w:
        assert w.getnchannels() == 1, "fixture must be mono"
        assert w.getframerate() == 16000, "fixture must be 16 kHz"
        assert w.getsampwidth() == 2, "fixture must be PCM16"
        return w.readframes(w.getnframes())


async def _recv(ws, timeout: float = RECV_TIMEOUT) -> dict:
    return json.loads(await asyncio.wait_for(ws.recv(), timeout=timeout))


async def _transcribe(url: str, headers: dict) -> str:
    """Stream the fixture audio and return the final transcription text."""
    pcm = _load_pcm16(AUDIO_WAV)
    async with websockets.connect(url, additional_headers=headers, max_size=None) as ws:
        created = await _recv(ws)
        assert created.get("type") == "session.created", created

        await ws.send(json.dumps({"type": "session.update", "model": MODEL}))
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        for i in range(0, len(pcm), AUDIO_CHUNK):
            chunk = base64.b64encode(pcm[i : i + AUDIO_CHUNK]).decode()
            await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": chunk}))
        await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))

        text = ""
        deadline = asyncio.get_running_loop().time() + RECV_TIMEOUT
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TimeoutError("timed out waiting for transcription.done")
            event = json.loads(await asyncio.wait_for(ws.recv(), timeout=remaining))
            etype = event.get("type")
            if etype == "transcription.delta":
                text += event.get("delta", "")
            elif etype == "transcription.done":
                return event.get("text", text)
            elif etype == "error":
                raise RuntimeError(f"realtime error: {json.dumps(event)}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.gpu(1)
@pytest.mark.engine("vllm")
class TestRealtimeLocalWebSocket:
    """Realtime WS transcription through the HTTP router to a local vLLM worker."""

    def test_session_created_on_connect(self, ws_url, ws_headers):
        """Connecting should upgrade and yield a session.created event."""

        async def _run():
            async with websockets.connect(ws_url, additional_headers=ws_headers) as ws:
                event = await _recv(ws)
                assert event.get("type") == "session.created", event

        asyncio.run(_run())

    def test_transcription_round_trip(self, ws_url, ws_headers):
        """Full round-trip: stream audio -> transcription text back through smg.

        This validates the HTTP router's realtime proxy, so it asserts a
        non-empty transcription is streamed back. The exact wording is left to
        the backend ASR model (it varies by model revision / vLLM build and even
        emits raw format markers on some builds), so it is logged, not asserted.
        """
        text = asyncio.run(_transcribe(ws_url, ws_headers))
        logger.info("transcription: %s", text)
        assert text.strip(), "expected a non-empty transcription"

    def test_missing_auth_rejected(self, ws_url):
        """No Authorization header -> the gateway rejects the upgrade (401)."""

        async def _run():
            with pytest.raises(websockets.exceptions.InvalidStatus):
                async with websockets.connect(ws_url):
                    pass

        asyncio.run(_run())

    def test_missing_model_rejected(self, realtime_gateway, ws_headers):
        """No ?model= query -> the gateway rejects the upgrade."""

        async def _run():
            url = f"{realtime_gateway.ws_base_url}/v1/realtime"
            with pytest.raises(websockets.exceptions.InvalidStatus):
                async with websockets.connect(url, additional_headers=ws_headers):
                    pass

        asyncio.run(_run())
