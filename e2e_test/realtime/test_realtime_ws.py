"""E2E tests for the Realtime WebSocket proxy (/v1/realtime).

Tests the gateway's ability to proxy OpenAI Realtime API WebSocket sessions:
- Session lifecycle (connect, session.created, session.update)
- Text generation (single-turn and multi-turn conversations)
- Response cancellation mid-stream
- Response format validation (session.created, response.done, response.output_text.delta)
- Error handling (invalid events, missing model, missing auth)

Prerequisites:
- OPENAI_API_KEY environment variable set
- ``websockets`` pip package installed

Usage:
    pytest e2e_test/realtime/test_realtime_ws.py -v
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager

import pytest
import websockets

logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
REALTIME_MODEL = "gpt-realtime"
RECV_TIMEOUT = 30  # seconds — cloud latency can be high


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event(event_type: str, **payload) -> str:
    """Build a JSON event string."""
    return json.dumps({"type": event_type, **payload})


def _make_user_message(text: str) -> str:
    """Build a conversation.item.create event with a user text message."""
    return _make_event(
        "conversation.item.create",
        item={
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": text}],
        },
    )


def _parse_event(raw: str) -> dict | None:
    """Parse a JSON event, returning None on decode errors."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Non-JSON message: %s", raw[:200])
        return None


async def _recv_event(ws, *, event_type: str | None = None, timeout: float = RECV_TIMEOUT) -> dict:
    """Receive the next JSON event, optionally filtering by type."""
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise TimeoutError(f"Timed out waiting for event type={event_type}")
        raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
        event = _parse_event(raw)
        if event is None:
            continue
        if event_type is None or event.get("type") == event_type:
            return event
        logger.debug(
            "Skipping event %s while waiting for %s",
            event.get("type"),
            event_type,
        )


async def _collect_response_text(ws, *, timeout: float = RECV_TIMEOUT) -> str:
    """Collect text deltas until response.done, return the full text."""
    parts: list[str] = []
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise TimeoutError("Timed out waiting for response.done")
        raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
        event = _parse_event(raw)
        if event is None:
            continue
        etype = event.get("type", "")
        if etype == "response.output_text.delta" and event.get("delta"):
            parts.append(event["delta"])
        elif etype == "response.done":
            break
        elif etype == "error":
            raise RuntimeError(f"Upstream error: {json.dumps(event)}")
    return "".join(parts)


@asynccontextmanager
async def _realtime_session(ws_url: str, ws_headers: dict):
    """Connect, wait for session.created, configure text modality, yield ws."""
    async with websockets.connect(ws_url, additional_headers=ws_headers) as ws:
        await _recv_event(ws, event_type="session.created")
        await ws.send(
            _make_event(
                "session.update",
                session={"type": "realtime", "output_modalities": ["text"]},
            )
        )
        await _recv_event(ws, event_type="session.updated")
        yield ws


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gateway():
    """Launch a cloud gateway in OpenAI mode for realtime tests."""
    from infra import launch_cloud_gateway

    if not OPENAI_API_KEY:
        pytest.skip("OPENAI_API_KEY not set")

    gw = launch_cloud_gateway("openai", history_backend="memory")
    yield gw
    gw.shutdown()


@pytest.fixture()
def ws_url(gateway):
    """Build the realtime WebSocket URL."""
    return f"{gateway.ws_base_url}/v1/realtime?model={REALTIME_MODEL}"


@pytest.fixture()
def ws_headers():
    """Build the WebSocket connection headers."""
    return {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.vendor("openai")
@pytest.mark.gpu(0)
class TestRealtimeWebSocket:
    """E2E tests for the Realtime WebSocket proxy."""

    def test_session_created_on_connect(self, ws_url, ws_headers):
        """Connecting should receive a session.created event."""

        async def _run():
            async with websockets.connect(ws_url, additional_headers=ws_headers) as ws:
                event = await _recv_event(ws, event_type="session.created")
                assert event["type"] == "session.created"
                assert "session" in event
                assert event["session"].get("model") is not None
                logger.info("Session created: id=%s", event["session"].get("id"))

        asyncio.run(_run())

    def test_session_update(self, ws_url, ws_headers):
        """Sending session.update should receive session.updated."""

        async def _run():
            async with websockets.connect(ws_url, additional_headers=ws_headers) as ws:
                await _recv_event(ws, event_type="session.created")
                await ws.send(
                    _make_event(
                        "session.update",
                        session={"type": "realtime", "output_modalities": ["text"]},
                    )
                )
                event = await _recv_event(ws, event_type="session.updated")
                assert event["type"] == "session.updated"
                assert "session" in event
                assert event["session"].get("output_modalities") == ["text"]
                logger.info("Session updated successfully")

        asyncio.run(_run())

    def test_text_response(self, ws_url, ws_headers):
        """Full text round-trip: user message -> text response."""

        async def _run():
            async with _realtime_session(ws_url, ws_headers) as ws:
                await ws.send(_make_user_message("Say hello in one short sentence."))
                await ws.send(
                    _make_event(
                        "response.create",
                        response={"output_modalities": ["text"]},
                    )
                )

                text = await _collect_response_text(ws)
                assert len(text) > 0, "Expected non-empty text response"
                logger.info("Got text response: %s", text[:100])

        asyncio.run(_run())

    def test_multi_turn_conversation(self, ws_url, ws_headers):
        """Multi-turn: send two user messages, verify both get responses."""

        async def _run():
            async with _realtime_session(ws_url, ws_headers) as ws:
                # Turn 1
                await ws.send(_make_user_message("My name is Alice."))
                await ws.send(
                    _make_event("response.create", response={"output_modalities": ["text"]})
                )
                text1 = await _collect_response_text(ws)
                assert len(text1) > 0
                logger.info("Turn 1: %s", text1[:100])

                # Turn 2 — model should remember the name
                await ws.send(_make_user_message("What is my name?"))
                await ws.send(
                    _make_event("response.create", response={"output_modalities": ["text"]})
                )
                text2 = await _collect_response_text(ws)
                assert "alice" in text2.lower(), f"Expected 'Alice' in response, got: {text2}"
                logger.info("Turn 2: %s", text2[:100])

        asyncio.run(_run())

    def test_conversation_item_added_event(self, ws_url, ws_headers):
        """Sending conversation.item.create should echo conversation.item.added.

        GA renamed the legacy `conversation.item.created` event to
        `conversation.item.added` (emitted when an item is added to the default
        conversation).
        """

        async def _run():
            async with _realtime_session(ws_url, ws_headers) as ws:
                await ws.send(_make_user_message("Hi"))
                event = await _recv_event(ws, event_type="conversation.item.added")
                assert event["type"] == "conversation.item.added"
                assert event["item"]["role"] == "user"
                logger.info("conversation.item.added received: id=%s", event["item"].get("id"))

        asyncio.run(_run())

    def test_response_cancel(self, ws_url, ws_headers):
        """Cancelling a response mid-stream should produce response.done with cancelled status."""

        async def _run():
            async with _realtime_session(ws_url, ws_headers) as ws:
                await ws.send(
                    _make_user_message("Write a very long essay about the history of computing.")
                )
                await ws.send(
                    _make_event("response.create", response={"output_modalities": ["text"]})
                )

                # Wait for first delta to confirm streaming started
                await _recv_event(ws, event_type="response.output_text.delta")

                # Cancel mid-stream
                await ws.send(_make_event("response.cancel"))

                event = await _recv_event(ws, event_type="response.done")
                status = event.get("response", {}).get("status")
                assert status == "cancelled", f"Expected cancelled status, got: {status}"
                logger.info("Response cancelled successfully")

        asyncio.run(_run())

    def test_session_created_format(self, ws_url, ws_headers):
        """Validate session.created event has the expected schema."""

        async def _run():
            async with websockets.connect(ws_url, additional_headers=ws_headers) as ws:
                event = await _recv_event(ws, event_type="session.created")
                # Top-level fields
                assert "event_id" in event, "Missing event_id"
                assert event["type"] == "session.created"
                # Session object (GA shape)
                session = event["session"]
                assert isinstance(session, dict)
                assert isinstance(session.get("id"), str)
                assert len(session["id"]) > 0
                assert isinstance(session.get("model"), str)
                assert isinstance(session.get("output_modalities"), list)
                # In GA, voice/turn_detection moved under audio.{output,input}.
                audio = session.get("audio")
                assert isinstance(audio, dict), f"Expected session.audio dict, got: {audio!r}"
                output = audio.get("output") or {}
                assert isinstance(output.get("voice"), str)
                input_cfg = audio.get("input") or {}
                assert isinstance(input_cfg.get("turn_detection"), (dict, type(None)))
                logger.info(
                    "session.created schema OK: id=%s model=%s",
                    session["id"],
                    session["model"],
                )

        asyncio.run(_run())

    def test_response_done_format(self, ws_url, ws_headers):
        """Validate response.done event has the expected schema."""

        async def _run():
            async with _realtime_session(ws_url, ws_headers) as ws:
                await ws.send(_make_user_message("Say hi."))
                await ws.send(
                    _make_event("response.create", response={"output_modalities": ["text"]})
                )

                event = await _recv_event(ws, event_type="response.done")
                # Top-level
                assert "event_id" in event
                assert event["type"] == "response.done"
                # Response object
                resp = event["response"]
                assert isinstance(resp, dict)
                assert isinstance(resp.get("id"), str)
                assert resp.get("status") == "completed"
                assert isinstance(resp.get("output"), list)
                assert len(resp["output"]) > 0
                # Output item — GA shape uses content type "output_text".
                item = resp["output"][0]
                assert item.get("type") == "message"
                assert item.get("role") == "assistant"
                assert isinstance(item.get("content"), list)
                assert len(item["content"]) > 0
                content = item["content"][0]
                assert content.get("type") == "output_text"
                assert isinstance(content.get("text"), str)
                assert len(content["text"]) > 0
                # Usage
                usage = resp.get("usage")
                assert isinstance(usage, dict), f"Missing usage in response.done: {resp.keys()}"
                assert isinstance(usage.get("total_tokens"), int)
                assert usage["total_tokens"] > 0
                logger.info(
                    "response.done schema OK: id=%s tokens=%d",
                    resp["id"],
                    usage["total_tokens"],
                )

        asyncio.run(_run())

    def test_response_text_delta_format(self, ws_url, ws_headers):
        """Validate response.output_text.delta events have the expected schema."""

        async def _run():
            async with _realtime_session(ws_url, ws_headers) as ws:
                await ws.send(_make_user_message("Say hello."))
                await ws.send(
                    _make_event("response.create", response={"output_modalities": ["text"]})
                )

                # Collect a few deltas and validate schema
                delta_count = 0
                deadline = asyncio.get_running_loop().time() + RECV_TIMEOUT
                while True:
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        break
                    raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
                    event = _parse_event(raw)
                    if event is None:
                        continue
                    if event.get("type") == "response.output_text.delta":
                        assert "event_id" in event
                        assert isinstance(event.get("delta"), str)
                        assert len(event["delta"]) > 0
                        assert isinstance(event.get("response_id"), str)
                        assert isinstance(event.get("item_id"), str)
                        assert isinstance(event.get("output_index"), int)
                        assert isinstance(event.get("content_index"), int)
                        delta_count += 1
                    elif event.get("type") == "response.done":
                        break

                assert delta_count > 0, "Expected at least one response.output_text.delta"
                logger.info("response.output_text.delta schema OK: %d deltas received", delta_count)

        asyncio.run(_run())

    def test_invalid_event_returns_error(self, ws_url, ws_headers):
        """Sending an unknown event type should return an error event."""

        async def _run():
            async with websockets.connect(ws_url, additional_headers=ws_headers) as ws:
                await _recv_event(ws, event_type="session.created")
                await ws.send(json.dumps({"type": "totally.bogus.event"}))
                event = await _recv_event(ws, event_type="error")
                assert event["type"] == "error"
                logger.info("Error received: %s", event.get("error", {}).get("message", ""))

        asyncio.run(_run())

    def test_missing_model_returns_error(self, gateway, ws_headers):
        """Connecting without ?model= should fail with a reject status."""

        async def _run():
            url = f"{gateway.ws_base_url}/v1/realtime"
            with pytest.raises(websockets.exceptions.InvalidStatus):
                async with websockets.connect(url, additional_headers=ws_headers):
                    pass

        asyncio.run(_run())

    def test_missing_auth_returns_error(self, gateway):
        """Connecting without Authorization header should fail."""

        async def _run():
            url = f"{gateway.ws_base_url}/v1/realtime?model={REALTIME_MODEL}"
            try:
                async with websockets.connect(url) as ws:
                    # Connection accepted — upstream must still send an error event
                    event = await _recv_event(ws, event_type="error", timeout=10)
                    assert event["type"] == "error", f"Expected error event, got: {event}"
            except websockets.exceptions.InvalidStatus:
                pass  # Handshake rejected — expected

        asyncio.run(_run())
