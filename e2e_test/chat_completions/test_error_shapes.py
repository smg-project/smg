"""Error responses to streaming requests keep the worker's JSON shape.

When ``stream: true`` is set and the worker rejects the request before
generation starts, it answers with a JSON error and an ``application/json``
content type. The gateway used to relay the status and body but relabel the
response as ``text/event-stream``, so an SSE client could drop the error body
(#2404). The upstream content type must survive; only a successful stream is
an event stream.

The rejection is triggered by a tool whose ``parameters`` is not a valid JSON
Schema. The gateway does not validate tool schemas, so the request reaches the
worker, and SGLang's OpenAI server rejects it before opening a stream.

Usage:
    E2E_RUNTIME=sglang pytest e2e_test/chat_completions/test_error_shapes.py -v
"""

from __future__ import annotations

import logging

import httpx
import pytest

logger = logging.getLogger(__name__)

_INVALID_SCHEMA_TOOL = {
    "type": "function",
    "function": {
        "name": "lookup",
        "description": "A tool with a parameters block that is not a JSON Schema.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "not-a-json-schema-type"}},
        },
    },
}


@pytest.mark.engine("sglang")
@pytest.mark.gpu(1)
@pytest.mark.e2e
@pytest.mark.model("meta-llama/Llama-3.2-1B-Instruct")
@pytest.mark.parametrize("setup_backend", ["http"], indirect=True)
class TestStreamingErrorShape:
    """Worker-side rejections of a streaming request stay JSON; real streams stay SSE."""

    def test_worker_rejection_keeps_json_content_type(self, setup_backend):
        _, model, _, gateway = setup_backend
        body = {
            "model": model,
            "stream": True,
            "messages": [{"role": "user", "content": "Look up the weather."}],
            "tools": [_INVALID_SCHEMA_TOOL],
            "max_tokens": 16,
        }

        resp = httpx.post(f"{gateway.base_url}/v1/chat/completions", json=body, timeout=60.0)

        logger.info(
            "status=%s content-type=%s body=%s",
            resp.status_code,
            resp.headers.get("content-type"),
            resp.text[:300],
        )
        assert resp.status_code == 400, (
            f"expected the worker's 400, got {resp.status_code}: {resp.text[:300]}"
        )
        content_type = resp.headers.get("content-type", "")
        assert content_type.startswith("application/json"), (
            f"error content type was relabelled: {content_type!r}"
        )
        error = resp.json()
        message = error.get("message") or error.get("error", {}).get("message", "")
        assert "schema" in message.lower(), f"unexpected error body: {error}"

    def test_successful_stream_is_an_event_stream(self, setup_backend):
        _, model, _, gateway = setup_backend
        body = {
            "model": model,
            "stream": True,
            "messages": [{"role": "user", "content": "Say hello."}],
            "max_tokens": 8,
        }

        with httpx.stream(
            "POST", f"{gateway.base_url}/v1/chat/completions", json=body, timeout=60.0
        ) as resp:
            assert resp.status_code == 200, resp.read()[:300]
            assert resp.headers.get("content-type", "").startswith("text/event-stream")
            frames = [line for line in resp.iter_lines() if line.startswith("data:")]

        assert frames, "stream carried no data frames"
        assert frames[-1].strip() == "data: [DONE]", f"stream did not terminate: {frames[-1]!r}"
