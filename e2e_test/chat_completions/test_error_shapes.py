"""Error responses to streaming requests keep the worker's JSON shape.

When ``stream: true`` is set and the worker rejects the request before
generation starts, it answers with a JSON error and an ``application/json``
content type. The gateway used to relay the status and body but relabel the
response as ``text/event-stream``, so an SSE client could drop the error body
(#2404). The upstream content type must survive; only a successful stream is
an event stream.

The rejection is the worker's, so each engine supplies its own. On SGLang a
tool whose ``parameters`` is not a valid JSON Schema: the gateway does not
validate tool schemas, so the request reaches the worker, and SGLang's OpenAI
server rejects it before opening a stream. On the mock engine the realistic
worker's fault hook answers the request with a 400 and a JSON body, which
puts the relay's rule on every pull request's CPU row.

Usage:
    E2E_RUNTIME=sglang E2E_ENGINE=sglang pytest e2e_test/chat_completions/test_error_shapes.py -v
    E2E_RUNTIME=mock E2E_ENGINE=mock pytest e2e_test/chat_completions/test_error_shapes.py -v
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator

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


def _streaming_chat(model: str, **extra) -> dict:
    return {
        "model": model,
        "stream": True,
        "messages": [{"role": "user", "content": "Look up the weather."}],
        "max_tokens": 16,
        **extra,
    }


class _StreamingErrorShapeBehavior:
    """Worker-side rejections of a streaming request stay JSON; real streams stay SSE.

    A class per engine below supplies the rejection: ``rejection`` yields the
    request body the worker rejects, armed for as long as the block runs, and
    ``rejection_word`` is a word the worker's error message carries.
    """

    rejection_word: str

    @contextlib.contextmanager
    def rejection(self, model: str, mock_answers) -> Iterator[dict]:
        raise NotImplementedError

    def test_worker_rejection_keeps_json_content_type(self, setup_backend, mock_answers):
        _, model, _, gateway = setup_backend
        with self.rejection(model, mock_answers) as body:
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
        assert self.rejection_word in message.lower(), f"unexpected error body: {error}"

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


@pytest.mark.engine("sglang")
@pytest.mark.gpu(1)
@pytest.mark.e2e
@pytest.mark.model("meta-llama/Llama-3.2-1B-Instruct")
@pytest.mark.parametrize("setup_backend", ["http"], indirect=True)
class TestStreamingErrorShape(_StreamingErrorShapeBehavior):
    """SGLang's OpenAI server rejects a tool whose parameters are not a JSON Schema."""

    rejection_word = "schema"

    @contextlib.contextmanager
    def rejection(self, model: str, mock_answers) -> Iterator[dict]:
        yield _streaming_chat(model, tools=[_INVALID_SCHEMA_TOOL])


@pytest.mark.engine("mock")
@pytest.mark.gpu(1)
@pytest.mark.e2e
@pytest.mark.model("RedHatAI/Llama-3.2-1B-Instruct-FP8")
@pytest.mark.workers(extra_engine_args=["--engine", "realistic"])
@pytest.mark.parametrize("setup_backend", ["http"], indirect=True)
class TestStreamingErrorShapeMock(_StreamingErrorShapeBehavior):
    """The mock worker's fault hook answers every request with a 400 and its
    JSON error body while armed (``fail?status=400``, cleared after the turn),
    so the relay's rule runs on the CPU row of every pull request."""

    rejection_word = "injected"

    @contextlib.contextmanager
    def rejection(self, model: str, mock_answers) -> Iterator[dict]:
        _arm(mock_answers, status=400)
        try:
            yield _streaming_chat(model)
        finally:
            _arm(mock_answers, status=0)


def _arm(mock_answers, **params) -> None:
    """The class's mock workers' ``fail`` hook with the given query (``status=0`` clears)."""
    for worker in mock_answers.workers:
        response = httpx.post(
            f"{worker.admin_url}/admin/fault/all/fail", params=params, timeout=10.0
        )
        response.raise_for_status()
