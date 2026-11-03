"""A panic inside the gateway's reader is that one request's error, never a cut stream.

The reader runs over the model's output. When it panicked, the gateway used
to lose the request's task: a streamed response ended without ``[DONE]`` or
an error event (a cut stream), and a non-streamed request lost its connection
without an answer. The panic is now caught at the request's boundary: the
stream ends with an error event under ``reader_panicked`` followed by
``[DONE]``, the non-streamed request is a 500 under the same code, and the
gateway counts the output under ``smg_symphony_outputs_total`` with
``outcome="panicked"``.

The panic is driven through the gateway's test trap: with
``SMG_TEST_READER_PANIC_ON=<marker>`` in its environment the reader panics on
an output that carries the marker, and the mock engine answers with one. The
requests ask for Symphony by header, so the reader runs whatever parser the
lane starts the gateway with.

Usage:
    E2E_RUNTIME=mock pytest e2e_test/chat_completions/test_reader_panic.py -v
"""

from __future__ import annotations

import json
import logging

import httpx
import pytest

logger = logging.getLogger(__name__)

# The text the trap springs on: no model's answer carries it by chance.
MARKER = "TRAP-7f3a-READER"
ANSWER = f"The forecast is {MARKER} and sunny."
SYMPHONY = {"x-smg-parser": "symphony"}


def _request(model: str, stream: bool) -> dict:
    return {
        "model": model,
        "stream": stream,
        "messages": [{"role": "user", "content": "What is the forecast?"}],
        "max_tokens": 32,
    }


def _panicked_outputs(gateway) -> int:
    """The gateway's count of outputs whose reading panicked, every table."""
    scrape = httpx.get(f"{gateway.metrics_url}/metrics", timeout=10.0).text
    total = 0
    for line in scrape.splitlines():
        if line.startswith("smg_symphony_outputs_total{") and 'outcome="panicked"' in line:
            total += int(float(line.rsplit(" ", 1)[1]))
    return total


@pytest.mark.engine("mock")
@pytest.mark.gpu(1)
@pytest.mark.model("RedHatAI/Llama-3.2-1B-Instruct-FP8")
@pytest.mark.gateway(
    extra_args=["--tool-call-parser", "llama"],
    # The trap armed; the panic hook without a backtrace, since symbolizing one
    # of a debug build takes tens of seconds on a loaded host and the hook's
    # cost is not what this class measures.
    env={"SMG_TEST_READER_PANIC_ON": MARKER, "RUST_BACKTRACE": "0"},
)
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestReaderPanicIsolation:
    """A reader panic ends its own request with an error, on both paths."""

    def test_streamed_reader_panic_ends_with_an_error_event_and_done(
        self, setup_backend, mock_answers
    ):
        _, model, _, gateway = setup_backend
        before = _panicked_outputs(gateway)
        # One engine chunk carries the whole answer, so one step of the reader sees the marker.
        mock_answers.enqueue(ANSWER, chunk_tokens=64)

        with httpx.stream(
            "POST",
            f"{gateway.base_url}/v1/chat/completions",
            json=_request(model, stream=True),
            headers=SYMPHONY,
            timeout=60.0,
        ) as resp:
            assert resp.status_code == 200, resp.read()
            assert resp.headers.get("x-smg-parser") == "symphony", dict(resp.headers)
            payloads = [
                line[len("data: ") :] for line in resp.iter_lines() if line.startswith("data: ")
            ]
        logger.info("streamed payloads: %s", payloads)

        # The terminal frames: the error event under its code, then [DONE].
        assert payloads, "no frames at all"
        assert payloads[-1].strip() == "[DONE]", payloads
        error = json.loads(payloads[-2])["error"]
        assert error["code"] == "reader_panicked", error
        assert error["type"] == "internal_error", error
        assert "panicked" in error["message"], error
        assert MARKER not in error["message"], error
        assert _panicked_outputs(gateway) == before + 1

    def test_non_streamed_reader_panic_is_a_500_under_its_code(self, setup_backend, mock_answers):
        _, model, _, gateway = setup_backend
        before = _panicked_outputs(gateway)
        mock_answers.enqueue(ANSWER)

        resp = httpx.post(
            f"{gateway.base_url}/v1/chat/completions",
            json=_request(model, stream=False),
            headers=SYMPHONY,
            timeout=60.0,
        )
        logger.info("non-streamed response: %s %s", resp.status_code, resp.text)

        # The trap lives in the Symphony reader, so the code says which parser read the answer.
        assert resp.status_code == 500, resp.text
        assert resp.headers.get("x-smg-error-code") == "reader_panicked", dict(resp.headers)
        error = resp.json()["error"]
        assert error["code"] == "reader_panicked", error
        assert "panicked" in error["message"], error
        assert MARKER not in error["message"], error
        assert _panicked_outputs(gateway) == before + 1

    def test_an_answer_without_the_marker_is_read_as_before(self, setup_backend, mock_answers):
        _, model, _, gateway = setup_backend
        before = _panicked_outputs(gateway)
        mock_answers.enqueue("The forecast is sunny.")

        resp = httpx.post(
            f"{gateway.base_url}/v1/chat/completions",
            json=_request(model, stream=False),
            headers=SYMPHONY,
            timeout=60.0,
        )

        assert resp.status_code == 200, resp.text
        assert resp.json()["choices"][0]["message"]["content"] == "The forecast is sunny."
        assert _panicked_outputs(gateway) == before
