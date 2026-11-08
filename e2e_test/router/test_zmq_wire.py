"""The gateway's ZMQ engine wire, driven end to end on the CPU row.

The scripted mock worker's ZMQ ranks speak the wire the gateway's ZMQ adapter
reads (``mock-worker --zmq-handshake … --zmq-count 1``): the gateway binds the
sockets and the rank dials in, as an engine does, so the harness pins the
gateway to that wire's backend. One class runs the rank's default shape, one
output message per token; the other has it send an engine's steps
(``--zmq-step-ms 5``: one message per five-millisecond step carrying every
running request's new token), so the adapter's reading of a message of many
outputs runs on every push, not only when a card lane with an engine does. Each class checks the
worker registered on the ZMQ wire and streams eight concurrent turns, each
with its own scripted answer, that must all come back whole, token by token;
the rank's output counts (`GET /admin/zmq/zmq:0`) then show the shape that
carried them: one output per message, or steps of several.

Usage:
    E2E_RUNTIME=mock E2E_ENGINE=mock pytest e2e_test/router/test_zmq_wire.py -v
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
from infra.streams import collect_streamed_chat

_MODEL = "RedHatAI/Llama-3.2-1B-Instruct-FP8"
_STREAMS = 8
_WORDS = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima".split()
# Eight answers of forty words, each filed under the prompt that asks for it;
# five milliseconds per token, the stepping class's step period, so the eight
# advance together as an engine's running requests do.
_ANSWERS = {
    i: f"Stream {i}: " + " ".join(_WORDS[(i + k) % len(_WORDS)] for k in range(40)) + "."
    for i in range(_STREAMS)
}
_TOKEN_DELAY_MS = 5


def _prompt(i: int) -> str:
    return f"Count to {i + 50} slowly."


def _output_counts(worker) -> tuple[int, int]:
    """The rank's messages that carried request outputs, and the outputs in them."""
    resp = httpx.get(f"{worker.admin_url}/admin/zmq/zmq:0", timeout=5.0)
    assert resp.status_code == 200, f"GET /admin/zmq/zmq:0 returned {resp.status_code}: {resp.text}"
    counts = resp.json()
    return counts["batch_messages"], counts["outputs"]


class _ZmqWireTurns:
    # Whether the class's rank sends steps of several outputs per message.
    steps: bool

    def test_the_worker_registered_on_the_zmq_wire(self, setup_backend):
        _, _, _, gateway = setup_backend
        resp = httpx.get(f"{gateway.base_url}/workers", timeout=5.0)
        assert resp.status_code == 200, f"GET /workers returned {resp.status_code}: {resp.text}"
        workers = resp.json().get("workers")
        assert workers and len(workers) == 1, f"one ZMQ worker expected: {resp.text}"
        assert workers[0]["url"].startswith("ipc://"), workers[0]
        assert workers[0].get("connection_mode") == "zmq", workers[0]
        assert workers[0].get("runtime_type") == "vllm", workers[0]

    def test_eight_concurrent_streams_come_back_whole(self, setup_backend, mock_answers):
        _, model_path, client, _ = setup_backend
        worker = mock_answers.workers[0]
        messages_before, outputs_before = _output_counts(worker)
        for i, answer in _ANSWERS.items():
            mock_answers.enqueue(answer, prompt_contains=_prompt(i), delay_ms=_TOKEN_DELAY_MS)

        def turn(i: int):
            stream = client.chat.completions.create(
                model=model_path,
                messages=[{"role": "user", "content": _prompt(i)}],
                max_tokens=200,
                stream=True,
                stream_options={"include_usage": True},
            )
            return i, collect_streamed_chat(stream)

        with ThreadPoolExecutor(max_workers=_STREAMS) as pool:
            turns = list(pool.map(turn, range(_STREAMS)))
        for i, streamed in turns:
            assert streamed.content == _ANSWERS[i], f"stream {i} did not come back whole"
            assert streamed.finish_reason == "stop", (
                f"stream {i} ended with {streamed.finish_reason}"
            )
            content_chunks = [
                chunk
                for chunk in streamed.chunks
                if chunk.choices and chunk.choices[0].delta.content
            ]
            assert len(content_chunks) > 1, f"stream {i} came in one delta"
            usage_chunks = [chunk for chunk in streamed.chunks if chunk.usage is not None]
            assert len(usage_chunks) == 1, f"stream {i} carried {len(usage_chunks)} usage chunks"
            assert usage_chunks[0].usage.completion_tokens == len(content_chunks), (
                f"stream {i}: usage counts {usage_chunks[0].usage.completion_tokens} tokens, "
                f"{len(content_chunks)} deltas came"
            )
        # The shape that carried the tokens: every output in a message of its
        # own, or steps of several (the eight streams pace at the period, so a
        # step carries several of them; two per message is the floor asserted).
        messages, outputs = _output_counts(worker)
        messages -= messages_before
        outputs -= outputs_before
        assert outputs >= _STREAMS * 2, f"{outputs} outputs pushed for {_STREAMS} streams"
        if self.steps:
            assert outputs >= 2 * messages, (
                f"{outputs} outputs in {messages} messages: the rank did not send steps"
            )
        else:
            assert outputs == messages, (
                f"{outputs} outputs in {messages} messages: not one output per message"
            )


@pytest.mark.engine("mock")
@pytest.mark.gpu(1)
@pytest.mark.model(_MODEL)
@pytest.mark.parametrize("setup_backend", ["zmq"], indirect=True)
class TestZmqWire(_ZmqWireTurns):
    """The rank's default shape: one output message per token."""

    steps = False


@pytest.mark.engine("mock")
@pytest.mark.gpu(1)
@pytest.mark.model(_MODEL)
@pytest.mark.workers(extra_engine_args=["--zmq-step-ms", str(_TOKEN_DELAY_MS)])
@pytest.mark.parametrize("setup_backend", ["zmq"], indirect=True)
class TestZmqWireStepBatches(_ZmqWireTurns):
    """The rank sends an engine's steps: one message per period, every running request's token in it."""

    steps = True
