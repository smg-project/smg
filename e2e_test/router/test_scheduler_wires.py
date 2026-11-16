"""The gateway's three gRPC scheduler wires, each driven end to end on the CPU row.

The scripted mock worker serves one of the gateway's three scheduler protos
per process (``mock-worker --grpc-wire tokenspeed|vllm|sglang``), the same
scripted answers behind each. One class per wire runs the same turns through
the gateway: a chat turn, a streamed turn and a tool-call turn, so each
proto's request builders, response wrappers and load conversion run on every
push, not only when a card lane with that engine does. The gateway detects a
worker's runtime from the wire it answers on, and each class asserts the
worker it registered is of its wire: a mock that silently served another
proto would fail here, not pass by accident.

Usage:
    E2E_RUNTIME=mock E2E_ENGINE=mock pytest e2e_test/router/test_scheduler_wires.py -v
"""

from __future__ import annotations

import json
import time

import httpx
import pytest
from infra.streams import collect_streamed_chat

_MODEL = "RedHatAI/Llama-3.2-1B-Instruct-FP8"

# Short poll interval so the worker's first load report lands within seconds
# of the class's start; the gateway default is 10 s.
_LOAD_MONITOR_INTERVAL_SECS = 1
_GATEWAY_ARGS = ["--load-monitor-interval", str(_LOAD_MONITOR_INTERVAL_SECS)]
_REPORT_TIMEOUT_SECS = 8 * _LOAD_MONITOR_INTERVAL_SECS

_ANSWER = "The capital of France is Paris."
_QUESTION = [{"role": "user", "content": "What is the capital of France?"}]

# The llama 3.2 tool-call prompt and syntax of the chat suite's function-calling
# class: a bare JSON object per call, a list of them under a forced choice.
_SYSTEM_MESSAGE = (
    "You are a helpful assistant with tool calling capabilities. "
    "To call a function, please respond with JSON for a function call. "
    'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}. '
    "Do not use variables.\n\n"
)
_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Compute the sum of two numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "A number"},
                    "b": {"type": "integer", "description": "A number"},
                },
                "required": ["a", "b"],
            },
        },
    }
]
_ADD_CALL = {"name": "add", "parameters": {"a": 3, "b": 5}}
_TOOL_QUESTION = [
    {"role": "system", "content": _SYSTEM_MESSAGE},
    {"role": "user", "content": "Compute (3+5)"},
]


def _registered_workers(gateway) -> list[dict]:
    resp = httpx.get(f"{gateway.base_url}/workers", timeout=5.0)
    assert resp.status_code == 200, f"GET /workers returned {resp.status_code}: {resp.text}"
    workers = resp.json().get("workers")
    assert workers, f"the gateway registered no worker: {resp.text}"
    return workers


def _wait_for_load_report(gateway, worker_url: str, timeout: float) -> dict:
    """Poll ``/loads`` until the worker's rank report is there, and return it."""
    deadline = time.monotonic() + timeout
    while True:
        resp = httpx.get(f"{gateway.base_url}/loads", timeout=5.0)
        assert resp.status_code == 200, f"GET /loads returned {resp.status_code}: {resp.text}"
        loads = resp.json().get("loads") or []
        for entry in loads:
            if entry.get("worker") == worker_url:
                return entry
        if time.monotonic() >= deadline:
            raise AssertionError(
                f"no load report from {worker_url} within {timeout:.0f} s: {loads}"
            )
        time.sleep(0.25)


class _SchedulerWireTurns:
    """The turns every wire's class runs. Not collected itself (underscore prefix)."""

    WIRE = ""

    def test_a_chat_turn_through_this_wire(self, setup_backend, mock_answers):
        """The worker registered under this wire's runtime answers a chat turn
        with the scripted text, and its load report reaches the fleet view
        through this wire's load RPC."""
        _backend, model, client, gateway = setup_backend
        workers = _registered_workers(gateway)
        assert [worker["runtime_type"] for worker in workers] == [self.WIRE], (
            f"the gateway registered {workers} under another runtime than the wire the mock serves"
        )

        mock_answers.enqueue(_ANSWER)
        response = client.chat.completions.create(
            model=model, messages=_QUESTION, max_tokens=64, temperature=0
        )
        choice = response.choices[0]
        assert choice.message.content == _ANSWER
        assert choice.finish_reason == "stop"
        assert response.usage is not None
        assert response.usage.prompt_tokens > 0 and response.usage.completion_tokens > 0
        assert response.usage.total_tokens == (
            response.usage.prompt_tokens + response.usage.completion_tokens
        )

        # The mock reports a token capacity of 1,000,000 on every wire: a
        # conversion that dropped or zeroed the field would show here.
        report = _wait_for_load_report(gateway, workers[0]["url"], _REPORT_TIMEOUT_SECS)
        assert report.get("max_total_num_tokens") == 1_000_000, (
            f"the rank report did not carry the mock's token capacity through this wire: {report}"
        )

    def test_a_streamed_turn_through_this_wire(self, setup_backend, mock_answers):
        """The streamed deltas join to the scripted text, the stream ends on
        ``stop``, and the usage chunk counts the completion as the non-streamed
        turn does: the count the gateway derives on this wire (the chunk ids
        summed on the vLLM wire, the completion's total on the others) read
        against the other reading of the same answer."""
        _backend, model, client, _gateway = setup_backend
        mock_answers.enqueue(_ANSWER)
        whole = client.chat.completions.create(
            model=model, messages=_QUESTION, max_tokens=64, temperature=0
        )
        assert whole.usage is not None

        mock_answers.enqueue(_ANSWER)
        stream = client.chat.completions.create(
            model=model,
            messages=_QUESTION,
            max_tokens=64,
            temperature=0,
            stream=True,
            stream_options={"include_usage": True},
        )
        streamed = collect_streamed_chat(stream)
        assert streamed.content == _ANSWER
        assert streamed.finish_reason == "stop"
        usage_chunks = [chunk for chunk in streamed.chunks if chunk.usage is not None]
        assert len(usage_chunks) == 1, "one usage chunk, after the choices"
        usage = usage_chunks[0].usage
        assert usage.completion_tokens == whole.usage.completion_tokens, (
            f"the stream counted {usage.completion_tokens} completion tokens, "
            f"the non-streamed turn {whole.usage.completion_tokens}"
        )
        assert usage.prompt_tokens == whole.usage.prompt_tokens
        content_chunks = [
            chunk for chunk in streamed.chunks if chunk.choices and chunk.choices[0].delta.content
        ]
        assert len(content_chunks) > 1, "a streamed answer comes in several content deltas"

    def test_a_tool_call_turn_through_this_wire(self, setup_backend, mock_answers):
        """A tool call in the model's syntax is read into ``tool_calls`` with
        its arguments and ends on ``tool_calls``, and a forced choice (the
        grammar the gateway shapes for this wire) reads the same call."""
        _backend, model, client, _gateway = setup_backend
        mock_answers.enqueue(json.dumps(_ADD_CALL))
        response = client.chat.completions.create(
            model=model,
            messages=_TOOL_QUESTION,
            max_tokens=256,
            temperature=0,
            tools=_TOOLS,
        )
        choice = response.choices[0]
        assert choice.finish_reason == "tool_calls"
        assert choice.message.tool_calls, "a tool call was expected"
        call = choice.message.tool_calls[0].function
        assert call.name == "add"
        assert json.loads(call.arguments) == {"a": 3, "b": 5}

        # Under a named choice the legacy path constrains the engine to the
        # function's parameters object (a json_schema in this wire's field of
        # the request, which the mock does not enforce) and names the call
        # itself: the scripted answer is that object. The legacy path reads
        # this checkpoint because Symphony's registry has no table for the
        # FP8 repository and the class names no --tool-call-parser; a table
        # for it would constrain to the list holding the call instead (see
        # test_function_call_specific).
        mock_answers.enqueue(json.dumps(_ADD_CALL["parameters"]))
        forced = client.chat.completions.create(
            model=model,
            messages=_TOOL_QUESTION,
            max_tokens=256,
            temperature=0,
            tools=_TOOLS,
            tool_choice={"type": "function", "function": {"name": "add"}},
        )
        forced_call = forced.choices[0].message.tool_calls[0].function
        assert forced_call.name == "add"
        assert json.loads(forced_call.arguments) == {"a": 3, "b": 5}


@pytest.mark.engine("mock")
@pytest.mark.gpu(1)
@pytest.mark.model(_MODEL)
@pytest.mark.gateway(extra_args=_GATEWAY_ARGS)
@pytest.mark.workers(extra_engine_args=["--grpc-wire", "tokenspeed"])
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestTokenSpeedWire(_SchedulerWireTurns):
    WIRE = "tokenspeed"


@pytest.mark.engine("mock")
@pytest.mark.gpu(1)
@pytest.mark.model(_MODEL)
@pytest.mark.gateway(extra_args=_GATEWAY_ARGS)
@pytest.mark.workers(extra_engine_args=["--grpc-wire", "vllm"])
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestVllmWire(_SchedulerWireTurns):
    WIRE = "vllm"


@pytest.mark.engine("mock")
@pytest.mark.gpu(1)
@pytest.mark.model(_MODEL)
@pytest.mark.gateway(extra_args=_GATEWAY_ARGS)
@pytest.mark.workers(extra_engine_args=["--grpc-wire", "sglang"])
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestSglangWire(_SchedulerWireTurns):
    WIRE = "sglang"
