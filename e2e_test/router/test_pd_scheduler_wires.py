"""The disaggregated gRPC path on each scheduler wire the mock serves, on the CPU row.

The gateway speaks one prefill-decode protocol per scheduler proto
(``pd_protocol.rs``). SGLang dispatches both sides in parallel and writes the
gateway-minted bootstrap host, port and room into each; vLLM dispatches the
prefill side, then the decode side, with no rendezvous; TokenSpeed dispatches in
parallel over a bootstrap room. The disaggregated topology classes run on the
mock's default wire, TokenSpeed, so the SGLang and vLLM branches ran only on
the card lanes. Each class here starts one prefill and one decode mock worker
on one wire (``--grpc-wire``, which the harness passes to every pool) and
drives a chat turn and a streamed turn through that wire's protocol, with the
gateway's pair selection read from its log.

Usage:
    E2E_RUNTIME=mock E2E_ENGINE=mock pytest e2e_test/router/test_pd_scheduler_wires.py -v
"""

from __future__ import annotations

import os
import re
import tempfile
import time
from pathlib import Path

import httpx
import pytest
from infra.pd_logs import LOG_FLUSH_TIMEOUT_S, read_logs
from infra.streams import collect_streamed_chat

# The mock engine reads this model's tokenizer and chat template from an
# ungated twin of the chat suite's model, as the topology sweep does.
_MODEL = "RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8"

# The gateway's pair selections land here through the gateway marker (rolling
# files named smg.YYYY-MM-DD); the debug level carries the marker.
_LOG_DIR = Path(tempfile.gettempdir()) / f"smg-e2e-pd-wires-{os.getpid()}"
_PAIR_MARKER = "Selected PD pair"
_PAIR_RE = re.compile(r"prefill=(\S+)\s+decode=(\S+)")
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_GATEWAY_ARGS = ["--prefill-policy", "round_robin", "--decode-policy", "round_robin"]

_ANSWER = "The capital of France is Paris."
_QUESTION = [{"role": "user", "content": "What is the capital of France?"}]


def _registered_workers(gateway) -> list[dict]:
    resp = httpx.get(f"{gateway.base_url}/workers", timeout=5.0)
    assert resp.status_code == 200, f"GET /workers returned {resp.status_code}: {resp.text}"
    workers = resp.json().get("workers")
    assert workers, f"the gateway registered no worker: {resp.text}"
    return workers


def _pairs_logged() -> list[tuple[str, str]]:
    """(prefill, decode) URL pairs the gateway has logged so far, in order."""
    pairs: list[tuple[str, str]] = []
    for raw in read_logs(_LOG_DIR, "smg*").splitlines():
        if _PAIR_MARKER not in raw:
            continue
        match = _PAIR_RE.search(_ANSI.sub("", raw))
        if match:
            pairs.append((match.group(1), match.group(2)))
    return pairs


def _wait_for_pairs(minimum: int, timeout: float = LOG_FLUSH_TIMEOUT_S) -> list[tuple[str, str]]:
    deadline = time.monotonic() + timeout
    pairs = _pairs_logged()
    while len(pairs) < minimum and time.monotonic() < deadline:
        time.sleep(0.5)
        pairs = _pairs_logged()
    assert len(pairs) >= minimum, (
        f"the gateway logged {len(pairs)} PD pair selections, expected at least {minimum}; "
        f"is --log-level debug reaching {_LOG_DIR}/smg*?"
    )
    return pairs


class _DisaggregatedWireTurns:
    """The turns every wire's class runs. Not collected itself (underscore prefix)."""

    WIRE = ""

    def test_both_roles_register_under_this_wire(self, setup_backend):
        """The prefill and the decode worker register under this wire's runtime,
        each with its role, both healthy: a mock that served another proto, or
        a role the discovery missed, fails here and not by a timeout later."""
        _backend, _model, _client, gateway = setup_backend
        workers = _registered_workers(gateway)
        by_role = {worker.get("worker_type"): worker for worker in workers}
        assert set(by_role) == {"prefill", "decode"}, f"the roles registered: {workers}"
        runtimes = {worker["url"]: worker.get("runtime_type") for worker in workers}
        assert set(runtimes.values()) == {self.WIRE}, (
            f"the gateway registered {runtimes} under another runtime than the wire the mock serves"
        )
        unhealthy = [worker["url"] for worker in workers if not worker.get("is_healthy")]
        assert not unhealthy, f"unhealthy workers after startup: {unhealthy}"
        assert by_role["prefill"]["url"] in {w.base_url for w in gateway.prefill_workers}
        assert by_role["decode"]["url"] in {w.base_url for w in gateway.decode_workers}

    def test_a_chat_turn_runs_this_wire_s_protocol(self, setup_backend, mock_answers):
        """A chat turn runs the two sides of this wire's protocol: the gateway
        pairs the prefill and the decode worker (its log says so) and the
        decode side's scripted text comes back with a stop and a usage."""
        _backend, model, client, gateway = setup_backend
        before = len(_pairs_logged())
        mock_answers.enqueue(_ANSWER)
        response = client.chat.completions.create(
            model=model, messages=_QUESTION, max_tokens=64, temperature=0
        )
        choice = response.choices[0]
        assert choice.message.content == _ANSWER
        assert choice.finish_reason == "stop"
        assert response.usage is not None
        assert response.usage.prompt_tokens > 0 and response.usage.completion_tokens > 0

        pairs = _wait_for_pairs(before + 1)[before:]
        prefills = {w.base_url for w in gateway.prefill_workers}
        decodes = {w.base_url for w in gateway.decode_workers}
        outside = [(p, d) for p, d in pairs if p not in prefills or d not in decodes]
        assert not outside, f"pairs {outside} outside the fleet prefill={prefills} decode={decodes}"

    def test_a_streamed_turn_runs_this_wire_s_protocol(self, setup_backend, mock_answers):
        """The streamed deltas join to the scripted text, the stream ends on
        ``stop`` and its usage chunk counts the completion as the non-streamed
        turn does, through this wire's prefill-decode protocol."""
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
        assert usage_chunks[0].usage.completion_tokens == whole.usage.completion_tokens, (
            f"the stream counted {usage_chunks[0].usage.completion_tokens} completion tokens, "
            f"the non-streamed turn {whole.usage.completion_tokens}"
        )


@pytest.mark.engine("mock")
@pytest.mark.gpu(2)
@pytest.mark.e2e
@pytest.mark.model(_MODEL)
@pytest.mark.workers(parallel_start=True, extra_engine_args=["--grpc-wire", "sglang"])
@pytest.mark.gateway(log_level="debug", log_dir=str(_LOG_DIR), extra_args=_GATEWAY_ARGS)
@pytest.mark.parametrize(
    "setup_backend", [pytest.param(("pd_grpc", (1, 1)), id="1p1d")], indirect=True
)
class TestSglangWireDisaggregated(_DisaggregatedWireTurns):
    WIRE = "sglang"


@pytest.mark.engine("mock")
@pytest.mark.gpu(2)
@pytest.mark.e2e
@pytest.mark.model(_MODEL)
@pytest.mark.workers(parallel_start=True, extra_engine_args=["--grpc-wire", "vllm"])
@pytest.mark.gateway(log_level="debug", log_dir=str(_LOG_DIR), extra_args=_GATEWAY_ARGS)
@pytest.mark.parametrize(
    "setup_backend", [pytest.param(("pd_grpc", (1, 1)), id="1p1d")], indirect=True
)
class TestVllmWireDisaggregated(_DisaggregatedWireTurns):
    WIRE = "vllm"
