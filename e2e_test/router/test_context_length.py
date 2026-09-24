"""An input longer than the model's context window is a 400 from the gateway.

Before this check, an over-long prompt on the disaggregated gRPC path reached
the prefill engine, whose rejection came back as::

    HTTP 500 {"error": {"code": "prefill_worker_failed_to_start", ...}}

a server error for a client mistake, blamed on a healthy worker (#2380).
The gateway learns the engine's window at discovery (``max_context_length``
on the model card), tokenizes the prompt itself on the gRPC path, and can
therefore answer ``400 context_length_exceeded`` without dispatching.

The engines run with a deliberately small window so a modest prompt overruns
it. A prompt that fits still goes through; the engine stays the arbiter of
whether ``input + max_tokens`` fits.

Usage:
    E2E_RUNTIME=vllm pytest e2e_test/router/test_context_length.py -v
"""

from __future__ import annotations

import logging

import httpx
import pytest
from infra.constants import get_runtime

logger = logging.getLogger(__name__)

_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
_MODEL_BY_ENGINE = {"tokenspeed": "Qwen/Qwen3.5-9B"}
# Small enough that a few hundred words overrun it, large enough for the
# chat template plus a short answer.
_WINDOW = 2048
_WINDOW_ARGS = {
    "sglang": ["--context-length", str(_WINDOW)],
    "vllm": ["--max-model-len", str(_WINDOW)],
    "tokenspeed": ["--max-model-len", str(_WINDOW)],
}.get(get_runtime(), [])
# Common English words are one token each on these tokenizers; twice the
# window's worth of them is over the window on any of them.
_OVER_LONG_WORDS = _WINDOW * 2
_FITS_WORDS = 64


def _chat(gateway, model: str, words: int, *, stream: bool = False) -> httpx.Response:
    prompt = " ".join(["hello"] * words)
    return httpx.post(
        f"{gateway.base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 8,
            "stream": stream,
        },
        timeout=120.0,
    )


def _assert_context_length_rejection(resp: httpx.Response) -> None:
    logger.info(
        "status=%s content-type=%s body=%s",
        resp.status_code,
        resp.headers.get("content-type"),
        resp.text[:300],
    )
    assert resp.status_code == 400, f"expected 400, got {resp.status_code}: {resp.text[:300]}"
    content_type = resp.headers.get("content-type", "")
    assert content_type.startswith("application/json"), (
        f"rejection must be a JSON error, got {content_type!r}: {resp.text[:300]}"
    )
    error = resp.json()["error"]
    assert error["code"] == "context_length_exceeded", resp.text
    assert "prefill_worker_failed_to_start" not in resp.text, (
        f"a client mistake must not be reported as a worker failure: {resp.text[:300]}"
    )
    assert f"{_WINDOW} tokens" in error["message"], error["message"]


class _ContextLengthContract:
    """Shared bodies; subclasses pin the topology via the setup marker."""

    def test_over_long_prompt_is_a_400_context_length_exceeded(self, setup_backend):
        _, model, _, gateway = setup_backend
        _assert_context_length_rejection(_chat(gateway, model, _OVER_LONG_WORDS))

    def test_over_long_streaming_prompt_is_a_json_400(self, setup_backend):
        _, model, _, gateway = setup_backend
        _assert_context_length_rejection(_chat(gateway, model, _OVER_LONG_WORDS, stream=True))

    def test_prompt_within_the_window_is_served(self, setup_backend):
        _, model, _, gateway = setup_backend
        resp = _chat(gateway, model, _FITS_WORDS)
        assert resp.status_code == 200, f"{resp.status_code}: {resp.text[:300]}"
        assert resp.json()["choices"], resp.text

    def test_workers_still_healthy_after_rejection(self, setup_backend):
        """The rejection never reached an engine: every worker stays healthy
        and serves the next request."""
        _, model, _, gateway = setup_backend
        _assert_context_length_rejection(_chat(gateway, model, _OVER_LONG_WORDS))
        workers = gateway.list_workers()
        assert workers, "no workers registered"
        unhealthy = [(w.url, w.status) for w in workers if w.status.lower() != "healthy"]
        assert not unhealthy, f"workers unhealthy after a rejected request: {unhealthy}"
        resp = _chat(gateway, model, _FITS_WORDS)
        assert resp.status_code == 200, f"{resp.status_code}: {resp.text[:300]}"


@pytest.mark.engine("sglang", "vllm", "tokenspeed")
@pytest.mark.gpu(2)
@pytest.mark.e2e
@pytest.mark.model(_MODEL, tokenspeed=_MODEL_BY_ENGINE["tokenspeed"])
@pytest.mark.workers(extra_engine_args=_WINDOW_ARGS)
@pytest.mark.parametrize(
    "setup_backend", [pytest.param(("pd_grpc", (1, 1)), id="1p1d")], indirect=True
)
class TestPDContextLength(_ContextLengthContract):
    """The disaggregated gRPC path: the case #2380 reported."""


@pytest.mark.engine("sglang", "vllm", "tokenspeed")
@pytest.mark.gpu(1)
@pytest.mark.e2e
@pytest.mark.model(_MODEL, tokenspeed=_MODEL_BY_ENGINE["tokenspeed"])
@pytest.mark.workers(extra_engine_args=_WINDOW_ARGS)
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestRegularContextLength(_ContextLengthContract):
    """The single-worker gRPC path is bounded the same way."""
