"""Overload protection sheds with its own error code and recovers.

``--worker-overload-protection`` takes a worker whose waiting queue crosses
``--worker-overload-waiting-requests`` out of routing; when every worker for
the model is overloaded the gateway sheds immediately with 503 instead of
queueing. The shed used to reuse the generic ``no_available_workers`` code, so
a downstream gateway could not tell an overload from a dead fleet (#2417).

The engine is pinned to one running request so a burst piles up in its
waiting queue, the gateway polls loads every second, and a probe sent while
the queue is deep must be shed with ``worker_overload_protection_shed`` and a
``Retry-After``. Once the burst drains the worker must serve again.

Usage:
    E2E_RUNTIME=sglang pytest e2e_test/router/test_overload.py -v
"""

from __future__ import annotations

import logging
import threading
import time

import httpx
import pytest

logger = logging.getLogger(__name__)

SHED_CODE = "worker_overload_protection_shed"
WAITING_THRESHOLD = 2
BURST_SIZE = 12
_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
_GATEWAY_ARGS = [
    "--worker-overload-protection",
    "--worker-overload-waiting-requests",
    str(WAITING_THRESHOLD),
    "--load-monitor-interval",
    "1",
]


def _chat_body(model: str, max_tokens: int, *, ignore_eos: bool) -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "Write a long story about the sea."}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": ignore_eos,
    }


def _post(gateway, body: dict, timeout: float) -> httpx.Response:
    return httpx.post(f"{gateway.base_url}/v1/chat/completions", json=body, timeout=timeout)


def _error_code(resp: httpx.Response) -> str | None:
    try:
        body = resp.json()
    except ValueError:
        return None
    error = body.get("error", body)
    return error.get("code") if isinstance(error, dict) else None


def _wait_until_served(gateway, model: str, timeout: float) -> None:
    """Retry a tiny request until the worker answers 200 again."""
    deadline = time.monotonic() + timeout
    last = "no attempt"
    while time.monotonic() < deadline:
        try:
            resp = _post(gateway, _chat_body(model, 4, ignore_eos=False), timeout=60.0)
        except httpx.HTTPError as exc:
            last = repr(exc)
        else:
            if resp.status_code == 200:
                return
            last = f"{resp.status_code} {resp.text[:200]}"
        time.sleep(1.0)
    pytest.fail(f"worker did not serve within {timeout:.0f}s; last: {last}")


def _wait_for_deep_queue(gateway, min_waiting: int, timeout: float) -> dict:
    """Poll /loads until a report shows at least ``min_waiting`` queued requests."""
    deadline = time.monotonic() + timeout
    last: list[dict] = []
    while time.monotonic() < deadline:
        resp = httpx.get(f"{gateway.base_url}/loads", timeout=5.0)
        assert resp.status_code == 200, resp.text
        last = resp.json().get("loads", [])
        for entry in last:
            if entry.get("num_waiting_reqs", 0) >= min_waiting:
                return entry
        time.sleep(0.2)
    pytest.fail(f"engine queue never reached {min_waiting} waiting requests; last reports: {last}")


class _OverloadShedBase:
    """Shared body; subclasses pin the engine and its single-slot flag."""

    def test_shed_carries_its_own_code_and_recovers(self, setup_backend):
        _, model, _, gateway = setup_backend
        _wait_until_served(gateway, model, timeout=60.0)

        results: list[httpx.Response | BaseException] = []

        def _long_request() -> None:
            try:
                results.append(_post(gateway, _chat_body(model, 256, ignore_eos=True), 240.0))
            except BaseException as exc:  # reported below
                results.append(exc)

        threads = [threading.Thread(target=_long_request, daemon=True) for _ in range(BURST_SIZE)]
        for t in threads:
            t.start()

        try:
            queue = _wait_for_deep_queue(gateway, WAITING_THRESHOLD, timeout=20.0)
            logger.info("queue is deep: %s", queue)

            probe = _post(gateway, _chat_body(model, 4, ignore_eos=False), timeout=240.0)
            logger.info(
                "probe: status=%s retry-after=%s code=%s",
                probe.status_code,
                probe.headers.get("retry-after"),
                _error_code(probe),
            )
            assert probe.status_code == 503, (
                f"probe was not shed while the queue was deep: {probe.status_code} {probe.text[:200]}"
            )
            assert _error_code(probe) == SHED_CODE, (
                f"shed carried the wrong code: {probe.text[:300]}"
            )
            retry_after = probe.headers.get("retry-after")
            assert retry_after is not None and int(retry_after) >= 1, (
                f"shed without a usable Retry-After: {retry_after!r}"
            )
        finally:
            for t in threads:
                t.join(timeout=300)

        transport_errors = [r for r in results if isinstance(r, BaseException)]
        assert not transport_errors, (
            f"burst requests failed at the transport: {transport_errors[:3]}"
        )
        responses = [r for r in results if isinstance(r, httpx.Response)]
        for resp in responses:
            assert resp.status_code == 200 or (
                resp.status_code == 503 and _error_code(resp) == SHED_CODE
            ), f"burst request neither served nor shed: {resp.status_code} {resp.text[:200]}"
        assert any(r.status_code == 200 for r in responses), "no burst request was served"

        _wait_until_served(gateway, model, timeout=90.0)


@pytest.mark.engine("sglang")
@pytest.mark.gpu(1)
@pytest.mark.e2e
@pytest.mark.model(_MODEL)
@pytest.mark.workers(count=1, extra_engine_args=["--max-running-requests", "1"])
@pytest.mark.gateway(extra_args=_GATEWAY_ARGS)
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestOverloadShedSglang(_OverloadShedBase):
    """SGLang runs one request at a time; the rest queue."""


@pytest.mark.engine("vllm")
@pytest.mark.gpu(1)
@pytest.mark.e2e
@pytest.mark.model(_MODEL)
@pytest.mark.workers(count=1, extra_engine_args=["--max-num-seqs", "1"])
@pytest.mark.gateway(extra_args=_GATEWAY_ARGS)
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestOverloadShedVllm(_OverloadShedBase):
    """vLLM runs one sequence at a time; the rest queue."""
