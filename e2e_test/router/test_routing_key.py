"""Routing keys stay sticky and a body ``rid`` outranks the header.

With ``--routing-key-override`` every request must take the buffered path so
the body's ``rid`` can win over the routing-key header. A streamed
pass-through had silently dropped that precedence, so two requests from one
``rid`` lineage could land on different workers (#2355). Which worker served
a request is observed through the gateway's in-flight load while a long
generation is running.

Usage:
    E2E_RUNTIME=sglang pytest e2e_test/router/test_routing_key.py -v
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

import httpx
import pytest

logger = logging.getLogger(__name__)

KEY_HEADER = "x-smg-routing-key"
_GATEWAY_ARGS = ["--routing-key-override"]


def _request(gateway, model: str, *, key: str, rid: str | None) -> Callable[[], None]:
    """Build a sender for one long generation carrying a header key and maybe a body rid."""
    body: dict = {
        "model": model,
        "messages": [{"role": "user", "content": "Write a long story about a river."}],
        "max_tokens": 200,
        "temperature": 0.0,
        "ignore_eos": True,
    }
    if rid is not None:
        body["rid"] = rid

    def _send() -> None:
        resp = httpx.post(
            f"{gateway.base_url}/v1/chat/completions",
            headers={KEY_HEADER: key},
            json=body,
            timeout=120.0,
        )
        assert resp.status_code == 200, f"{resp.status_code} {resp.text[:200]}"

    return _send


def _serving_worker(gateway, send: Callable[[], None]) -> str:
    """Run ``send`` and return the URL of the worker that carried it in flight."""
    failures: list[BaseException] = []

    def _run() -> None:
        try:
            send()
        except BaseException as exc:  # surfaced after the join
            failures.append(exc)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    busy: set[str] = set()
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline and thread.is_alive():
        busy.update(w.url for w in gateway.list_workers(strict=True) if w.pending_requests > 0)
        if busy:
            break
        time.sleep(0.02)
    thread.join(timeout=120)
    assert not failures, f"request failed: {failures[0]!r}"
    assert len(busy) == 1, f"expected exactly one worker to carry the request, saw {sorted(busy)}"
    return busy.pop()


@pytest.mark.engine("sglang", "vllm")
@pytest.mark.gpu(2)
@pytest.mark.e2e
@pytest.mark.model("meta-llama/Llama-3.2-1B-Instruct")
@pytest.mark.workers(count=2)
@pytest.mark.gateway(policy="manual", extra_args=_GATEWAY_ARGS)
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestRoutingKeyPinning:
    """Two workers, manual policy: keys pin, and the body rid decides the key."""

    def test_header_key_is_sticky(self, setup_backend):
        _, model, _, gateway = setup_backend
        assert len(gateway.list_workers(strict=True)) == 2

        served = {
            _serving_worker(gateway, _request(gateway, model, key="session-sticky", rid=None))
            for _ in range(3)
        }

        assert len(served) == 1, f"one key reached more than one worker: {sorted(served)}"

    def test_body_rid_outranks_header_key(self, setup_backend):
        _, model, _, gateway = setup_backend

        # Find two keys the policy assigned to different workers.
        homes: dict[str, str] = {}
        for i in range(8):
            key = f"lineage-{i}"
            homes[key] = _serving_worker(gateway, _request(gateway, model, key=key, rid=None))
            if len(set(homes.values())) == 2:
                break
        assert len(set(homes.values())) == 2, f"every key landed on one worker: {homes}"
        key_a, key_b = list(homes)[-2:]
        worker_a, worker_b = homes[key_a], homes[key_b]
        if worker_a == worker_b:  # the last two keys share a home; pick a differing pair
            key_a = next(k for k, w in homes.items() if w != worker_b)
            worker_a = homes[key_a]
        logger.info("homes: %s -> %s, %s -> %s", key_a, worker_a, key_b, worker_b)

        assert _serving_worker(gateway, _request(gateway, model, key=key_a, rid=key_b)) == worker_b
        assert _serving_worker(gateway, _request(gateway, model, key=key_b, rid=key_a)) == worker_a
