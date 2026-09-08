"""A worker that dies and returns on the same port rejoins rotation.

The gateway must notice a dead gRPC worker through its health probes, fail
fast while it is gone rather than hang, and put the worker back into rotation
once it is healthy again. The SGLang gRPC server used to race its own IPC
binding during startup (#2427); restarting the worker under a live gateway is
the closest end-to-end reproduction of that path.

Usage:
    E2E_RUNTIME=sglang pytest e2e_test/router/test_worker_restart.py -v
"""

from __future__ import annotations

import logging
import os
import time

import httpx
import pytest
from infra import ConnectionMode, Gateway, get_pool
from infra.constants import get_runtime
from infra.model_specs import get_model_spec

logger = logging.getLogger(__name__)

MODEL = "meta-llama/Llama-3.2-1B-Instruct"
_HEALTH_ARGS = [
    "--health-check-interval-secs",
    "1",
    "--health-check-timeout-secs",
    "2",
    "--health-failure-threshold",
    "1",
    "--health-success-threshold",
    "1",
]


def _status(gateway: Gateway, url: str) -> str | None:
    for worker in gateway.list_workers(strict=True):
        if worker.url == url:
            return worker.status
    return None


def _wait_for_status(gateway: Gateway, url: str, wanted: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    seen: str | None = None
    while time.monotonic() < deadline:
        seen = _status(gateway, url)
        if seen == wanted:
            return
        time.sleep(0.5)
    pytest.fail(f"worker {url} never became {wanted} within {timeout:.0f}s (last: {seen})")


def _chat(gateway: Gateway, model_path: str, timeout: float) -> httpx.Response:
    return httpx.post(
        f"{gateway.base_url}/v1/chat/completions",
        json={
            "model": model_path,
            "messages": [{"role": "user", "content": "Say hello."}],
            "max_tokens": 4,
        },
        timeout=timeout,
    )


def _error_code(resp: httpx.Response) -> str | None:
    try:
        body = resp.json()
    except ValueError:
        return None
    error = body.get("error", body)
    return error.get("code") if isinstance(error, dict) else None


def _wait_until_served(gateway: Gateway, model_path: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last = "no attempt"
    while time.monotonic() < deadline:
        try:
            resp = _chat(gateway, model_path, timeout=30.0)
        except httpx.HTTPError as exc:
            last = repr(exc)
        else:
            if resp.status_code == 200:
                return
            last = f"{resp.status_code} {resp.text[:200]}"
        time.sleep(1.0)
    pytest.fail(f"gateway did not serve within {timeout:.0f}s; last: {last}")


@pytest.mark.engine("sglang", "vllm")
@pytest.mark.gpu(1)
@pytest.mark.e2e
class TestWorkerRestart:
    """Stop the pooled gRPC worker under a live gateway, then bring it back."""

    def test_worker_returns_to_rotation_after_restart(self):
        engine = get_runtime()
        model_path = get_model_spec(MODEL)["model"]
        # The session pool owns the GPU; borrowing its worker avoids racing a
        # cached worker for memory and leaves a healthy worker behind for the
        # next class.
        worker = get_pool().acquire(
            model_id=MODEL,
            engine=engine,
            mode=ConnectionMode.GRPC,
            count=1,
            log_dir=os.environ.get("E2E_LOG_DIR"),
        )[0]
        gateway = Gateway()
        try:
            gateway.start(
                worker_urls=[worker.base_url], model_path=model_path, extra_args=_HEALTH_ARGS
            )
            _wait_until_served(gateway, model_path, timeout=120.0)

            worker.stop()
            _wait_for_status(gateway, worker.base_url, "unhealthy", timeout=30.0)
            started = time.monotonic()
            resp = _chat(gateway, model_path, timeout=30.0)
            elapsed = time.monotonic() - started
            logger.info(
                "while down: status=%s after %.1fs body=%s",
                resp.status_code,
                elapsed,
                resp.text[:200],
            )
            # The model exists and its worker is merely down: that is a 503
            # no_available_workers (#2465), not a 404 that tells the client the
            # model is gone.
            assert resp.status_code == 503, (
                f"request during the outage should be a 503, got {resp.status_code}: {resp.text[:200]}"
            )
            assert _error_code(resp) == "no_available_workers", resp.text[:200]
            assert elapsed < 20.0, f"request during the outage hung for {elapsed:.1f}s"

            worker.start()  # same port, same URL
            _wait_for_status(gateway, worker.base_url, "healthy", timeout=240.0)
            _wait_until_served(gateway, model_path, timeout=60.0)
        finally:
            gateway.shutdown()
