"""Tests for gateway admin operations (cache flush, profiling).

Covers the worker-abstracted admin fan-out behind /flush_cache,
/start_profile, and /stop_profile: the gateway dispatches per worker on
connection mode (HTTP endpoint vs gRPC RPC), so both modes are exercised
with the same assertions, including that profiling actually exports trace
artifacts to the requested output directory.

Usage:
    pytest e2e_test/router/test_admin_ops.py -v
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import httpx
import pytest

logger = logging.getLogger(__name__)

FLUSH_TIMEOUT = 60.0
PROFILE_START_TIMEOUT = 120.0
PROFILE_STOP_TIMEOUT = 300.0
TRACE_EXPORT_WAIT = 60.0


def _wait_for_trace_artifacts(output_dir: Path) -> list[Path]:
    """Poll for profiler artifacts — trace export may lag the RPC reply."""
    deadline = time.time() + TRACE_EXPORT_WAIT
    while time.time() < deadline:
        if output_dir.is_dir():
            files = [f for f in output_dir.rglob("*") if f.is_file() and f.stat().st_size > 0]
            if files:
                return files
        time.sleep(1.0)
    return []


class FlushCacheBehavior:
    """Shared cache-flush assertions for engines exposing FlushCache."""

    def test_flush_cache_full_cycle(self, setup_backend):
        backend, model, client, gateway = setup_backend

        # Populate the prefix cache so there is something to flush.
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say hi"}],
            max_tokens=8,
        )

        resp = httpx.post(f"{gateway.base_url}/flush_cache", timeout=FLUSH_TIMEOUT)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        logger.info("flush_cache response: %s", body)
        assert body["status"] == "success"
        assert body["workers_flushed"] == 1
        assert body["total_workers"] == 1
        if backend == "grpc":
            assert body["total_grpc_workers"] == 1
        else:
            assert body["total_http_workers"] == 1

        # Generation still works against the flushed cache.
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say hi again"}],
            max_tokens=8,
        )
        assert completion.choices, "generation after cache flush should succeed"


class AdminOpsBehavior(FlushCacheBehavior):
    """Shared flush and profiling assertions, parametrized per engine below."""

    def test_start_and_stop_profile_produces_traces(self, setup_backend, tmp_path):
        _backend, model, client, gateway = setup_backend

        output_dir = tmp_path / "traces"
        output_dir.mkdir()
        resp = httpx.post(
            f"{gateway.base_url}/start_profile",
            json={"output_dir": str(output_dir)},
            timeout=PROFILE_START_TIMEOUT,
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        logger.info("start_profile response: %s", body)
        assert body["status"] == "success"
        assert body["workers_profiled"] == 1

        # Give the profiler some forward steps to record.
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say hi"}],
            max_tokens=8,
        )

        resp = httpx.post(f"{gateway.base_url}/stop_profile", timeout=PROFILE_STOP_TIMEOUT)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        logger.info("stop_profile response: %s", body)
        assert body["status"] == "success"
        assert body["workers_profiled"] == 1

        artifacts = _wait_for_trace_artifacts(output_dir)
        assert artifacts, (
            f"profiler reported success but produced no artifacts in {output_dir} "
            f"within {TRACE_EXPORT_WAIT}s"
        )
        logger.info(
            "profiler produced %d artifact(s): %s",
            len(artifacts),
            [f.name for f in artifacts[:5]],
        )

    def test_profile_url_filter_without_match_returns_404(self, setup_backend):
        _backend, _model, _client, gateway = setup_backend

        resp = httpx.post(
            f"{gateway.base_url}/start_profile",
            json={"url": "http://nonexistent:9999"},
            timeout=30.0,
        )
        assert resp.status_code == 404, resp.text

    def test_start_profile_rejects_malformed_json(self, setup_backend):
        _backend, _model, _client, gateway = setup_backend

        resp = httpx.post(
            f"{gateway.base_url}/start_profile",
            content=b"{not json",
            headers={"content-type": "application/json"},
            timeout=30.0,
        )
        assert resp.status_code == 400, resp.text


@pytest.mark.engine("sglang")
@pytest.mark.gpu(1)
@pytest.mark.e2e
@pytest.mark.parametrize("setup_backend", ["grpc", "http"], indirect=True)
class TestAdminOps(AdminOpsBehavior):
    """SGLang serves both connection modes behind the gateway."""


@pytest.mark.engine("tokenspeed")
@pytest.mark.gpu(1)
@pytest.mark.e2e
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestAdminOpsTokenSpeed(AdminOpsBehavior):
    """TokenSpeed runs gRPC-only behind the gateway."""


@pytest.mark.engine("vllm")
@pytest.mark.gpu(1)
@pytest.mark.e2e
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestFlushCacheVllm(FlushCacheBehavior):
    """vLLM exposes cache flush over gRPC."""
