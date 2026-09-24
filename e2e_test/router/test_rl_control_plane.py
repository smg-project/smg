"""E2E tests for the RL control plane (/v1/rl) against a real SGLang engine.

Usage:
    pytest e2e_test/router/test_rl_control_plane.py -v
"""

from __future__ import annotations

import logging

import httpx
import pytest

logger = logging.getLogger(__name__)

TIMEOUT = 60.0


@pytest.mark.engine("sglang")
@pytest.mark.gpu(1)
@pytest.mark.e2e
@pytest.mark.gateway(policy="round_robin", extra_args=["--enable-rl"])
@pytest.mark.parametrize("setup_backend", ["http"], indirect=True)
class TestRlControlPlane:
    def test_discovery_reports_engine_and_topology(self, setup_backend):
        _backend, _model, _client, gateway = setup_backend
        resp = httpx.get(f"{gateway.base_url}/v1/rl/workers", timeout=TIMEOUT)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["total"] >= 1
        w = body["workers"][0]
        assert w["engine"] == "sglang"
        assert w["tp_size"] is not None
        assert w["health"] == "ready"
        assert "abort" in w["capabilities"]["pause_modes"]

    def test_fanout_flush_cache(self, setup_backend):
        _backend, _model, _client, gateway = setup_backend
        resp = httpx.post(
            f"{gateway.base_url}/v1/rl/engine/flush_cache",
            params={"selector": "engine=sglang"},
            timeout=TIMEOUT,
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["succeeded"] == body["total"] >= 1
        assert body["failed"] == []

    def test_pause_then_continue_keeps_serving(self, setup_backend):
        _backend, model, client, gateway = setup_backend
        for op in ("pause_generation", "continue_generation"):
            resp = httpx.post(
                f"{gateway.base_url}/v1/rl/engine/{op}",
                params={"selector": "engine=sglang"},
                json={},
                timeout=TIMEOUT,
            )
            assert resp.status_code == 200, f"{op}: {resp.text}"
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say hi"}],
            max_tokens=4,
        )
        assert completion.choices

    def test_single_worker_proxy_server_info(self, setup_backend):
        _backend, _model, _client, gateway = setup_backend
        workers_resp = httpx.get(f"{gateway.base_url}/v1/rl/workers", timeout=TIMEOUT)
        wid = workers_resp.json()["workers"][0]["id"]
        resp = httpx.get(
            f"{gateway.base_url}/v1/rl/workers/{wid}/engine/server_info", timeout=TIMEOUT
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["body"].get("tp_size") is not None


@pytest.mark.engine("sglang")
@pytest.mark.gpu(1)
@pytest.mark.e2e
@pytest.mark.gateway(policy="round_robin")
@pytest.mark.parametrize("setup_backend", ["http"], indirect=True)
class TestRlControlPlaneDisabled:
    def test_v1_rl_is_404_without_flag(self, setup_backend):
        _backend, _model, _client, gateway = setup_backend
        resp = httpx.get(f"{gateway.base_url}/v1/rl/workers", timeout=TIMEOUT)
        assert resp.status_code == 404
        assert resp.content == b""
