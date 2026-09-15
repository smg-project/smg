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


@pytest.mark.engine("sglang")
@pytest.mark.gpu(1)
@pytest.mark.e2e
@pytest.mark.gateway(
    policy="round_robin",
    extra_args=[
        "--enable-rl",
        "--rl-version-policy",
        "latest-only",
        "--retry-max-retries",
        "3",
        "--retry-initial-backoff-ms",
        "50",
        "--retry-max-backoff-ms",
        "200",
    ],
)
@pytest.mark.parametrize("setup_backend", ["http"], indirect=True)
class TestRlVersionRouting:
    def _chat(self, gateway, model, **kwargs):
        return httpx.post(
            f"{gateway.base_url}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Say hi"}],
                "max_tokens": 4,
            },
            timeout=TIMEOUT,
            **kwargs,
        )

    def test_responses_are_stamped_and_follow_a_refit(self, setup_backend):
        _backend, model, _client, gateway = setup_backend
        # Fans out update_weight_version. There is no fallback here: if the
        # engine in the harness is an older SGLang that 404s on this RPC,
        # switch this call to update_weights_from_disk with model_path=model
        # and a weight_version field (mirrors examples/rl/refit_from_disk.py).
        resp = httpx.post(
            f"{gateway.base_url}/v1/rl/engine/update_weight_version",
            params={"selector": "engine=sglang"},
            json={"new_version": "7", "abort_all_requests": False},
            timeout=TIMEOUT,
        )
        assert resp.status_code == 200, resp.text
        resp = self._chat(gateway, model)
        assert resp.status_code == 200, resp.text
        assert resp.headers.get("x-smg-weight-version") == "7"
        assert resp.headers.get("x-smg-routed-worker-id")
        rows = httpx.get(f"{gateway.base_url}/v1/rl/workers", timeout=TIMEOUT).json()["workers"]
        assert {w["weight_version"] for w in rows} == {"7"}
        assert {w["version_source"] for w in rows} == {"passthrough"}
        assert {w["control"] for w in rows} == {"active"}

    def test_a_paused_engine_is_not_routed(self, setup_backend):
        _backend, model, _client, gateway = setup_backend
        pause = httpx.post(
            f"{gateway.base_url}/v1/rl/engine/pause_generation",
            params={"selector": "engine=sglang"},
            json={},
            timeout=TIMEOUT,
        )
        assert pause.status_code == 200, pause.text
        try:
            resp = self._chat(gateway, model)
            assert resp.status_code == 503, resp.text
            assert resp.headers.get("x-smg-routed-worker-id") is None
        finally:
            resume = httpx.post(
                f"{gateway.base_url}/v1/rl/engine/continue_generation",
                params={"selector": "engine=sglang"},
                json={},
                timeout=TIMEOUT,
            )
            assert resume.status_code == 200, resume.text
        resp = self._chat(gateway, model)
        assert resp.status_code == 200, resp.text

    def test_invalid_policy_header_is_rejected(self, setup_backend):
        _backend, model, _client, gateway = setup_backend
        resp = self._chat(gateway, model, headers={"x-smg-version-policy": "freshest"})
        assert resp.status_code == 400
        assert resp.json()["error"] == "invalid_version_policy"
