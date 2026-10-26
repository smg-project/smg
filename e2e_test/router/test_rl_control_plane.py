"""E2E tests for the RL control plane (/v1/rl) against real engines.

The same behavior runs on an SGLang HTTP worker (controlled through itself)
and on a TokenSpeed gRPC worker (controlled through its in-engine control app,
discovered from server info).

Usage:
    pytest e2e_test/router/test_rl_control_plane.py -v
"""

from __future__ import annotations

import logging

import httpx
import pytest
from infra import get_servicer_impl

logger = logging.getLogger(__name__)

TIMEOUT = 60.0
REFIT_TIMEOUT = 600.0


class RlControlPlaneBehavior:
    ENGINE: str
    CONNECTION_MODE: str

    @property
    def selector(self) -> str:
        return f"engine={self.ENGINE}"

    def _engine(self, gateway, op: str, body: dict, timeout: float = TIMEOUT) -> httpx.Response:
        return httpx.post(
            f"{gateway.base_url}/v1/rl/engine/{op}",
            params={"selector": self.selector},
            json=body,
            timeout=timeout,
        )

    def _workers(self, gateway):
        resp = httpx.get(f"{gateway.base_url}/v1/rl/workers", timeout=TIMEOUT)
        assert resp.status_code == 200, resp.text
        return resp.json()

    def test_discovery_reports_engine_topology_and_control_endpoint(self, setup_backend):
        _backend, _model, _client, gateway = setup_backend
        body = self._workers(gateway)
        assert body["total"] >= 1
        w = body["workers"][0]
        assert w["engine"] == self.ENGINE
        assert w["connection_mode"] == self.CONNECTION_MODE
        assert w["tp_size"] is not None
        assert w["health"] == "ready"
        assert w["control_url"], "every engine in this lane has a control endpoint"
        assert "abort" in w["capabilities"]["pause_modes"]

    def test_fanout_flush_cache(self, setup_backend):
        _backend, _model, _client, gateway = setup_backend
        resp = httpx.post(
            f"{gateway.base_url}/v1/rl/engine/flush_cache",
            params={"selector": self.selector},
            json={},
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
                params={"selector": self.selector},
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


@pytest.mark.engine("sglang")
@pytest.mark.gpu(1)
@pytest.mark.e2e
@pytest.mark.gateway(policy="round_robin", extra_args=["--enable-rl"])
@pytest.mark.parametrize("setup_backend", ["http"], indirect=True)
class TestRlControlPlaneSglang(RlControlPlaneBehavior):
    ENGINE = "sglang"
    CONNECTION_MODE = "http"

    def test_single_worker_proxy_server_info(self, setup_backend):
        _backend, _model, _client, gateway = setup_backend
        wid = self._workers(gateway)["workers"][0]["id"]
        resp = httpx.get(
            f"{gateway.base_url}/v1/rl/workers/{wid}/engine/server_info", timeout=TIMEOUT
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["body"].get("tp_size") is not None

    def test_refit_from_disk_is_visible_on_the_next_generate(self, setup_backend):
        _backend, model, _client, gateway = setup_backend
        w = self._workers(gateway)["workers"][0]
        model_path = w["labels"].get("model_path") or w["labels"].get("model")
        assert model_path, w["labels"]
        resp = self._engine(gateway, "pause_generation", {})
        assert resp.status_code == 200, f"pause_generation: {resp.text}"
        try:
            resp = self._engine(
                gateway,
                "update_weights_from_disk",
                {"model_path": model_path, "weight_version": "e2e-refit", "flush_cache": True},
                timeout=REFIT_TIMEOUT,
            )
            assert resp.status_code == 200, f"update_weights_from_disk: {resp.text}"
        finally:
            # Resume even when the refit assertion fails: the worker is shared
            # with the tests that follow, and a paused one would fail them all.
            resumed = self._engine(gateway, "continue_generation", {})
        assert resumed.status_code == 200, f"continue_generation: {resumed.text}"
        resp = httpx.post(
            f"{gateway.base_url}/generate",
            json={"model": model, "text": "Hello", "sampling_params": {"max_new_tokens": 4}},
            timeout=TIMEOUT,
        )
        assert resp.status_code == 200, resp.text
        out = resp.json()
        first = out[0] if isinstance(out, list) else out
        assert first["meta_info"]["weight_version"] == "e2e-refit", first["meta_info"]


@pytest.mark.skipif(
    get_servicer_impl("tokenspeed") == "rust",
    reason="the Rust TokenSpeed servicer runs a headless scheduler with no in-engine "
    "control app, so its worker advertises no rl.control_url",
)
@pytest.mark.engine("tokenspeed")
@pytest.mark.gpu(1)
@pytest.mark.e2e
@pytest.mark.gateway(policy="round_robin", extra_args=["--enable-rl"])
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestRlControlPlaneTokenSpeed(RlControlPlaneBehavior):
    """TokenSpeed runs gRPC-only behind the gateway; control goes to its in-engine app."""

    ENGINE = "tokenspeed"
    CONNECTION_MODE = "grpc"

    def test_single_worker_proxy_model_info(self, setup_backend):
        _backend, _model, _client, gateway = setup_backend
        wid = self._workers(gateway)["workers"][0]["id"]
        resp = httpx.get(
            f"{gateway.base_url}/v1/rl/workers/{wid}/engine/model_info", timeout=TIMEOUT
        )
        assert resp.status_code == 200, resp.text
        assert "weight_version" in resp.json()["body"]

    def test_discovery_reports_the_refits_the_engine_advertises(self, setup_backend):
        _backend, _model, _client, gateway = setup_backend
        w = self._workers(gateway)["workers"][0]
        # The engine lists the loads its scheduler implements, and which ones
        # depends on the build (older builds: distributed only; current
        # nightlies add disk and mooncake). Discovery reports that list.
        assert "rl.update_from" in w["labels"], w["labels"]
        advertised = [x.strip() for x in w["labels"]["rl.update_from"].split(",") if x.strip()]
        assert w["capabilities"]["source"] == "label"
        assert set(w["capabilities"]["update_from"]) == set(advertised)
        assert "distributed" in advertised

    def test_a_disk_refit_is_answered_as_advertised_and_the_engine_keeps_serving(
        self, setup_backend
    ):
        _backend, model, client, gateway = setup_backend
        workers = self._workers(gateway)["workers"]
        model_path = workers[0]["labels"].get("model_path") or workers[0]["labels"].get("model")
        assert model_path, workers[0]["labels"]
        accepts = {"disk" in w["capabilities"]["update_from"] for w in workers}
        assert len(accepts) == 1, "the lane's workers disagree on disk refits"
        accepts_disk = accepts.pop()

        resp = self._engine(gateway, "pause_generation", {})
        assert resp.status_code == 200, f"pause_generation: {resp.text}"
        try:
            resp = self._engine(
                gateway,
                "update_weights_from_disk",
                {"model_path": model_path, "weight_version": "e2e-disk", "flush_cache": True},
                timeout=REFIT_TIMEOUT,
            )
            if accepts_disk:
                assert resp.status_code == 200, resp.text
                body = resp.json()
                assert body["succeeded"] == len(workers), body
                for outcome in body["results"].values():
                    assert outcome["body"]["success"] is True, outcome
            else:
                assert resp.status_code == 207, resp.text
                body = resp.json()
                assert body["succeeded"] == 0
                assert len(body["failed"]) == len(workers), body["failed"]
                for failure in body["failed"]:
                    assert failure["error"] == "upstream_error"
                    assert failure["status"] == 501
                    outcome = body["results"][failure["worker_id"]]
                    assert "distributed" in outcome["body"]["message"]
        finally:
            # A refit, accepted or refused, leaves the engine paused like any
            # other control call would; resume it whatever the assertions
            # above decided.
            resumed = self._engine(gateway, "continue_generation", {})
        assert resumed.status_code == 200, f"continue_generation: {resumed.text}"

        if accepts_disk:
            resp = httpx.post(
                f"{gateway.base_url}/generate",
                json={"model": model, "text": "Hello", "sampling_params": {"max_new_tokens": 4}},
                timeout=TIMEOUT,
            )
            assert resp.status_code == 200, resp.text
            out = resp.json()
            first = out[0] if isinstance(out, list) else out
            assert first["meta_info"]["weight_version"] == "e2e-disk", first["meta_info"]

        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say hi"}],
            max_tokens=4,
        )
        assert completion.choices


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
        # An unmounted route answers the JSON not-found envelope, never an empty body.
        body = resp.json()
        assert body["error"]["code"] == "unknown_url", body
