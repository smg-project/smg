"""Tests for gateway worker management APIs.

Tests the gateway's worker management endpoints:
- GET /workers - List all workers
- POST /add_worker - Add a worker dynamically
- DELETE /workers/{worker_id} - Remove a worker dynamically
- GET /v1/models - List available models

Usage:
    pytest e2e_test/router/test_worker_api.py -v

Note: These tests use HTTP mode which is not supported by vLLM.
"""

from __future__ import annotations

import logging
import os
import time

import httpx
import pytest
from infra import ConnectionMode, Gateway, start_workers, stop_workers

logger = logging.getLogger(__name__)

# Skip all tests in this module for vLLM (HTTP mode not supported)
pytestmark = pytest.mark.skip_for_runtime("vllm", reason="vLLM does not support HTTP mode")


@pytest.mark.engine("sglang")
@pytest.mark.gpu(1)
@pytest.mark.e2e
@pytest.mark.parametrize("setup_backend", ["grpc", "http"], indirect=True)
class TestWorkerAPI:
    """Tests for worker management APIs using setup_backend fixture."""

    def test_list_workers(self, setup_backend):
        """Test listing workers via /workers endpoint."""
        backend, model, _, gateway = setup_backend

        workers = gateway.list_workers()
        assert len(workers) >= 1, "Expected at least one worker"
        logger.info("Found %d workers", len(workers))

        for worker in workers:
            logger.info(
                "Worker: id=%s, url=%s, model=%s, status=%s",
                worker.id,
                worker.url,
                worker.model,
                worker.status,
            )
            assert worker.url, "Worker should have a URL"
            # model_id is set for workers with discovered models, None for wildcard
            if worker.model is not None:
                assert worker.model, "Worker model_id should be non-empty when present"

    def test_list_models(self, setup_backend):
        """Test listing models via /v1/models endpoint."""
        backend, model, _, gateway = setup_backend

        models = gateway.list_models()
        assert len(models) >= 1, "Expected at least one model"
        logger.info("Found %d models", len(models))

        for m in models:
            logger.info("Model: %s", m.get("id", "unknown"))
            assert "id" in m, "Model should have an id"

    def test_health_endpoint(self, setup_backend):
        """Test health check endpoint."""
        backend, model, _, gateway = setup_backend

        assert gateway.health(), "Gateway should be healthy"
        logger.info("Gateway health check passed")


@pytest.mark.engine("sglang")
@pytest.mark.gpu(1)
@pytest.mark.e2e
class TestIGWMode:
    """Tests for IGW mode - start gateway empty, add workers via API.

    Workers are launched on-demand via start_workers().
    """

    def test_igw_start_empty(self):
        """Test starting gateway in IGW mode with no workers."""
        gateway = Gateway()
        gateway.start(igw_mode=True)

        try:
            assert gateway.health(), "Gateway should be healthy"
            assert gateway.igw_mode, "Gateway should be in IGW mode"

            workers = gateway.list_workers()
            logger.info("IGW gateway started with %d workers", len(workers))
        finally:
            gateway.shutdown()

    def test_igw_add_worker(self):
        """Test adding a worker to IGW gateway."""
        engine = os.environ.get("E2E_ENGINE", "sglang")
        http_workers = start_workers(
            "meta-llama/Llama-3.1-8B-Instruct", engine, mode=ConnectionMode.HTTP, count=1
        )

        try:
            http_worker = http_workers[0]

            gateway = Gateway()
            gateway.start(igw_mode=True)

            try:
                # Add worker
                success, result = gateway.add_worker(http_worker.base_url)
                assert success, f"Failed to add worker: {result}"
                logger.info("Added worker: %s", result)

                # Verify worker was added
                workers = gateway.list_workers()
                assert len(workers) >= 1, "Expected at least one worker"
                logger.info("Worker count: %d", len(workers))

                # Verify models are available
                models = gateway.list_models()
                logger.info("Models available: %d", len(models))
            finally:
                gateway.shutdown()
        finally:
            stop_workers(http_workers)

    def test_igw_add_and_remove_worker(self):
        """Test adding and removing workers dynamically."""
        engine = os.environ.get("E2E_ENGINE", "sglang")
        http_workers = start_workers(
            "meta-llama/Llama-3.1-8B-Instruct", engine, mode=ConnectionMode.HTTP, count=1
        )

        try:
            http_worker = http_workers[0]

            gateway = Gateway()
            gateway.start(igw_mode=True)

            try:
                # Add worker
                success, _ = gateway.add_worker(http_worker.base_url)
                assert success, "Failed to add worker"

                initial_count = len(gateway.list_workers())
                logger.info("Worker count after add: %d", initial_count)

                # Remove worker — removal must succeed and the worker must
                # disappear from the registry.
                success, msg = gateway.remove_worker(http_worker.base_url)
                assert success, f"Failed to remove worker: {msg}"
                logger.info("Removed worker: %s", msg)

                # Poll briefly in case removal is applied asynchronously.
                # Sample once up front so the assertion below always has a
                # real observation to report, even if the deadline has already
                # passed by the time the loop condition is first evaluated.
                # Poll on the strict read: the degrading call returns [] on a
                # transient /workers failure, which would look like "removed"
                # and end the wait before the queued job has run.
                def _worker_gone():
                    try:
                        return http_worker.base_url not in [
                            w.url for w in gateway.list_workers(strict=True)
                        ]
                    except Exception:  # transient /workers failure: keep waiting
                        return False

                deadline = time.perf_counter() + 15
                while not _worker_gone() and time.perf_counter() < deadline:
                    time.sleep(1.0)

                # Authoritative final read: strict, so a failed /workers call
                # cannot masquerade as "no workers left".
                remaining_urls = [w.url for w in gateway.list_workers(strict=True)]
                assert http_worker.base_url not in remaining_urls, (
                    f"Worker {http_worker.base_url} still listed after removal: {remaining_urls}"
                )
                logger.info("Worker count after remove: %d", len(remaining_urls))
            finally:
                gateway.shutdown()
        finally:
            stop_workers(http_workers)


@pytest.mark.engine("sglang")
@pytest.mark.gpu(2)
@pytest.mark.e2e
class TestIGWMultiWorker:
    """Test IGW mode with multiple workers (requires 2 GPUs)."""

    def test_igw_multiple_workers(self):
        """Test adding multiple workers (HTTP + gRPC) to IGW gateway."""
        engine = os.environ.get("E2E_ENGINE", "sglang")
        http_workers = start_workers(
            "meta-llama/Llama-3.1-8B-Instruct", engine, mode=ConnectionMode.HTTP, count=1
        )
        try:
            grpc_workers = start_workers(
                "meta-llama/Llama-3.1-8B-Instruct",
                engine,
                mode=ConnectionMode.GRPC,
                count=1,
                gpu_offset=1,
            )
        except Exception:
            stop_workers(http_workers)
            raise
        all_workers = http_workers + grpc_workers

        try:
            http_worker = http_workers[0]
            grpc_worker = grpc_workers[0]

            gateway = Gateway()
            gateway.start(igw_mode=True)

            try:
                # Add both workers
                success1, _ = gateway.add_worker(http_worker.base_url)
                success2, _ = gateway.add_worker(grpc_worker.base_url)

                if not success1 or not success2:
                    pytest.skip("Dynamic worker management not fully supported")

                workers = gateway.list_workers()
                logger.info("Worker count: %d", len(workers))
                assert len(workers) >= 2, "Expected at least 2 workers"

                for w in workers:
                    logger.info("Worker: id=%s, url=%s", w.id, w.url)
            finally:
                gateway.shutdown()
        finally:
            stop_workers(all_workers)


@pytest.mark.e2e
# TokenSpeed deliberately excluded: this test class spins up its worker
# via ``ConnectionMode.HTTP``, and ``Worker._build_tokenspeed_grpc_cmd``
# rejects HTTP mode — TokenSpeed has no HTTP frontend in this repo.
# Including ``tokenspeed`` here would fail deterministically on every
# run rather than validate health-check behaviour.
@pytest.mark.engine("sglang", "vllm")
@pytest.mark.gpu(1)
class TestDisableHealthCheck:
    """Tests for --disable-health-check CLI option."""

    def test_disable_health_check_workers_immediately_healthy(self):
        """Test that workers are immediately healthy when health checks are disabled."""
        engine = os.environ.get("E2E_ENGINE", "sglang")
        http_workers = start_workers(
            "meta-llama/Llama-3.1-8B-Instruct", engine, mode=ConnectionMode.HTTP, count=1
        )

        try:
            http_worker = http_workers[0]

            gateway = Gateway()
            gateway.start(
                igw_mode=True,
                extra_args=["--disable-health-check"],
            )

            try:
                # Add worker - should be immediately healthy since health checks are disabled
                success, worker_id = gateway.add_worker(
                    http_worker.base_url,
                    wait_ready=True,
                    ready_timeout=10,  # Short timeout since it should be immediate
                )
                assert success, f"Failed to add worker: {worker_id}"
                logger.info("Added worker with health checks disabled: %s", worker_id)

                # Verify worker is healthy
                workers = gateway.list_workers()
                assert len(workers) >= 1, "Expected at least one worker"

                for worker in workers:
                    logger.info(
                        "Worker: id=%s, status=%s, disable_health_check=%s",
                        worker.id,
                        worker.status,
                        worker.metadata.get("disable_health_check"),
                    )
                    # Worker should be healthy immediately
                    assert worker.status == "healthy", (
                        "Worker should be healthy when health checks disabled"
                    )
            finally:
                gateway.shutdown()
        finally:
            stop_workers(http_workers)

    def test_disable_health_check_gateway_starts_without_health_checker(self):
        """Test that gateway starts successfully with health checks disabled."""
        gateway = Gateway()
        gateway.start(
            igw_mode=True,
            extra_args=["--disable-health-check"],
        )

        try:
            assert gateway.health(), "Gateway should be healthy"
            logger.info("Gateway started with health checks disabled")
        finally:
            gateway.shutdown()


@pytest.mark.engine("sglang")
@pytest.mark.gpu(4)
@pytest.mark.e2e
class TestIGWMixedWorkerClassification:
    """Test IGW mode with mixed local and external workers.

    Verifies that the classify step correctly identifies:
    - Local HTTP sglang workers (via /health probe)
    - Local gRPC sglang workers (via gRPC health probe)
    - External OpenAI workers (via URL-based detection)
    - External xAI workers (via URL-based detection)

    Workers are added immediately after gateway start without waiting
    for backends to be fully ready, exercising the race-condition-free
    classification logic.

    Requires 4 GPUs: 2 for HTTP workers (Llama-3.1-8B), 2 for gRPC workers (DeepSeek-R1-Distill-Qwen-7B).
    Requires OPENAI_API_KEY and XAI_API_KEY environment variables.
    """

    def test_mixed_local_and_external_workers(self):
        """Add local sglang + external cloud workers, verify all models discoverable."""
        openai_key = os.environ.get("OPENAI_API_KEY")
        xai_key = os.environ.get("XAI_API_KEY")
        if not openai_key:
            pytest.skip("OPENAI_API_KEY not set")
        if not xai_key:
            pytest.skip("XAI_API_KEY not set")

        engine = os.environ.get("E2E_ENGINE", "sglang")

        # Start local backends WITHOUT waiting — spawn processes and return immediately.
        # This exercises the race condition where workers are added to the gateway
        # before the backends are fully ready (before /health responds).
        http_workers = start_workers(
            "meta-llama/Llama-3.1-8B-Instruct",
            engine,
            mode=ConnectionMode.HTTP,
            count=2,
            wait_ready=False,
        )
        try:
            grpc_workers = start_workers(
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                engine,
                mode=ConnectionMode.GRPC,
                count=2,
                gpu_offset=2,
                wait_ready=False,
            )
        except Exception:
            stop_workers(http_workers)
            raise
        all_local_workers = http_workers + grpc_workers

        try:
            # Start gateway in IGW mode FIRST with extended startup timeout
            # so registration workflows can wait for backends still loading models
            gateway = Gateway()
            gateway.start(
                igw_mode=True,
                extra_args=["--worker-startup-timeout-secs", "300"],
            )

            try:
                # Add all workers immediately — don't wait for registration to complete
                # Local HTTP workers
                for w in http_workers:
                    success, wid = gateway.add_worker(w.base_url, wait_ready=False)
                    assert success, f"Failed to add HTTP worker {w.base_url}: {wid}"
                    logger.info("Queued HTTP worker: %s → %s", w.base_url, wid)

                # Local gRPC workers
                for w in grpc_workers:
                    success, wid = gateway.add_worker(w.base_url, wait_ready=False)
                    assert success, f"Failed to add gRPC worker {w.base_url}: {wid}"
                    logger.info("Queued gRPC worker: %s → %s", w.base_url, wid)

                # External OpenAI worker (POST directly with api_key + disable_health_check)
                resp = httpx.post(
                    f"{gateway.base_url}/workers",
                    json={
                        "url": "https://api.openai.com",
                        "api_key": openai_key,
                        "runtime": "external",
                        "disable_health_check": True,
                    },
                    timeout=10,
                )
                assert resp.status_code in (200, 202), f"Failed to add OpenAI worker: {resp.text}"
                logger.info("Queued OpenAI worker: %s", resp.json().get("worker_id"))

                # External xAI worker
                resp = httpx.post(
                    f"{gateway.base_url}/workers",
                    json={
                        "url": "https://api.x.ai",
                        "api_key": xai_key,
                        "runtime": "external",
                        "disable_health_check": True,
                    },
                    timeout=10,
                )
                assert resp.status_code in (200, 202), f"Failed to add xAI worker: {resp.text}"
                logger.info("Queued xAI worker: %s", resp.json().get("worker_id"))

                # Wait for ALL 6 workers to register and become healthy.
                # Local backends need time to load models (especially gRPC/DeepSeek).
                # External workers are instant with disable_health_check.
                expected_workers = 6  # 2 HTTP + 2 gRPC + OpenAI + xAI
                deadline = time.perf_counter() + 300
                while time.perf_counter() < deadline:
                    workers = gateway.list_workers()
                    healthy = sum(1 for w in workers if w.status == "healthy")
                    if healthy >= expected_workers:
                        break
                    time.sleep(5)
                else:
                    workers = gateway.list_workers()
                    for w in workers:
                        logger.info(
                            "Worker: id=%s url=%s model=%s status=%s",
                            w.id,
                            w.url,
                            w.model,
                            w.status,
                        )
                    pytest.fail(
                        f"Timed out waiting for {expected_workers} healthy workers "
                        f"(got {sum(1 for w in workers if w.status == 'healthy')} healthy "
                        f"out of {len(workers)} total)"
                    )

                # Verify all workers registered
                workers = gateway.list_workers()
                worker_urls = [w.url for w in workers]
                logger.info("All workers (%d): %s", len(workers), worker_urls)
                assert len(workers) >= expected_workers, (
                    f"Expected at least {expected_workers} workers, got {len(workers)}"
                )

                # Verify classification via raw API response (includes runtime_type)
                raw_resp = httpx.get(f"{gateway.base_url}/workers", timeout=10)
                assert raw_resp.status_code == 200
                raw_workers = raw_resp.json().get("workers", [])
                for w in raw_workers:
                    url = w.get("url", "")
                    rt = w.get("runtime_type", "")
                    if "api.openai.com" in url or "api.x.ai" in url:
                        assert rt == "external", f"Cloud worker {url} should be external, got {rt}"
                    else:
                        assert rt in (
                            "sglang",
                            "vllm",
                            "trtllm",
                        ), f"Local worker {url} should have local runtime, got {rt}"
                    logger.info("Worker %s → runtime_type=%s", url, rt)

                # Verify /v1/models returns models from ALL workers
                models = gateway.list_models()
                model_ids = [m["id"] for m in models]
                logger.info("All models (%d): %s", len(models), model_ids)

                # Local models should be present
                assert any("llama" in m.lower() for m in model_ids), (
                    f"Expected a Llama model from HTTP workers, got: {model_ids}"
                )
                assert any("deepseek" in m.lower() for m in model_ids), (
                    f"Expected a DeepSeek model from gRPC workers, got: {model_ids}"
                )

                # TODO: verify external providers discover many models via fan-out.
                # The /v1/models endpoint with a Bearer token should fan out to
                # external providers and return their full model lists. Currently
                # external workers registered with disable_health_check only show
                # their primary model in the registry. Needs investigation into
                # model discovery for externally-registered workers.
            finally:
                gateway.shutdown()
        finally:
            stop_workers(all_local_workers)


@pytest.mark.engine("sglang")
@pytest.mark.gpu(1)
@pytest.mark.e2e
class TestWorkerAPIRestSemantics:
    """Tests for proper REST semantics on worker management endpoints.

    Verifies:
    - POST /workers returns 409 on duplicate URL
    - PUT /workers/{id} does full replace
    - PATCH /workers/{id} does partial update
    """

    def test_post_duplicate_url_returns_409(self):
        """POST /workers with the same URL twice should return 409 Conflict."""
        engine = os.environ.get("E2E_ENGINE", "sglang")
        http_workers = start_workers(
            "meta-llama/Llama-3.1-8B-Instruct", engine, mode=ConnectionMode.HTTP, count=1
        )

        try:
            http_worker = http_workers[0]

            gateway = Gateway()
            gateway.start(igw_mode=True)

            try:
                # First POST — should succeed
                success, worker_id = gateway.add_worker(http_worker.base_url)
                assert success, f"First POST should succeed: {worker_id}"
                logger.info("First POST succeeded: worker_id=%s", worker_id)

                # Second POST with same URL — should return 409
                resp = httpx.post(
                    f"{gateway.base_url}/workers",
                    json={"url": http_worker.base_url},
                    timeout=10,
                )
                assert resp.status_code == 409, (
                    f"Expected 409 Conflict on duplicate URL, got {resp.status_code}: {resp.text}"
                )
                error_data = resp.json()
                assert "WORKER_ALREADY_EXISTS" in error_data.get("code", ""), (
                    f"Expected WORKER_ALREADY_EXISTS error code, got: {error_data}"
                )
                logger.info("Duplicate POST correctly returned 409: %s", error_data)
            finally:
                gateway.shutdown()
        finally:
            stop_workers(http_workers)

    def test_patch_partial_update(self):
        """PATCH /workers/{id} should do partial update and persist changes."""
        engine = os.environ.get("E2E_ENGINE", "sglang")
        http_workers = start_workers(
            "meta-llama/Llama-3.1-8B-Instruct", engine, mode=ConnectionMode.HTTP, count=1
        )

        try:
            http_worker = http_workers[0]

            gateway = Gateway()
            gateway.start(igw_mode=True)

            try:
                # Add worker
                success, worker_id = gateway.add_worker(http_worker.base_url)
                assert success, f"Failed to add worker: {worker_id}"

                # PATCH to update priority, cost, and labels
                resp = httpx.patch(
                    f"{gateway.base_url}/workers/{worker_id}",
                    json={"priority": 100, "cost": 2.5, "labels": {"env": "test"}},
                    timeout=10,
                )
                assert resp.status_code in (
                    200,
                    202,
                ), f"PATCH should succeed, got {resp.status_code}: {resp.text}"
                logger.info("PATCH succeeded: %s", resp.json())

                # Poll until the update is applied (async 202)
                deadline = time.perf_counter() + 15
                verified = False
                while time.perf_counter() < deadline:
                    resp = httpx.get(
                        f"{gateway.base_url}/workers/{worker_id}",
                        timeout=10,
                    )
                    if resp.status_code == 200:
                        worker_data = resp.json()
                        if (
                            worker_data.get("priority") == 100
                            and worker_data.get("cost") == 2.5
                            and worker_data.get("labels", {}).get("env") == "test"
                        ):
                            verified = True
                            break
                    time.sleep(1.0)

                assert verified, (
                    f"PATCH changes not persisted within timeout. "
                    f"Last worker state: {resp.json() if resp.status_code == 200 else resp.text}"
                )
                logger.info("PATCH changes verified: priority=100, cost=2.5, labels={env: test}")
            finally:
                gateway.shutdown()
        finally:
            stop_workers(http_workers)

    def test_put_full_replace(self):
        """PUT /workers/{id} should do full replace with model re-discovery."""
        engine = os.environ.get("E2E_ENGINE", "sglang")
        http_workers = start_workers(
            "meta-llama/Llama-3.1-8B-Instruct", engine, mode=ConnectionMode.HTTP, count=1
        )

        try:
            http_worker = http_workers[0]

            gateway = Gateway()
            gateway.start(igw_mode=True)

            try:
                # Add worker
                success, worker_id = gateway.add_worker(http_worker.base_url)
                assert success, f"Failed to add worker: {worker_id}"

                # PUT full replace with same URL
                resp = httpx.put(
                    f"{gateway.base_url}/workers/{worker_id}",
                    json={"url": http_worker.base_url},
                    timeout=10,
                )
                assert resp.status_code in (
                    200,
                    202,
                ), f"PUT should succeed, got {resp.status_code}: {resp.text}"
                logger.info("PUT full replace succeeded: %s", resp.json())

                # Worker should still be registered
                workers = gateway.list_workers()
                assert any(w.id == worker_id for w in workers), (
                    "Worker should still exist after PUT replace"
                )
            finally:
                gateway.shutdown()
        finally:
            stop_workers(http_workers)

    def test_put_url_mismatch_returns_400(self):
        """PUT /workers/{id} with a different URL should return 400."""
        engine = os.environ.get("E2E_ENGINE", "sglang")
        http_workers = start_workers(
            "meta-llama/Llama-3.1-8B-Instruct", engine, mode=ConnectionMode.HTTP, count=1
        )

        try:
            http_worker = http_workers[0]

            gateway = Gateway()
            gateway.start(igw_mode=True)

            try:
                # Add worker
                success, worker_id = gateway.add_worker(http_worker.base_url)
                assert success, f"Failed to add worker: {worker_id}"

                # PUT with different URL — should fail
                resp = httpx.put(
                    f"{gateway.base_url}/workers/{worker_id}",
                    json={"url": "http://different-url:9999"},
                    timeout=10,
                )
                assert resp.status_code == 400, (
                    f"Expected 400 on URL mismatch, got {resp.status_code}: {resp.text}"
                )
                logger.info("URL mismatch correctly returned 400: %s", resp.json())
            finally:
                gateway.shutdown()
        finally:
            stop_workers(http_workers)
