"""CPU-only unit tests for external worker attachment, health checks, and pool lifecycle."""

from __future__ import annotations

import json
import socket
import threading
from collections.abc import Iterator
from concurrent import futures
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import pytest
from infra import constants, worker_pool
from infra.constants import (
    DEFAULT_HOST,
    ENV_EXTERNAL_WORKERS,
    ENV_EXTERNAL_WORKERS_FLUSH_CACHE,
    ENV_MM_PROCESSING,
    ENV_VLLM_SERVICER_IMPL,
    ConnectionMode,
    WorkerType,
    host_port,
)
from infra.external_workers import (
    ExternalWorkerAttachError,
    ExternalWorkerRegistry,
    load_external_workers,
    parse_external_worker_url,
)
from infra.gateway import Gateway
from infra.process_utils import _reserved_ports, reserve_port
from infra.worker import Worker, stop_workers


@contextmanager
def _http_server(
    *,
    health_status: int = 200,
    flush_status: int = 200,
) -> Iterator[tuple[str, list[tuple[str, str]]]]:
    """Run a loopback HTTP stub serving ``/health`` and ``/flush_cache``."""
    requests_seen: list[tuple[str, str]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            requests_seen.append(("GET", self.path))
            code = health_status if self.path == "/health" else 404
            self.send_response(code)
            self.end_headers()

        def do_POST(self) -> None:
            requests_seen.append(("POST", self.path))
            code = flush_status if self.path == "/flush_cache" else 404
            body = b'{"status":"success"}' if code == 200 else b"unsupported"
            self.send_response(code)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}", requests_seen
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@contextmanager
def _grpc_health_server(status_name: str) -> Iterator[str]:
    """Run a loopback gRPC health server answering with ``status_name``."""
    pytest.importorskip("grpc")
    pytest.importorskip("grpc_health")
    import grpc
    from grpc_health.v1 import health_pb2, health_pb2_grpc

    status = getattr(health_pb2.HealthCheckResponse, status_name)

    class Health(health_pb2_grpc.HealthServicer):
        def Check(self, request, context):
            return health_pb2.HealthCheckResponse(status=status)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    health_pb2_grpc.add_HealthServicer_to_server(Health(), server)
    port = server.add_insecure_port(host_port(DEFAULT_HOST, 0))
    server.start()
    try:
        yield f"grpc://{host_port(DEFAULT_HOST, port)}"
    finally:
        server.stop(0)


def _unused_port() -> int:
    """Return an ephemeral loopback port that nothing is listening on."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(autouse=True)
def _clean_external_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure external worker env vars do not leak across tests."""
    for var in (
        ENV_EXTERNAL_WORKERS,
        ENV_EXTERNAL_WORKERS_FLUSH_CACHE,
        ENV_VLLM_SERVICER_IMPL,
        ENV_MM_PROCESSING,
    ):
        monkeypatch.delenv(var, raising=False)


# ---------------------------------------------------------------------------
# URL & configuration parsing
# ---------------------------------------------------------------------------


def test_load_external_workers_empty_when_unset_or_blank(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset or blank configuration returns an empty, falsy registry."""
    assert not load_external_workers()
    monkeypatch.setenv(ENV_EXTERNAL_WORKERS, "   ")
    assert not load_external_workers()


def test_parse_external_worker_url_schemes_ports_and_ipv6() -> None:
    """URL parser infers modes, normalizes slashes, and supports IPv6 literals."""
    assert parse_external_worker_url("grpc://10.0.0.5:30000/") == (
        "grpc://10.0.0.5:30000",
        ConnectionMode.GRPC,
        "10.0.0.5",
        30000,
    )
    assert parse_external_worker_url("http://engine.local") == (
        "http://engine.local",
        ConnectionMode.HTTP,
        "engine.local",
        80,
    )
    assert parse_external_worker_url("https://engine.local/") == (
        "https://engine.local",
        ConnectionMode.HTTP,
        "engine.local",
        443,
    )
    assert parse_external_worker_url("HTTPS://Engine.Local:443/") == (
        "https://engine.local:443",
        ConnectionMode.HTTP,
        "engine.local",
        443,
    )
    assert parse_external_worker_url("grpc://[::1]:31234", ConnectionMode.GRPC) == (
        "grpc://[::1]:31234",
        ConnectionMode.GRPC,
        "::1",
        31234,
    )


@pytest.mark.parametrize(
    ("url", "mode", "match"),
    [
        ("", None, "non-empty string"),
        ("ipc:///tmp/sock", None, "Unsupported external worker URL scheme"),
        ("grpc://127.0.0.1", None, "explicit port"),
        ("grpc://:30000", None, "missing a hostname"),
        ("http://127.0.0.1:30000", ConnectionMode.GRPC, "does not match"),
        ("grpc://127.0.0.1:30000", ConnectionMode.ZMQ, "only support HTTP and gRPC"),
    ],
)
def test_parse_external_worker_url_rejects_invalid(
    url: str, mode: ConnectionMode | None, match: str
) -> None:
    """Malformed URLs and incompatible schemes/modes fail loudly."""
    with pytest.raises(ValueError, match=match):
        parse_external_worker_url(url, mode)


def test_load_external_workers_inline_object_list_and_file(tmp_path) -> None:
    """Loader accepts inline objects, bare lists, and @-prefixed or plain JSON files."""
    payload = {
        "workers": [
            {
                "engine": "VLLM",
                "model_id": "meta-llama/Llama-3.1-8B-Instruct",
                "url": "grpc://10.0.0.1:30000/",
            },
            {
                "engine": "sglang",
                "model_id": "Qwen/Qwen2.5-7B-Instruct",
                "mode": "http",
                "urls": ["http://10.0.0.2:8000", "http://10.0.0.2:8001"],
                "gpus": 1,
                "extra_engine_args": ["--context-length", "8192"],
                "flush_cache": False,
                "health_timeout": 2.5,
            },
        ],
        "flush_cache": True,
        "health_timeout": 7.5,
        "fallback_to_local": False,
    }
    reg = load_external_workers(json.dumps(payload))
    assert len(reg.entries) == 2
    assert reg.flush_cache is True
    assert reg.health_timeout == 7.5
    assert reg.fallback_to_local is False
    assert reg.entries[0].engine == "vllm"
    assert reg.entries[0].mode == ConnectionMode.GRPC
    assert reg.entries[0].urls == ("grpc://10.0.0.1:30000",)
    assert reg.entries[1].extra_engine_args == ("--context-length", "8192")
    assert reg.entries[1].flush_cache is False

    list_reg = load_external_workers(json.dumps(payload["workers"]))
    assert len(list_reg.entries) == 2
    assert list_reg.flush_cache is True
    assert list_reg.fallback_to_local is True

    cfg_path = tmp_path / "external.json"
    cfg_path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_external_workers(str(cfg_path)) == reg
    assert load_external_workers(f"@{cfg_path}") == reg


def test_load_external_workers_flush_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """``E2E_EXTERNAL_WORKERS_FLUSH_CACHE`` overrides both top-level and per-entry flush_cache."""
    with _http_server(health_status=200) as (url, _seen):
        raw = json.dumps(
            {
                "workers": [
                    {
                        "engine": "vllm",
                        "model_id": "m",
                        "url": url,
                        "flush_cache": True,
                    },
                ],
                "flush_cache": True,
            }
        )
        monkeypatch.setenv(ENV_EXTERNAL_WORKERS_FLUSH_CACHE, "0")
        reg_off = load_external_workers(raw)
        assert reg_off.flush_cache is False
        attached_off = reg_off.attach(model_id="m", engine="vllm", mode=ConnectionMode.HTTP)
        assert attached_off is not None and attached_off[0].flush_cache is False

        monkeypatch.setenv(ENV_EXTERNAL_WORKERS_FLUSH_CACHE, "true")
        reg_on = load_external_workers(raw)
        assert reg_on.flush_cache is True
        attached_on = reg_on.attach(model_id="m", engine="vllm", mode=ConnectionMode.HTTP)
        assert attached_on is not None and attached_on[0].flush_cache is True

        monkeypatch.setenv(ENV_EXTERNAL_WORKERS_FLUSH_CACHE, "maybe")
        with pytest.raises(ValueError, match="not a valid boolean"):
            load_external_workers(raw)


@pytest.mark.parametrize(
    "urls",
    [
        ["http://127.0.0.1:30000", "http://127.0.0.1:30000", "http://127.0.0.1:30001"],
        ["http://127.0.0.1:30000", "http://127.0.0.1:30000/", "http://127.0.0.1:30001"],
        ["HTTP://LOCALHOST:30000", "http://localhost:30000/"],
        ["http://engine.local", "http://engine.local:80/"],
        ["https://engine.local:443", "HTTPS://ENGINE.LOCAL/"],
    ],
)
def test_load_external_workers_rejects_duplicate_normalized_urls(urls: list[str]) -> None:
    """Duplicate endpoints after scheme/host/slash/default-port normalization are rejected."""
    raw = json.dumps([{"engine": "vllm", "model_id": "m", "urls": urls}])
    with pytest.raises(ValueError, match="duplicate URL"):
        load_external_workers(raw)


def test_external_worker_http_url_preserves_https_case_insensitively() -> None:
    """``Worker.http_url`` preserves TLS even if constructed with an uppercase HTTPS scheme."""
    worker = Worker(
        model_id="org/model",
        engine="vllm",
        port=443,
        gpu_ids=[],
        mode=ConnectionMode.HTTP,
        host="engine.local",
        external=True,
        external_url="HTTPS://engine.local:443/",
    )
    assert worker.http_url == "HTTPS://engine.local:443"


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ("/nonexistent/external-workers.json", "Failed to read"),
        ("{bad json", "Invalid JSON"),
        ('{"not_workers": []}', "'workers' list"),
        ('[{"engine": "", "model_id": "m", "url": "http://127.0.0.1:8000"}]', "'engine'"),
        (
            '[{"engine": "vllm", "model_id": "m", "mode": "zmq", "url": "grpc://127.0.0.1:1"}]',
            "unsupported mode",
        ),
        (
            '[{"engine": "vllm", "model_id": "m", "worker_type": "prefill", '
            '"url": "grpc://127.0.0.1:1"}]',
            "unsupported worker_type",
        ),
        (
            '[{"engine": "vllm", "model_id": "m", "url": "http://127.0.0.1:1", '
            '"urls": ["http://127.0.0.1:2"]}]',
            "not both",
        ),
        ('[{"engine": "vllm", "model_id": "m", "urls": []}]', "non-empty list"),
        (
            '{"workers": [], "health_timeout": 0}',
            "positive number",
        ),
        (
            '{"workers": [], "fallback_to_local": "yes"}',
            "must be a boolean",
        ),
    ],
)
def test_load_external_workers_validation_errors(raw: str, match: str) -> None:
    """Invalid files, JSON, or schema values fail fast with ValueError."""
    with pytest.raises(ValueError, match=match):
        load_external_workers(raw)


# ---------------------------------------------------------------------------
# External Worker lifecycle protections & live health probes
# ---------------------------------------------------------------------------


def test_external_worker_http_health_and_non_destructive_lifecycle() -> None:
    """External HTTP workers probe /health, forbid start, and treat stop/kill as no-ops."""
    with _http_server(health_status=200) as (url, seen):
        port = int(url.rsplit(":", 1)[1])
        reserve_port(port)
        try:
            worker = Worker(
                model_id="org/model",
                engine="sglang",
                port=port,
                gpu_ids=[],
                mode=ConnectionMode.HTTP,
                host="127.0.0.1",
                external=True,
                external_url=url,
            )
            assert worker.base_url == url
            assert worker.http_url == url
            assert worker.is_healthy() is True
            assert worker.is_alive() is True
            assert ("GET", "/health") in seen
            worker.wait_ready(timeout=5)
            worker.raise_if_exited()
            assert worker.kill_engine() == []
            with pytest.raises(RuntimeError, match="Cannot start externally managed worker"):
                worker.start()

            # stop() and stop_workers() must not touch reserved ports or fail
            worker.stop()
            stop_workers([worker])
            assert port in _reserved_ports
        finally:
            _reserved_ports.discard(port)


def test_external_worker_http_unhealthy_raises_on_wait_ready() -> None:
    """Unhealthy external HTTP workers report False from is_healthy and raise in wait_ready."""
    with _http_server(health_status=503) as (url, _seen):
        port = int(url.rsplit(":", 1)[1])
        worker = Worker(
            model_id="org/model",
            engine="vllm",
            port=port,
            gpu_ids=[],
            mode=ConnectionMode.HTTP,
            host="127.0.0.1",
            external=True,
            external_url=url,
            health_timeout=1.0,
        )
        assert worker.is_healthy() is False
        assert worker.is_alive() is False
        with pytest.raises(TimeoutError, match="unhealthy"):
            worker.wait_ready(timeout=1)


def test_external_worker_grpc_health_live_server() -> None:
    """External gRPC workers verify grpc.health.v1 SERVING status over the target host/port."""
    with _grpc_health_server("SERVING") as serving_url:
        reg = load_external_workers(
            json.dumps(
                [{"engine": "vllm", "model_id": "org/model", "mode": "grpc", "url": serving_url}]
            )
        )
        attached = reg.attach(
            model_id="org/model", engine="vllm", mode=ConnectionMode.GRPC, count=1
        )
        assert attached is not None
        assert len(attached) == 1
        assert attached[0].external is True
        assert attached[0].base_url == serving_url
        assert attached[0].is_alive() is True

    with _grpc_health_server("NOT_SERVING") as down_url:
        reg = load_external_workers(
            json.dumps(
                {
                    "workers": [
                        {
                            "engine": "vllm",
                            "model_id": "org/model",
                            "mode": "grpc",
                            "url": down_url,
                        }
                    ],
                    "fallback_to_local": False,
                }
            )
        )
        with pytest.raises(ExternalWorkerAttachError, match="failed health checks"):
            reg.attach(model_id="org/model", engine="vllm", mode=ConnectionMode.GRPC, count=1)


# ---------------------------------------------------------------------------
# WorkerPool integration: attach, reuse, eviction, teardown, fallback
# ---------------------------------------------------------------------------


@pytest.fixture
def pool_harness(monkeypatch: pytest.MonkeyPatch):
    """Patch ``worker_pool.start_workers`` / ``stop_workers`` to observe calls."""
    launched: list[tuple[dict, SimpleNamespace]] = []
    stopped: list[object] = []

    def fake_start_workers(**kwargs):
        count = kwargs.get("count", 1)
        batch = []
        for idx in range(count):
            worker = SimpleNamespace(
                alive=True,
                external=False,
                base_url=f"grpc://127.0.0.1:{40000 + len(launched) + idx}",
                model_id=kwargs["model_id"],
            )
            worker.is_alive = lambda w=worker: w.alive
            launched.append((kwargs, worker))
            batch.append(worker)
        return batch

    monkeypatch.setattr(worker_pool, "start_workers", fake_start_workers)
    monkeypatch.setattr(worker_pool, "stop_workers", lambda workers: stopped.extend(workers))
    return launched, stopped


def test_worker_pool_attaches_reuses_and_never_stops_external_workers(
    pool_harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pool attaches healthy external workers, reuses them, and never stops them on eviction/cleanup."""
    launched, stopped = pool_harness
    with _http_server(health_status=200) as (url, _seen):
        monkeypatch.setenv(
            ENV_EXTERNAL_WORKERS,
            json.dumps(
                [
                    {
                        "engine": "vllm",
                        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
                        "mode": "http",
                        "url": url,
                    }
                ]
            ),
        )
        pool = worker_pool.WorkerPool()
        first = pool.acquire(
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            engine="vllm",
            mode=ConnectionMode.HTTP,
            count=1,
        )
        assert len(first) == 1
        assert first[0].external is True
        assert first[0].base_url == url
        assert first[0].flush_cache is True
        assert launched == []

        # Second acquire with same key reuses attached external worker
        second = pool.acquire(
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            engine="vllm",
            mode=ConnectionMode.HTTP,
            count=1,
        )
        assert second == first
        assert launched == []

        # Switching to an unmatched model evicts external slot WITHOUT stopping it, then launches local
        local = pool.acquire(
            model_id="Qwen/Qwen2.5-7B-Instruct",
            engine="vllm",
            mode=ConnectionMode.HTTP,
            count=1,
        )
        assert len(local) == 1
        assert local[0].external is False
        assert len(launched) == 1
        assert stopped == []

        # Switching back to external stops the local worker, then re-attaches external
        reattached = pool.acquire(
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            engine="vllm",
            mode=ConnectionMode.HTTP,
            count=1,
        )
        assert reattached[0].external is True
        assert stopped == local

        # Cleanup detaches external workers without calling stop_workers on them
        pool.cleanup()
        assert stopped == local


def test_worker_pool_falls_back_on_mismatch_zmq_pd_and_extra_engine_args(
    pool_harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ZMQ, PD/EPD, extra_engine_args, gpus, and count mismatches all fall back to local launch."""
    launched, stopped = pool_harness
    with _http_server(health_status=200) as (url, _seen):
        reg = load_external_workers(
            json.dumps(
                [
                    {
                        "engine": "vllm",
                        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
                        "mode": "http",
                        "url": url,
                    }
                ]
            )
        )
        pool = worker_pool.WorkerPool(external_workers=reg)

        # Custom extra_engine_args -> local launch
        w_extra = pool.acquire(
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            engine="vllm",
            mode=ConnectionMode.HTTP,
            extra_engine_args=["--max-model-len", "2048"],
        )
        assert w_extra[0].external is False

        # Count exceeds available URLs -> local launch
        w_count = pool.acquire(
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            engine="vllm",
            mode=ConnectionMode.HTTP,
            count=2,
        )
        assert len(w_count) == 2
        assert all(not w.external for w in w_count)

        # Explicit gpus mismatch -> local launch
        w_gpus = pool.acquire(
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            engine="vllm",
            mode=ConnectionMode.HTTP,
            gpus=2,
        )
        assert w_gpus[0].external is False

        # ZMQ mode -> local launch (never external)
        w_zmq = pool.acquire(
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            engine="vllm",
            mode=ConnectionMode.ZMQ,
        )
        assert w_zmq[0].external is False

        # PD worker_type -> local launch (never external)
        w_pd = pool.acquire(
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            engine="vllm",
            mode=ConnectionMode.HTTP,
            worker_type=WorkerType.PREFILL,
        )
        assert w_pd[0].external is False
        assert len(launched) == 6
        pool.cleanup()
        assert len(stopped) >= 4


def test_worker_pool_picks_healthy_subset_and_falls_back_when_unhealthy(pool_harness) -> None:
    """Registry skips dead endpoints in a multi-URL entry and falls back when too few are healthy."""
    launched, _stopped = pool_harness
    dead_url = f"http://127.0.0.1:{_unused_port()}"
    with (
        _http_server(health_status=200) as (live1, _),
        _http_server(health_status=200) as (
            live2,
            _,
        ),
    ):
        reg = load_external_workers(
            json.dumps(
                [
                    {
                        "engine": "vllm",
                        "model_id": "m",
                        "mode": "http",
                        "urls": [dead_url, live1, live2],
                        "health_timeout": 1.0,
                    }
                ]
            )
        )
        pool = worker_pool.WorkerPool(external_workers=reg)
        attached = pool.acquire(model_id="m", engine="vllm", mode=ConnectionMode.HTTP, count=2)
        assert [w.base_url for w in attached] == [live1, live2]
        assert launched == []
        pool.cleanup()

    # Now all endpoints are down -> fallback_to_local=True launches local worker
    dead_reg = load_external_workers(
        json.dumps(
            {
                "workers": [
                    {
                        "engine": "vllm",
                        "model_id": "m",
                        "mode": "http",
                        "url": dead_url,
                        "health_timeout": 0.5,
                    }
                ],
                "fallback_to_local": True,
            }
        )
    )
    pool2 = worker_pool.WorkerPool(external_workers=dead_reg)
    fallback_workers = pool2.acquire(model_id="m", engine="vllm", mode=ConnectionMode.HTTP, count=1)
    assert fallback_workers[0].external is False
    assert len(launched) == 1
    pool2.cleanup()


def test_external_registry_respects_servicer_impl_and_mm_processing_constraints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Entries pinned to a servicer_impl or mm_processing mode only match compatible lanes."""
    monkeypatch.setenv(ENV_VLLM_SERVICER_IMPL, "rust")
    monkeypatch.setenv(ENV_MM_PROCESSING, "worker")
    monkeypatch.setattr(constants, "_package_servicer_switch", lambda: None)

    reg: ExternalWorkerRegistry = load_external_workers(
        json.dumps(
            [
                {
                    "engine": "vllm",
                    "model_id": "m",
                    "mode": "grpc",
                    "url": "grpc://127.0.0.1:30001",
                    "servicer_impl": "python",
                },
                {
                    "engine": "vllm",
                    "model_id": "m",
                    "mode": "grpc",
                    "url": "grpc://127.0.0.1:30002",
                    "servicer_impl": "rust",
                    "mm_processing": "router",
                },
                {
                    "engine": "vllm",
                    "model_id": "m",
                    "mode": "grpc",
                    "url": "grpc://127.0.0.1:30003",
                    "servicer_impl": "rust",
                    "mm_processing": "worker",
                },
            ]
        )
    )
    matches = reg.match(model_id="m", engine="vllm", mode=ConnectionMode.GRPC, count=1)
    assert [m.urls for m in matches] == [("grpc://127.0.0.1:30003",)]


def test_gateway_flush_cache_handles_success_unsupported_and_connection_error() -> None:
    """``Gateway.flush_cache`` returns True on HTTP 200 and gracefully returns False otherwise."""
    with _http_server(flush_status=200) as (ok_url, seen):
        gw = Gateway(port=int(ok_url.rsplit(":", 1)[1]))
        assert gw.flush_cache(timeout=2.0) is True
        assert ("POST", "/flush_cache") in seen

    with _http_server(flush_status=501) as (unimpl_url, _):
        gw = Gateway(port=int(unimpl_url.rsplit(":", 1)[1]))
        assert gw.flush_cache(timeout=2.0) is False

    gw_closed = Gateway(port=_unused_port())
    assert gw_closed.flush_cache(timeout=1.0) is False
