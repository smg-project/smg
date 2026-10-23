"""Actual Python caller/API; the HTTP registry and engine processes are stand-ins."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import import_module
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from infra import model_specs
from infra.constants import ConnectionMode, WorkerType
from infra.gateway import Gateway
from infra.worker import Worker

setup = import_module("fixtures.setup_backend")
MODEL = "Qwen/Qwen3.5-9B"


@pytest.fixture
def registry(monkeypatch):
    state = SimpleNamespace(
        workers=[], posts=[], launches=[], shutdowns=[], stopped=[], reject=False, ready=True
    )

    class Handler(BaseHTTPRequestHandler):
        def answer(self, status, body):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state.posts.append(body)
            if state.reject:
                return self.answer(422, {"error": "registration rejected"})
            state.workers.append({**body, "id": "encoder", "healthy": True})
            return self.answer(202, {"worker_id": "encoder"})

        def do_GET(self):
            url = urlsplit(self.path)
            if url.path == "/readiness":
                return self.answer(
                    200 if state.ready else 503,
                    {
                        "status": "ready" if state.ready else "not_ready",
                        "healthy_workers": len(state.workers),
                    },
                )
            model = parse_qs(url.query).get("model", [None])[0]
            workers = [w for w in state.workers if model is None or w["models"][0]["id"] == model]
            return self.answer(200, {"total": len(workers), "workers": workers})

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    gateway = Gateway(port=server.server_port, prometheus_port=50199)
    monkeypatch.setattr(gateway, "_launch", lambda **kwargs: state.launches.append(kwargs))
    monkeypatch.setattr(gateway, "shutdown", lambda: state.shutdowns.append(True))
    monkeypatch.setattr(setup, "stop_workers", lambda workers: state.stopped.extend(workers))
    monkeypatch.setattr(setup, "_make_openai_client", lambda _: None)
    monkeypatch.setattr(Worker, "raise_if_exited", lambda _: None)
    try:
        yield state, gateway
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def workers():
    return [
        Worker(
            model_id=MODEL,
            engine="sglang",
            port=50200 + i,
            gpu_ids=[i],
            mode=ConnectionMode.GRPC,
            worker_type=role,
        )
        for i, role in enumerate((WorkerType.ENCODE, WorkerType.PREFILL, WorkerType.DECODE))
    ]


def test_real_fixture_registers_native_encoder_explicitly(registry, monkeypatch):
    state, gateway = registry
    fleet = workers()
    for worker in fleet[1:]:
        state.workers.append(
            {
                "id": worker.worker_type.value,
                "url": worker.base_url,
                "models": [{"id": MODEL}],
                "healthy": True,
            }
        )
    monkeypatch.setattr(
        setup,
        "_start_workers_tracked",
        lambda **kw: [w for w in fleet if w.worker_type == kw["worker_type"]],
    )
    backend = setup._setup_epd(
        MODEL,
        MODEL,
        "sglang",
        ConnectionMode.GRPC,
        (1, 1, 1),
        dict(setup._GW_DEFAULTS),
        gateway,
        None,
    )
    try:
        next(backend)
        args = state.launches[0]["mode_args"]
        assert "--epd-disaggregation" in args and "--encode" not in args
        assert "--prefill" in args and "--decode" in args
        assert state.posts == [
            {
                "url": fleet[0].base_url,
                "worker_type": "encode",
                "connection_mode": "grpc",
                "runtime_type": "sglang",
                "models": [{"id": MODEL}],
                "labels": {"model_path": MODEL},
            }
        ]
        assert gateway.encode_workers == fleet[:1]
        same = httpx.get(gateway.base_url + "/workers", params={"model": MODEL}).json()
        other = httpx.get(gateway.base_url + "/workers", params={"model": "wrong"}).json()
        assert same["total"] == 3 and other["total"] == 0
    finally:
        backend.close()
    assert state.shutdowns and state.stopped == fleet


@pytest.mark.parametrize("index", [0, 1, 2])
def test_mismatched_role_model_is_rejected_before_registration(registry, index):
    state, gateway = registry
    fleet = workers()
    fleet[index].model_id = "wrong"
    with pytest.raises(ValueError, match="same configured model"):
        setup._register_sglang_epd_encoders(gateway, fleet, MODEL, MODEL, 1)
    assert state.posts == []


def test_registration_rejection_is_fatal_and_fixture_cleans_up(registry, monkeypatch):
    state, gateway = registry
    state.reject = True
    fleet = workers()
    monkeypatch.setattr(
        setup,
        "_start_workers_tracked",
        lambda **kw: [w for w in fleet if w.worker_type == kw["worker_type"]],
    )
    backend = setup._setup_epd(
        MODEL,
        MODEL,
        "sglang",
        ConnectionMode.GRPC,
        (1, 1, 1),
        dict(setup._GW_DEFAULTS),
        gateway,
        None,
    )
    with pytest.raises(RuntimeError, match="registration rejected"):
        next(backend)
    assert state.shutdowns and state.stopped == fleet


def test_registration_waits_for_fleet_readiness_after_accepted_api(registry):
    state, gateway = registry
    state.ready = False
    fleet = workers()
    state.workers = [
        {"id": str(i), "url": w.base_url, "models": [{"id": MODEL}]}
        for i, w in enumerate(fleet[1:])
    ]
    with pytest.raises(TimeoutError, match="readiness: not_ready"):
        setup._register_sglang_epd_encoders(gateway, fleet, MODEL, MODEL, 0.1)
    assert len(state.posts) == 1


def test_configured_model_key_may_map_to_a_distinct_snapshot_path(registry, monkeypatch):
    state, gateway = registry
    snapshot = "/models/models--Qwen--Qwen3.5-9B/snapshots/fixed"
    monkeypatch.setitem(
        model_specs.MODEL_SPECS, MODEL, {**model_specs.MODEL_SPECS[MODEL], "model": snapshot}
    )
    monkeypatch.setattr(setup, "wait_for_workers_ready", lambda *a, **kw: None)
    setup._register_sglang_epd_encoders(gateway, workers(), MODEL, snapshot, 1)
    assert state.posts[0]["models"] == [{"id": snapshot}]
    assert state.posts[0]["labels"] == {"model_path": snapshot}


def test_duplicate_encoder_model_path_override_is_rejected(registry):
    state, gateway = registry
    fleet = workers()
    fleet[0].extra_engine_args = ["--model-path", "wrong"]
    with pytest.raises(ValueError, match="encoder command"):
        setup._register_sglang_epd_encoders(gateway, fleet, MODEL, MODEL, 1)
    assert state.posts == []
