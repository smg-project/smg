"""Real Gateway HTTP API; registry, engine processes and router launch are stand-ins."""

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
from infra.process_utils import ProcessExitedError
from infra.worker import Worker

setup = import_module("fixtures.setup_backend")
MODEL = "Qwen/Qwen3.5-9B"


def fleet(engine="sglang"):
    result = []
    for i, role in enumerate((WorkerType.ENCODE, WorkerType.PREFILL, WorkerType.DECODE)):
        worker = Worker(MODEL, engine, 50200 + i, [i], ConnectionMode.GRPC, worker_type=role)
        worker.process = SimpleNamespace(poll=lambda: None, returncode=None)
        result.append(worker)
    return result


def row(worker, model=MODEL):
    return dict(
        id=worker.worker_type.value,
        url=worker.base_url,
        model_id=model,
        models=[{"id": model}],
        runtime_type="sglang",
        connection_mode="grpc",
        worker_type=worker.worker_type.value,
        is_healthy=True,
    )


@pytest.fixture
def registry(monkeypatch):
    state = SimpleNamespace(
        rows=[],
        posts=[],
        launches=[],
        stopped=[],
        shutdowns=[],
        status=202,
        ready=True,
        visible=None,
        missing_id=False,
        wrong_id=False,
    )

    class Handler(BaseHTTPRequestHandler):
        def answer(self, status, body):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            assert self.path == "/workers"
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state.posts.append(body)
            if state.status != 202:
                return self.answer(state.status, {"error": "registration rejected"})
            identity = "encoder-" + str(len(state.posts))
            state.rows.append(
                dict(body, id=identity, model_id=body["models"][0]["id"], is_healthy=True)
            )
            return self.answer(
                202,
                {} if state.missing_id else {"worker_id": "wrong" if state.wrong_id else identity},
            )

        def do_GET(self):
            address = urlsplit(self.path)
            rows = state.rows if state.visible is None else state.visible
            if address.path == "/readiness":
                return self.answer(
                    200 if state.ready else 503,
                    dict(
                        status="ready" if state.ready else "not_ready",
                        healthy_workers=sum(w["is_healthy"] for w in rows),
                    ),
                )
            assert address.path == "/workers"
            model = parse_qs(address.query).get("model", [None])[0]
            selected = [w for w in rows if model is None or w["model_id"] == model]
            return self.answer(200, {"total": len(selected), "workers": selected})

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    gateway = Gateway(port=server.server_port, prometheus_port=50199)
    gateway.process = SimpleNamespace(poll=lambda: None, returncode=None)
    monkeypatch.setattr(gateway, "_launch", lambda **kw: state.launches.append(kw))
    monkeypatch.setattr(gateway, "shutdown", lambda: state.shutdowns.append(True))
    monkeypatch.setattr(setup, "stop_workers", lambda workers: state.stopped.extend(workers))
    monkeypatch.setattr(setup, "_make_openai_client", lambda _: None)
    try:
        yield state, gateway
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def backend(registry, monkeypatch, workers, engine="sglang"):
    state, gateway = registry
    monkeypatch.setattr(
        setup,
        "_start_workers_tracked",
        lambda **kw: [w for w in workers if w.worker_type == kw["worker_type"]],
    )
    return setup._setup_epd(
        MODEL,
        MODEL,
        engine,
        ConnectionMode.GRPC,
        (1, 1, 1),
        {**setup._GW_DEFAULTS, "timeout": 1},
        gateway,
        None,
    )


def test_real_api_registration_and_same_model_fleet(registry, monkeypatch):
    state, gateway = registry
    workers = fleet()
    state.rows = [row(w) for w in workers[1:]]
    with_backend = backend(registry, monkeypatch, workers)
    try:
        next(with_backend)
        args = state.launches[0]["mode_args"]
        assert "--epd-disaggregation" in args and "--encode" not in args
        assert "--prefill" in args and "--decode" in args
        assert state.posts == [
            dict(
                url=workers[0].base_url,
                worker_type="encode",
                connection_mode="grpc",
                runtime_type="sglang",
                models=[{"id": MODEL}],
                labels={"model_path": MODEL},
            )
        ]
        assert gateway.encode_workers == workers[:1]
        assert (
            httpx.get(gateway.base_url + "/workers", params={"model": "wrong"}).json()["workers"]
            == []
        )
    finally:
        with_backend.close()
    assert state.shutdowns and state.stopped == workers


@pytest.mark.parametrize("index", [0, 1, 2])
def test_each_role_model_mismatch_fails_before_post(registry, index):
    state, gateway = registry
    workers = fleet()
    workers[index].model_id = "wrong"
    with pytest.raises(ValueError, match="same configured model"):
        setup._register_sglang_epd_encoders(gateway, workers, MODEL, MODEL, 0.5)
    assert not state.posts


@pytest.mark.parametrize("fault", ["wrong_fleet", "model", "role", "runtime", "wire", "unhealthy"])
def test_count_and_readiness_cannot_replace_expected_fleet(registry, fault):
    state, gateway = registry
    workers = fleet()
    state.visible = [row(w) for w in workers]
    state.visible[0]["id"] = "encoder-1"
    if fault == "wrong_fleet":
        for i, worker in enumerate(state.visible):
            worker["url"] = f"grpc://other-{i}:6000"
    else:
        key, value = {
            "model": ("model_id", "other"),
            "role": ("worker_type", "regular"),
            "runtime": ("runtime_type", "vllm"),
            "wire": ("connection_mode", "http"),
            "unhealthy": ("is_healthy", False),
        }[fault]
        state.visible[1][key] = value
    with pytest.raises(TimeoutError, match="missing or unhealthy"):
        setup._register_sglang_epd_encoders(gateway, workers, MODEL, MODEL, 0.5)


@pytest.mark.parametrize("fault", ["reject", "missing_id", "wrong_id", "not_ready"])
def test_async_acceptance_is_not_ready_and_failure_cleans_up(registry, monkeypatch, fault):
    state, gateway = registry
    workers = fleet()
    state.rows = [row(w) for w in workers[1:]]
    if fault == "reject":
        state.status = 422
    elif fault == "missing_id":
        state.missing_id = True
    elif fault == "wrong_id":
        state.wrong_id = True
    else:
        state.ready = False
    with pytest.raises((RuntimeError, TimeoutError)):
        next(backend(registry, monkeypatch, workers))
    assert state.shutdowns and state.stopped == workers


def test_configured_snapshot_path_and_extra_healthy_worker(registry, monkeypatch):
    state, gateway = registry
    snapshot = "/models/models--Qwen--Qwen3.5-9B/snapshots/fixed"
    monkeypatch.setitem(
        model_specs.MODEL_SPECS, MODEL, {**model_specs.MODEL_SPECS[MODEL], "model": snapshot}
    )
    workers = fleet()
    state.rows = [row(w, snapshot) for w in workers[1:]] + [
        dict(row(workers[1], "other"), url="grpc://other:6000")
    ]
    setup._register_sglang_epd_encoders(gateway, workers, MODEL, snapshot, 1)
    assert state.posts[0]["models"] == [{"id": snapshot}]
    assert state.posts[0]["labels"] == {"model_path": snapshot}


def test_model_path_override_is_not_registered(registry):
    state, gateway = registry
    workers = fleet()
    workers[0].extra_engine_args = ["--model-path", "wrong"]
    with pytest.raises(ValueError, match="encoder command"):
        setup._register_sglang_epd_encoders(gateway, workers, MODEL, MODEL, 1)
    assert not state.posts


def test_process_exit_fails_even_if_registry_claims_healthy(registry):
    state, gateway = registry
    state.rows = [row(w) for w in fleet()]
    gateway.process = SimpleNamespace(poll=lambda: 7, returncode=7)
    with pytest.raises(ProcessExitedError, match="Gateway exited"):
        setup._register_sglang_epd_encoders(gateway, fleet(), MODEL, MODEL, 1)
    assert not state.posts


def test_two_encoders_have_distinct_async_registration_identities(registry):
    state, gateway = registry
    workers = fleet()
    second = fleet()[0]
    second.port = 50300
    workers.append(second)
    state.rows = [row(w) for w in workers[1:3]]
    setup._register_sglang_epd_encoders(gateway, workers, MODEL, MODEL, 2)
    assert {w["url"] for w in state.posts} == {workers[0].base_url, second.base_url}
    assert len({w["id"] for w in state.rows if w["worker_type"] == "encode"}) == 2


def test_tokenspeed_keeps_static_encoder_registration(registry, monkeypatch):
    state, gateway = registry
    with_backend = backend(registry, monkeypatch, fleet("tokenspeed"), "tokenspeed")
    try:
        next(with_backend)
        assert "--encode" in state.launches[0]["mode_args"]
        assert not state.posts
    finally:
        with_backend.close()


@pytest.mark.parametrize(
    "engine,mode", [("sglang", ConnectionMode.HTTP), ("vllm", ConnectionMode.GRPC)]
)
def test_unimplemented_registration_lane_is_rejected_before_post(registry, engine, mode):
    state, gateway = registry
    workers = fleet()
    workers[0].engine, workers[0].mode = engine, mode
    with pytest.raises(ValueError, match="same configured model and gRPC runtime"):
        setup._register_sglang_epd_encoders(gateway, workers, MODEL, MODEL, 1)
    assert not state.posts
