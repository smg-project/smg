"""Unit tests for smg.rl against a local stub server (no gateway, no GPU)."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import pytest
from smg.rl import RL, FanoutError, RlError, Worker

WORKER = {
    "id": "w1",
    "url": "http://e:1",
    "base_url": "http://e:1",
    "engine": "sglang",
    "engine_version": "0.5",
    "model_id": "m",
    "worker_type": "regular",
    "connection_mode": "http",
    "tp_size": 1,
    "dp_size": 1,
    "pp_size": 1,
    "dp_ranks": 1,
    "role": None,
    "health": "ready",
    "weight_version": "7",
    "labels": {"tp_size": "1"},
    "capabilities": {
        "source": "static",
        "pause_modes": ["abort"],
        "update_from": ["disk"],
        "abort": True,
        "flush_cache": True,
        "sleep_wake": True,
        "reports_weight_version": True,
    },
}


class _Stub(BaseHTTPRequestHandler):
    seen: list[dict] = []
    responses: dict[str, tuple[int, dict]] = {}

    def _handle(self):
        length = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(length) if length else b""
        url = urlparse(self.path)
        _Stub.seen.append(
            {
                "method": self.command,
                "path": url.path,
                "query": parse_qs(url.query),
                "auth": self.headers.get("authorization"),
                "body": json.loads(body) if body else None,
            }
        )
        status, payload = _Stub.responses.get(url.path, (404, {"error": "not_found"}))
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    do_GET = _handle
    do_POST = _handle

    def log_message(self, *_):  # silence
        pass


@pytest.fixture
def stub():
    _Stub.seen = []
    _Stub.responses = {}
    server = HTTPServer(("127.0.0.1", 0), _Stub)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def test_workers_and_worker(stub):
    _Stub.responses["/v1/rl/workers"] = (200, {"workers": [WORKER], "total": 1})
    _Stub.responses["/v1/rl/workers/w1"] = (200, WORKER)
    rl = RL(stub, api_key="k")
    ws = rl.workers()
    assert len(ws) == 1 and ws[0].id == "w1" and ws[0].engine == "sglang"
    assert ws[0].capabilities["pause_modes"] == ["abort"]
    assert rl.worker("w1").weight_version == "7"
    assert _Stub.seen[0]["auth"] == "Bearer k"


def test_call_sends_method_params_body(stub):
    _Stub.responses["/v1/rl/workers/w1/engine/pause"] = (
        200,
        {
            "worker_id": "w1",
            "url": "http://e:1",
            "status": 200,
            "latency_ms": 3,
            "body": {"ok": True},
        },
    )
    rl = RL(stub)
    r = rl.call("w1", "pause", {"x": 1}, params={"mode": "keep"})
    assert r.status == 200 and r.body == {"ok": True} and r.latency_ms == 3
    s = _Stub.seen[-1]
    assert s["method"] == "POST" and s["query"] == {"mode": ["keep"]} and s["body"] == {"x": 1}
    rl.call("w1", "/pause", method="GET")
    assert _Stub.seen[-1]["method"] == "GET" and _Stub.seen[-1]["body"] is None


def test_fanout_raises_on_partial_unless_allowed(stub):
    _Stub.responses["/v1/rl/engine/flush_cache"] = (
        207,
        {
            "results": {"w1": {"url": "u", "status": 200, "latency_ms": 1, "body": {}}},
            "failed": [
                {
                    "worker_id": "w2",
                    "url": "u2",
                    "status": 500,
                    "error": "upstream_error",
                    "message": "HTTP 500",
                }
            ],
            "total": 2,
            "succeeded": 1,
        },
    )
    rl = RL(stub)
    with pytest.raises(FanoutError) as ei:
        rl.fanout("flush_cache", selector="engine=sglang")
    assert [f.worker_id for f in ei.value.result.failed] == ["w2"]
    res = rl.fanout("flush_cache", selector="engine=sglang", allow_partial=True)
    assert res.succeeded == 1 and res.total == 2 and "w1" in res.results
    assert _Stub.seen[-1]["query"] == {"selector": ["engine=sglang"]}


def test_smg_errors_raise_rlerror(stub):
    _Stub.responses["/v1/rl/engine/pause"] = (
        400,
        {"error": "no_workers_match", "message": "none", "selector": "x=y"},
    )
    with pytest.raises(RlError) as ei:
        RL(stub).fanout("pause", selector="x=y")
    assert ei.value.code == "no_workers_match" and ei.value.status == 400


def test_worker_from_json_defaults_missing_dicts():
    d = dict(WORKER)
    del d["labels"]
    del d["capabilities"]
    del d["role"]
    d["future_field"] = 1
    w = Worker.from_json(d)
    assert w.labels == {}
    assert w.capabilities == {}
    assert w.role is None


def test_call_raises_on_smg_error_envelope(stub):
    _Stub.responses["/v1/rl/workers/w1/engine/pause"] = (
        502,
        {
            "error": "upstream_unreachable",
            "message": "connect refused",
            "worker_id": "w1",
            "url": "http://e:1",
        },
    )
    with pytest.raises(RlError) as ei:
        RL(stub).call("w1", "pause")
    assert ei.value.code == "upstream_unreachable"
    assert ei.value.status == 502


def test_call_returns_mirrored_upstream_error(stub):
    _Stub.responses["/v1/rl/workers/w1/engine/pause"] = (
        500,
        {
            "worker_id": "w1",
            "url": "http://e:1",
            "status": 500,
            "latency_ms": 2,
            "body": {"error": "boom"},
        },
    )
    r = RL(stub).call("w1", "pause")
    assert r.status == 500
    assert r.body == {"error": "boom"}


def test_fanout_error_reports_response_status(stub):
    _Stub.responses["/v1/rl/engine/pause"] = (
        200,
        {
            "results": {"w1": {"url": "u", "status": 200, "latency_ms": 1, "body": {}}},
            "failed": [
                {
                    "worker_id": "w2",
                    "url": "u2",
                    "status": 500,
                    "error": "upstream_error",
                    "message": "HTTP 500",
                }
            ],
            "total": 2,
            "succeeded": 1,
        },
    )
    rl = RL(stub)
    with pytest.raises(FanoutError) as ei:
        rl.fanout("pause", selector="engine=sglang")
    assert ei.value.status == 200
    assert ei.value.result.failed[0].worker_id == "w2"
