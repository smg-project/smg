import contextlib
import importlib.util
import json
import os
import socketserver
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "bfcl" / "arm_probe.py"
SPEC = importlib.util.spec_from_file_location("bfcl_arm_probe", SOURCE)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)
MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"


@contextlib.contextmanager
def server(responses):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            calls.append(self.path)
            status, body, headers = responses.get(self.path, (404, b"missing", {}))
            self.send_response(status)
            for key, value in headers.items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{http.server_port}", calls
    finally:
        http.shutdown()
        thread.join()
        http.server_close()


def response(data, status=200):
    return status, json.dumps(data).encode(), {}


@contextlib.contextmanager
def broken_http_server(payload):
    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            self.request.recv(8192)
            self.request.sendall(payload)

    http = socketserver.TCPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{http.server_address[1]}"
    finally:
        http.shutdown()
        thread.join()
        http.server_close()


def readiness(model=MODEL, healthy=1):
    return {
        "status": "ready",
        "models": [
            {
                "model": model,
                "ready": True,
                "healthy_workers": healthy,
                "total_workers": 1,
                "unknown": "PRIVATE",
            }
        ],
    }


def test_a_live_gateway_needs_the_target_model_and_healthy_worker():
    for data, status, expected in (
        (readiness(), 200, True),
        (readiness("another/model"), 200, False),
        (readiness(healthy=0), 200, False),
        (readiness(), 503, False),
        ({"status": "ready", "healthy_workers": 4}, 200, False),
    ):
        with server({"/readiness": response(data, status)}) as (url, _):
            actual = probe.selected_json(url, "/readiness", MODEL)
            assert probe.ready(actual) is expected
            assert "PRIVATE" not in json.dumps(actual)


def test_unready_json_remains_observable_without_becoming_ready():
    with server({"/readiness": response({"status": "not ready", "models": []}, 503)}) as (url, _):
        actual = probe.selected_json(url, "/readiness", MODEL)
    assert actual == {"http_status": 503, "ready": False, "models": []}


def test_worker_observation_excludes_urls_keys_labels_prompts_and_other_models():
    data = {
        "workers": [
            {
                "model_id": MODEL,
                "url": "grpc://user:PRIVATE@localhost:1234",
                "api_key": "PRIVATE",
                "labels": {"prompt": "PRIVATE"},
                "status": "not_ready",
                "is_healthy": False,
                "load": 2,
                "stalled": "wedged",
                "engine_load": {"secret": "PRIVATE"},
            },
            {"model_id": "other/model", "load": 999},
        ]
    }
    with server({"/workers": response(data)}) as (url, _):
        actual = probe.selected_json(url, "/workers", MODEL)
    assert actual == {
        "http_status": 200,
        "workers": [
            {
                "is_healthy": False,
                "status": "not_ready",
                "load": 2,
                "stalled": "wedged",
            }
        ],
    }
    assert "PRIVATE" not in json.dumps(actual)


def test_redirect_is_not_followed_and_error_body_is_not_retained():
    with server(
        {
            "/readiness": (302, b"PRIVATE", {"Location": "/private"}),
            "/private": response({"secret": "PRIVATE"}),
        }
    ) as (url, calls):
        actual = probe.selected_json(url, "/readiness", MODEL)
    assert calls == ["/readiness"]
    assert actual == {"http_status": 302, "error": "invalid_json_schema"}


@pytest.mark.parametrize("body", [b"not-json-PRIVATE", b"[]", b"x" * (probe.MAX_BYTES + 1)])
def test_bad_or_oversize_answer_is_unknown_and_bounded(body):
    with server({"/readiness": (200, body, {})}) as (url, _):
        actual = probe.selected_json(url, "/readiness", MODEL)
    assert actual.get("error") in ("invalid_json_schema", "oversize_response")
    assert not probe.ready(actual)
    assert "PRIVATE" not in json.dumps(actual)


@pytest.mark.parametrize("endpoint", ["readiness", "metrics"])
@pytest.mark.parametrize(
    "payload,error",
    [
        (b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n10\r\nPRIVATE", "IncompleteRead"),
        (b"PRIVATE\r\n\r\n", "BadStatusLine"),
    ],
)
def test_broken_http_is_unknown_without_ending_observation(endpoint, payload, error):
    with broken_http_server(payload) as url:
        if endpoint == "metrics":
            actual = probe.routing_states(url, 1234)
        else:
            actual = probe.selected_json(url, "/readiness", MODEL)
    assert actual == {"error": error}
    assert "PRIVATE" not in json.dumps(actual)


def test_routing_state_retains_only_this_worker_active_reason():
    body = (
        b'smg_worker_state{worker="grpc://127.0.0.1:1234",state="available",reason="none"} 0\n'
        b'smg_worker_state{worker="grpc://127.0.0.1:1234",state="excluded",reason="circuit_open"} 1\n'
        b'smg_worker_state{worker="grpc://PRIVATE@127.0.0.1:99",state="excluded",reason="failed"} 1\n'
        b'some_secret{key="PRIVATE"} 1\n'
    )
    with server({"/metrics": (200, body, {})}) as (url, _):
        actual = probe.routing_states(url, 1234)
    assert actual == {
        "http_status": 200,
        "states": [{"state": "excluded", "reason": "circuit_open"}],
    }
    assert "PRIVATE" not in json.dumps(actual)


def test_pid_birth_is_compared_and_zombie_does_not_imply_an_exit_code(tmp_path, monkeypatch):
    (tmp_path / "arm_b_worker.pid").write_text(str(os.getpid()))
    owned = probe.owned_processes(tmp_path)
    assert probe.current_processes(owned)["arm_b_worker"]["same_process_present"]
    original = owned["arm_b_worker"]
    monkeypatch.setattr(probe, "pid_identity", lambda pid: {**original, "state": "Z"})
    row = probe.current_processes(owned)["arm_b_worker"]
    assert row["state"] == "Z" and "exit_code" not in row
    monkeypatch.setattr(
        probe,
        "pid_identity",
        lambda pid: {
            **original,
            "start_ticks": original["start_ticks"] + 1,
        },
    )
    assert not probe.current_processes(owned)["arm_b_worker"]["same_process_present"]
    monkeypatch.setattr(probe, "pid_identity", lambda pid: None)
    assert not probe.current_processes(owned)["arm_b_worker"]["same_process_present"]


def test_monitor_cli_writes_bounded_snapshots_without_generation(tmp_path):
    with server(
        {
            "/readiness": response(readiness()),
            "/workers": response({"workers": []}),
        }
    ) as (url, calls):
        result = subprocess.run(
            [
                sys.executable,
                str(SOURCE),
                "--base-url",
                url,
                "--model",
                MODEL,
                "--run-dir",
                str(tmp_path),
                "--duration",
                "0.01",
            ],
            capture_output=True,
            timeout=5,
        )
    rows = [json.loads(line) for line in (tmp_path / "arm_b_state.log").read_text().splitlines()]
    assert result.returncode == 0, result.stderr
    assert 1 <= len(rows) <= 2
    assert set(calls) == {"/readiness", "/workers"}
    assert all(probe.ready(row["readiness"]) for row in rows)


def test_wait_cli_timeout_does_not_accept_process_liveness(tmp_path):
    with server({"/readiness": response({"status": "ready", "healthy_workers": 2})}) as (
        url,
        calls,
    ):
        result = subprocess.run(
            [
                sys.executable,
                str(SOURCE),
                "--base-url",
                url,
                "--model",
                MODEL,
                "--run-dir",
                str(tmp_path),
                "--wait-ready",
                "0.01",
            ],
            capture_output=True,
            timeout=5,
        )
    assert result.returncode == 1
    assert set(calls) == {"/readiness"}
    assert b"TIMEOUT" in result.stderr
    assert b'"http_status": 200' in result.stderr
    assert b'"models": []' in result.stderr


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com:80",
        "http://user:PRIVATE@127.0.0.1:80",
        "http://127.0.0.1:80/private",
        "http://127.0.0.1:80?key=PRIVATE",
        "https://127.0.0.1:80",
        "http://127.0.0.1",
    ],
)
def test_probe_rejects_remote_or_authenticated_targets(url):
    with pytest.raises(ValueError):
        probe.loopback_url(url)


def launcher_cli_without_engines(tmp_path, arm, run_timeout):
    """Run the actual launcher, replacing only process launch and arm-A curl.

    setsid records commands but never executes an engine, gateway or observer.
    The loopback fixture satisfies arm-B TCP/readiness checks. No model, GPU,
    production process or launcher stop/cleanup path is used.
    """
    stub_bin = tmp_path / "bin"
    stub_bin.mkdir()
    events_path = tmp_path / "launch-events.jsonl"
    setsid = stub_bin / "setsid"
    setsid.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "with open(os.environ['BFCL_FIXTURE_EVENTS'], 'a') as out:\n"
        "    out.write(json.dumps({'argv': sys.argv[1:], "
        "'RUN_TIMEOUT': os.environ.get('RUN_TIMEOUT')}) + '\\n')\n"
    )
    setsid.chmod(0o755)
    curl = stub_bin / "curl"
    curl.write_text("#!/bin/sh\nexit 0\n")
    curl.chmod(0o755)
    run_dir = tmp_path / "state with spaces"
    env = {
        **os.environ,
        "PATH": f"{stub_bin}{os.pathsep}{os.environ['PATH']}",
        "BFCL_FIXTURE_EVENTS": str(events_path),
        "BFCL_RUN_DIR": str(run_dir),
        "BFCL_MODEL": MODEL,
        "BFCL_STARTUP_TIMEOUT": "1",
        "VLLM_BIN": "/must-not-run-a-real-engine",
        "VLLM_PYTHON": "/must-not-run-a-real-engine",
        "SMG_LAUNCH": "/must-not-run-a-real-gateway",
        "CUDA_VISIBLE_DEVICES": "",
        "HIP_VISIBLE_DEVICES": "",
        "ROCR_VISIBLE_DEVICES": "",
    }
    if run_timeout is None:
        env.pop("RUN_TIMEOUT", None)
    else:
        env["RUN_TIMEOUT"] = run_timeout
    with server({"/readiness": response(readiness())}) as (url, _):
        port = url.rsplit(":", 1)[1]
        for name in ("ARM_A_PORT", "ARM_B_GRPC_PORT", "ARM_B_GW_PORT", "ARM_B_METRICS_PORT"):
            env[f"BFCL_{name}"] = port
        result = subprocess.run(
            ["bash", str(SOURCE.with_name("launch_arm.sh")), arm],
            env=env,
            capture_output=True,
            timeout=10,
        )
    expected_events = 1 if arm == "a" else 3
    deadline = time.monotonic() + 1
    while True:
        events = (
            [json.loads(line) for line in events_path.read_text().splitlines()]
            if events_path.exists()
            else []
        )
        if len(events) >= expected_events or result.returncode or time.monotonic() >= deadline:
            return result, events, run_dir
        time.sleep(0.01)


@pytest.mark.parametrize("arm", ["a", "b"])
@pytest.mark.parametrize(
    "run_timeout,expected",
    [
        pytest.param(None, 14400, id="default"),
        pytest.param("", 14400, id="empty-default"),
        pytest.param("0", 21600, id="uncapped-scoring"),
        pytest.param("14400", 14400, id="normal"),
        pytest.param("21600", 21600, id="observer-cap"),
        pytest.param("21601", 21600, id="above-cap"),
        pytest.param("00008", 8, id="leading-zero-decimal"),
        pytest.param("0000", 21600, id="leading-zero-uncapped"),
        pytest.param("9" * 100, 21600, id="beyond-shell-integer-range"),
        pytest.param(" +0008 ", 8, id="integer-whitespace-and-plus"),
        pytest.param("1_000", 1000, id="python-integer-separator"),
    ],
)
def test_launcher_bounds_observer_before_start_without_changing_scoring(
    tmp_path, arm, run_timeout, expected
):
    result, events, _ = launcher_cli_without_engines(tmp_path, arm, run_timeout)
    assert result.returncode == 0, result.stderr
    assert len(events) == (1 if arm == "a" else 3)
    # Preserve the exact scoring setting, including zero and decimal spelling.
    assert all(event["RUN_TIMEOUT"] == run_timeout for event in events)
    bounded = f"RUN_TIMEOUT validated; arm B observer bound: {expected}s".encode()
    assert bounded in result.stderr
    assert result.stderr.index(bounded) < result.stderr.index(b"started arm_")
    if arm == "b":
        observer = next(event for event in events if "--duration" in event["argv"])
        assert observer["argv"][observer["argv"].index("--duration") + 1] == str(expected)


@pytest.mark.parametrize("arm", ["a", "b"])
@pytest.mark.parametrize("run_timeout", ["-1", "1.5", "1e4", "not-an-integer", "   "])
def test_launcher_rejects_invalid_timeout_before_any_process_start(tmp_path, arm, run_timeout):
    result, events, run_dir = launcher_cli_without_engines(tmp_path, arm, run_timeout)
    assert result.returncode == 2
    assert b"RUN_TIMEOUT" in result.stderr
    assert events == []
    assert not run_dir.exists()
