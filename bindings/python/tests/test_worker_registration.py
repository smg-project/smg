"""Exercise startup registration through the real Python launcher and Rust binding."""

import json
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


@contextmanager
def worker(model):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            payloads = {
                "/health": {},
                "/version": {"version": "0.10.0"},
                "/v1/models": {
                    "object": "list",
                    "data": [{"id": model, "object": "model", "owned_by": "vllm"}],
                },
            }
            payload = payloads.get(self.path)
            body = json.dumps(payload if payload is not None else {}).encode()
            self.send_response(200 if payload is not None else 404)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def get_json(url):
    with urllib.request.urlopen(url, timeout=2) as response:
        return json.load(response)


def wait_for(process, log, probe):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and process.poll() is None:
        try:
            if probe():
                return
        except (urllib.error.URLError, TimeoutError):
            pass  # The listener and worker registration start asynchronously.
        time.sleep(0.1)
    pytest.fail(f"Gateway did not reach the expected state:\n{log.read_text()}")


@pytest.mark.parametrize("igw,static_workers", [(False, True), (True, True), (True, False)])
def test_python_launcher_registers_workers(tmp_path, igw, static_workers):
    """Static URLs survive IGW conversion and restarts; an empty IGW stays dynamic."""
    with worker("model-a") as first, worker("model-b") as second:
        urls = [first, second]
        # A fresh process must discover the same configured workers without POST /workers.
        for restart in range(2 if igw and static_workers else 1):
            port = free_port()
            metrics_port = free_port()
            while metrics_port == port:
                metrics_port = free_port()
            base = f"http://127.0.0.1:{port}"
            command = [
                sys.executable,
                "-m",
                "smg.launch_router",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--prometheus-port",
                str(metrics_port),
                "--policy",
                "round_robin",
                "--disable-tokenizer-autoload",
                "--shutdown-grace-period-secs",
                "1",
            ]
            if igw:
                command.append("--enable-igw")
            if static_workers:
                command.extend(["--worker-urls", *urls])
            log = tmp_path / f"gateway-{restart}.log"
            with log.open("w") as output:
                process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
                try:
                    wait_for(process, log, lambda: get_json(f"{base}/workers") is not None)
                    if not static_workers:
                        assert get_json(f"{base}/workers")["total"] == 0
                        request = urllib.request.Request(
                            f"{base}/workers",
                            json.dumps({"url": first}).encode(),
                            {"Content-Type": "application/json"},
                        )
                        with urllib.request.urlopen(request, timeout=2) as response:
                            assert response.status == 202
                    expected_urls = set(urls if static_workers else [first])
                    wait_for(
                        process,
                        log,
                        lambda: (
                            {w["url"] for w in get_json(f"{base}/workers")["workers"]}
                            == expected_urls
                        ),
                    )
                    expected_models = {"model-a", "model-b"} if static_workers else {"model-a"}
                    wait_for(
                        process,
                        log,
                        lambda: (
                            {m["id"] for m in get_json(f"{base}/v1/models")["data"]}
                            == expected_models
                        ),
                    )
                finally:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
