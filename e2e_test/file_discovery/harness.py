"""Shared pieces of the host-process file-discovery e2e: the in-process fake
engine, the manifest writer, and the gateway driver."""

from __future__ import annotations

import http.server
import json
import os
import random
import socket
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path

import requests

MODEL = "file-e2e-model"
# Rereads, health checks and removals all run on one-second clocks here, so
# convergence is a few seconds; this bounds a scenario that never converges.
CONVERGE = 60.0


def binary(env: str, default: str) -> Path:
    path = Path(os.environ.get(env, default)).resolve()
    assert path.exists(), f"{path} not found (cargo build -p smg -p mock-worker first)"
    return path


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def free_port_range(count: int) -> int:
    """The first of `count` consecutive free ports."""
    for _ in range(50):
        base = random.randint(30000, 60000 - count)
        sockets = []
        try:
            for port in range(base, base + count):
                sock = socket.socket()
                sockets.append(sock)
                sock.bind(("127.0.0.1", port))
            return base
        except OSError:
            continue
        finally:
            for sock in sockets:
                sock.close()
    raise AssertionError(f"no {count} consecutive free ports")


class FakeEngine:
    """An in-process engine: healthy until told otherwise, and every answer
    names the port that served it."""

    def __init__(self):
        engine = self
        self.healthy = True

        class Handler(http.server.BaseHTTPRequestHandler):
            def _respond(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))
                path = self.path.split("?")[0]
                if path == "/health":
                    code = 200 if engine.healthy else 503
                    body = {"status": "ok" if engine.healthy else "down"}
                elif path == "/model_info":
                    code, body = 200, {"model_path": MODEL, "is_generation": True}
                elif path == "/server_info":
                    code, body = 200, {"version": "file-e2e"}
                elif path == "/v1/chat/completions":
                    code, body = (
                        200,
                        {
                            "id": "cmpl-file-e2e",
                            "object": "chat.completion",
                            "created": 0,
                            "model": MODEL,
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": f"served-by-{engine.port}",
                                    },
                                    "finish_reason": "stop",
                                }
                            ],
                            "usage": {
                                "prompt_tokens": 1,
                                "completion_tokens": 1,
                                "total_tokens": 2,
                            },
                        },
                    )
                else:
                    # Unknown paths must 404: the gateway's metadata fetch
                    # falls back between endpoint variants on it.
                    code, body = 404, {}
                payload = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            do_GET = _respond
            do_POST = _respond

            def log_message(self, *args):
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()


class Manifest:
    def __init__(self, path: Path):
        self.path = path

    def write(self, workers: list[dict]) -> None:
        """Replace the manifest as writers should: a temporary file beside
        it, renamed over it."""
        staged = self.path.with_suffix(".staged")
        staged.write_text(json.dumps({"version": 1, "workers": workers}))
        staged.replace(self.path)


class Gateway:
    """One smg process with file discovery, plus /workers assertions."""

    def __init__(self, manifest: Manifest, log_path: Path):
        self.manifest = manifest
        self.log_path = log_path
        self.proc: subprocess.Popen | None = None
        self.base_url = ""

    def start(self, *extra_args: str) -> None:
        port = free_port()
        self.base_url = f"http://127.0.0.1:{port}"
        log = open(self.log_path, "a")
        self.proc = subprocess.Popen(
            [
                str(binary("SMG_BIN", "target/debug/smg")),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--prometheus-port",
                str(free_port()),
                "--discovery-provider",
                "file",
                "--discovery-file",
                str(self.manifest.path),
                "--discovery-check-interval-secs",
                "1",
                "--policy",
                "round_robin",
                "--health-check-interval-secs",
                "1",
                "--health-failure-threshold",
                "2",
                "--health-success-threshold",
                "1",
                "--drain-settle-secs",
                "0",
                "--worker-startup-timeout-secs",
                "5",
                *extra_args,
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + 60
        while True:
            assert self.proc.poll() is None, f"smg exited early; log: {self.log_tail()}"
            try:
                if requests.get(f"{self.base_url}/health", timeout=2).ok:
                    return
            except requests.RequestException:
                pass
            assert time.monotonic() < deadline, f"smg never became healthy; {self.log_tail()}"
            time.sleep(0.2)

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(timeout=10)

    def restart(self, *extra_args: str) -> None:
        self.stop()
        self.start(*extra_args)

    def log_tail(self, lines: int = 40) -> str:
        return "\n".join(self.log_path.read_text().splitlines()[-lines:])

    def workers(self) -> list[dict] | None:
        """The registered workers, or None when the gateway could not be
        asked; an unanswered request must never read as an empty fleet."""
        try:
            response = requests.get(f"{self.base_url}/workers", timeout=5)
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError):
            return None
        return payload.get("workers", payload if isinstance(payload, list) else [])

    def wait_for(self, what: str, predicate: Callable[[list[dict]], bool]) -> list[dict]:
        deadline = time.monotonic() + CONVERGE
        while time.monotonic() < deadline:
            workers = self.workers()
            if workers is not None and predicate(workers):
                return workers
            time.sleep(0.25)
        raise AssertionError(
            f"timed out waiting for {what}; workers: {self.workers()}\n{self.log_tail()}"
        )

    def wait_for_urls(self, expected: set[str], what: str) -> list[dict]:
        return self.wait_for(what, lambda workers: {w["url"] for w in workers} == expected)

    def chat(self) -> requests.Response:
        return requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}]},
            timeout=15,
        )


def served_by(gateway: Gateway, count: int) -> set[str]:
    """Which engines answered `count` requests, once the fleet takes traffic.
    Health promotion lags registration, so wait for the first answer."""
    deadline = time.monotonic() + CONVERGE
    while (response := gateway.chat()).status_code != 200:
        assert time.monotonic() < deadline, f"no worker became routable: {response.text}"
        time.sleep(0.5)
    answers = set()
    for _ in range(count):
        response = gateway.chat()
        assert response.status_code == 200, response.text
        answers.add(response.json()["choices"][0]["message"]["content"])
    return answers
