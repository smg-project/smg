"""What the file and Slurm discovery e2e lanes share: the binaries, free
ports, the in-process fake engine, the mock-worker launcher, and the gateway
driver.

The file lane runs its gateway and engines as processes on this host. The
Slurm lane runs them in its cluster's containers, so it takes only the model,
a free port, the gateway driver (subclassed to run on the controller) and
`served_by`. Each lane adds the source it drives and the flags that select
its provider.
"""

from __future__ import annotations

import http.server
import json
import os
import random
import socket
import subprocess
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
import requests

MODEL = "discovery-e2e-model"
# Reads, health checks and removals all run on one-second clocks here, so
# convergence is a few seconds; this bounds a scenario that never converges.
CONVERGE = 60.0


def binary(env: str, default: str) -> Path:
    path = Path(os.environ.get(env, default)).resolve()
    assert path.exists(), f"{path} not found (cargo build -p smg -p mock-worker first)"
    return path


def free_port(host: str = "127.0.0.1") -> int:
    with socket.socket() as sock:
        sock.bind((host, 0))
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
    names the address that served it."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0):
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
                    code, body = 200, {"version": "discovery-e2e"}
                elif path == "/v1/chat/completions":
                    code, body = (
                        200,
                        {
                            "id": "cmpl-discovery-e2e",
                            "object": "chat.completion",
                            "created": 0,
                            "model": MODEL,
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": engine.served_by,
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

        self.server = http.server.ThreadingHTTPServer((host, port), Handler)
        self.host = host
        self.port = self.server.server_address[1]
        self.url = f"http://{host}:{self.port}"
        # What a chat completion it serves answers.
        self.served_by = f"served-by-{host}:{self.port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def engines() -> Iterator[Callable[[int], list[FakeEngine]]]:
    """Start `count` fake engines on free loopback ports; they are stopped
    after the test."""
    started: list[FakeEngine] = []

    def start(count: int) -> list[FakeEngine]:
        new = [FakeEngine() for _ in range(count)]
        started.extend(new)
        return new

    yield start
    for engine in started:
        engine.stop()


@pytest.fixture
def mock_worker() -> Iterator[Callable[..., None]]:
    """Start the mock-worker binary; it is stopped after the test."""
    procs: list[subprocess.Popen] = []

    def start(*args: str, ready_port: int, ready_host: str = "127.0.0.1") -> None:
        proc = subprocess.Popen(
            [str(binary("MOCK_WORKER_BIN", "target/debug/mock-worker")), *args],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        procs.append(proc)
        deadline = time.monotonic() + 30
        while True:
            assert proc.poll() is None, f"mock-worker exited ({proc.returncode})"
            with socket.socket() as sock:
                if sock.connect_ex((ready_host, ready_port)) == 0:
                    return
            assert time.monotonic() < deadline, f"mock-worker never listened on {ready_port}"
            time.sleep(0.2)

    yield start
    for proc in procs:
        proc.kill()
        proc.wait(timeout=10)


class Gateway:
    """One smg process discovering its workers, plus /workers assertions.

    `discovery_args` gives the flags that select and configure the provider,
    for a check interval; `env` is the process environment.
    """

    def __init__(
        self,
        discovery_args: Callable[[int], list[str]],
        log_path: Path,
        env: dict[str, str] | None = None,
    ):
        self.discovery_args = discovery_args
        self.log_path = log_path
        self.env = env
        self.proc: subprocess.Popen | None = None
        self.base_url = ""

    def start(self, *extra_args: str, check_interval_secs: int = 1) -> None:
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
                *self.discovery_args(check_interval_secs),
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
            env=self.env,
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

    def restart(self, *extra_args: str, check_interval_secs: int = 1) -> None:
        self.stop()
        self.start(*extra_args, check_interval_secs=check_interval_secs)

    def log_tail(self, lines: int = 40) -> str:
        return "\n".join(self.log_path.read_text().splitlines()[-lines:])

    def wait_for_log(self, needle: str, what: str, within: float = CONVERGE) -> None:
        deadline = time.monotonic() + within
        while needle not in self.log_path.read_text():
            assert time.monotonic() < deadline, f"timed out waiting for {what}; {self.log_tail()}"
            time.sleep(0.25)

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

    def wait_for(
        self,
        what: str,
        predicate: Callable[[list[dict]], bool],
        within: float = CONVERGE,
    ) -> list[dict]:
        deadline = time.monotonic() + within
        while time.monotonic() < deadline:
            workers = self.workers()
            if workers is not None and predicate(workers):
                return workers
            time.sleep(0.25)
        raise AssertionError(
            f"timed out waiting for {what}; workers: {self.workers()}\n{self.log_tail()}"
        )

    def wait_for_urls(self, expected: set[str], what: str, within: float = CONVERGE) -> list[dict]:
        return self.wait_for(
            what, lambda workers: {w["url"] for w in workers} == expected, within=within
        )

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
