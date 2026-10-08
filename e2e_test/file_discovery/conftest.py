"""Fixtures for the host-process file-discovery e2e.

Each test drives the real smg binary against engines on this host and edits
the manifest file directly, so a change reaches the gateway on its next
reread. No cluster is involved: the in-cluster ConfigMap path is covered by
e2e_test/kind_discovery/test_file_discovery_in_cluster.py.

Engines are either in-process fakes, whose health a test flips directly and
whose answers name their port, or the real mock-worker binary, for scale and
gRPC. Every test gets its own gateway on its own ports, so a failure in one
scenario cannot cascade into the next.

Gated behind SMG_FILE_E2E=1. Run locally:
    cargo build -p smg -p mock-worker && \\
        SMG_FILE_E2E=1 pytest e2e_test/file_discovery --confcutdir e2e_test/file_discovery
"""

from __future__ import annotations

import os
import socket
import subprocess
import time
from collections.abc import Callable, Iterator

import pytest

from .harness import FakeEngine, Gateway, Manifest, binary


def pytest_collection_modifyitems(config, items):
    if os.environ.get("SMG_FILE_E2E") == "1":
        return
    skip = pytest.mark.skip(reason="file-discovery e2e disabled (set SMG_FILE_E2E=1)")
    for item in items:
        item.add_marker(skip)


@pytest.fixture
def engines() -> Iterator[Callable[[int], list[FakeEngine]]]:
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

    def start(*args: str, ready_port: int) -> None:
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
                if sock.connect_ex(("127.0.0.1", ready_port)) == 0:
                    return
            assert time.monotonic() < deadline, f"mock-worker never listened on {ready_port}"
            time.sleep(0.2)

    yield start
    for proc in procs:
        proc.kill()
        proc.wait(timeout=10)


@pytest.fixture
def manifest(tmp_path) -> Manifest:
    return Manifest(tmp_path / "workers.json")


@pytest.fixture
def gateway(manifest, tmp_path) -> Iterator[Gateway]:
    """A gateway not yet started; a test starts it with any extra flags.
    Timeouts carry its log tail, so a failure explains itself."""
    gw = Gateway(manifest, tmp_path / "smg.log")
    yield gw
    gw.stop()
