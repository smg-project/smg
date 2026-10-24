"""Unit tests for ``Worker.stop()``: whatever the process state, the log handle
is closed and the reserved ports go back to the pool.

Each case spawns a real stub process through the harness's own output
redirection, or none at all (a launch that failed before the spawn).
"""

import os
import sys

import pytest
from infra.constants import ENV_SHOW_WORKER_LOGS, ConnectionMode
from infra.process_utils import _reserved_ports, live_process_group_members, reserve_port
from infra.worker import Worker

PORTS = {
    "port": 31001,
    "bootstrap_port": 31002,
    "nixl_port": 31003,
    "dist_init_port": 31004,
    "rl_control_port": 31005,
    "admin_port": 31006,
}
MORIIO_PORTS = [31007, 31008]
EVERY_PORT = set(PORTS.values()) | set(MORIIO_PORTS)


@pytest.fixture
def worker(tmp_path, monkeypatch):
    """A worker holding every kind of port the harness reserves for one."""
    monkeypatch.setenv(ENV_SHOW_WORKER_LOGS, "0")
    worker = Worker(
        model_id="org/model",
        engine="vllm",
        gpu_ids=[0],
        mode=ConnectionMode.HTTP,
        log_dir=str(tmp_path / "e2e-logs"),
        **PORTS,
    )
    worker._admin_port_auto = True
    worker._moriio_ports = list(MORIIO_PORTS)
    for port in EVERY_PORT:
        reserve_port(port)
    yield worker
    _reserved_ports.difference_update(EVERY_PORT)


def _spawn(worker: Worker, script: str) -> None:
    worker.process = worker._spawn_process([sys.executable, "-c", script], env=dict(os.environ))
    # As ``start()`` keeps it: the group ``stop()`` reaps once the launcher is gone.
    worker._pgid = worker.process.pid


def test_stop_of_a_process_that_exited_on_its_own_closes_the_log_and_releases_the_ports(worker):
    _spawn(worker, "import sys; print('INFO loading weights'); sys.exit(1)")
    worker.process.wait(timeout=30)
    handle = worker._log_file
    assert handle is not None and not handle.closed

    worker.stop()

    assert handle.closed
    assert worker._log_file is None
    assert not (EVERY_PORT & _reserved_ports)


def test_stop_of_a_worker_whose_launch_failed_before_the_spawn_releases_the_ports(worker):
    assert worker.process is None

    worker.stop()

    assert worker._log_file is None
    assert not (EVERY_PORT & _reserved_ports)


def test_stop_of_a_live_process_terminates_it_then_releases(worker):
    _spawn(worker, "import time; time.sleep(60)")
    handle = worker._log_file

    worker.stop()

    assert worker.process.poll() is not None
    assert handle is not None and handle.closed
    assert not (EVERY_PORT & _reserved_ports)


def test_stop_of_a_process_that_exited_drains_the_children_it_left_in_its_group(worker):
    # The launcher leads its own session; a child it started (an engine
    # process tearing down, in the real case) stays in that group after it
    # exits, and stop() waits for the group to empty as it does after a
    # termination: here the child ends on its own, inside the wait's bound.
    _spawn(
        worker,
        "import subprocess, sys;"
        " subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(3)']); sys.exit(0)",
    )
    worker.process.wait(timeout=30)
    assert live_process_group_members(worker.process.pid), "the stub left no child to drain"

    worker.stop()

    assert not live_process_group_members(worker.process.pid)
    assert worker._log_file is None
    assert not (EVERY_PORT & _reserved_ports)
