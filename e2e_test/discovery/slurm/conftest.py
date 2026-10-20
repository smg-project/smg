"""Fixtures for the Slurm-discovery CLI-contract e2e.

Each test drives the real smg binary with strict stand-ins for `squeue` and
`scontrol` first on its PATH (fake_bin/). They answer from a state file the
test edits and record every call, so the test crosses the real command
boundary: the arguments the provider passes, the JSON it reads back, and
`scontrol`'s hostlist expansion. It is not a real-Slurm test.

Every node runs its server on one shared port, as Slurm deployments do, so
each node is a loopback address of its own: 127.0.0.2, 127.0.0.3 and so on.
Linux routes all of 127.0.0.0/8 to the loopback interface; elsewhere the
lane stops with an error rather than passing without running.

Gated behind SMG_SLURM_E2E=1. Run locally (Linux):
    cargo build -p smg -p mock-worker && \\
        SMG_SLURM_E2E=1 pytest e2e_test/discovery/slurm --confcutdir e2e_test/discovery/slurm
"""

from __future__ import annotations

import os
import socket
from collections.abc import Callable, Iterator

import pytest

from ..common import FakeEngine, Gateway, free_port
from ..common import mock_worker as mock_worker  # noqa: F401  (fixture)
from .harness import FakeSlurm, slurm_gateway

# Loopback addresses standing in for the nodes.
NODE_ADDRESSES = [f"127.0.0.{n}" for n in range(2, 9)]


def _loopback_range_usable() -> bool:
    try:
        with socket.socket() as sock:
            sock.bind((NODE_ADDRESSES[0], 0))
        return True
    except OSError:
        return False


def pytest_collection_modifyitems(config, items):
    if os.environ.get("SMG_SLURM_E2E") != "1":
        skip = pytest.mark.skip(reason="Slurm-discovery e2e disabled (set SMG_SLURM_E2E=1)")
        for item in items:
            item.add_marker(skip)
    elif not _loopback_range_usable():
        # Asked for, so not skipped: a lane that cannot run must not pass.
        pytest.exit(
            f"{NODE_ADDRESSES[0]} is not a loopback address here; the Slurm e2e needs "
            "Linux, which routes all of 127.0.0.0/8 to the loopback interface",
            returncode=4,
        )


@pytest.fixture
def port() -> int:
    """The port every node's server listens on."""
    return free_port(NODE_ADDRESSES[0])


@pytest.fixture
def node_engines(port) -> Iterator[Callable[..., list[FakeEngine]]]:
    """Start a fake engine on each given node address, on the shared port."""
    started: list[FakeEngine] = []

    def start(*addresses: str) -> list[FakeEngine]:
        new = [FakeEngine(address, port) for address in addresses]
        started.extend(new)
        return new

    yield start
    for engine in started:
        engine.stop()


@pytest.fixture
def slurm(tmp_path) -> FakeSlurm:
    return FakeSlurm(tmp_path)


@pytest.fixture
def gateway(slurm, port, tmp_path) -> Iterator[Gateway]:
    """A gateway not yet started; a test starts it with its mappings.
    Timeouts carry its log tail, so a failure explains itself."""
    gw = slurm_gateway(slurm, port, tmp_path / "smg.log")
    yield gw
    gw.stop()
