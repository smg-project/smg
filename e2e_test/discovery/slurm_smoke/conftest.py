"""Fixtures for the real-Slurm smoke.

A real one-node Slurm runs in one container (Dockerfile), with the workspace
smg and mock-worker binaries beside it. The test submits a batch job that
starts mock-worker, and the gateway, inside the same container, discovers it
through the real squeue and scontrol.

It checks Slurm's own commands and job lifecycle against the provider, not
the provider's rules: those, and every multi-node or multi-role case, are in
the CLI-contract lane (e2e_test/discovery/slurm). One node hosts one worker on
the shared port, so the smoke runs one job.

Gated behind SMG_SLURM_SMOKE=1, with SMG_SLURM_SMOKE_IMAGE naming the image
built from the Dockerfile; needs docker. Slurm's logs, the gateway's, the
job's, and each step's raw squeue and scontrol answers are copied to
SMG_SLURM_SMOKE_LOGS when it is set. Build and run (Linux):
    cargo build -p smg -p mock-worker
    stage=$(mktemp -d) && cp e2e_test/discovery/slurm_smoke/{Dockerfile,slurm.conf,cgroup.conf,entrypoint.sh} \\
        target/debug/smg target/debug/mock-worker "$stage" && docker build -t smg-slurm-smoke "$stage"
    SMG_SLURM_SMOKE=1 SMG_SLURM_SMOKE_IMAGE=smg-slurm-smoke \\
        pytest e2e_test/discovery/slurm_smoke --confcutdir e2e_test/discovery/slurm_smoke
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from .harness import ContainerGateway, SlurmNode


def pytest_collection_modifyitems(config, items):
    if os.environ.get("SMG_SLURM_SMOKE") == "1":
        return
    skip = pytest.mark.skip(reason="real-Slurm smoke disabled (set SMG_SLURM_SMOKE=1)")
    for item in items:
        item.add_marker(skip)


@pytest.fixture
def slurm_node() -> Iterator[SlurmNode]:
    image = os.environ.get("SMG_SLURM_SMOKE_IMAGE")
    assert image, "SMG_SLURM_SMOKE_IMAGE names the image built from the Dockerfile"
    logs = os.environ.get("SMG_SLURM_SMOKE_LOGS")
    node = SlurmNode(image, Path(logs) if logs else None)
    # A start that fails still leaves Slurm's logs to copy, and a container
    # to remove.
    try:
        node.start()
        yield node
    finally:
        node.stop()


@pytest.fixture
def gateway(slurm_node) -> Iterator[ContainerGateway]:
    """A gateway, not yet started, inside the Slurm container."""
    gw = ContainerGateway(slurm_node)
    yield gw
    gw.stop()
