"""Fixtures for the Slurm-discovery e2e, against a real Slurm.

A Slurm 23.11 cluster runs in containers on a network of their own: a
controller (munge, slurmctld) and six one-CPU nodes (munge, slurmd), from the
image built from this directory's Dockerfile. The gateway runs on the
controller and polls the real squeue and scontrol. Each job starts
mock-worker on every node of its allocation with srun, so every worker
listens on the same port on its own node, as a Slurm deployment's do.

Slurm 23.11 ignores squeue's filters under --json, all but --user (--me
included), so the provider's own filtering is what these tests watch. One cluster serves every
test; each test gets an empty queue and a gateway of its own.

Gated behind SMG_SLURM_E2E=1, with SMG_SLURM_IMAGE naming the image; needs
docker, with privileged containers (23.11's slurmd sets up its cgroups
itself). Slurm's logs, every gateway's and job's, and raw squeue and scontrol
answers at the main steps are copied to SMG_SLURM_LOGS when it is set. Build
and run (Linux):
    cargo build -p smg -p mock-worker
    stage=$(mktemp -d) && cp e2e_test/discovery/slurm/{Dockerfile,slurm.conf,cgroup.conf,entrypoint.sh} \\
        target/debug/smg target/debug/mock-worker "$stage" && docker build -t smg-slurm "$stage"
    SMG_SLURM_E2E=1 SMG_SLURM_IMAGE=smg-slurm \\
        pytest e2e_test/discovery/slurm --confcutdir e2e_test/discovery/slurm
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from .harness import ControllerGateway, SlurmCluster

HERE = Path(__file__).parent


def pytest_collection_modifyitems(config, items):
    if os.environ.get("SMG_SLURM_E2E") == "1":
        return
    skip = pytest.mark.skip(reason="Slurm-discovery e2e disabled (set SMG_SLURM_E2E=1)")
    # Every conftest gets the whole session's items: mark this lane's alone.
    for item in items:
        if item.path.is_relative_to(HERE):
            item.add_marker(skip)


@pytest.fixture(scope="module")
def cluster() -> Iterator[SlurmCluster]:
    image = os.environ.get("SMG_SLURM_IMAGE")
    assert image, "SMG_SLURM_IMAGE names the image built from the Dockerfile"
    logs = os.environ.get("SMG_SLURM_LOGS")
    cluster = SlurmCluster(image, Path(logs) if logs else None)
    # A start that fails still leaves Slurm's logs to copy, and containers
    # to remove.
    try:
        cluster.start()
        yield cluster
    finally:
        cluster.stop()


@pytest.fixture
def slurm(cluster) -> SlurmCluster:
    """The cluster, with no job left from an earlier test."""
    cluster.cancel_all()
    return cluster


@pytest.fixture
def gateway(slurm) -> Iterator[ControllerGateway]:
    """A gateway, not yet started, on the controller."""
    gw = ControllerGateway(slurm)
    yield gw
    gw.stop()
