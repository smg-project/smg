"""Fixtures for the host-process file-discovery e2e.

Each test drives the real smg binary against engines on this host and edits
the manifest file directly, so a change reaches the gateway when its watch
sees the file change, or on its next periodic reread. No cluster is
involved: the in-cluster ConfigMap path is covered by
e2e_test/discovery/kind/test_file_discovery_in_cluster.py.

Engines are either in-process fakes, whose health a test flips directly and
whose answers name their port, or the real mock-worker binary, for scale and
gRPC. Every test gets its own gateway on its own ports, so a failure in one
scenario cannot cascade into the next.

Gated behind SMG_FILE_E2E=1. Run locally:
    cargo build -p smg -p mock-worker && \\
        SMG_FILE_E2E=1 pytest e2e_test/discovery/file --confcutdir e2e_test/discovery/file
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from ..common import Gateway
from ..common import engines as engines  # noqa: F401  (fixture)
from ..common import mock_worker as mock_worker  # noqa: F401  (fixture)
from .harness import Manifest, file_gateway

HERE = Path(__file__).parent


def pytest_collection_modifyitems(config, items):
    if os.environ.get("SMG_FILE_E2E") == "1":
        return
    skip = pytest.mark.skip(reason="file-discovery e2e disabled (set SMG_FILE_E2E=1)")
    # Every conftest gets the whole session's items: mark this lane's alone.
    for item in items:
        if item.path.is_relative_to(HERE):
            item.add_marker(skip)


@pytest.fixture
def manifest(tmp_path) -> Manifest:
    return Manifest(tmp_path / "workers.json")


@pytest.fixture
def gateway(manifest, tmp_path) -> Iterator[Gateway]:
    """A gateway not yet started; a test starts it with any extra flags.
    Timeouts carry its log tail, so a failure explains itself."""
    gw = file_gateway(manifest, tmp_path / "smg.log")
    yield gw
    gw.stop()
