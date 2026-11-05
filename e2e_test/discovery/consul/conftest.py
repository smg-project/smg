"""Fixtures for the Consul-discovery e2e, against a real Consul.

A dev-mode Consul agent, from the image SMG_CONSUL_IMAGE names, runs in a
container with its HTTP API published on a local port. The gateway and the
engines are processes on this host. Each instance is registered through the
catalog API, as a node without an agent is, at the address of an engine on
this host, with a health check whose status the test sets. So the provider's
blocking queries, Consul's passing filter, its indexes, ACLs and TLS are
Consul's own; only the checks' results are scripted.

One agent serves the module's plain tests, each starting on an empty catalog;
the ACL and TLS tests start agents of their own. Gated behind
SMG_CONSUL_E2E=1; the agents' logs are copied to SMG_CONSUL_LOGS when it is
set. Run locally (docker, and the binaries built):
    cargo build -p smg -p mock-worker && \\
        SMG_CONSUL_E2E=1 SMG_CONSUL_IMAGE=hashicorp/consul:2.0.4 \\
        pytest e2e_test/discovery/consul --confcutdir e2e_test/discovery/consul
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from ..common import Gateway
from ..common import engines as engines  # noqa: F401  (fixture)
from ..common import mock_worker as mock_worker  # noqa: F401  (fixture)
from .harness import ConsulAgent, consul_gateway, make_certificates

HERE = Path(__file__).parent


def pytest_collection_modifyitems(config, items):
    if os.environ.get("SMG_CONSUL_E2E") == "1":
        return
    skip = pytest.mark.skip(reason="Consul-discovery e2e disabled (set SMG_CONSUL_E2E=1)")
    # Every conftest gets the whole session's items: mark this lane's alone.
    for item in items:
        if item.path.is_relative_to(HERE):
            item.add_marker(skip)


@pytest.fixture(scope="module")
def image() -> str:
    image = os.environ.get("SMG_CONSUL_IMAGE")
    assert image, "SMG_CONSUL_IMAGE names the Consul image"
    return image


def logs() -> Path | None:
    path = os.environ.get("SMG_CONSUL_LOGS")
    return Path(path) if path else None


def running(agent: ConsulAgent) -> Iterator[ConsulAgent]:
    # A start that fails still leaves the container to remove, and its log.
    try:
        agent.start()
        yield agent
    finally:
        agent.remove()


@pytest.fixture(scope="module")
def consul(image) -> Iterator[ConsulAgent]:
    yield from running(ConsulAgent(image, logs()))


@pytest.fixture
def agent(consul) -> ConsulAgent:
    """The module's agent, with no instance left from an earlier test."""
    consul.clear()
    return consul


@pytest.fixture(scope="module")
def acl_agent(image) -> Iterator[ConsulAgent]:
    """An agent with ACLs on, denying by default."""
    yield from running(ConsulAgent(image, logs(), acl=True))


@pytest.fixture(scope="module")
def tls_dir(image, tmp_path_factory) -> Path:
    return make_certificates(image, tmp_path_factory.mktemp("consul-tls"))


@pytest.fixture(scope="module")
def tls_agent(image, tls_dir) -> Iterator[ConsulAgent]:
    """An agent whose API is HTTPS only and asks for a client certificate."""
    yield from running(ConsulAgent(image, logs(), tls_dir=tls_dir))


@pytest.fixture
def gateway(agent, tmp_path) -> Iterator[Gateway]:
    """A gateway, not yet started, on the module's agent."""
    gw = consul_gateway(agent, tmp_path / "smg.log")
    yield gw
    gw.stop()
