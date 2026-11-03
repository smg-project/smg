"""Unit tests for the harness's address handling: the ``E2E_HOST`` switch, the
bracket-aware ``host_port`` join, the family-aware port reservation and the
range the engines' late-bound ports come from."""

import socket
from types import SimpleNamespace

import pytest
from infra import process_utils
from infra.constants import DEFAULT_HOST, ENV_HOST, default_host, host_port
from infra.process_utils import get_engine_port, get_open_port, release_port


def test_default_host_is_the_ipv4_loopback_unless_e2e_host_says_otherwise(monkeypatch):
    monkeypatch.delenv(ENV_HOST, raising=False)
    assert default_host() == "127.0.0.1"
    monkeypatch.setenv(ENV_HOST, " ")
    assert default_host() == "127.0.0.1"
    monkeypatch.setenv(ENV_HOST, "::1")
    assert default_host() == "::1"


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("127.0.0.1", "127.0.0.1:30000"),
        ("localhost", "localhost:30000"),
        ("::1", "[::1]:30000"),
        ("fd00::1", "[fd00::1]:30000"),
        ("[::1]", "[::1]:30000"),
    ],
)
def test_host_port_brackets_an_ipv6_literal_once(host, expected):
    assert host_port(host, 30000) == expected
    assert host_port(host, "30000") == expected


def _binds(host: str, port: int) -> bool:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
        except OSError:
            return False
    return True


def test_a_reserved_port_binds_on_the_ipv4_loopback():
    port = get_open_port(host="127.0.0.1")
    try:
        assert _binds("127.0.0.1", port)
    finally:
        release_port(port)


def test_a_reserved_port_binds_on_the_ipv6_loopback():
    if not _binds("::1", 0):
        pytest.skip("no IPv6 loopback on this host")
    port = get_open_port(host="::1")
    try:
        assert _binds("::1", port)
    finally:
        release_port(port)


def test_reserved_ports_are_not_handed_out_twice():
    ports = [get_open_port(host="127.0.0.1") for _ in range(5)]
    try:
        assert len(set(ports)) == len(ports)
    finally:
        for port in ports:
            release_port(port)


def _ephemeral_range() -> tuple[int, int]:
    try:
        with open("/proc/sys/net/ipv4/ip_local_port_range") as f:
            low, high = f.read().split()
    except (OSError, ValueError):
        pytest.skip("no readable ip_local_port_range on this host")
    return int(low), int(high)


def test_an_engines_port_lies_below_the_kernels_ephemeral_range():
    """An engine binds its port after its model load, so the number must be
    one no other process's ``bind(0)`` can be handed meanwhile: below the
    range the kernel allots those from. The number is free and distinct."""
    low, _ = _ephemeral_range()
    if low <= 20001:
        pytest.skip(f"no room below the ephemeral range (floor {low}) on this host")
    ports = [get_engine_port() for _ in range(5)]
    try:
        assert len(set(ports)) == len(ports)
        for port in ports:
            assert 1024 <= port < low, (port, low)
            assert _binds(DEFAULT_HOST, port)
    finally:
        for port in ports:
            release_port(port)


@pytest.mark.parametrize(
    ("ephemeral_floor", "ceiling"),
    [
        pytest.param(32768, 30000, id="stock-kernel-node-ports-inside-the-window"),
        pytest.param(28000, 28000, id="ephemeral-range-starting-below-the-node-ports"),
    ],
)
def test_an_engines_port_lies_below_the_node_port_range_too(monkeypatch, ephemeral_floor, ceiling):
    """Kubernetes keeps 30000-32767 for its node ports. A runner on its node's
    network namespace binds one without an error, but the node's service proxy
    answers the connections to it: an engine listened on such a port for the
    whole start-up bound while every probe was refused. The draw window ends
    at the node-port floor, or at the kernel's ephemeral floor when that is
    lower."""
    monkeypatch.setattr(process_utils, "_ephemeral_port_floor", lambda: ephemeral_floor)
    monkeypatch.setattr(process_utils, "port_is_free", lambda port, host=DEFAULT_HOST: True)
    windows: list[tuple[int, int]] = []

    def randrange(start: int, stop: int) -> int:
        windows.append((start, stop))
        return stop - 1

    monkeypatch.setattr(process_utils, "random", SimpleNamespace(randrange=randrange))
    port = get_engine_port()
    try:
        assert windows == [(20000, ceiling - 1)]
        assert port == ceiling - 2
    finally:
        release_port(port)


def test_the_fallback_without_room_below_the_ephemeral_range_skips_a_node_port_too(monkeypatch):
    """A kernel whose ephemeral range reaches down to the low floor leaves the
    allocator its fallback, a kernel-chosen port; a node port drawn there is
    refused the same way and the next draw is taken."""
    monkeypatch.setattr(process_utils, "_ephemeral_port_floor", lambda: 1024)
    monkeypatch.setattr(process_utils, "port_is_free", lambda port, host=DEFAULT_HOST: True)
    draws = iter([30565, 25000])
    monkeypatch.setattr(process_utils, "get_open_port", lambda *args, **kwargs: next(draws))
    port = get_engine_port()
    try:
        assert port == 25000
    finally:
        release_port(port)
