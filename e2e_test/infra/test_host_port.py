"""Unit tests for the harness's address handling: the ``E2E_HOST`` switch, the
bracket-aware ``host_port`` join and the family-aware port reservation."""

import socket

import pytest
from infra.constants import ENV_HOST, default_host, host_port
from infra.process_utils import get_open_port, release_port


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
