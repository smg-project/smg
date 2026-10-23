"""IPv6-safe ``host:port`` joins shared by the Python servicers."""

import socket
from concurrent import futures

import pytest
from smg_grpc_servicer.hostport import bracket_if_ipv6, host_port, loopback_target


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("::", "[::]:50051"),
        ("[::]", "[::]:50051"),
        ("::1", "[::1]:50051"),
        ("fd00::1", "[fd00::1]:50051"),
        ("[fd00::1]", "[fd00::1]:50051"),
        ("0.0.0.0", "0.0.0.0:50051"),
        ("127.0.0.1", "127.0.0.1:50051"),
        ("localhost", "localhost:50051"),
        ("engine-0.engines.svc", "engine-0.engines.svc:50051"),
    ],
)
def test_host_port_brackets_ipv6_literals_only(host, expected):
    assert host_port(host, 50051) == expected


def test_bracket_if_ipv6_is_idempotent():
    assert bracket_if_ipv6(bracket_if_ipv6("fd00::1")) == "[fd00::1]"
    assert bracket_if_ipv6("10.0.0.7") == "10.0.0.7"


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("0.0.0.0", "127.0.0.1:50051"),
        ("::", "[::1]:50051"),
        ("[::]", "[::1]:50051"),
        ("::1", "[::1]:50051"),
        ("[::1]", "[::1]:50051"),
        ("fd00::1", "[fd00::1]:50051"),
        ("10.0.0.7", "10.0.0.7:50051"),
        ("localhost", "localhost:50051"),
    ],
)
def test_loopback_target_maps_wildcard_binds_to_loopback(host, expected):
    assert loopback_target(host, 50051) == expected


def _ipv6_loopback_available() -> bool:
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("::1", 0))
        return True
    except OSError:
        return False


@pytest.mark.skipif(not _ipv6_loopback_available(), reason="no IPv6 loopback on this host")
def test_grpc_server_bound_on_a_bare_ipv6_host_is_reachable():
    """``--host ::`` end to end with grpc-python: the bind string the servicers
    build must bind, and the warm-up target must connect to it."""
    grpc = pytest.importorskip("grpc")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    port = server.add_insecure_port(host_port("::", 0))
    assert port != 0
    server.start()
    try:
        channel = grpc.insecure_channel(loopback_target("::", port))
        try:
            grpc.channel_ready_future(channel).result(timeout=10)
        finally:
            channel.close()
    finally:
        server.stop(0)


def test_a_bare_ipv6_join_is_what_grpc_rejects():
    """The string the servicers used to build for ``--host ::``."""
    grpc = pytest.importorskip("grpc")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    try:
        port = server.add_insecure_port(f"{'::'}:{0}")
    except RuntimeError:
        port = 0
    assert port == 0
