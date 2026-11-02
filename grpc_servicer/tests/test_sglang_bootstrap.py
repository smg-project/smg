"""Engine-free bootstrap startup contracts using real TCP listeners."""

import json
import socket
import subprocess
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from smg_grpc_servicer.sglang import bootstrap
from smg_grpc_servicer.sglang.bootstrap import wait_for_bootstrap_listener


class Listener:
    def __init__(self, *, delayed=False, bind=True, host="127.0.0.1"):
        self.host = host
        self.socket = socket.socket()
        self.socket.bind((host, 0))
        self.port = self.socket.getsockname()[1]
        self.socket.close()
        self.socket = socket.socket()
        self.release_start = threading.Event()
        self.stop = threading.Event()
        self.bound = threading.Event()
        self.closed = False
        self._runner = SimpleNamespace(addresses=[])

        def run():
            self.release_start.wait()
            if bind:
                self.socket.bind((host, self.port))
                self.socket.listen()
                self._runner.addresses = [self.socket.getsockname()]
                self.bound.set()
            self.stop.wait()
            self.socket.close()

        self.thread = threading.Thread(target=run)
        self.thread.start()
        if not delayed:
            self.release_start.set()
            if bind:
                assert self.bound.wait(1)

    def close(self):
        self.closed = True
        self.stop.set()
        self.release_start.set()
        self.thread.join(timeout=1)
        assert not self.thread.is_alive()


def test_waits_for_background_bind_before_returning():
    server = Listener(delayed=True)
    checked = threading.Event()
    errors = []

    def check():
        try:
            wait_for_bootstrap_listener(server, server.host, server.port, timeout=1)
            checked.set()
        except BaseException as error:
            errors.append(error)

    check_thread = threading.Thread(target=check)
    try:
        check_thread.start()
        assert not checked.wait(0.05)
        server.release_start.set()
        check_thread.join(timeout=2)
        assert checked.is_set() and not errors
        assert not server.closed
    finally:
        server.close()
        check_thread.join(timeout=2)


def test_dead_background_thread_is_failure_even_if_port_was_taken():
    listener = Listener()
    dead_thread = threading.Thread(target=lambda: None)
    dead_thread.start()
    dead_thread.join()
    server = SimpleNamespace(port=listener.port, thread=dead_thread, close=Mock())
    try:
        with pytest.raises(RuntimeError, match="exited before its listener was ready"):
            wait_for_bootstrap_listener(server, listener.host, listener.port)
        server.close.assert_called_once_with()
    finally:
        listener.close()


def test_foreign_listener_does_not_qualify_a_live_unbound_backend():
    listener = Listener()
    stopped = threading.Event()
    own_thread = threading.Thread(target=stopped.wait)
    own_thread.start()

    def close():
        stopped.set()
        own_thread.join(timeout=1)

    server = SimpleNamespace(
        port=listener.port,
        thread=own_thread,
        _runner=SimpleNamespace(addresses=[]),
        close=Mock(side_effect=close),
    )
    try:
        with pytest.raises(RuntimeError, match="owned addresses: \\[\\]"):
            wait_for_bootstrap_listener(server, listener.host, listener.port, timeout=0.05)
        server.close.assert_called_once_with()
        assert not own_thread.is_alive()
    finally:
        close()
        listener.close()


def test_owned_listener_on_another_host_does_not_qualify_declared_host():
    server = Listener()
    server._runner.addresses = [("127.0.0.2", server.port)]
    with pytest.raises(RuntimeError, match="did not listen"):
        wait_for_bootstrap_listener(server, server.host, server.port, timeout=0.05)
    assert server.closed


def test_unknown_ownership_interface_does_not_fall_back_to_foreign_tcp():
    listener = Listener()
    server = SimpleNamespace(port=listener.port, thread=listener.thread, close=Mock())
    try:
        with pytest.raises(RuntimeError, match="expected backend-owned _runner.addresses"):
            wait_for_bootstrap_listener(server, listener.host, listener.port, timeout=0.05)
        server.close.assert_called_once_with()
    finally:
        listener.close()


def test_live_thread_with_no_listener_times_out_and_closes():
    server = Listener(bind=False)
    with pytest.raises(RuntimeError, match="did not listen.*last connection error"):
        wait_for_bootstrap_listener(server, server.host, server.port, timeout=0.05)
    assert server.closed


def test_returned_port_must_match_declared_rendezvous():
    server = Listener()
    with pytest.raises(RuntimeError, match="started with port"):
        wait_for_bootstrap_listener(server, server.host, server.port + 1)
    assert server.closed


def test_prefill_must_return_a_bootstrap_handle():
    with pytest.raises(RuntimeError, match="returned no server"):
        wait_for_bootstrap_listener(None, "127.0.0.1", 8998)


def test_wildcard_listener_is_probed_through_loopback():
    server = Listener(host="0.0.0.0")
    try:
        wait_for_bootstrap_listener(server, server.host, server.port)
        assert not server.closed
    finally:
        server.close()


def test_failed_backend_close_preserves_startup_error():
    server = SimpleNamespace(port=123, close=Mock(side_effect=RuntimeError("close failed")))
    with pytest.raises(RuntimeError, match="started with port 123"):
        wait_for_bootstrap_listener(server, "127.0.0.1", 456)
    server.close.assert_called_once_with()


def test_numeric_listener_never_enters_a_slow_hostname_resolver(monkeypatch):
    """The original helper exceeds its timeout even for a ready numeric listener."""
    server = Listener()
    original = socket.getaddrinfo

    def slow_resolution(*args, **kwargs):
        time.sleep(0.2)
        return original(*args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", slow_resolution)
    started = time.monotonic()
    try:
        wait_for_bootstrap_listener(server, server.host, server.port, timeout=0.05)
        assert time.monotonic() - started < 0.15
        assert not server.closed
    finally:
        server.close()


def test_hostname_resolution_reaches_only_the_owned_listener(monkeypatch):
    server = Listener()
    resolver = Mock(side_effect=AssertionError("the parent must not resolve the hostname again"))
    monkeypatch.setattr(socket, "getaddrinfo", resolver)
    try:
        wait_for_bootstrap_listener(server, "localhost", server.port, timeout=2)
        resolver.assert_not_called()
        assert not server.closed
    finally:
        server.close()


def test_slow_resolver_child_is_killed_reaped_and_backend_closed(monkeypatch):
    server = Listener()
    server.close = Mock(wraps=server.close)
    popen = subprocess.Popen
    children = []

    def stalled_resolver(argv, **kwargs):
        assert argv[-2:] == ["localhost", str(server.port)]
        assert argv[1:4] == ["-I", "-S", "-c"]
        assert kwargs["close_fds"] is True and "shell" not in kwargs
        child = popen([argv[0], "-I", "-S", "-c", "import time; time.sleep(30)"], **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(bootstrap.subprocess, "Popen", stalled_resolver)
    started = time.monotonic()
    try:
        with pytest.raises(RuntimeError, match="resolution deadline expired"):
            wait_for_bootstrap_listener(server, "localhost", server.port, timeout=0.05)
        assert time.monotonic() - started < 1.2
        assert len(children) == 1 and children[0].poll() is not None
        server.close.assert_called_once_with()
        assert server.closed
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=1)
        if not server.closed:
            server.close()


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1"])
def test_zero_deadline_starts_neither_resolver_nor_probe(monkeypatch, host):
    child = Mock(side_effect=AssertionError("expired readiness must not start DNS"))
    monkeypatch.setattr(bootstrap.subprocess, "Popen", child)
    server = SimpleNamespace(port=8998, close=Mock())
    with pytest.raises(RuntimeError, match="resolution deadline expired"):
        wait_for_bootstrap_listener(server, host, 8998, timeout=0)
    child.assert_not_called()
    server.close.assert_called_once_with()


@pytest.mark.parametrize(
    ("host", "family", "address"),
    [
        ("127.0.0.1", socket.AF_INET, ("127.0.0.1", 8998)),
        ("0:0:0:0:0:0:0:1", socket.AF_INET6, ("::1", 8998, 0, 0)),
        ("fe80::1%7", socket.AF_INET6, ("fe80::1", 8998, 0, 7)),
        ("fe80::1%test-interface", socket.AF_INET6, ("fe80::1", 8998, 0, 9)),
    ],
)
def test_numeric_resolution_preserves_address_family_and_ipv6_scope(
    monkeypatch, host, family, address
):
    monkeypatch.setattr(socket, "if_nametoindex", lambda name: 9)
    child = Mock(side_effect=AssertionError("numeric addresses need no resolver"))
    monkeypatch.setattr(bootstrap.subprocess, "Popen", child)
    assert bootstrap._resolve_addresses(host, 8998, deadline=time.monotonic() + 1) == [
        (family, address)
    ]
    child.assert_not_called()


@pytest.mark.parametrize("connect_succeeds_after_deadline", [False, True])
def test_all_addresses_share_one_deadline_and_late_connect_cannot_qualify(
    monkeypatch, connect_succeeds_after_deadline
):
    now = [0.0]
    budgets = []
    connected = []
    server = SimpleNamespace(
        port=8998,
        thread=SimpleNamespace(is_alive=lambda: True),
        _runner=SimpleNamespace(addresses=[("0.0.0.0", 8998)]),
        close=Mock(),
    )

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def settimeout(self, value):
            budgets.append(value)

        def connect(self, address):
            connected.append(address)
            now[0] += 0.1
            if not connect_succeeds_after_deadline:
                raise OSError("controlled connection exhaustion")

    monkeypatch.setattr(
        bootstrap,
        "time",
        SimpleNamespace(monotonic=lambda: now[0], sleep=lambda seconds: None),
    )
    monkeypatch.setattr(
        bootstrap,
        "_resolve_addresses",
        lambda *args, **kwargs: [
            (socket.AF_INET, ("127.0.0.1", 8998)),
            (socket.AF_INET, ("127.0.0.2", 8998)),
        ],
    )
    monkeypatch.setattr(socket, "socket", lambda *args: Connection())
    with pytest.raises(RuntimeError, match="did not listen within 0.1s"):
        wait_for_bootstrap_listener(server, "localhost", 8998, timeout=0.1)
    assert budgets == [0.1] and connected == [("127.0.0.1", 8998)]
    server.close.assert_called_once_with()


def test_an_owned_address_does_not_authorize_a_different_resolved_listener(monkeypatch):
    server = Listener()
    monkeypatch.setattr(
        bootstrap,
        "_resolve_addresses",
        lambda *args, **kwargs: [
            (socket.AF_INET, ("127.0.0.2", server.port)),
            (socket.AF_INET, ("127.0.0.1", server.port)),
        ],
    )
    socket_type = socket.socket
    connected = []

    class Connection:
        def __init__(self, *args):
            self.socket = socket_type(*args)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.socket.close()

        def settimeout(self, value):
            self.socket.settimeout(value)

        def connect(self, address):
            connected.append(address)
            self.socket.connect(address)

    monkeypatch.setattr(socket, "socket", Connection)
    try:
        wait_for_bootstrap_listener(server, "localhost", server.port)
        assert connected == [("127.0.0.1", server.port)]
        assert not server.closed
    finally:
        server.close()


def test_ipv6_listener_on_another_scope_does_not_qualify_the_target(monkeypatch):
    now = [0.0]

    def monotonic():
        now[0] += 0.01
        return now[0]

    monkeypatch.setattr(
        bootstrap, "time", SimpleNamespace(monotonic=monotonic, sleep=lambda _: None)
    )
    monkeypatch.setattr(
        bootstrap,
        "_resolve_addresses",
        lambda *args, **kwargs: [(socket.AF_INET6, ("fe80::1", 8998, 0, 7))],
    )
    connection = Mock(side_effect=AssertionError("the backend owns another interface"))
    monkeypatch.setattr(socket, "socket", connection)
    server = SimpleNamespace(
        port=8998,
        thread=SimpleNamespace(is_alive=lambda: True),
        _runner=SimpleNamespace(addresses=[("fe80::1", 8998, 0, 9)]),
        close=Mock(),
    )
    with pytest.raises(RuntimeError, match="did not listen"):
        wait_for_bootstrap_listener(server, "fe80::1%7", 8998, timeout=0.03)
    connection.assert_not_called()
    server.close.assert_called_once_with()


@pytest.mark.parametrize(
    "frame",
    [
        b"",
        b"not JSON",
        b"[]",
        b'{"addresses":[]}',
        b'{"addresses":[[2,["localhost",8998]]]}',
        b'{"addresses":[[2,["127.0.0.1",9000]]]}',
        b'{"addresses":[[true,["127.0.0.1",8998]]]}',
        b'{"addresses":[[10,["::1",8998,0,-1]]]}',
        b'{"addresses":[[10,["::1",8998,1048576,0]]]}',
        b'{"error":["not an errno","bad"]}',
        b"x" * 65537,
    ],
)
def test_invalid_resolver_output_fails_startup_and_closes_backend(monkeypatch, frame):
    class Resolver:
        returncode = 0

        def wait(self, *, timeout):
            return 0

        def poll(self):
            return 0

    def resolver(*args, **kwargs):
        kwargs["stdout"].write(frame)
        return Resolver()

    monkeypatch.setattr(bootstrap.subprocess, "Popen", resolver)
    server = SimpleNamespace(port=8998, close=Mock())
    with pytest.raises(RuntimeError, match="Invalid bootstrap resolver response"):
        wait_for_bootstrap_listener(server, "localhost", 8998)
    server.close.assert_called_once_with()


def test_resolver_dns_failure_preserves_typed_error_and_backend_cleanup(monkeypatch):
    def resolver(*args, **kwargs):
        frame = {"error": [socket.EAI_NONAME, "controlled hostname failure"]}
        kwargs["stdout"].write(json.dumps(frame).encode())
        return SimpleNamespace(returncode=0, wait=lambda **kwargs: 0, poll=lambda: 0)

    monkeypatch.setattr(bootstrap.subprocess, "Popen", resolver)
    server = SimpleNamespace(port=8998, close=Mock())
    with pytest.raises(socket.gaierror, match="controlled hostname failure"):
        wait_for_bootstrap_listener(server, "localhost", 8998)
    server.close.assert_called_once_with()


def test_nonzero_resolver_exit_fails_before_scheduler_startup(monkeypatch):
    resolver = SimpleNamespace(returncode=3, wait=lambda **kwargs: 3, poll=lambda: 3)
    monkeypatch.setattr(bootstrap.subprocess, "Popen", lambda *args, **kwargs: resolver)
    server = SimpleNamespace(port=8998, close=Mock())
    with pytest.raises(RuntimeError, match="resolver exited with code 3"):
        wait_for_bootstrap_listener(server, "localhost", 8998)
    server.close.assert_called_once_with()


def test_failed_resolver_reap_and_backend_close_preserve_primary_timeout(monkeypatch, caplog):
    class Resolver:
        returncode = None

        def __init__(self):
            self.waits = []
            self.killed = False

        def wait(self, *, timeout):
            self.waits.append(timeout)
            if len(self.waits) == 1:
                raise subprocess.TimeoutExpired("owned resolver", timeout)
            raise OSError("controlled reap failure")

        def poll(self):
            return None

        def kill(self):
            self.killed = True

    resolver = Resolver()
    monkeypatch.setattr(bootstrap.subprocess, "Popen", lambda *args, **kwargs: resolver)
    server = SimpleNamespace(port=8998, close=Mock(side_effect=RuntimeError("close failed")))
    with pytest.raises(RuntimeError, match="resolution deadline expired"):
        wait_for_bootstrap_listener(server, "localhost", 8998)
    assert resolver.killed and len(resolver.waits) == 2 and resolver.waits[-1] == 1.0
    assert "Could not reap the owned bootstrap resolver" in caplog.text
    server.close.assert_called_once_with()
