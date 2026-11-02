"""Engine-free bootstrap startup contracts using real TCP listeners."""

import socket
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
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
