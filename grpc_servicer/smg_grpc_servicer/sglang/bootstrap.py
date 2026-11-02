"""Wait for SGLang's background bootstrap listener before starting its ranks.

The backend constructor starts a daemon thread; returning a handle does not
mean it bound its port. Prefill ranks register only during their startup, so
launching them against a dead listener can leave a healthy gRPC worker whose
KV rendezvous never works.
"""

from __future__ import annotations

import socket
import time
from typing import Any


def wait_for_bootstrap_listener(server: Any, host: str, port: int, *, timeout: float = 5.0) -> None:
    """Require the declared listener, and release a failed backend's handle.

    This precedes scheduler/model startup and does not wait for the ranks to
    register. Keep the returned handle alive after this call, as before.
    The backend-owned runner is present on CommonKVBootstrapServer in the
    minimum supported SGLang 0.5.21 and current release 0.5.22. A backend
    without that ownership signal fails closed, rather than trusting a
    foreign listener at the expected port.
    """
    endpoint = f"{host}:{port}"
    try:
        if server is None:
            raise RuntimeError(f"SGLang prefill bootstrap {endpoint} returned no server")
        actual_port = getattr(server, "port", port)
        if actual_port != port:
            raise RuntimeError(
                f"SGLang prefill bootstrap {endpoint} started with port {actual_port}"
            )
        probe_host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(host, host)
        target_hosts = {
            address[4][0].split("%", 1)[0]
            for address in socket.getaddrinfo(probe_host, port, type=socket.SOCK_STREAM)
        }
        deadline = time.monotonic() + timeout
        last_error: OSError | None = None
        addresses = ()
        while True:
            thread = getattr(server, "thread", None)
            if thread is not None and not thread.is_alive():
                raise RuntimeError(
                    f"SGLang prefill bootstrap {endpoint} exited before its listener was ready"
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    f"SGLang prefill bootstrap {endpoint} did not listen within {timeout:g}s; "
                    f"owned addresses: {addresses}; last connection error: {last_error}; "
                    f"backend: {type(server).__module__}.{type(server).__qualname__}; "
                    "expected backend-owned _runner.addresses"
                )
            # A different process may take the port while the bootstrap
            # thread is still starting. Its accepting socket is not proof
            # that our backend bound. SGLang's CommonKVBootstrapServer keeps
            # its aiohttp runner here; addresses is the runner's public list
            # of sockets owned by its sites, empty before bind.
            runner = getattr(server, "_runner", None)
            addresses = getattr(runner, "addresses", ())
            owned = any(
                address[1] == port
                and address[0].split("%", 1)[0] in target_hosts | {"0.0.0.0", "::"}
                for address in addresses
            )
            if owned:
                try:
                    with socket.create_connection((probe_host, port), timeout=min(0.2, remaining)):
                        return
                except OSError as error:
                    last_error = error
            time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
    except BaseException:
        # SGLang's bootstrap handle uses close(), not shutdown(). Do not
        # replace the startup error if an already failed listener cannot close.
        close = getattr(server, "close", None)
        if close is not None:
            try:
                close()
            except Exception:
                pass
        raise
