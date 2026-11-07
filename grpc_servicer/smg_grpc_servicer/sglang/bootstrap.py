"""Wait for SGLang's background bootstrap listener before starting its ranks.

The backend constructor starts a daemon thread; returning a handle does not
mean it bound its port. Prefill ranks register only during their startup, so
launching them against a dead listener can leave a healthy gRPC worker whose
KV rendezvous never works.
"""

from __future__ import annotations

import json
import logging
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any

logger = logging.getLogger(__name__)
_RESOLVER_MAX_BYTES = 65536
_RESOLVER_REAP_SECONDS = 1.0
_RESOLVER_CODE = """
import json
import socket
import sys

try:
    answers = socket.getaddrinfo(sys.argv[1], int(sys.argv[2]), type=socket.SOCK_STREAM)
    addresses = []
    for family, kind, protocol, name, address in answers:
        item = [family, list(address)]
        if family in (socket.AF_INET, socket.AF_INET6) and item not in addresses:
            addresses.append(item)
        if len(addresses) > 64:
            raise OSError(socket.EAI_FAIL, "too many bootstrap addresses")
    result = {"addresses": addresses}
except (OSError, UnicodeError) as error:
    code = getattr(error, "errno", None)
    code = code if isinstance(code, int) else socket.EAI_FAIL
    message = getattr(error, "strerror", None) or str(error)
    result = {"error": [code, message[:1024]]}
frame = json.dumps(result, separators=(",", ":")).encode("ascii")
if len(frame) > 65536:
    raise SystemExit(1)
sys.stdout.buffer.write(frame)
"""


def _numeric_address(host: str, port: int) -> tuple[int, tuple] | None:
    """Build numeric socket addresses without invoking a hostname resolver."""
    address, separator, scope = host.partition("%")
    for family in (socket.AF_INET, socket.AF_INET6):
        try:
            packed = socket.inet_pton(family, address)
        except OSError:
            continue
        numeric = socket.inet_ntop(family, packed)
        if family == socket.AF_INET:
            if separator:
                raise socket.gaierror(socket.EAI_NONAME, "IPv4 address has an IPv6 scope")
            return family, (numeric, port)
        scope_id = 0
        if separator:
            scope_id = int(scope) if scope.isdecimal() else socket.if_nametoindex(scope)
            if not 0 <= scope_id <= 0xFFFFFFFF:
                raise socket.gaierror(socket.EAI_NONAME, "invalid IPv6 scope")
        return family, (numeric, port, 0, scope_id)
    return None


def _read_addresses(frame: bytes, port: int) -> list[tuple[int, tuple]]:
    """Accept only a bounded resolver frame containing numeric TCP addresses."""
    try:
        if not frame or len(frame) > _RESOLVER_MAX_BYTES:
            raise ValueError("empty or oversized frame")
        result = json.loads(frame)
        if not isinstance(result, dict):
            raise ValueError("expected an object")
        if set(result) == {"error"}:
            error = result["error"]
            if (
                not isinstance(error, list)
                or len(error) != 2
                or type(error[0]) is not int
                or not isinstance(error[1], str)
                or len(error[1]) > 1024
            ):
                raise ValueError("invalid resolver error")
            raise socket.gaierror(error[0], error[1])
        if set(result) != {"addresses"} or not isinstance(result["addresses"], list):
            raise ValueError("expected addresses")
        if not 1 <= len(result["addresses"]) <= 64:
            raise ValueError("invalid address count")
        addresses = []
        for item in result["addresses"]:
            if not isinstance(item, list) or len(item) != 2:
                raise ValueError("invalid address entry")
            family, address = item
            if type(family) is not int or family not in (socket.AF_INET, socket.AF_INET6):
                raise ValueError("invalid address family")
            size = 2 if family == socket.AF_INET else 4
            if not isinstance(address, list) or len(address) != size:
                raise ValueError("invalid socket address")
            if not isinstance(address[0], str) or type(address[1]) is not int:
                raise ValueError("invalid host or port")
            if address[1] != port:
                raise ValueError("unexpected port")
            socket.inet_pton(family, address[0])
            if any(type(value) is not int or not 0 <= value <= 0xFFFFFFFF for value in address[2:]):
                raise ValueError("invalid IPv6 flow or scope")
            if family == socket.AF_INET6 and address[2] > 0xFFFFF:
                raise ValueError("invalid IPv6 flow information")
            addresses.append((family, tuple(address)))
        return addresses
    except (ValueError, TypeError, OSError) as error:
        if isinstance(error, socket.gaierror):
            raise
        raise RuntimeError(f"Invalid bootstrap resolver response: {error}") from error


def _resolve_addresses(host: str, port: int, *, deadline: float) -> list[tuple[int, tuple]]:
    """Resolve within the readiness deadline; a failed child gets at most 1s to reap."""
    if time.monotonic() >= deadline:
        raise RuntimeError(f"SGLang prefill bootstrap {host}:{port} resolution deadline expired")
    numeric = _numeric_address(host, port)
    if numeric is not None:
        return [numeric]
    with tempfile.TemporaryFile() as output:
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"SGLang prefill bootstrap {host}:{port} resolution deadline expired"
            )
        process = subprocess.Popen(
            [sys.executable, "-I", "-S", "-c", _RESOLVER_CODE, host, str(port)],
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        )
        try:
            try:
                process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired as error:
                raise RuntimeError(
                    f"SGLang prefill bootstrap {host}:{port} resolution deadline expired"
                ) from error
            if process.returncode != 0:
                raise RuntimeError(f"Bootstrap resolver exited with code {process.returncode}")
            output.seek(0)
            return _read_addresses(output.read(_RESOLVER_MAX_BYTES + 1), port)
        except BaseException:
            try:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=_RESOLVER_REAP_SECONDS)
            except Exception as cleanup_error:
                # Keep the startup failure primary; never turn failed cleanup into readiness.
                logger.warning("Could not reap the owned bootstrap resolver: %s", cleanup_error)
            raise


def wait_for_bootstrap_listener(server: Any, host: str, port: int, *, timeout: float = 5.0) -> None:
    """Require the declared listener, and release a failed backend's handle.

    This precedes scheduler/model startup and does not wait for the ranks to
    register. Keep the returned handle alive after this call, as before.
    The backend-owned runner is present on CommonKVBootstrapServer in the
    minimum supported SGLang 0.5.21 and current release 0.5.22. A backend
    without that ownership signal fails closed, rather than trusting a
    foreign listener at the expected port.

    The same readiness deadline covers DNS and all connection attempts. A
    timed-out resolver is killed and gets a separate, at-most-one-second reap
    allowance; this cleanup allowance cannot qualify a late listener.
    """
    endpoint = f"{host}:{port}"
    deadline = time.monotonic() + timeout
    try:
        if server is None:
            raise RuntimeError(f"SGLang prefill bootstrap {endpoint} returned no server")
        actual_port = getattr(server, "port", port)
        if actual_port != port:
            raise RuntimeError(
                f"SGLang prefill bootstrap {endpoint} started with port {actual_port}"
            )
        probe_host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(host, host)
        targets = _resolve_addresses(probe_host, port, deadline=deadline)
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
            for family, target in targets:
                wildcard = "0.0.0.0" if family == socket.AF_INET else "::"
                owned = any(
                    address[1] == port
                    and address[0].split("%", 1)[0] in {target[0], wildcard}
                    and (
                        family == socket.AF_INET
                        or address[0] == wildcard
                        or target[3] == 0
                        or (len(address) == 4 and address[3] == target[3])
                    )
                    for address in addresses
                )
                remaining = deadline - time.monotonic()
                if not owned or remaining <= 0:
                    continue
                try:
                    with socket.socket(family, socket.SOCK_STREAM) as connection:
                        connection.settimeout(min(0.2, remaining))
                        connection.connect(target)
                    if time.monotonic() < deadline:
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
