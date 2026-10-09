"""``host:port`` strings for grpc-python, with IPv6 literals in brackets.

grpc-core splits a ``host:port`` target at its last colon, so a bare IPv6
literal does not survive the join: ``::`` becomes ``:::50051`` (a bind that
fails with "Misformatted domain name") and ``::1`` becomes ``::1:50051`` (a
dial that never connects). Every Python servicer joins its listen address and
its warm-up target through these helpers, so ``--host ::`` and ``--host [::]``
both work, as they do for the gateway.
"""

from __future__ import annotations

_LOOPBACK = {"0.0.0.0": "127.0.0.1", "::": "::1"}


def bracket_if_ipv6(host: str) -> str:
    """Return ``host`` ready for a ``host:port`` join: an IPv6 literal in
    brackets; IPv4 addresses, hostnames and already-bracketed literals as
    they are."""
    if ":" in host and not host.startswith("["):
        return f"[{host}]"
    return host


def host_port(host: str, port: int) -> str:
    """``host:port`` with an IPv6 ``host`` bracketed."""
    return f"{bracket_if_ipv6(host)}:{port}"


def loopback_target(host: str, port: int) -> str:
    """The target to dial a server bound to ``host`` from the same machine:
    a wildcard bind is not routable as a destination, so it maps to the
    loopback address of its family; any other host is dialed as given."""
    bare = host[1:-1] if host.startswith("[") and host.endswith("]") else host
    return host_port(_LOOPBACK.get(bare, bare), port)
