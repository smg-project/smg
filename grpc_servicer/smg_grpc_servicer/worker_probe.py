"""Liveness of a multi-node engine's worker ranks: are they still attached to
their leader?

An engine whose tensor-parallel ranks span pods runs its request path and the
gRPC servicer on a leader pod and the other ranks on worker pods started with
the same engine flags and their own node rank. Every rank joins the process
group through a TCP store on the leader, the rendezvous the engine's command
line names, and keeps that connection for its lifetime. When the leader's
engine goes away (the pod restarts, a servicer gives up on a start), the ranks
on the worker pods are not told: their store connection is dead and they wait
on it, the next leader waits for ranks that never join, and a worker probe
that only asks whether the engine process exists keeps the orphans running
until someone deletes the pod.

This module is the probe each engine's ``worker_probe`` module is built on::

    python3 -I -m smg_grpc_servicer.<engine>.worker_probe [--leader HOST] [--port PORT]

It passes (exit 0) while at least one TCP connection from this network
namespace to the leader's store port is established, and fails (exit 1)
otherwise, so the kubelet restarts the worker and fresh ranks join the new
leader within ``periodSeconds x failureThreshold``. It needs only the
standard library; ``-I`` keeps the engine's ``PYTHONPATH`` site
customizations (which may import the engine) out of the probe's
interpreter, and a ``timeoutSeconds`` of 30 or more covers an interpreter
start on a node saturated by the weight stream. Without arguments the leader
is read off the worker's own command line (``/proc/1/cmdline``) through the
engine's :class:`Rendezvous`, the flags that engine's launcher takes it from,
so a manifest states it once; ``--leader`` names the leader by its stable DNS
name instead, re-resolved on every run, so that a recreated leader pod at a
new address fails the probe even while the stale connections to the old
address linger.

The engine modules: :mod:`smg_grpc_servicer.vllm.worker_probe` reads
``--master-addr`` and ``--master-port``; :mod:`smg_grpc_servicer.sglang.worker_probe`
and :mod:`smg_grpc_servicer.tokenspeed.worker_probe` read ``--dist-init-addr
host:port``.
"""

from __future__ import annotations

import argparse
import ipaddress
import shlex
import socket
import struct
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from smg_grpc_servicer.hostport import host_port

#: Where the kernel lists this network namespace's TCP sockets (IPv6 first:
#: an IPv6-only cluster has nothing in the IPv4 table).
DEFAULT_TABLES: tuple[str, ...] = ("/proc/net/tcp6", "/proc/net/tcp")
DEFAULT_CMDLINE = "/proc/1/cmdline"
#: ``st`` of an established connection in ``/proc/net/tcp*``.
TCP_ESTABLISHED = "01"
TCP_STATES = {
    "01": "ESTABLISHED",
    "02": "SYN_SENT",
    "03": "SYN_RECV",
    "04": "FIN_WAIT1",
    "05": "FIN_WAIT2",
    "06": "TIME_WAIT",
    "07": "CLOSE",
    "08": "CLOSE_WAIT",
    "09": "LAST_ACK",
    "0A": "LISTEN",
    "0B": "CLOSING",
}

Address = ipaddress.IPv4Address | ipaddress.IPv6Address


@dataclass(frozen=True)
class Connection:
    """One TCP socket of this network namespace, by its remote end."""

    remote: Address
    port: int
    state: str

    @property
    def established(self) -> bool:
        return self.state == TCP_ESTABLISHED

    @property
    def state_name(self) -> str:
        return TCP_STATES.get(self.state, self.state)


@dataclass(frozen=True)
class Verdict:
    alive: bool
    reason: str


@dataclass(frozen=True)
class Rendezvous:
    """How one engine's command line names the leader's store, for the probe's
    defaults and its messages."""

    #: The engine's probe module, ``python -m`` style, for the CLI's name.
    module: str
    #: The flag the leader's host is read from, as it appears in messages.
    leader_flag: str
    #: The flag the store port is read from (the same flag when one
    #: ``host:port`` carries both).
    port_flag: str
    #: The leader's host and store port as the engine's command line carries
    #: them; ``None`` for a part that is absent or unreadable.
    leader_from_cmdline: Callable[[bytes], tuple[str | None, int | None]]
    #: The engine's own default for the store port, if it has one.
    default_port: int | None = None


def _parse_address(hex_address: str) -> Address:
    """An address as ``/proc/net/tcp*`` prints it: the in-memory words in
    host byte order, as hex."""
    words = [hex_address[i : i + 8] for i in range(0, len(hex_address), 8)]
    packed = b"".join(struct.pack("=I", int(word, 16)) for word in words)
    return normalize(ipaddress.ip_address(packed))


def normalize(address: Address) -> Address:
    """An IPv4 address mapped into IPv6 (``::ffff:a.b.c.d``, how a dual-stack
    socket reports an IPv4 peer) compares equal to the IPv4 address."""
    mapped = getattr(address, "ipv4_mapped", None)
    return mapped if mapped is not None else address


def parse_tcp_table(text: str) -> list[Connection]:
    """The connections listed by one ``/proc/net/tcp`` or ``/proc/net/tcp6``;
    the header and malformed lines are skipped."""
    connections: list[Connection] = []
    for line in text.splitlines()[1:]:
        fields = line.split()
        if len(fields) < 4:
            continue
        remote_hex, _, port_hex = fields[2].partition(":")
        try:
            connection = Connection(
                remote=_parse_address(remote_hex), port=int(port_hex, 16), state=fields[3]
            )
        except (ValueError, struct.error):
            continue
        connections.append(connection)
    return connections


def read_connections(tables: Iterable[str] = DEFAULT_TABLES) -> list[Connection]:
    """Every TCP socket of this network namespace; a table the kernel does
    not provide (no IPv6, no IPv4) is skipped."""
    connections: list[Connection] = []
    for table in tables:
        try:
            with open(table, encoding="ascii") as handle:
                connections.extend(parse_tcp_table(handle.read()))
        except OSError:
            continue
    return connections


def arguments(cmdline: bytes) -> list[str]:
    """The command line's words. An entry that holds a whole shell command
    (``sh -c "<engine> serve ... --flag value"``) is split into its words
    too, so the flags inside it are found."""
    words: list[str] = []
    for part in cmdline.split(b"\0"):
        if not part:
            continue
        text = part.decode("utf-8", "replace")
        if any(c.isspace() for c in text):
            try:
                words.extend(shlex.split(text))
                continue
            except ValueError:
                pass
        words.append(text)
    return words


def flag_values(cmdline: bytes, *flags: str) -> dict[str, str]:
    """The value of each of ``flags`` on the command line (``--flag value``
    or ``--flag=value``, with ``-`` or ``_`` in the flag's name, inside a
    shell wrapper's one argument too), keyed by the dashed spelling; a flag
    that is absent has no entry, the last occurrence of a repeated one wins."""
    words = arguments(cmdline)
    values: dict[str, str] = {}
    for index, argument in enumerate(words):
        name, _, inline = argument.partition("=")
        flag = name.replace("_", "-")
        if flag not in flags:
            continue
        if "=" in argument:
            values[flag] = inline
        elif index + 1 < len(words):
            values[flag] = words[index + 1]
    return values


def split_host_port(value: str) -> tuple[str | None, int | None]:
    """A ``host:port`` as a rendezvous flag carries it (an IPv6 literal in
    brackets, a ``tcp://`` scheme tolerated): the host, and the port when it
    is there and a number."""
    text = value.strip()
    if text.startswith("tcp://"):
        text = text[len("tcp://") :]
    if not text:
        return None, None
    if text.startswith("["):
        host, bracket, rest = text[1:].partition("]")
        port_text = rest[1:] if bracket and rest.startswith(":") else ""
    elif ":" in text:
        host, _, port_text = text.rpartition(":")
    else:
        host, port_text = text, ""
    try:
        port = int(port_text) if port_text else None
    except ValueError:
        port = None
    return host or None, port


def resolve_leader(host: str) -> set[Address]:
    """The addresses ``host`` resolves to now (both families); a literal
    address, with or without brackets, is itself."""
    literal = host.strip("[]")
    try:
        return {normalize(ipaddress.ip_address(literal))}
    except ValueError:
        pass
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        return set()
    addresses: set[Address] = set()
    for _family, _type, _proto, _canonical, sockaddr in infos:
        try:
            addresses.add(normalize(ipaddress.ip_address(sockaddr[0])))
        except ValueError:
            continue
    return addresses


def check(leader_addresses: set[Address], port: int, connections: Iterable[Connection]) -> Verdict:
    """Pass while at least one connection to the leader's store is established."""
    if not leader_addresses:
        return Verdict(False, "the leader does not resolve to any address")
    to_leader = [
        connection
        for connection in connections
        if connection.port == port and connection.remote in leader_addresses
    ]
    established = sum(1 for connection in to_leader if connection.established)
    where = ", ".join(
        host_port(str(address), port) for address in sorted(leader_addresses, key=str)
    )
    if established:
        return Verdict(
            True, f"{established} established connection(s) to the leader's store {where}"
        )
    stale = ", ".join(sorted(connection.state_name for connection in to_leader)) or "none"
    return Verdict(
        False,
        f"no established connection to the leader's store {where} "
        f"(connections to it: {stale}): the ranks are not attached to a live leader",
    )


def probe(
    leader: str,
    port: int,
    *,
    tables: Iterable[str] = DEFAULT_TABLES,
    resolver: Callable[[str], set[Address]] = resolve_leader,
) -> Verdict:
    """The verdict for ``leader``'s store at ``port`` from this namespace's
    live socket tables."""
    return check(resolver(leader), port, read_connections(tables))


def _read_cmdline(path: str) -> bytes:
    try:
        with open(path, "rb") as handle:
            return handle.read()
    except OSError:
        return b""


def main(rendezvous: Rendezvous, argv: Sequence[str] | None = None) -> int:
    """The probe's command line for one engine: the verdict's exit code."""
    port_default = (
        f", else {rendezvous.default_port}" if rendezvous.default_port is not None else ""
    )
    parser = argparse.ArgumentParser(
        prog=f"python -m {rendezvous.module}",
        description=(
            "Liveness probe for a multi-node engine's worker pod: passes while a rank "
            "holds an established connection to the leader's TCP store."
        ),
    )
    parser.add_argument(
        "--leader",
        help=(
            "the leader's host name or address, re-resolved on every run "
            f"(default: {rendezvous.leader_flag} off the engine's command line)"
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        help=(
            f"the leader's TCP store port (default: {rendezvous.port_flag} off the engine's "
            f"command line{port_default})"
        ),
    )
    parser.add_argument(
        "--cmdline",
        default=DEFAULT_CMDLINE,
        help=f"the engine's command line to read the defaults from (default {DEFAULT_CMDLINE})",
    )
    args = parser.parse_args(argv)
    leader, port = args.leader, args.port
    port_source = "--port"
    if leader is None or port is None:
        cmdline_leader, cmdline_port = rendezvous.leader_from_cmdline(_read_cmdline(args.cmdline))
        leader = leader or cmdline_leader
        if port is None:
            port = cmdline_port
            port_source = f"{rendezvous.port_flag} in {args.cmdline}"
        if port is None and rendezvous.default_port is not None:
            port = rendezvous.default_port
            port_source = f"the default; no {rendezvous.port_flag} in {args.cmdline}"
    if leader is None:
        print(
            f"worker probe: no leader: pass --leader, or {rendezvous.leader_flag} in {args.cmdline}",
            file=sys.stderr,
        )
        return 1
    if port is None:
        print(
            f"worker probe: no store port: pass --port, or {rendezvous.port_flag} with a port "
            f"in {args.cmdline}",
            file=sys.stderr,
        )
        return 1
    verdict = probe(leader, port)
    print(
        f"worker probe: {'alive' if verdict.alive else 'NOT alive'}: {verdict.reason} "
        f"(store port {port} from {port_source})"
    )
    return 0 if verdict.alive else 1
