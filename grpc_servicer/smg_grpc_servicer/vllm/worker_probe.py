"""Liveness of a multi-node engine's worker ranks: are they still attached to
their leader?

An engine whose tensor-parallel ranks span pods runs its EngineCore and the
gRPC servicer on a leader pod and the other ranks on worker pods started with
``vllm serve --headless --node-rank <k> --master-addr <leader>``. Every rank
joins the process group through the leader's TCP store (``--master-port``)
and keeps that connection for its lifetime. When the leader's engine goes
away (the pod restarts, a servicer gives up on a start), the ranks on the
worker pods are not told: their store connection is dead and they wait on
it, the next leader waits for ranks that never join, and a worker probe that
only asks whether the engine process exists (``grep vllm /proc/1/cmdline``)
keeps the orphans running until someone deletes the pod.

This module is the worker pod's probe::

    python3 -I -m smg_grpc_servicer.vllm.worker_probe [--leader HOST] [--port PORT]

It passes (exit 0) while at least one TCP connection from this network
namespace to the leader's store port is established, and fails (exit 1)
otherwise, so the kubelet restarts the worker and fresh ranks join the new
leader within ``periodSeconds x failureThreshold``. It needs only the
standard library; ``-I`` keeps the engine's ``PYTHONPATH`` site
customizations (which may import the engine) out of the probe's
interpreter, and a ``timeoutSeconds`` of 30 or more covers an interpreter
start on a node saturated by the weight stream. Without arguments the
leader is read off the worker's own command line (``--master-addr`` and
``--master-port`` in ``/proc/1/cmdline``), so a manifest states it once;
``--leader`` names the leader by its stable DNS name instead, re-resolved on
every run, so that a recreated leader pod at a new address fails the probe
even while the stale connections to the old address linger.
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

#: vLLM's ``--master-port`` default.
DEFAULT_MASTER_PORT = 29501
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


def _arguments(cmdline: bytes) -> list[str]:
    """The command line's words. An entry that holds a whole shell command
    (``sh -c "vllm serve ... --master-port 29600"``) is split into its words
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


def master_from_cmdline(cmdline: bytes) -> tuple[str | None, int | None]:
    """``--master-addr`` and ``--master-port`` as the engine's command line
    carries them (``--flag value`` or ``--flag=value``, with ``-`` or ``_``
    in the flag's name, inside a shell wrapper's one argument too); ``None``
    for a flag that is absent or unreadable."""
    arguments = _arguments(cmdline)
    values: dict[str, str] = {}
    for index, argument in enumerate(arguments):
        name, _, inline = argument.partition("=")
        flag = name.replace("_", "-")
        if flag not in ("--master-addr", "--master-port"):
            continue
        if inline or "=" in argument:
            values[flag] = inline
        elif index + 1 < len(arguments):
            values[flag] = arguments[index + 1]
    address = values.get("--master-addr") or None
    try:
        port = int(values["--master-port"]) if "--master-port" in values else None
    except ValueError:
        port = None
    return address, port


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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m smg_grpc_servicer.vllm.worker_probe",
        description=(
            "Liveness probe for a multi-node engine's worker pod: passes while a rank "
            "holds an established connection to the leader's TCP store."
        ),
    )
    parser.add_argument(
        "--leader",
        help=(
            "the leader's host name or address, re-resolved on every run "
            "(default: --master-addr off the engine's command line)"
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        help=(
            "the leader's TCP store port (default: --master-port off the engine's "
            f"command line, else {DEFAULT_MASTER_PORT})"
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
        cmdline_leader, cmdline_port = master_from_cmdline(_read_cmdline(args.cmdline))
        leader = leader or cmdline_leader
        if port is None:
            port = cmdline_port
            port_source = f"--master-port in {args.cmdline}"
        if port is None:
            port = DEFAULT_MASTER_PORT
            port_source = f"the default; no --master-port in {args.cmdline}"
    if leader is None:
        print(
            f"worker probe: no leader: pass --leader, or --master-addr in {args.cmdline}",
            file=sys.stderr,
        )
        return 1
    verdict = probe(leader, port)
    print(
        f"worker probe: {'alive' if verdict.alive else 'NOT alive'}: {verdict.reason} "
        f"(store port {port} from {port_source})"
    )
    return 0 if verdict.alive else 1


if __name__ == "__main__":
    raise SystemExit(main())
