"""The multi-node worker pod's liveness probe: attached to a live leader, or not."""

from __future__ import annotations

import ipaddress
import os
import socket
import struct

import pytest
from smg_grpc_servicer.vllm import worker_probe

HEADER = (
    "  sl  local_address                         remote_address                        st "
    "tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n"
)
PORT = 29501


def _hex(address: str) -> str:
    """An address as ``/proc/net/tcp*`` prints it on this host."""
    packed = ipaddress.ip_address(address).packed
    words = struct.unpack(f"={len(packed) // 4}I", packed)
    return "".join(f"{word:08X}" for word in words)


def _row(remote: str, port: int, state: str, local: str = "::", local_port: int = 40000) -> str:
    return (
        f"   0: {_hex(local)}:{local_port:04X} {_hex(remote)}:{port:04X} {state} "
        "00000000:00000000 00:00000000 00000000  1000        0 12345 1 0000000000000000 20 4 30 10 -1\n"
    )


def test_parses_ipv6_and_ipv4_tables_by_remote_end_and_state():
    table6 = HEADER + _row("fd00::1", PORT, "01") + _row("fd00::1", PORT, "08")
    table4 = HEADER + _row("10.0.0.7", PORT, "01", local="0.0.0.0")
    v6 = worker_probe.parse_tcp_table(table6)
    v4 = worker_probe.parse_tcp_table(table4)
    assert [(str(c.remote), c.port, c.state_name) for c in v6] == [
        ("fd00::1", PORT, "ESTABLISHED"),
        ("fd00::1", PORT, "CLOSE_WAIT"),
    ]
    assert [(str(c.remote), c.port, c.established) for c in v4] == [("10.0.0.7", PORT, True)]
    # A header-only table and a malformed line are nothing.
    assert worker_probe.parse_tcp_table(HEADER) == []
    assert worker_probe.parse_tcp_table(HEADER + "garbage\n") == []


def test_ipv4_peers_of_a_dual_stack_socket_match_an_ipv4_leader():
    table6 = HEADER + _row("::ffff:10.0.0.7", PORT, "01")
    (connection,) = worker_probe.parse_tcp_table(table6)
    assert connection.remote == ipaddress.ip_address("10.0.0.7")
    assert worker_probe.resolve_leader("10.0.0.7") == {ipaddress.ip_address("10.0.0.7")}
    assert worker_probe.resolve_leader("[fd00::1]") == {ipaddress.ip_address("fd00::1")}


def _cmdline(*arguments: str) -> bytes:
    """A command line as ``/proc/<pid>/cmdline`` stores it: NUL-terminated arguments."""
    return b"".join(argument.encode() + b"\x00" for argument in arguments)


def test_master_from_cmdline_reads_both_spellings():
    spaced = _cmdline("vllm", "serve", "m", "--headless", "--master-addr", "fd00::1")
    spaced += _cmdline("--master-port", "29501")
    joined = _cmdline("python3", "-m", "x", "--master-addr=leader.svc", "--master-port=29502")
    assert worker_probe.master_from_cmdline(spaced) == ("fd00::1", 29501)
    assert worker_probe.master_from_cmdline(joined) == ("leader.svc", 29502)
    assert worker_probe.master_from_cmdline(_cmdline("vllm", "serve", "m")) == (None, None)
    assert worker_probe.master_from_cmdline(_cmdline("x", "--master-port", "many")) == (None, None)
    # The underscore spellings the engine's parser also takes.
    underscored = _cmdline("vllm", "serve", "m", "--master_addr", "fd00::2", "--master_port=29503")
    assert worker_probe.master_from_cmdline(underscored) == ("fd00::2", 29503)
    # A shell wrapper carries the whole engine command in one argument.
    wrapped = _cmdline(
        "/bin/sh", "-c", 'exec vllm serve "$M" --headless --master-addr fd00::3 --master-port 29600'
    )
    assert worker_probe.master_from_cmdline(wrapped) == ("fd00::3", 29600)


def _connections(*rows: tuple[str, int, str]) -> list[worker_probe.Connection]:
    return worker_probe.parse_tcp_table(HEADER + "".join(_row(*row) for row in rows))


def test_check_passes_only_with_an_established_connection_to_the_leader_store():
    leader = {ipaddress.ip_address("fd00::1")}
    alive = worker_probe.check(leader, PORT, _connections(("fd00::1", PORT, "01")))
    assert alive.alive and "1 established" in alive.reason
    assert f"[fd00::1]:{PORT}" in alive.reason  # host and port readable in the probe's event text

    # The leader's engine died: the kernel moved the ranks' sockets to CLOSE_WAIT.
    orphaned = worker_probe.check(leader, PORT, _connections(("fd00::1", PORT, "08")))
    assert not orphaned.alive and "CLOSE_WAIT" in orphaned.reason

    # Connections to the leader on other ports (NCCL, the message queues) do not count.
    other_port = worker_probe.check(leader, PORT, _connections(("fd00::1", 41234, "01")))
    assert not other_port.alive

    # Neither do connections to other peers on the store port.
    other_peer = worker_probe.check(leader, PORT, _connections(("fd00::2", PORT, "01")))
    assert not other_peer.alive and "none" in other_peer.reason


def test_check_fails_when_the_leader_moved_or_vanished():
    # A recreated leader pod at a new address: the stale connection to the old
    # one is still established (no FIN reached this pod), and does not count.
    stale = _connections(("fd00::1", PORT, "01"))
    moved = worker_probe.check({ipaddress.ip_address("fd00::9")}, PORT, stale)
    assert not moved.alive
    vanished = worker_probe.check(set(), PORT, stale)
    assert not vanished.alive and "does not resolve" in vanished.reason


def test_probe_resolves_the_leader_on_every_run(tmp_path):
    table = tmp_path / "tcp6"
    table.write_text(HEADER + _row("fd00::1", PORT, "01"))
    resolved = {"leader": {ipaddress.ip_address("fd00::1")}}
    verdict = worker_probe.probe(
        "leader", PORT, tables=(str(table),), resolver=lambda host: resolved[host]
    )
    assert verdict.alive
    resolved["leader"] = {ipaddress.ip_address("fd00::9")}
    assert not worker_probe.probe(
        "leader", PORT, tables=(str(table),), resolver=lambda host: resolved[host]
    ).alive
    # A missing table (no IPv4 on an IPv6-only host, or the reverse) is skipped.
    assert worker_probe.read_connections((str(tmp_path / "absent"), str(table)))


@pytest.mark.skipif(not os.path.exists("/proc/net/tcp"), reason="needs the kernel's socket tables")
def test_the_live_kernel_tables_show_a_store_connection_until_the_leader_closes_it():
    """A stand-in leader store on the loopback: the probe passes while the
    rank's connection is established, and fails once the leader side closed
    it (the rank's socket sits in CLOSE_WAIT, as the orphaned ranks' did)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as store:
        store.bind(("127.0.0.1", 0))
        store.listen(1)
        port = store.getsockname()[1]
        with socket.create_connection(("127.0.0.1", port)) as rank:
            accepted, _ = store.accept()
            assert worker_probe.probe("127.0.0.1", port).alive
            assert worker_probe.main(["--leader", "127.0.0.1", "--port", str(port)]) == 0

            accepted.close()
            store.close()
            # The FIN moves the rank's socket out of ESTABLISHED.
            assert worker_probe.probe("127.0.0.1", port).alive is False
            assert worker_probe.main(["--leader", "127.0.0.1", "--port", str(port)]) == 1
            rank.close()


def test_main_takes_the_leader_from_the_engines_command_line(tmp_path, capsys):
    cmdline = tmp_path / "cmdline"
    cmdline.write_bytes(
        _cmdline("vllm", "serve", "m", "--headless", "--master-addr", "127.0.0.1")
        + _cmdline("--master-port", "1")
    )
    # Port 1 on the loopback: nothing is connected to it, so the probe fails,
    # naming the leader it read off the command line.
    assert worker_probe.main(["--cmdline", str(cmdline)]) == 1
    out = capsys.readouterr().out
    assert "127.0.0.1:1" in out and "store port 1 from --master-port" in out
    # Without the flag the default port is used, and the verdict says so.
    (tmp_path / "noport").write_bytes(_cmdline("vllm", "serve", "m", "--master-addr", "127.0.0.1"))
    assert worker_probe.main(["--cmdline", str(tmp_path / "noport")]) == 1
    assert "from the default; no --master-port" in capsys.readouterr().out
    # No leader anywhere: the probe says what it needs.
    (tmp_path / "bare").write_bytes(_cmdline("vllm", "serve", "m"))
    assert worker_probe.main(["--cmdline", str(tmp_path / "bare")]) == 1
    assert "--leader" in capsys.readouterr().err
