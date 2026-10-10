"""The multi-node worker pod's liveness probe: attached to a live leader, or
not; the same contract for every engine the servicer launches."""

from __future__ import annotations

import importlib
import ipaddress
import os
import re
import socket
import struct
from pathlib import Path

import pytest
from smg_grpc_servicer import worker_probe
from smg_grpc_servicer.sglang import worker_probe as sglang_probe
from smg_grpc_servicer.tokenspeed import worker_probe as tokenspeed_probe
from smg_grpc_servicer.vllm import worker_probe as vllm_probe

ENGINES = {"sglang": sglang_probe, "tokenspeed": tokenspeed_probe, "vllm": vllm_probe}
README = Path(__file__).resolve().parents[1] / "README.md"
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


def _connections(*rows: tuple[str, int, str]) -> list[worker_probe.Connection]:
    return worker_probe.parse_tcp_table(HEADER + "".join(_row(*row) for row in rows))


def _cmdline(*arguments: str) -> bytes:
    """A command line as ``/proc/<pid>/cmdline`` stores it: NUL-terminated arguments."""
    return b"".join(argument.encode() + b"\x00" for argument in arguments)


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


def test_flag_values_reads_both_spellings_and_shell_wrappers():
    spaced = _cmdline("engine", "serve", "m", "--leader-addr", "fd00::1", "--store-port", "5000")
    assert worker_probe.flag_values(spaced, "--leader-addr", "--store-port") == {
        "--leader-addr": "fd00::1",
        "--store-port": "5000",
    }
    # The ``=`` and underscore spellings, the last of a repeated flag, a
    # flag whose value is missing, and flags that are not asked for.
    mixed = _cmdline("x", "--leader_addr=a", "--leader-addr", "b", "--other", "c", "--store-port")
    assert worker_probe.flag_values(mixed, "--leader-addr", "--store-port") == {
        "--leader-addr": "b"
    }
    # A shell wrapper carries the whole engine command in one argument.
    wrapped = _cmdline(
        "/bin/sh", "-c", 'exec engine serve "$M" --leader-addr fd00::3 --store-port=1'
    )
    assert worker_probe.flag_values(wrapped, "--leader-addr", "--store-port") == {
        "--leader-addr": "fd00::3",
        "--store-port": "1",
    }
    assert worker_probe.flag_values(b"", "--leader-addr") == {}


def test_split_host_port_takes_brackets_schemes_and_missing_ports():
    assert worker_probe.split_host_port("10.0.0.7:5000") == ("10.0.0.7", 5000)
    assert worker_probe.split_host_port("leader.svc:5000") == ("leader.svc", 5000)
    assert worker_probe.split_host_port("[fd00::1]:5000") == ("fd00::1", 5000)
    assert worker_probe.split_host_port("tcp://leader:5000") == ("leader", 5000)
    # No port, or one that is not a number: the host still names the leader.
    assert worker_probe.split_host_port("leader") == ("leader", None)
    assert worker_probe.split_host_port("[fd00::1]") == ("fd00::1", None)
    assert worker_probe.split_host_port("leader:many") == ("leader", None)
    assert worker_probe.split_host_port(":5000") == (None, 5000)
    assert worker_probe.split_host_port("") == (None, None)


def test_check_passes_only_with_an_established_connection_to_the_leader_store():
    leader = {ipaddress.ip_address("fd00::1")}
    alive = worker_probe.check(leader, PORT, _connections(("fd00::1", PORT, "01")))
    assert alive.alive and "1 established" in alive.reason
    assert f"[fd00::1]:{PORT}" in alive.reason  # host and port readable in the probe's event text

    # The leader's engine died: the kernel moved the ranks' sockets to CLOSE_WAIT.
    orphaned = worker_probe.check(leader, PORT, _connections(("fd00::1", PORT, "08")))
    assert not orphaned.alive and "CLOSE_WAIT" in orphaned.reason

    # Connections to the leader on other ports (the collective's own, the
    # message queues) do not count.
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


@pytest.mark.parametrize("engine", sorted(ENGINES))
def test_every_engine_probe_is_the_shared_one_under_its_own_name(engine, capsys):
    module = ENGINES[engine]
    assert module.RENDEZVOUS.module == f"smg_grpc_servicer.{engine}.worker_probe"
    assert importlib.import_module(module.RENDEZVOUS.module) is module
    with pytest.raises(SystemExit) as stopped:
        module.main(["--help"])
    assert stopped.value.code == 0
    usage = capsys.readouterr().out
    assert f"python -m smg_grpc_servicer.{engine}.worker_probe" in usage
    assert module.RENDEZVOUS.leader_flag in usage and module.RENDEZVOUS.port_flag in usage
    # Explicit --leader and --port need no command line at all.
    assert module.main(["--leader", "127.0.0.1", "--port", "1", "--cmdline", "/nonexistent"]) == 1
    assert "127.0.0.1:1" in capsys.readouterr().out


@pytest.mark.parametrize("engine", sorted(ENGINES))
@pytest.mark.skipif(not os.path.exists("/proc/net/tcp"), reason="needs the kernel's socket tables")
def test_the_live_kernel_tables_show_a_store_connection_until_the_leader_closes_it(engine):
    """A stand-in leader store on the loopback: the probe passes while the
    rank's connection is established, and fails once the leader side closed
    it (the rank's socket sits in CLOSE_WAIT, as the orphaned ranks' did)."""
    main = ENGINES[engine].main
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as store:
        store.bind(("127.0.0.1", 0))
        store.listen(1)
        port = store.getsockname()[1]
        with socket.create_connection(("127.0.0.1", port)) as rank:
            accepted, _ = store.accept()
            assert worker_probe.probe("127.0.0.1", port).alive
            assert main(["--leader", "127.0.0.1", "--port", str(port)]) == 0

            accepted.close()
            store.close()
            # The FIN moves the rank's socket out of ESTABLISHED.
            assert worker_probe.probe("127.0.0.1", port).alive is False
            assert main(["--leader", "127.0.0.1", "--port", str(port)]) == 1
            rank.close()


PROBE_COMMAND = re.compile(
    r"python3 -I -m smg_grpc_servicer\.(\w+)\.worker_probe --leader \"\$LEADER_HOST\""
)


def _manifests(readme: str) -> dict[str, dict[str, dict[str, str]]]:
    """The README's worker-pod manifest snippet of each engine: its probes
    (``startupProbe``, ``livenessProbe``) and their fields."""
    manifests: dict[str, dict[str, dict[str, str]]] = {}
    for block in re.findall(r"```yaml\n(.*?)```", readme, flags=re.S):
        engines = set(PROBE_COMMAND.findall(block))
        if not engines:
            continue
        (engine,) = engines
        probes: dict[str, dict[str, str]] = {}
        fields: dict[str, str] = {}
        for line in block.splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if not line.startswith(" "):
                fields = probes.setdefault(line.strip().rstrip(":"), {})
                continue
            key, _, value = line.strip().partition(":")
            fields[key] = value.strip()
        assert engine not in manifests, f"two worker-pod manifests for {engine}"
        manifests[engine] = probes
    return manifests


def test_the_readme_gives_every_engine_the_same_worker_pod_manifest():
    """The manifest a worker pod of each engine runs: ``python3 -I`` and the
    engine's own probe module in both probes, a ``timeoutSeconds`` of 30 or
    more, a start budget of at least an hour, three missed periods for the
    liveness verdict."""
    manifests = _manifests(README.read_text(encoding="utf-8"))
    assert sorted(manifests) == sorted(ENGINES)
    for engine, probes in manifests.items():
        assert sorted(probes) == ["livenessProbe", "startupProbe"], engine
        command = f'python3 -I -m smg_grpc_servicer.{engine}.worker_probe --leader "$LEADER_HOST"'
        for kind, fields in probes.items():
            assert command in fields["exec"], (engine, kind)
            assert int(fields["timeoutSeconds"]) >= 30, (engine, kind)
            assert int(fields["periodSeconds"]) > 0, (engine, kind)
        startup, liveness = probes["startupProbe"], probes["livenessProbe"]
        assert int(startup["periodSeconds"]) * int(startup["failureThreshold"]) >= 3600, engine
        assert int(liveness["failureThreshold"]) == 3, engine
