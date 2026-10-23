"""The TokenSpeed servicer's worker probe: the leader read off the launcher's
``--dist-init-addr`` (the shared verdict is tested in ``test_worker_probe.py``)."""

from __future__ import annotations

from smg_grpc_servicer.tokenspeed import worker_probe


def _cmdline(*arguments: str) -> bytes:
    """A command line as ``/proc/<pid>/cmdline`` stores it: NUL-terminated arguments."""
    return b"".join(argument.encode() + b"\x00" for argument in arguments)


def test_leader_from_cmdline_reads_dist_init_addr_in_every_spelling():
    served = _cmdline("python", "-m", "smg_grpc_servicer.tokenspeed", "--model", "m")
    served += _cmdline("--host", "0.0.0.0", "--port", "50051", "--nnodes", "2", "--node-rank", "1")
    served += _cmdline("--dist-init-addr", "10.0.0.7:20000")
    # The engine's own --port is its gRPC port; the store's comes from the rendezvous flag.
    assert worker_probe.leader_from_cmdline(served) == ("10.0.0.7", 20000)
    joined = _cmdline("ts", "serve", "--model", "m", "--dist-init-addr=head.svc:20000")
    assert worker_probe.leader_from_cmdline(joined) == ("head.svc", 20000)
    underscored = _cmdline("ts", "serve", "--dist_init_addr", "head:20001")
    assert worker_probe.leader_from_cmdline(underscored) == ("head", 20001)
    # A shell wrapper carries the whole launcher command in one argument.
    wrapped = _cmdline(
        "/bin/sh",
        "-c",
        'exec python -m smg_grpc_servicer.tokenspeed --model "$M" --port 50051 '
        "--nnodes 2 --node-rank 1 --dist-init-addr head:20000 --servicer-impl rust",
    )
    assert worker_probe.leader_from_cmdline(wrapped) == ("head", 20000)
    # No flag (derived from the launcher environment, or a single-node engine),
    # no port, a port that is not a number.
    assert worker_probe.leader_from_cmdline(_cmdline("ts", "serve", "--model", "m")) == (None, None)
    assert worker_probe.leader_from_cmdline(_cmdline("x", "--dist-init-addr", "head")) == (
        "head",
        None,
    )
    assert worker_probe.leader_from_cmdline(_cmdline("x", "--dist-init-addr", "head:many")) == (
        "head",
        None,
    )


def test_main_takes_the_leader_and_port_from_the_engines_command_line(tmp_path, capsys):
    cmdline = tmp_path / "cmdline"
    cmdline.write_bytes(
        _cmdline("python", "-m", "smg_grpc_servicer.tokenspeed", "--model", "m", "--port", "50051")
        + _cmdline("--nnodes", "2", "--node-rank", "1", "--dist-init-addr", "127.0.0.1:1")
    )
    # Port 1 on the loopback: nothing is connected to it, so the probe fails,
    # naming the leader and the store port it read off the command line (the
    # engine's own --port is not the store's).
    assert worker_probe.main(["--cmdline", str(cmdline)]) == 1
    out = capsys.readouterr().out
    assert "127.0.0.1:1" in out and "store port 1 from --dist-init-addr" in out
    # --leader keeps the command line's port; --port keeps its leader.
    assert worker_probe.main(["--cmdline", str(cmdline), "--leader", "127.0.0.2"]) == 1
    assert "127.0.0.2:1" in capsys.readouterr().out
    assert worker_probe.main(["--cmdline", str(cmdline), "--port", "2"]) == 1
    out = capsys.readouterr().out
    assert "127.0.0.1:2" in out and "store port 2 from --port" in out
    # A flag without a port: the engine has no default, so the probe says what it needs.
    (tmp_path / "noport").write_bytes(_cmdline("ts", "serve", "--dist-init-addr", "127.0.0.1"))
    assert worker_probe.main(["--cmdline", str(tmp_path / "noport")]) == 1
    assert "--port" in capsys.readouterr().err
    # No leader anywhere either: a launch that derived its rendezvous from the
    # environment gives the manifest --leader and --port.
    (tmp_path / "bare").write_bytes(_cmdline("ts", "serve", "--model", "m"))
    assert worker_probe.main(["--cmdline", str(tmp_path / "bare")]) == 1
    assert "--leader" in capsys.readouterr().err
