"""The vLLM servicer's worker probe: the leader read off ``vllm serve``'s own
command line (the shared verdict is tested in ``test_worker_probe.py``)."""

from __future__ import annotations

from smg_grpc_servicer.vllm import worker_probe


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
