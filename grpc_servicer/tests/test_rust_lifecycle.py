"""Unit tests for the lifecycle helpers the Rust servicer launchers share."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import types

import pytest
from smg_grpc_servicer import rust_lifecycle


def test_a_local_directory_is_the_tokenizer_directory(tmp_path):
    assert rust_lifecycle.resolve_tokenizer_dir(str(tmp_path)) == str(tmp_path)


def test_a_cached_snapshot_counts_only_once_it_holds_a_tokenizer(tmp_path, monkeypatch):
    """The engine downloads the same repo in parallel; a snapshot it has
    started (config present, tokenizer not yet) must not satisfy the local
    lookup, the Hub attempt that follows fetches the tokenizer files."""
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}")
    calls: list[bool] = []

    def fake_snapshot_download(repo, revision=None, allow_patterns=None, local_files_only=False):
        calls.append(local_files_only)
        assert "*.json" in allow_patterns and "*.safetensors" not in str(allow_patterns)
        if not local_files_only:
            (snapshot / "tokenizer.json").write_text("{}")
        return str(snapshot)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=fake_snapshot_download),
    )
    assert rust_lifecycle.resolve_tokenizer_dir("org/model") == str(snapshot)
    assert calls == [True, False]

    # Once the tokenizer is cached the local lookup is enough.
    calls.clear()
    assert rust_lifecycle.resolve_tokenizer_dir("org/model") == str(snapshot)
    assert calls == [True]


def test_an_unresolvable_tokenizer_is_none(tmp_path, monkeypatch):
    def failing(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=failing)
    )
    assert rust_lifecycle.resolve_tokenizer_dir("org/missing") is None


@pytest.mark.parametrize("name", ["tokenizer.json", "tokenizer.model", "x.tiktoken", "vocab.json"])
def test_holds_tokenizer_recognises_each_loader_input(tmp_path, name):
    assert not rust_lifecycle.holds_tokenizer(str(tmp_path))
    (tmp_path / name).write_text("")
    assert rust_lifecycle.holds_tokenizer(str(tmp_path))


class _PopenAsSpawned:
    """A ``multiprocessing``-shaped view over a ``Popen`` for the lifecycle helpers."""

    def __init__(self, popen: subprocess.Popen):
        self._popen = popen

    @property
    def pid(self) -> int:
        return self._popen.pid

    @property
    def exitcode(self) -> int | None:
        return self._popen.poll()

    def join(self, timeout: float | None = None) -> None:
        try:
            self._popen.wait(timeout)
        except subprocess.TimeoutExpired:
            pass

    def terminate(self) -> None:
        self._popen.terminate()

    def kill(self) -> None:
        self._popen.kill()


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    with open(f"/proc/{pid}/stat") as stat:  # a zombie is as dead as we need
        return stat.read().rsplit(")", 1)[1].split()[0] != "Z"


def test_kill_reaches_the_engine_process_group(tmp_path):
    """The headless child is a session leader (``new_session``), so killing the
    engine kills the workers it spawned, not just the launcher."""
    pid_file = tmp_path / "worker.pid"
    launcher = subprocess.Popen(
        ["bash", "-c", f"sleep 300 & echo $! > {pid_file}; wait"],
        start_new_session=True,
    )
    deadline = time.monotonic() + 10
    while not pid_file.exists() or not pid_file.read_text().strip():
        assert time.monotonic() < deadline, "the launcher never spawned its worker"
        time.sleep(0.05)
    worker_pid = int(pid_file.read_text())
    engine = rust_lifecycle.EngineProcess(_PopenAsSpawned(launcher))

    engine.kill()

    assert engine.wait(timeout=10) == -signal.SIGKILL
    deadline = time.monotonic() + 10
    while _alive(worker_pid):
        assert time.monotonic() < deadline, "the worker outlived the engine kill"
        time.sleep(0.05)


def test_kill_of_a_child_outside_its_own_group_is_a_plain_kill(tmp_path):
    """Before the child reaches ``new_session`` it shares this group: only the
    child itself is killed."""
    launcher = subprocess.Popen(["sleep", "300"])
    engine = rust_lifecycle.EngineProcess(_PopenAsSpawned(launcher))
    engine.kill()
    assert engine.wait(timeout=10) == -signal.SIGKILL
