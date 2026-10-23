"""A worker relaunched on its port keeps the previous engine's log beside the new one."""

from __future__ import annotations

import os
from pathlib import Path

from infra import worker as worker_module
from infra.constants import ENV_SHOW_WORKER_LOGS, ConnectionMode
from infra.worker import Worker, rotate_previous_log


class _FakePopen:
    """Stands in for the engine: writes one line to the log it was handed."""

    def __init__(self, cmd, env, stdout, stderr, start_new_session):  # noqa: ARG002
        self.pid = 4242
        stdout.write(f"engine {cmd[-1]} starting\n")
        stdout.flush()

    def poll(self):
        return None


def _worker(log_dir: Path) -> Worker:
    return Worker(
        model_id="m",
        engine="vllm",
        port=50112,
        gpu_ids=[0],
        mode=ConnectionMode.GRPC,
        log_dir=str(log_dir),
    )


def _spawn(worker: Worker, run: str) -> None:
    worker.process = worker._spawn_process(["engine", run], {})


def _end_run(worker: Worker, last_words: str) -> None:
    """What the engine wrote before it died, then the launcher's close (as stop() does)."""
    assert worker._log_file is not None
    worker._log_file.write(last_words + "\n")
    worker._log_file.close()
    worker._log_file = None


def test_a_relaunch_on_the_same_port_keeps_the_previous_log(monkeypatch, tmp_path):
    monkeypatch.setattr(worker_module.subprocess, "Popen", _FakePopen)
    monkeypatch.setenv(ENV_SHOW_WORKER_LOGS, "0")
    worker = _worker(tmp_path)

    _spawn(worker, "one")
    _end_run(worker, "EngineCore died: the trace")
    _spawn(worker, "two")
    _end_run(worker, "clean stop")
    _spawn(worker, "three")

    live = Path(worker.log_path)
    assert live == tmp_path / "worker-m_vllm_grpc_50112.log"
    assert live.read_text() == "engine three starting\n"
    assert (tmp_path / "worker-m_vllm_grpc_50112.log.1").read_text() == (
        "engine one starting\nEngineCore died: the trace\n"
    )
    assert (tmp_path / "worker-m_vllm_grpc_50112.log.2").read_text() == (
        "engine two starting\nclean stop\n"
    )
    # The transfer assertions glob the live logs only; the rotated ones stay
    # out of their way and inside the uploaded directory.
    assert sorted(p.name for p in tmp_path.glob("worker-*.log")) == [live.name]


def test_the_temp_fallback_without_a_log_dir_is_overwritten_not_rotated(monkeypatch, tmp_path):
    """The shared temp path (no E2E_LOG_DIR) keeps one file per worker across sessions."""
    monkeypatch.setattr(worker_module.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(worker_module.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setenv(ENV_SHOW_WORKER_LOGS, "0")
    worker = Worker(model_id="m", engine="vllm", port=50112, gpu_ids=[0], mode=ConnectionMode.GRPC)

    _spawn(worker, "one")
    _end_run(worker, "EngineCore died: the trace")
    _spawn(worker, "two")

    assert sorted(os.listdir(tmp_path)) == ["smg-worker-m_vllm_grpc_50112.log"]
    assert (tmp_path / "smg-worker-m_vllm_grpc_50112.log").read_text() == "engine two starting\n"


def test_an_empty_or_absent_previous_log_is_not_rotated(tmp_path):
    path = tmp_path / "worker-x.log"
    assert rotate_previous_log(str(path)) is None
    path.write_text("")
    assert rotate_previous_log(str(path)) is None
    assert sorted(os.listdir(tmp_path)) == ["worker-x.log"]
    path.write_text("something\n")
    assert rotate_previous_log(str(path)) == str(path) + ".1"
    assert sorted(os.listdir(tmp_path)) == ["worker-x.log.1"]
