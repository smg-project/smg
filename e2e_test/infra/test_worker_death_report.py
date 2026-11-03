"""Unit tests for the failure text of a worker that exits before it reports SERVING.

Each case spawns a real stub process through the harness's own output
redirection, lets it exit, and reads the error the start-up wait raises.
"""

import os
import socket
import sys
import time

import pytest
from infra.constants import DEFAULT_HOST, ENV_SHOW_WORKER_LOGS, ConnectionMode
from infra.worker import LOG_EXCERPT_CHARS, Worker

pytest.importorskip("grpc")
pytest.importorskip("grpc_health")

# The shape of an engine log whose core died at a start-up check: the core's
# traceback at error level, the bare multiprocessing copy, then the server's
# own wrapper error that points back at it.
NIGHTLY_SHAPE = r"""
import sys
core = "(EngineCore pid=2) ERROR 10-22 11:03:45 [core.py:1433] "
main = "(MainProcess pid=1) ERROR 10-22 11:03:45 [server.py:204] "
guard = "ValueError: Free memory on device cuda:0 (25.2/79.18 GiB) on startup is less than desired (0.9, 71.26 GiB)."
lines = [
    "(MainProcess pid=1) WARNING 10-22 11:03:40 [platforms.py:31] Failed to import flashinfer:"
    " ModuleNotFoundError: No module named 'flashinfer'",
    "(MainProcess pid=1) INFO 10-22 11:03:41 [server.py:72] transfer backend is available",
    core + "EngineCore failed to start.",
    core + "Traceback (most recent call last):",
    core + '  File "worker/utils.py", line 548, in request_memory',
    core + "    raise ValueError(",
    core + guard,
    "Process EngineCore:",
    "Traceback (most recent call last):",
    '  File "/usr/lib/python3.12/multiprocessing/process.py", line 314, in _bootstrap',
    "    self.run()",
    guard,
    main + "Failed to start gRPC server",
    main + "Traceback (most recent call last):",
    main + "    raise RuntimeError(",
    main + "RuntimeError: Engine core initialization failed. See root cause above.",
]
print("\n".join(lines))
sys.exit(1)
"""


def _spawned_worker(tmp_path, monkeypatch, script: str, mode: ConnectionMode, port: int) -> Worker:
    """A worker whose process is a stub running ``script``."""
    monkeypatch.setenv(ENV_SHOW_WORKER_LOGS, "0")
    worker = Worker(
        model_id="org/model",
        engine="vllm",
        port=port,
        gpu_ids=[0],
        mode=mode,
        log_dir=str(tmp_path / "e2e-logs"),
    )
    worker.process = worker._spawn_process([sys.executable, "-c", script], env=dict(os.environ))
    return worker


def _dead_worker(
    tmp_path,
    monkeypatch,
    script: str,
    mode: ConnectionMode = ConnectionMode.GRPC,
    port: int = 31001,
) -> Worker:
    """A worker whose process, a stub running ``script``, has already exited."""
    worker = _spawned_worker(tmp_path, monkeypatch, script, mode, port)
    worker.process.wait(timeout=30)
    return worker


def _refused_port() -> int:
    """A port nobody listens on: an HTTP health check there is refused at once."""
    family = socket.AF_INET6 if ":" in DEFAULT_HOST else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as probe:
        probe.bind((DEFAULT_HOST, 0))
        return probe.getsockname()[1]


def _death_text(worker: Worker, timeout: int = 5) -> str:
    try:
        with pytest.raises(RuntimeError, match="died during startup") as raised:
            worker.wait_ready(timeout=timeout)
    finally:
        if worker._log_file is not None:
            worker._log_file.close()
    return str(raised.value)


def test_the_text_names_the_first_and_last_error_of_the_log_and_its_path(tmp_path, monkeypatch):
    worker = _dead_worker(tmp_path, monkeypatch, NIGHTLY_SHAPE)
    text = _death_text(worker)
    assert text.startswith(
        f"Worker org/model (PID {worker.process.pid}) died during startup (exit status 1)"
    )
    # The root cause (the first exception) comes before the wrapper (the last).
    assert text.index("ValueError: Free memory on device cuda:0") < text.index(
        "RuntimeError: Engine core initialization failed"
    )
    assert text.count("ValueError: Free memory") == 1
    # Frames, messages at error level, the server's own lines and the warning
    # that names an exception (a missing optional import) are not quoted.
    for noise in (
        'File "',
        "EngineCore failed to start.",
        "Failed to start gRPC server",
        "INFO",
        "ModuleNotFoundError",
        "WARNING",
    ):
        assert noise not in text
    # The path is the one inside the uploaded log directory, not the absolute one.
    assert text.endswith("; log file worker-org__model_vllm_grpc_31001.log")
    assert str(tmp_path) not in text


# A TP worker whose NCCL init failed: the root cause names a level word inside its
# message (torch's "run with NCCL_DEBUG=INFO for details"), at error level.
NCCL_SHAPE = r"""
import sys
core = "(EngineCore pid=2) ERROR 10-22 11:03:45 [core.py:1433] "
main = "(MainProcess pid=1) ERROR 10-22 11:03:45 [server.py:204] "
nccl = (
    "torch.distributed.DistBackendError: NCCL error in: NCCLUtils.hpp:268, unhandled cuda error"
    " (run with NCCL_DEBUG=INFO for details), NCCL version 2.21.5"
)
lines = [
    "(MainProcess pid=1) INFO 10-22 11:03:41 [server.py:72] transfer backend is available",
    core + "EngineCore failed to start.",
    core + nccl,
    nccl,
    main + "RuntimeError: Engine core initialization failed. See root cause above.",
]
print("\n".join(lines))
sys.exit(1)
"""


def test_a_level_word_inside_an_error_level_exception_message_does_not_hide_it(
    tmp_path, monkeypatch
):
    worker = _dead_worker(tmp_path, monkeypatch, NCCL_SHAPE)
    text = _death_text(worker)
    # The NCCL line is the root cause: quoted first, the wrapper last.
    assert text.index("DistBackendError: NCCL error in") < text.index(
        "RuntimeError: Engine core initialization failed"
    )
    assert text.count("DistBackendError") == 1
    assert "run with NCCL_DEBUG=INFO for details" in text


def test_an_empty_log_leaves_the_exit_status_alone(tmp_path, monkeypatch):
    worker = _dead_worker(tmp_path, monkeypatch, "import sys; sys.exit(3)")
    assert _death_text(worker) == (
        f"Worker org/model (PID {worker.process.pid}) died during startup (exit status 3)"
    )


def test_a_long_line_is_bounded(tmp_path, monkeypatch):
    script = "import sys; print('RuntimeError: ' + 'x' * 5000, file=sys.stderr); sys.exit(1)"
    text = _death_text(_dead_worker(tmp_path, monkeypatch, script))
    assert '"RuntimeError: ' + "x" * (LOG_EXCERPT_CHARS - 17) + '..."' in text
    assert "x" * LOG_EXCERPT_CHARS not in text
    assert len(text) < 2 * LOG_EXCERPT_CHARS


def test_without_an_exception_the_last_error_level_line_speaks(tmp_path, monkeypatch):
    script = (
        "import sys; print('INFO loading weights'); print('ERROR the model path is unreadable');"
        " print('INFO shutting down'); sys.exit(2)"
    )
    text = _death_text(_dead_worker(tmp_path, monkeypatch, script))
    assert '(exit status 2); the log says: "ERROR the model path is unreadable"; log file ' in text


def test_without_any_error_line_the_last_line_speaks(tmp_path, monkeypatch):
    script = (
        "import sys; print('INFO loading weights'); print('Killed by the launcher'); sys.exit(2)"
    )
    text = _death_text(_dead_worker(tmp_path, monkeypatch, script))
    assert 'the log says: "Killed by the launcher"' in text
    assert "loading weights" not in text


def test_a_signal_death_names_the_signal(tmp_path, monkeypatch):
    script = "import os, signal; print('INFO loading weights', flush=True); os.kill(os.getpid(), signal.SIGKILL)"
    text = _death_text(_dead_worker(tmp_path, monkeypatch, script))
    assert "died during startup (signal SIGKILL); the log says:" in text


def test_an_http_worker_that_died_gets_the_same_report_at_once(tmp_path, monkeypatch):
    worker = _dead_worker(
        tmp_path, monkeypatch, NIGHTLY_SHAPE, mode=ConnectionMode.HTTP, port=_refused_port()
    )
    started = time.perf_counter()
    text = _death_text(worker, timeout=30)
    # The exit ends the wait, not the bound; the text is the gRPC branch's.
    assert time.perf_counter() - started < 10
    assert text == worker._death_report()
    assert "ValueError: Free memory on device cuda:0" in text


def test_an_http_worker_that_dies_during_the_wait_ends_it_within_seconds(tmp_path, monkeypatch):
    script = (
        "import sys, time; time.sleep(3);"
        " print('RuntimeError: the model path is unreadable'); sys.exit(2)"
    )
    worker = _spawned_worker(tmp_path, monkeypatch, script, ConnectionMode.HTTP, _refused_port())
    started = time.perf_counter()
    text = _death_text(worker, timeout=30)
    elapsed = time.perf_counter() - started
    # The floor only shows the process was alive when the wait began. The stub
    # sleeps three seconds so that a busy runner, which may start this wait
    # late after the spawn, has two and a half seconds of room before the
    # stub's own exit could end the wait under the floor.
    assert 0.5 <= elapsed < 10, elapsed
    assert '(exit status 2); the log says: "RuntimeError: the model path is unreadable"' in text
