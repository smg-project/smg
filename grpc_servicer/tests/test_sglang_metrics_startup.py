"""Native metrics setup survives the Rust scheduler launch boundary."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from smg_grpc_servicer.sglang import rust


def test_scheduler_counter_and_scrape_survive_repeated_native_metrics_setup(monkeypatch):
    engine_module = pytest.importorskip("sglang.srt.entrypoints.engine")
    utils = pytest.importorskip("sglang.srt.utils")
    runtime_context = pytest.importorskip("sglang.srt.runtime_context")
    plugins = pytest.importorskip("sglang.srt.plugins")
    server_args = pytest.importorskip("sglang.srt.server_args")
    common = utils.common
    # Record absence too: native setup writes the variable without monkeypatch.
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", "")
    monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR")
    monkeypatch.setattr(common, "prometheus_multiproc_dir", None, raising=False)
    # The native gRPC entrypoint initializes metrics before delegating to SMG.
    common.set_prometheus_multiproc_dir()
    directory = Path(os.environ["PROMETHEUS_MULTIPROC_DIR"])

    # Exercise the native metrics setup again at its real launch call site;
    # model/GPU preparation and scheduler creation are outside this test.
    monkeypatch.setattr(
        engine_module, "_set_envs_and_config", lambda _args: common.set_prometheus_multiproc_dir()
    )
    monkeypatch.setattr(
        engine_module.Engine,
        "_launch_scheduler_processes",
        lambda *_args: (SimpleNamespace(wait_for_ready=lambda: None), []),
    )
    monkeypatch.setattr(utils, "configure_logger", lambda _args: None)
    monkeypatch.setattr(plugins, "load_plugins", lambda: None)
    monkeypatch.setattr(runtime_context, "publish", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server_args.PortArgs, "init_new", lambda _args: object())
    monkeypatch.setattr(rust, "ensure_spawn_start_method", lambda *_args: None)
    args = SimpleNamespace(
        resolve_once=lambda: None, check_server_args=lambda: None, disaggregation_mode="null"
    )
    previous_handler = signal.getsignal(signal.SIGQUIT)
    engine = None
    try:
        engine = rust.launch_scheduler_ranks(args, handshake_port=24321)
        engine._manager.ready_thread.join(5)
        # A fresh process selects Prometheus's multiprocess backend from the
        # inherited environment, just as a spawned scheduler does.
        code = (
            "from prometheus_client import Counter, CollectorRegistry, generate_latest, multiprocess\n"
            "Counter('smg_scheduler_dir_lifetime', 'test').inc()\n"
            "registry = CollectorRegistry()\n"
            "multiprocess.MultiProcessCollector(registry)\n"
            "print(generate_latest(registry).decode(), end='')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
        )
        assert result.returncode == 0, result.stderr
        assert "\nsmg_scheduler_dir_lifetime_total 1.0\n" in result.stdout
        assert directory.is_dir()
        assert list(directory.glob("counter_*.db"))
    finally:
        if engine is not None:
            engine._manager.shutdown(timeout=0)
            engine._manager.holding.clear()
        common.prometheus_multiproc_dir.cleanup()
        signal.signal(signal.SIGQUIT, previous_handler)
