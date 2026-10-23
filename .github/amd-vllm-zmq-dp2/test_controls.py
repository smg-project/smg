"""Focused CPU controls; fake engine titles are not real DP execution evidence."""

import importlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
activity = importlib.import_module("metrics_activity")
ranks = importlib.import_module("rank_processes")
reuse = importlib.import_module("reuse")


def test_original_running_gauge_requires_both_actual_labels_and_positive_values():
    def row(rank, value, model="m", worker="ipc:///owned"):
        return f'smg_engine_running_requests{{worker="{worker}",model="{model}",dp_rank="{rank}"}} {value}'

    text = row(0, 1) + "\n" + row(1, 2)
    assert activity.running_ranks(text, "ipc:///owned", "m")[0] == {0, 1}
    for bad in [
        row(0, 1),
        row(0, 1) + "\n" + row(1, 0),
        row(0, 1) + "\n" + row(1, 2, worker="ipc:///foreign"),
    ]:
        assert activity.running_ranks(bad, "ipc:///owned", "m")[0] != {0, 1}
    for bad in [row(1, "NaN"), row(2, 1), row(1, -1)]:
        with pytest.raises(ValueError):
            activity.running_ranks(bad, "ipc:///owned", "m")


def test_real_owned_rank_titles_and_marker_without_any_drm_gate(monkeypatch):
    monkeypatch.setenv("E2E_JOB_MARKER", "dp2-cpu-control")
    children = []
    try:
        for index in range(2):
            environment = dict(os.environ, SPT_NOENV="1")
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    f"import setproctitle,time; setproctitle.setproctitle('VLLM::EngineCore_DP{index}'); "
                    "print('ready',flush=True); time.sleep(20)",
                ],
                env=environment,
                stdout=subprocess.PIPE,
            )
            children.append(child)
            assert child.stdout.readline() == b"ready\n"
        observed = ranks.rank_processes(os.getpid())
        assert {r["rank"] for r in observed} == {0, 1}
        assert {r["pid"] for r in observed} == {c.pid for c in children}
        assert all(
            r["marker_visible"] and not r["physical_gpu_exclusivity_or_topology_proven"]
            for r in observed
        )
        monkeypatch.setenv("E2E_JOB_MARKER", "foreign")
        with pytest.raises(ValueError, match="marker"):
            ranks.rank_processes(os.getpid())
    finally:
        for child in children:
            child.terminate()
            child.wait(timeout=5)


def test_current_original_worker_builds_two_local_ranks_on_two_devices(monkeypatch):
    monkeypatch.syspath_prepend(str(HERE.resolve().parents[1] / "bindings/python/src"))
    monkeypatch.syspath_prepend(str(HERE.resolve().parents[1] / "e2e_test"))
    monkeypatch.setenv("E2E_ZMQ_ENGINE_COUNT", "2")
    monkeypatch.setenv("E2E_GPU_OFFSET", "0")
    monkeypatch.setenv("ROUTER_LOCAL_MODEL_PATH", "")

    # Only launch entrypoints need the unbuilt native module; this control never invokes them.
    def no_launch(*_args, **_kwargs):
        pytest.fail("unexpected native router launch")

    monkeypatch.setitem(sys.modules, "smg.launch_router", SimpleNamespace(launch_router=no_launch))
    monkeypatch.setitem(sys.modules, "smg.router_args", SimpleNamespace(RouterArgs=object))
    from infra import ConnectionMode
    from infra import worker as source

    monkeypatch.setattr(source, "get_engine_port", lambda: 32123)
    monkeypatch.setattr(source.Worker, "start", lambda self, **_: None)
    workers = source.start_workers(
        "Qwen/Qwen3-4B-Instruct-2507",
        engine="vllm",
        mode=ConnectionMode.ZMQ,
        count=1,
        wait_ready=False,
    )
    assert len(workers) == 1 and workers[0].gpu_ids == [0, 1]
    argv = workers[0]._build_cmd()
    for flag in ["--data-parallel-size", "--data-parallel-size-local"]:
        assert argv.count(flag) == 1 and argv[argv.index(flag) + 1] == "2"
    assert "--headless" in argv


def test_55_minute_absolute_budget_and_closed_flags(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "synthetic-cpu-consumer")
    monkeypatch.setenv("GITHUB_RUN_ID", "1")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    started = datetime.now(UTC)
    value = {
        "consumer_sha": "synthetic-cpu-consumer",
        "consumer_run": "1",
        "consumer_attempt": "1",
        "consumer_started_at": started.isoformat(),
        "diagnostic_deadline_epoch": started.timestamp() + 55 * 60 - 15,
        "cleanup_reserve_minutes": 5,
        "kill_grace_seconds": 15,
    }
    path = tmp_path / "budget.json"
    path.write_text(json.dumps(value))
    monkeypatch.setenv("SMG_VLLM_ZMQ_DP2_BUDGET_FILE", str(path))
    assert reuse.load_budget() == value
    path.write_text(
        json.dumps({**value, "diagnostic_deadline_epoch": started.timestamp() + 25 * 60 - 15})
    )
    with pytest.raises(ValueError, match="policy"):
        reuse.load_budget()
    past = started - timedelta(seconds=4000)
    path.write_text(
        json.dumps(
            {
                **value,
                "consumer_started_at": past.isoformat(),
                "diagnostic_deadline_epoch": past.timestamp() + 55 * 60 - 15,
            }
        )
    )
    with pytest.raises(ValueError, match="elapsed"):
        reuse.load_budget()
    monkeypatch.setattr(reuse, "api_get", lambda *_: pytest.fail("unadmitted API"))
    with pytest.raises(ValueError, match="not admitted"):
        reuse.main()
