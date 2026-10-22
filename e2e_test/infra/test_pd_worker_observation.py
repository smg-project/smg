"""Real harness command/env construction; no engine or GPU is started."""

import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
from infra import worker as worker_module
from infra.constants import ConnectionMode, WorkerType
from infra.pd_worker_observation import WorkerObservation
from infra.worker import Worker


@pytest.fixture
def harness(monkeypatch):
    monkeypatch.setattr(worker_module, "get_model_spec", lambda _: {"model": "/model", "tp": 1})
    monkeypatch.setattr(worker_module, "detect_rdma_fabric_devices", lambda: ["mlx5_1:1"])
    monkeypatch.setattr(worker_module, "get_open_port", lambda: 50234)
    baseline = []
    monkeypatch.setattr(
        worker_module, "gpu_memory_used_mib", lambda ids: baseline.append(ids) or {}
    )
    return baseline


def worker(tmp_path, *, role=WorkerType.PREFILL, observe=True, **kwargs):
    return Worker(
        model_id="m",
        engine="vllm",
        port=50111,
        gpu_ids=[3, 4],
        kv_backend="nixl",
        worker_type=role,
        scheduler_observation=WorkerObservation("run", "attempt", role.value, tmp_path)
        if observe
        else None,
        **kwargs,
    )


def capture_launch(monkeypatch):
    calls = []
    monkeypatch.setenv(worker_module.ENV_SHOW_WORKER_LOGS, "1")
    monkeypatch.setattr(
        worker_module.subprocess,
        "Popen",
        lambda cmd, **kw: calls.append((cmd, kw)) or SimpleNamespace(pid=123),
    )
    return calls


def test_start_keeps_final_gpu_and_plugin_environment(harness, tmp_path, monkeypatch):
    calls = capture_launch(monkeypatch)
    extra = {
        "ROCR_VISIBLE_DEVICES": "5,6",
        "HIP_VISIBLE_DEVICES": "5,6",
        "PYTHONPATH": "one::two",
        "VLLM_PLUGINS": "first,,last",
        "SMG_PAIRING_PROTOCOL": "kv-v1",
    }
    w = worker(tmp_path, extra_env=extra)
    expected_command = w._build_cmd()
    w.start(wait_ready=False)
    command, options = calls[0]
    env = options["env"]
    assert command == expected_command and "--scheduler-cls" not in command
    assert options["start_new_session"] is True and harness == [[3, 4]]
    assert env["CUDA_VISIBLE_DEVICES"] == "3,4"
    assert env["UCX_NET_DEVICES"] == "mlx5_1:1" and env["UCX_CUDA_IPC_CACHE"] == "n"
    assert env["VLLM_NIXL_SIDE_CHANNEL_PORT"] == "50234"
    assert all(
        env[key] == value
        for key, value in extra.items()
        if key not in ("PYTHONPATH", "VLLM_PLUGINS")
    )
    assert env["PYTHONPATH"].endswith(os.pathsep + "one::two")
    assert env["VLLM_PLUGINS"] == "first,,last,smg_pd_scheduler_observer"
    assert env["SMG_PD_OBSERVE"] == "1" and env["SMG_PD_TRACE_ROLE"] == "prefill"
    assert env["SMG_PD_TRACE_WORKER_INSTANCE"] == w.observer_instance
    assert extra["VLLM_PLUGINS"] == "first,,last"


def test_worker_context_and_restarts_do_not_share_environment(harness, tmp_path, monkeypatch):
    calls = capture_launch(monkeypatch)
    monkeypatch.setenv("SMG_PD_TRACE_RUN", "unrelated-global-run")
    monkeypatch.delenv("SMG_PD_OBSERVE", raising=False)
    monkeypatch.delenv("VLLM_PLUGINS", raising=False)
    first, second = worker(tmp_path), worker(tmp_path, role=WorkerType.DECODE)
    for w in (first, second, first):
        w.start(wait_ready=False)
    environments = [options["env"] for _, options in calls]
    assert len({env["SMG_PD_TRACE_WORKER_INSTANCE"] for env in environments}) == 3
    assert [env["SMG_PD_TRACE_ROLE"] for env in environments] == ["prefill", "decode", "prefill"]
    assert all(
        env["SMG_PD_TRACE_RUN"] == "run" and "VLLM_PLUGINS" not in env for env in environments
    )
    assert os.environ["SMG_PD_TRACE_RUN"] == "unrelated-global-run"
    assert "SMG_PD_OBSERVE" not in os.environ


def test_default_does_not_enable_from_global_or_extra_env(harness, tmp_path, monkeypatch):
    calls = capture_launch(monkeypatch)
    monkeypatch.setenv("SMG_PD_OBSERVE", "1")
    w = worker(tmp_path, observe=False, extra_env={"SMG_PD_OBSERVE": "1", "PYTHONPATH": "original"})
    w.start(wait_ready=False)
    env = calls[0][1]["env"]
    assert "SMG_PD_OBSERVE" not in env and env["PYTHONPATH"] == "original"
    assert w.observer_instance is None and os.environ["SMG_PD_OBSERVE"] == "1"


@pytest.mark.parametrize(
    "config",
    [
        ["--kv-transfer-config", '{"kv_role":"kv_consumer"}'],
        ['--kv-transfer-config={"kv_role":"kv_consumer"}'],
        ["--kv_transfer_config", '{"kv_role":"kv_consumer"}'],
        ["--kv-transfer-config.kv_role", "kv_consumer"],
        ["--kv_transfer_config.kv_role=kv_consumer"],
        ["--kv-transfer", '{"kv_role":"kv_consumer"}'],
        ["--config", "unread-config.yaml"],
        ["--conf=unread-config.yaml"],
        ["-c", "unread-config.yaml"],
    ],
)
def test_overridden_role_fails_before_gpu_baseline_or_spawn(harness, tmp_path, monkeypatch, config):
    calls = capture_launch(monkeypatch)
    w = worker(tmp_path, extra_engine_args=config)
    with pytest.raises(ValueError, match="KV role configuration"):
        w.start(wait_ready=False)
    assert harness == [] and calls == []


@pytest.mark.parametrize(
    ("engine", "mode", "role"),
    [
        ("sglang", ConnectionMode.HTTP, WorkerType.PREFILL),
        ("vllm", ConnectionMode.GRPC, WorkerType.PREFILL),
        ("vllm", ConnectionMode.ZMQ, WorkerType.PREFILL),
        ("vllm", ConnectionMode.HTTP, WorkerType.REGULAR),
    ],
)
def test_unsupported_launch_is_rejected(tmp_path, engine, mode, role):
    with pytest.raises(ValueError):
        WorkerObservation("r", "a", "w", tmp_path).child_environment(
            {}, engine=engine, mode=mode, worker_type=role, command=[]
        )


@pytest.mark.parametrize(
    "command",
    [
        [],
        ["--kv-transfer-config"],
        ["--kv-transfer-config", "{}"],
        ["--kv-transfer-config", "[]"],
        ["--kv-transfer-config", '{"kv_role":"kv_consumer"}'],
    ],
)
def test_missing_or_wrong_command_role_is_not_inferred(tmp_path, command):
    with pytest.raises(ValueError):
        WorkerObservation("r", "a", "w", tmp_path).child_environment(
            {},
            engine="vllm",
            mode=ConnectionMode.HTTP,
            worker_type=WorkerType.PREFILL,
            command=command,
        )


@pytest.mark.parametrize(
    ("run", "attempt", "name", "directory"),
    [
        (None, "a", "w", None),
        ("r", "", "w", None),
        ("r", "a", None, None),
        ("r", "a", "w", "relative"),
    ],
)
def test_incomplete_launch_context_is_rejected(tmp_path, run, attempt, name, directory):
    with pytest.raises(ValueError):
        WorkerObservation(run, attempt, name, directory or tmp_path)


def test_fresh_cpu_child_receives_launch_context_without_a_backend(harness, tmp_path, monkeypatch):
    monkeypatch.delenv("SMG_PD_OBSERVE", raising=False)
    w = worker(tmp_path)
    env = w.scheduler_observation.child_environment(
        w._build_env(),
        engine=w.engine,
        mode=w.mode,
        worker_type=w.worker_type,
        command=w._build_cmd(),
    )
    code = """
import importlib.metadata, json, os, sys
from infra.pd_scheduler_trace import SchedulerTrace
plugins=importlib.metadata.entry_points(group='vllm.general_plugins')
assert sum(p.name=='smg_pd_scheduler_observer' for p in plugins)==1
trace=SchedulerTrace.from_env(); trace.emit('cpu_context_only')
assert trace.unknown is None
assert 'vllm' not in sys.modules and 'torch' not in sys.modules
print(json.dumps(trace.context))
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", code],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    context = json.loads(result.stdout)
    assert context["worker_instance"] == env["SMG_PD_TRACE_WORKER_INSTANCE"]
    assert context["instance"] != context["worker_instance"]
    record = json.loads(next(tmp_path.glob("scheduler-*.jsonl")).read_text())
    assert record["run"] == "run" and record["role"] == "prefill"
    assert record["observation"] == "partial" and record["event"] == "cpu_context_only"
