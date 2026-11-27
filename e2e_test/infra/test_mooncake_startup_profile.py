"""CPU coverage of the opt-in AMD full-router startup contract."""

import json
import os
import subprocess
import sys
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from infra import constants, worker
from infra.constants import ConnectionMode, WorkerType

backend = import_module("fixtures.setup_backend")
ROOT = Path(__file__).resolve().parents[2]
MODELS = (
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "microsoft/Phi-3.5-vision-instruct",
    "Qwen/Qwen3-VL-8B-Instruct",
)
PROFILE_ENV = {
    "E2E_STARTUP_PROFILE": "vllm-mooncake-pd2",
    "E2E_ENGINE": "vllm",
    "E2E_RUNTIME": "vllm",
    "E2E_GPU_VENDOR": "amd",
    "E2E_GPU_TIER": "2",
    "E2E_MIN_SELECTED": "15",
    "E2E_VLLM_KV_BACKEND": "mooncake",
}


@pytest.fixture(autouse=True)
def isolated_profile(monkeypatch):
    for key in os.environ:
        if key.startswith("E2E_") or key.startswith("SMG_VLLM_"):
            monkeypatch.delenv(key)
    monkeypatch.setattr(constants, "_STARTUP_PROFILE_CONTEXT", None)
    monkeypatch.setattr(constants, "_STARTUP_PROFILE_SELECTION", None)
    monkeypatch.setattr(constants, "_RUNTIME_CACHE", None)


def bind(monkeypatch, *, options=None, args=None):
    for key, value in PROFILE_ENV.items():
        monkeypatch.setenv(key, value)
    config = SimpleNamespace(
        args=args if args is not None else ["e2e_test/router"],
        invocation_params=SimpleNamespace(dir=ROOT),
        getoption=lambda name, default=None: (options or {}).get(name, default),
    )
    constants.validate_startup_profile_selection(config)
    return config


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("role", [WorkerType.REGULAR, WorkerType.PREFILL, WorkerType.DECODE])
def test_original_four_models_and_grpc_roles_remain_supported(monkeypatch, model, role):
    bind(monkeypatch)
    worker.validate_startup_profile_worker(
        model, engine="vllm", mode=ConnectionMode.GRPC, worker_type=role
    )
    assert worker.worker_startup_timeout(model) == 300
    assert backend._gateway_readiness_timeout(ConnectionMode.GRPC, model, 60) == 60


def test_profile_does_not_accept_a_grpc_probe_after_its_deadline(monkeypatch):
    bind(monkeypatch)
    clock = [0.0]
    calls = []

    def probe(**kwargs):
        calls.append(kwargs)
        clock[0] += 0.2
        return True

    monkeypatch.setattr(worker.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(worker.time, "perf_counter", lambda: clock[0])
    instance = SimpleNamespace(
        engine="vllm",
        model_id=MODELS[0],
        port=1,
        raise_if_exited=lambda: None,
        grpc_serving=probe,
    )
    with pytest.raises(TimeoutError):
        worker.Worker._wait_grpc_healthy(instance, timeout=0.125)
    assert calls == [{"timeout": 0.125, "_deadline": 0.125}]


def test_original_dp2_rank_pinning_shape_remains_supported(monkeypatch):
    bind(monkeypatch)
    worker.validate_startup_profile_worker(
        MODELS[1],
        engine="vllm",
        mode=ConnectionMode.GRPC,
        gpus=2,
        extra_engine_args=["--data-parallel-size", "2"],
    )


@pytest.mark.parametrize(
    "override",
    [
        {"model_id": "Qwen/Qwen3-4B-Instruct-2507"},
        {"engine": "sglang"},
        {"mode": ConnectionMode.HTTP},
        {"worker_type": WorkerType.ENCODE},
        {"count": 2},
        {"count": True},
        {"tp": 2},
        {"tp": True},
        {"gpus": 2},
        {"kv_backend": "nixl"},
        {"extra_engine_args": ["--kv_transfer_config={}"]},
        {"extra_engine_args": ["--model=another-model"]},
        {"extra_engine_args": ["-tp", "2"]},
        {"extra_env": {"E2E_VLLM_KV_BACKEND": "nixl"}},
        {"extra_env": {"SMG_VLLM_SERVICER_WARMUP": "off"}},
        {"extra_env": {"SMG_VLLM_SERVICER_IMPL": "python"}},
        {"extra_env": {"SMG_VLLM_SERVICER_IMPL": "rust"}},
        {"extra_env": {"SMG_VLLM_SERVICER_IMPL": ""}},
    ],
)
def test_changed_worker_contract_fails_before_allocation(monkeypatch, override):
    bind(monkeypatch)
    monkeypatch.setattr(worker, "get_gpu_offset", lambda: pytest.fail("allocation reached"))
    arguments = {"model_id": MODELS[0], "engine": "vllm", "mode": ConnectionMode.GRPC}
    arguments.update(override)
    with pytest.raises(ValueError, match="vllm-mooncake-pd2"):
        worker.start_workers(**arguments)


def test_pool_hit_cannot_bypass_contract_validation(monkeypatch):
    bind(monkeypatch)
    monkeypatch.setattr(backend, "get_pool", lambda: pytest.fail("pool access reached"))
    with pytest.raises(ValueError, match="per-worker KV backend override"):
        backend._start_workers_tracked(
            model_id=MODELS[0], engine="vllm", mode=ConnectionMode.GRPC, kv_backend="nixl"
        )


@pytest.mark.parametrize("role,count", [("prefill", 2), ("decode", 0), ("decode", True)])
def test_pd_pool_counts_fail_before_starting_any_worker(monkeypatch, role, count):
    bind(monkeypatch)
    monkeypatch.setattr(
        backend, "_start_workers_tracked", lambda **kwargs: pytest.fail("worker start reached")
    )
    setup = backend._setup_pd(
        model_id=MODELS[0],
        model_path=MODELS[0],
        engine="vllm",
        connection_mode=ConnectionMode.GRPC,
        workers_config={role: count},
        gateway_config={"timeout": 60},
        gateway=None,
        log_dir=None,
    )
    with pytest.raises(ValueError, match="one prefill and one decode"):
        next(setup)


@pytest.mark.parametrize(
    "config",
    [
        {"decode_env": {"SMG_VLLM_SERVICER_WARMUP": "off"}},
        {"decode_env": {"SMG_VLLM_SERVICER_IMPL": "python"}},
        {"decode_args": ["--model=other"]},
        {"extra_engine_args": ["--tensor-parallel-size", "2"]},
    ],
)
def test_pd_role_overrides_fail_before_starting_either_pool(monkeypatch, config):
    bind(monkeypatch)
    monkeypatch.setattr(
        backend, "_start_pd_pool", lambda **kwargs: pytest.fail("pool start reached")
    )
    setup = backend._setup_pd(
        model_id=MODELS[0],
        model_path=MODELS[0],
        engine="vllm",
        connection_mode=ConnectionMode.GRPC,
        workers_config=config,
        gateway_config={"timeout": 60},
        gateway=None,
        log_dir=None,
    )
    with pytest.raises(ValueError, match="vllm-mooncake-pd2"):
        next(setup)


@pytest.mark.parametrize("option", ["keyword", "markexpr", "deselect", "lf", "stepwise"])
def test_restricted_collection_is_rejected(monkeypatch, option):
    with pytest.raises(ValueError, match="unfiltered pytest selection"):
        bind(monkeypatch, options={option: "selected"})


def test_full_router_path_and_collected_items_are_bound(monkeypatch):
    with pytest.raises(ValueError, match="unfiltered pytest directory"):
        bind(monkeypatch, args=["e2e_test/router/test_pd_mooncake.py::TestPDMooncakeKvTransfer"])
    config = bind(monkeypatch)
    outsider = SimpleNamespace(
        path=ROOT / "e2e_test/chat_completions/test_chat.py", nodeid="outside"
    )
    with pytest.raises(ValueError, match="outside"):
        constants.validate_startup_profile_selection(config, [outsider])


def test_lane_and_wait_cannot_change_after_binding(monkeypatch):
    bind(monkeypatch)
    assert constants.get_startup_profile() == "vllm-mooncake-pd2"
    monkeypatch.setenv("E2E_STARTUP_TIMEOUT", "61")
    with pytest.raises(ValueError, match="frozen wait"):
        constants.get_startup_profile()
    monkeypatch.delenv("E2E_STARTUP_TIMEOUT")
    monkeypatch.setenv("E2E_KV_BACKEND", "nixl")
    with pytest.raises(ValueError, match="lane mismatch"):
        constants.get_startup_profile()


def test_gateway_override_cannot_extend_the_frozen_wait(monkeypatch):
    bind(monkeypatch)
    with pytest.raises(ValueError, match="exceeds frozen wait"):
        backend._gateway_readiness_timeout(ConnectionMode.GRPC, MODELS[0], 61)


def test_unset_profile_retains_ordinary_startup_and_worker_arguments(monkeypatch):
    monkeypatch.setenv("E2E_STARTUP_TIMEOUT", "1800")
    monkeypatch.setattr(worker, "get_model_spec", lambda _: {"startup_timeout": 3600})
    assert worker.worker_startup_timeout("ordinary") == 3600
    worker.validate_startup_profile_worker("ordinary", engine="sglang", tp=4, count=2)
    assert backend._gateway_readiness_timeout(ConnectionMode.HTTP, "ordinary", 60) == 1800


@pytest.mark.parametrize(
    "override",
    [{}, {"timeout": 51}, {"test_timeout": 35}, {"min_selected": 14}, {"test_filter": "minted"}],
)
def test_real_workflow_input_guard_preserves_the_inherited_caller_contract(override):
    workflow = yaml.safe_load((ROOT / ".github/workflows/e2e-gpu-job.yml").read_text())
    step = next(
        entry
        for entry in workflow["jobs"]["run"]["steps"]
        if entry["name"] == "Validate Mooncake PD startup contract"
    )
    code = step["run"].split("python3 - <<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    inputs = {
        "engine": "vllm",
        "gpu_vendor": "amd",
        "gpu_tier": "2",
        "test_dirs": "e2e_test/router",
        "test_filter": "",
        "min_selected": 15,
        "timeout": 50,
        "test_timeout": 34,
        "vllm_kv_backend": "mooncake",
        "kv_backend": "",
        "vllm_extra_kv_backends": "",
        "mm_processing": "",
        "servicer_impl": "",
        "parser": "",
        "connection_mode": "",
        "zmq_engine_count": "",
    }
    inputs.update(override)
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "PROFILE_INPUTS": json.dumps(inputs)},
        capture_output=True,
        text=True,
        timeout=10,
    )
    if override:
        assert result.returncode != 0
        assert "Mooncake PD startup contract mismatch:" in result.stderr
    else:
        assert result.returncode == 0, result.stderr


def test_real_pytest_cli_rejects_filter_before_collection():
    env = {k: v for k, v in os.environ.items() if not k.startswith("E2E_")}
    env.update(PROFILE_ENV)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "e2e_test/router",
            "-k",
            "minted_params",
            "--collect-only",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "requires unfiltered pytest selection; keyword=" in result.stdout + result.stderr
