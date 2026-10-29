"""The worker environment's servicer-implementation switch, per engine: the
lane's explicit choice is exported as given, a lane that sets nothing exports
nothing and follows the servicer package's default, and the vLLM hook check
runs whenever the workers would run the Rust servicer."""

from __future__ import annotations

import pytest
from infra import constants
from infra import worker as worker_module
from infra.constants import (
    ENV_SGLANG_SERVICER_IMPL,
    ENV_TOKENSPEED_SERVICER_IMPL,
    ENV_VLLM_SERVICER_IMPL,
    ConnectionMode,
)
from infra.worker import Worker

ENGINES = [
    ("vllm", ENV_VLLM_SERVICER_IMPL, "SMG_VLLM_SERVICER_IMPL", True),
    ("tokenspeed", ENV_TOKENSPEED_SERVICER_IMPL, "SMG_TOKENSPEED_SERVICER_IMPL", False),
    ("sglang", ENV_SGLANG_SERVICER_IMPL, "SMG_SGLANG_SERVICER_IMPL", False),
]


def _worker(engine: str, mode: ConnectionMode) -> Worker:
    return Worker(model_id="m", engine=engine, port=50111, gpu_ids=[0], mode=mode)


def _package_default(monkeypatch, impl: str | None) -> None:
    """Pin what the servicer package's switch answers for a launch that names
    nothing (``impl``), or make the package absent (None)."""
    switch = None if impl is None else (lambda variable, environ=None, **_: (impl, "default"))
    monkeypatch.setattr(constants, "_package_servicer_switch", lambda: switch)


@pytest.fixture
def hook_checks(monkeypatch) -> list[bool]:
    checks: list[bool] = []
    monkeypatch.setattr(worker_module, "_require_rust_servicer_hook", lambda: checks.append(True))
    return checks


@pytest.mark.parametrize(("engine", "lane_setting", "variable", "needs_hook"), ENGINES)
@pytest.mark.parametrize("package_default", ["python", "rust", None])
def test_an_unset_lane_exports_nothing_and_follows_the_package_default(
    monkeypatch, hook_checks, engine, lane_setting, variable, needs_hook, package_default
):
    """No lane setting: the variable is absent from the worker's environment
    (an inherited value included, so the worker runs the package default the
    harness read), and the vLLM hook check runs exactly when that default is
    the Rust servicer. An absent package decides nothing."""
    _package_default(monkeypatch, package_default)
    monkeypatch.delenv(lane_setting, raising=False)
    monkeypatch.setenv(variable, "python")  # inherited from the shell: not the lane's word
    assert variable not in _worker(engine, ConnectionMode.GRPC)._build_env()
    assert hook_checks == ([True] if needs_hook and package_default == "rust" else [])


@pytest.mark.parametrize(("engine", "lane_setting", "variable", "needs_hook"), ENGINES)
@pytest.mark.parametrize("package_default", ["python", "rust"])
def test_an_explicit_python_lane_exports_python(
    monkeypatch, hook_checks, engine, lane_setting, variable, needs_hook, package_default
):
    """``python`` is exported as such whatever the package's default, so a
    deprecated-path row keeps running the Python servicer after the default
    moves; no hook check, since no worker runs the Rust servicer."""
    _package_default(monkeypatch, package_default)
    monkeypatch.setenv(lane_setting, "python")
    monkeypatch.setenv(variable, "rust")  # inherited: the lane's choice wins over it
    assert _worker(engine, ConnectionMode.GRPC)._build_env()[variable] == "python"
    assert hook_checks == []


@pytest.mark.parametrize(("engine", "lane_setting", "variable", "needs_hook"), ENGINES)
@pytest.mark.parametrize("package_default", ["python", "rust", None])
def test_an_explicit_rust_lane_exports_rust_and_checks_the_hook(
    monkeypatch, hook_checks, engine, lane_setting, variable, needs_hook, package_default
):
    _package_default(monkeypatch, package_default)
    monkeypatch.setenv(lane_setting, "rust")
    monkeypatch.delenv(variable, raising=False)
    assert _worker(engine, ConnectionMode.GRPC)._build_env()[variable] == "rust"
    assert hook_checks == ([True] if needs_hook else [])


@pytest.mark.parametrize(("engine", "lane_setting", "variable", "_needs_hook"), ENGINES)
def test_a_worker_off_the_grpc_path_keeps_the_inherited_setting(
    monkeypatch, hook_checks, engine, lane_setting, variable, _needs_hook
):
    _package_default(monkeypatch, "rust")
    monkeypatch.setenv(variable, "rust")
    monkeypatch.delenv(lane_setting, raising=False)
    assert _worker(engine, ConnectionMode.HTTP)._build_env()[variable] == "rust"
    assert hook_checks == []


@pytest.mark.parametrize("gpu_ids", [[0], [1], [0, 1]])
@pytest.mark.parametrize("rocr_devices", ["0,1", "3,1"])
def test_nonempty_hip_mask_follows_worker_logical_devices(monkeypatch, gpu_ids, rocr_devices):
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", rocr_devices)
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6,7")
    worker = _worker("sglang", ConnectionMode.HTTP)
    worker.gpu_ids = gpu_ids

    env = worker._build_env()

    # HIP indices address the parent's ROCR-selected order, not physical IDs.
    assert env["CUDA_VISIBLE_DEVICES"] == env["HIP_VISIBLE_DEVICES"] == ",".join(map(str, gpu_ids))
    assert env["ROCR_VISIBLE_DEVICES"] == rocr_devices
    assert worker_module.os.environ["HIP_VISIBLE_DEVICES"] == "0,1"
    assert worker_module.os.environ["CUDA_VISIBLE_DEVICES"] == "6,7"


@pytest.mark.parametrize(("engine", "inherited_hip"), [("vllm", None), ("sglang", "")])
def test_absent_or_empty_hip_mask_is_preserved(monkeypatch, engine, inherited_hip):
    if inherited_hip is None:
        monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "3,1")
    if inherited_hip is None:
        monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", inherited_hip)
    worker = _worker(engine, ConnectionMode.HTTP)
    worker.gpu_ids = [1]

    env = worker._build_env()

    assert env.get("HIP_VISIBLE_DEVICES") == inherited_hip
    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert env.get("ROCR_VISIBLE_DEVICES") == (None if inherited_hip is None else "3,1")


def test_explicit_extra_env_still_overrides_gpu_masks(monkeypatch):
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0,1")
    worker = _worker("sglang", ConnectionMode.HTTP)
    worker.extra_env = {
        "CUDA_VISIBLE_DEVICES": "1",
        "HIP_VISIBLE_DEVICES": "0",
        "ROCR_VISIBLE_DEVICES": "3,1",
    }

    env = worker._build_env()

    assert all(env[key] == value for key, value in worker.extra_env.items())
