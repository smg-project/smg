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
    ENV_TRTLLM_SERVICER_IMPL,
    ENV_VLLM_SERVICER_IMPL,
    ConnectionMode,
)
from infra.worker import Worker

ENGINES = [
    ("vllm", ENV_VLLM_SERVICER_IMPL, "SMG_VLLM_SERVICER_IMPL", True),
    ("tokenspeed", ENV_TOKENSPEED_SERVICER_IMPL, "SMG_TOKENSPEED_SERVICER_IMPL", False),
    ("sglang", ENV_SGLANG_SERVICER_IMPL, "SMG_SGLANG_SERVICER_IMPL", False),
    ("trtllm", ENV_TRTLLM_SERVICER_IMPL, "SMG_TRTLLM_SERVICER_IMPL", False),
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


def test_the_lanes_worker_log_filter_becomes_the_workers_rust_log(monkeypatch, hook_checks):
    """A lane's E2E_WORKER_RUST_LOG is the worker process's RUST_LOG (the
    tracing filter of the Rust servicer it hosts; the shell's own value yields
    to it); unset or blank, the worker keeps whatever the shell exported."""
    _package_default(monkeypatch, "rust")
    monkeypatch.setenv("RUST_LOG", "warn")
    monkeypatch.delenv(constants.ENV_WORKER_RUST_LOG, raising=False)
    assert _worker("vllm", ConnectionMode.GRPC)._build_env()["RUST_LOG"] == "warn"
    monkeypatch.setenv(constants.ENV_WORKER_RUST_LOG, " ")
    assert _worker("vllm", ConnectionMode.GRPC)._build_env()["RUST_LOG"] == "warn"
    monkeypatch.setenv(constants.ENV_WORKER_RUST_LOG, "info,engine_servicer::request=debug")
    env = _worker("vllm", ConnectionMode.GRPC)._build_env()
    assert env["RUST_LOG"] == "info,engine_servicer::request=debug"
    monkeypatch.delenv("RUST_LOG")
    assert _worker("vllm", ConnectionMode.GRPC)._build_env()["RUST_LOG"] == (
        "info,engine_servicer::request=debug"
    )


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


@pytest.mark.parametrize(
    "engine,model",
    [
        ("sglang", "openai/gpt-oss-20b"),
        ("sglang", "Qwen/Qwen3-VL-8B-Instruct"),
        ("vllm", "openai/gpt-oss-20b"),
    ],
)
@pytest.mark.parametrize("enabled", [False, True])
def test_fixed_sglang_fallback_changes_only_its_original_model_environment(
    monkeypatch, hook_checks, engine, model, enabled
):
    _package_default(monkeypatch, "python")
    monkeypatch.setenv("SGLANG_USE_AITER", "1")  # Original immutable image default.
    monkeypatch.delenv("SGLANG_GPTOSS_MXFP4_BF16_FALLBACK", raising=False)
    monkeypatch.setenv("E2E_SGLANG_GPTOSS_MXFP4_BF16", "1" if enabled else "0")
    env = Worker(
        model_id=model, engine=engine, port=50111, gpu_ids=[0], mode=ConnectionMode.GRPC
    )._build_env()
    target = enabled and engine == "sglang" and model == "openai/gpt-oss-20b"
    assert env["SGLANG_USE_AITER"] == ("0" if target else "1")
    assert env.get("SGLANG_GPTOSS_MXFP4_BF16_FALLBACK") == ("1" if target else None)


@pytest.mark.parametrize("mode", [ConnectionMode.HTTP, ConnectionMode.GRPC, ConnectionMode.ZMQ])
def test_fixed_sglang_fallback_uses_generic_moe_flags_without_mutating_model_spec(
    monkeypatch, mode
):
    spec = {"model": "openai/gpt-oss-20b", "features": ["chat"]}
    monkeypatch.setattr(worker_module, "get_model_spec", lambda model: spec)
    monkeypatch.setenv("E2E_SGLANG_GPTOSS_MXFP4_BF16", "1")
    worker = Worker(
        model_id="openai/gpt-oss-20b", engine="sglang", port=50111, gpu_ids=[0], mode=mode
    )
    if mode == ConnectionMode.ZMQ:
        monkeypatch.setattr(
            worker,
            "_build_sglang_zmq_cmd",
            lambda path, tp, actual, extras: [path, *actual["sglang_args"], *extras],
        )
    cmd = worker._build_cmd()
    assert "openai/gpt-oss-20b" in cmd
    assert cmd[cmd.index("--moe-runner-backend") + 1] == "triton"
    assert cmd[cmd.index("--moe-a2a-backend") + 1] == "none"
    assert spec == {"model": "openai/gpt-oss-20b", "features": ["chat"]}


def test_fixed_sglang_fallback_rejects_conflicting_explicit_backend_or_environment(
    monkeypatch, hook_checks
):
    _package_default(monkeypatch, "python")
    monkeypatch.setenv("E2E_SGLANG_GPTOSS_MXFP4_BF16", "1")
    monkeypatch.setattr(worker_module, "get_model_spec", lambda model: {"model": model})
    worker = Worker(
        model_id="openai/gpt-oss-20b",
        engine="sglang",
        port=50111,
        gpu_ids=[0],
        mode=ConnectionMode.GRPC,
    )
    worker.extra_engine_args = ["--moe-runner-backend=auto"]
    with pytest.raises(ValueError, match="owns"):
        worker._build_cmd()
    worker.extra_env = {"SGLANG_USE_AITER": "1"}
    with pytest.raises(ValueError, match="AITER"):
        worker._build_env()
