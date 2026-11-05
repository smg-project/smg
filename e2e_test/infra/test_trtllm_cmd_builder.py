"""Unit tests for the TensorRT-LLM gRPC worker command (no GPU): the engine's
own gRPC server unless the lane says ``rust``, then this repository's launcher
with the Rust servicer, both on the same LLM API options file."""

from __future__ import annotations

import os

import pytest
import yaml
from infra import constants
from infra.constants import DEFAULT_HOST, ENV_TRTLLM_SERVICER_IMPL, ConnectionMode
from infra.worker import Worker

_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
_SPEC = {"trtllm_extra_config": {"kv_cache_config": {"free_gpu_memory_fraction": 0.8}}}
_ENGINE_FLAGS = ["--max_batch_size", "8"]


@pytest.fixture(autouse=True)
def _hermetic_lane(monkeypatch):
    """No lane setting, nothing inherited and no servicer package, unless a
    test pins one of them."""
    monkeypatch.delenv(ENV_TRTLLM_SERVICER_IMPL, raising=False)
    monkeypatch.delenv("SMG_TRTLLM_SERVICER_IMPL", raising=False)
    monkeypatch.setattr(constants, "_package_servicer_switch", lambda: None)


def _worker() -> Worker:
    return Worker(
        model_id=_MODEL, engine="trtllm", port=50111, gpu_ids=[0], mode=ConnectionMode.GRPC
    )


def _options(cmd: list[str], flag: str) -> dict:
    """The LLM API options file the command names after ``flag``, read and removed."""
    value = cmd[cmd.index(flag) + 1]
    path = value[1:] if value.startswith("@") else value
    with open(path, encoding="utf-8") as handle:
        options = yaml.safe_load(handle)
    os.unlink(path)
    return options


def test_an_unset_lane_runs_the_engines_own_grpc_server():
    w = _worker()
    cmd = w._build_trtllm_cmd("/models/llama", 2, {**_SPEC, "trtllm_args": _ENGINE_FLAGS})
    assert cmd[:5] == ["python3", "-m", "tensorrt_llm.commands.serve", "serve", "/models/llama"]
    assert "--grpc" in cmd
    assert cmd[cmd.index("--host") + 1] == DEFAULT_HOST
    assert cmd[cmd.index("--port") + 1] == "50111"
    assert cmd[cmd.index("--tp_size") + 1] == "2"
    assert cmd[-2:] == _ENGINE_FLAGS
    assert _options(cmd, "--extra_llm_api_options") == {
        "guided_decoding_backend": "xgrammar",
        "kv_cache_config": {"free_gpu_memory_fraction": 0.8},
    }
    assert "SMG_TRTLLM_SERVICER_IMPL" not in w._build_env()


def test_an_explicit_python_lane_runs_the_engines_own_grpc_server(monkeypatch):
    monkeypatch.setenv(ENV_TRTLLM_SERVICER_IMPL, "python")
    w = _worker()
    cmd = w._build_trtllm_cmd("/models/llama", 1, _SPEC)
    assert "tensorrt_llm.commands.serve" in cmd
    assert "--servicer-impl" not in cmd
    _options(cmd, "--extra_llm_api_options")
    assert w._build_env()["SMG_TRTLLM_SERVICER_IMPL"] == "python"


def test_a_rust_lane_runs_the_launcher_with_the_rust_servicer(monkeypatch):
    """``E2E_TRTLLM_SERVICER_IMPL=rust`` is this repository's launcher: the
    LLM API's keywords, the parallel size among them, ride ``--llm-args``."""
    monkeypatch.setenv(ENV_TRTLLM_SERVICER_IMPL, "rust")
    w = _worker()
    cmd = w._build_trtllm_cmd("/models/llama", 2, _SPEC)
    assert cmd[:3] == ["python3", "-m", "smg_grpc_servicer.trtllm"]
    assert "tensorrt_llm.commands.serve" not in cmd
    assert cmd[cmd.index("--model") + 1] == "/models/llama"
    assert cmd[cmd.index("--host") + 1] == DEFAULT_HOST
    assert cmd[cmd.index("--port") + 1] == "50111"
    assert cmd[cmd.index("--servicer-impl") + 1] == "rust"
    assert cmd[cmd.index("--llm-args") + 1].startswith("@")
    assert _options(cmd, "--llm-args") == {
        "guided_decoding_backend": "xgrammar",
        "kv_cache_config": {"free_gpu_memory_fraction": 0.8},
        "tensor_parallel_size": 2,
    }
    assert w._build_env()["SMG_TRTLLM_SERVICER_IMPL"] == "rust"
    # The dispatch reaches it from the worker's own spec too.
    dispatched = w._build_cmd()
    assert dispatched[:3] == cmd[:3]
    _options(dispatched, "--llm-args")


def test_the_packages_rust_default_selects_the_launcher_too(monkeypatch):
    monkeypatch.setattr(
        constants,
        "_package_servicer_switch",
        lambda: lambda variable, environ=None, **_: ("rust", "default"),
    )
    cmd = _worker()._build_trtllm_cmd("/models/llama", 1, _SPEC)
    assert cmd[:3] == ["python3", "-m", "smg_grpc_servicer.trtllm"]
    _options(cmd, "--llm-args")


def test_the_launcher_refuses_the_engine_clis_flags_by_name(monkeypatch):
    monkeypatch.setenv(ENV_TRTLLM_SERVICER_IMPL, "rust")
    with pytest.raises(ValueError, match="trtllm_args"):
        _worker()._build_trtllm_cmd("/models/llama", 1, {"trtllm_args": _ENGINE_FLAGS})
