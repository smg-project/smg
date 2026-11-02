"""Unit tests for the default startup wait and the ``E2E_STARTUP_TIMEOUT``
floor on startup waits."""

import ast
from pathlib import Path

import pytest
from infra import worker as worker_module
from infra.constants import (
    DEFAULT_STARTUP_TIMEOUT,
    ENGINE_START_SECS,
    ENV_STARTUP_TIMEOUT,
    SERVICER_WARMUP_BUDGET_SECS,
    effective_startup_timeout,
    get_startup_timeout,
)
from infra.model_specs import MODEL_SPECS
from infra.worker import worker_startup_timeout

_SERVICER_WARMUP = (
    Path(__file__).resolve().parents[2] / "grpc_servicer" / "smg_grpc_servicer" / "warmup.py"
)


def _servicer_default_budget_secs() -> float:
    """``DEFAULT_BUDGET_SECS`` as the servicer's warm-up module assigns it, read
    off its source so the harness test needs none of the servicer's imports."""
    for node in ast.parse(_SERVICER_WARMUP.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "DEFAULT_BUDGET_SECS"
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"DEFAULT_BUDGET_SECS is not assigned at the top of {_SERVICER_WARMUP}")


def test_default_bound_is_the_engine_start_plus_the_servicer_warmup_budget():
    """The servicer holds SERVING until its planned warm-up ran, a pass bounded
    by its default budget; the harness waits for the engine's start plus that
    budget. The budget half is pinned to the servicer's own constant so the two
    cannot drift apart unnoticed."""
    assert DEFAULT_STARTUP_TIMEOUT == ENGINE_START_SECS + SERVICER_WARMUP_BUDGET_SECS
    if not _SERVICER_WARMUP.is_file():
        pytest.skip(f"{_SERVICER_WARMUP} is not in this tree")
    assert SERVICER_WARMUP_BUDGET_SECS == _servicer_default_budget_secs()


def test_a_spec_that_raises_the_bound_keeps_the_warm_up_budget():
    """A spec's ``startup_timeout`` replaces the default, so a value sized for
    the engine's start alone, at or under the default, would lose the warm-up
    headroom: every spec that sets one raises the default. That the budget is
    part of the value is said at the spec itself, as
    ``X + SERVICER_WARMUP_BUDGET_SECS``; no bound here can tell."""
    overrides = {
        name: spec["startup_timeout"]
        for name, spec in MODEL_SPECS.items()
        if "startup_timeout" in spec
    }
    assert overrides
    for name, bound in overrides.items():
        assert bound > DEFAULT_STARTUP_TIMEOUT, name


def test_unset_env_leaves_the_per_model_bound(monkeypatch):
    monkeypatch.delenv(ENV_STARTUP_TIMEOUT, raising=False)
    assert get_startup_timeout() is None
    assert effective_startup_timeout(DEFAULT_STARTUP_TIMEOUT) == DEFAULT_STARTUP_TIMEOUT
    monkeypatch.setenv(ENV_STARTUP_TIMEOUT, " ")
    assert effective_startup_timeout(600) == 600


def test_env_is_a_floor_not_a_cap(monkeypatch):
    monkeypatch.setenv(ENV_STARTUP_TIMEOUT, " 1800 ")
    assert get_startup_timeout() == 1800
    assert effective_startup_timeout(DEFAULT_STARTUP_TIMEOUT) == 1800
    assert effective_startup_timeout(3600) == 3600


def test_non_positive_values_are_rejected(monkeypatch):
    monkeypatch.setenv(ENV_STARTUP_TIMEOUT, "0")
    with pytest.raises(ValueError, match=ENV_STARTUP_TIMEOUT):
        get_startup_timeout()


def test_a_workers_bound_is_its_specs_timeout_or_the_default(monkeypatch):
    monkeypatch.delenv(ENV_STARTUP_TIMEOUT, raising=False)
    specs = {"slow": {"startup_timeout": 900}, "plain": {}}
    monkeypatch.setattr(worker_module, "get_model_spec", lambda model_id: specs[model_id])
    assert worker_startup_timeout("slow") == 900
    assert worker_startup_timeout("plain") == DEFAULT_STARTUP_TIMEOUT
    assert worker_startup_timeout("plain", timeout=450) == 450
    # the spec's own bound wins over a caller's default
    assert worker_startup_timeout("slow", timeout=450) == 900


def test_a_workers_bound_takes_the_env_floor(monkeypatch):
    monkeypatch.setenv(ENV_STARTUP_TIMEOUT, "1800")
    specs = {"slow": {"startup_timeout": 900}, "slower": {"startup_timeout": 3600}}
    monkeypatch.setattr(worker_module, "get_model_spec", lambda model_id: specs[model_id])
    assert worker_startup_timeout("slow") == 1800
    assert worker_startup_timeout("slower") == 3600


def _original_profile(monkeypatch, profile):
    from infra import constants as constants_module

    monkeypatch.setattr(constants_module, "_STARTUP_PROFILE_CONTEXT", None)
    monkeypatch.delenv(ENV_STARTUP_TIMEOUT, raising=False)
    values = {
        "E2E_STARTUP_PROFILE": profile,
        "E2E_ENGINE": "vllm",
        "E2E_RUNTIME": "vllm",
        "E2E_GPU_VENDOR": "amd",
        "E2E_GPU_TIER": "1" if profile == "vllm-zmq-dp2-chat" else "4",
        "E2E_CONNECTION_MODE": "zmq" if profile == "vllm-zmq-dp2-chat" else "",
        "E2E_ZMQ_ENGINE_COUNT": "2" if profile == "vllm-zmq-dp2-chat" else "",
        "TEST_DIRS": "e2e_test/chat_completions",
        "TEST_FILTER": "",
        "E2E_MIN_SELECTED": "202" if profile == "vllm-zmq-dp2-chat" else "32",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)


_ORIGINAL13_MODELS = (
    "Qwen/Qwen3-30B-A3B",
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen3-VL-8B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "openai/gpt-oss-20b",
)


def test_original13_all_models_can_run_sequentially_in_one_profile(monkeypatch):
    _original_profile(monkeypatch, "vllm-zmq-dp2-chat")
    assert [worker_startup_timeout(model) for model in _ORIGINAL13_MODELS] == [300] * 8


def test_original29_both_models_keep_distinct_frozen_bounds_in_one_process(monkeypatch):
    _original_profile(monkeypatch, "vllm-4gpu-chat")
    assert worker_startup_timeout("meta-llama/Llama-4-Scout-17B-16E-Instruct") == 1200
    assert worker_startup_timeout("openai/gpt-oss-120b") == 600
    assert worker_startup_timeout("meta-llama/Llama-4-Scout-17B-16E-Instruct") == 1200


@pytest.mark.parametrize(
    "profile,floor",
    [
        ("vllm-zmq-dp2-chat", ""),
        ("vllm-zmq-dp2-chat", " "),
        ("vllm-zmq-dp2-chat", "1"),
        ("vllm-zmq-dp2-chat", "300"),
        ("vllm-4gpu-chat", ""),
        ("vllm-4gpu-chat", "60"),
    ],
)
def test_a_compatible_floor_cannot_extend_a_profile(monkeypatch, profile, floor):
    _original_profile(monkeypatch, profile)
    monkeypatch.setenv(ENV_STARTUP_TIMEOUT, floor)
    model, bound = (
        (_ORIGINAL13_MODELS[0], 300)
        if profile == "vllm-zmq-dp2-chat"
        else ("openai/gpt-oss-120b", 600)
    )
    assert worker_startup_timeout(model, timeout=9999) == bound


@pytest.mark.parametrize("floor", ["301", "1201", "601", "0", "-1", "bad", "1.5"])
def test_original13_rejects_extended_or_invalid_floor(monkeypatch, floor):
    _original_profile(monkeypatch, "vllm-zmq-dp2-chat")
    monkeypatch.setenv(ENV_STARTUP_TIMEOUT, floor)
    with pytest.raises(ValueError):
        worker_startup_timeout(_ORIGINAL13_MODELS[0])


def test_original29_rejects_floor61_before_its_shorter_gateway_wait(monkeypatch):
    _original_profile(monkeypatch, "vllm-4gpu-chat")
    monkeypatch.setenv(ENV_STARTUP_TIMEOUT, "61")
    with pytest.raises(ValueError, match="frozen wait"):
        worker_startup_timeout("openai/gpt-oss-120b")


@pytest.mark.parametrize(
    "key,value",
    [
        ("E2E_STARTUP_PROFILE", "original38-12"),
        ("E2E_ENGINE", "sglang"),
        ("E2E_RUNTIME", "sglang"),
        ("E2E_GPU_VENDOR", "nvidia"),
        ("E2E_GPU_TIER", "4"),
        ("E2E_CONNECTION_MODE", "grpc"),
        ("E2E_ZMQ_ENGINE_COUNT", "1"),
        ("TEST_DIRS", "e2e_test/router"),
        ("TEST_FILTER", "-k subset"),
        ("E2E_MIN_SELECTED", "1"),
    ],
)
def test_unknown_or_mismatched_profile_never_falls_back(monkeypatch, key, value):
    _original_profile(monkeypatch, "vllm-zmq-dp2-chat")
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError):
        worker_startup_timeout(_ORIGINAL13_MODELS[0])


def test_unknown_profile_model_is_rejected_even_with_an_ordinary_spec(monkeypatch):
    _original_profile(monkeypatch, "vllm-zmq-dp2-chat")
    monkeypatch.setattr(worker_module, "get_model_spec", lambda _: {"tp": 1})
    with pytest.raises(ValueError, match="no frozen startup contract"):
        worker_startup_timeout("different-model")


def test_spec_tp_override_is_rejected_without_changing_the_model_spec(monkeypatch):
    _original_profile(monkeypatch, "vllm-zmq-dp2-chat")
    monkeypatch.setattr(worker_module, "get_model_spec", lambda _: {"tp": 2})
    with pytest.raises(ValueError, match="requires TP1"):
        worker_startup_timeout(_ORIGINAL13_MODELS[0])


@pytest.mark.parametrize("new_value", ["", "vllm-4gpu-chat"])
def test_selected_profile_cannot_change_in_process(monkeypatch, new_value):
    _original_profile(monkeypatch, "vllm-zmq-dp2-chat")
    assert worker_startup_timeout(_ORIGINAL13_MODELS[0]) == 300
    monkeypatch.setenv("E2E_STARTUP_PROFILE", new_value)
    with pytest.raises(ValueError):
        worker_startup_timeout(_ORIGINAL13_MODELS[0])


def test_profile_floor_cannot_change_after_a_model_was_resolved(monkeypatch):
    _original_profile(monkeypatch, "vllm-zmq-dp2-chat")
    assert worker_startup_timeout(_ORIGINAL13_MODELS[0]) == 300
    monkeypatch.setenv(ENV_STARTUP_TIMEOUT, "1")
    with pytest.raises(ValueError, match="context changed"):
        worker_startup_timeout(_ORIGINAL13_MODELS[1])


def test_blank_profile_retains_current_model_and_floor_semantics(monkeypatch):
    from infra import constants as constants_module

    monkeypatch.setattr(constants_module, "_STARTUP_PROFILE_CONTEXT", None)
    monkeypatch.setenv("E2E_STARTUP_PROFILE", " ")
    monkeypatch.delenv(ENV_STARTUP_TIMEOUT, raising=False)
    assert worker_startup_timeout(_ORIGINAL13_MODELS[0]) == 600
    assert worker_startup_timeout("meta-llama/Llama-4-Scout-17B-16E-Instruct") == 1500
    assert worker_startup_timeout("openai/gpt-oss-120b") == 900
    monkeypatch.setenv(ENV_STARTUP_TIMEOUT, "1800")
    assert worker_startup_timeout("openai/gpt-oss-120b") == 1800
