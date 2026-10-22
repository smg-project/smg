"""Unit tests for the default startup wait and the ``E2E_STARTUP_TIMEOUT``
floor on startup waits."""

import ast
from pathlib import Path

import pytest
from infra.constants import (
    DEFAULT_STARTUP_TIMEOUT,
    ENGINE_START_SECS,
    ENV_STARTUP_TIMEOUT,
    SERVICER_WARMUP_BUDGET_SECS,
    effective_startup_timeout,
    get_startup_timeout,
)

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
