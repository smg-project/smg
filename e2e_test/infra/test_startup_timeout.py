"""Unit tests for the ``E2E_STARTUP_TIMEOUT`` floor on startup waits."""

import pytest
from infra.constants import (
    DEFAULT_STARTUP_TIMEOUT,
    ENV_STARTUP_TIMEOUT,
    effective_startup_timeout,
    get_startup_timeout,
)


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
