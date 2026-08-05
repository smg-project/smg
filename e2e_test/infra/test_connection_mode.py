"""Unit tests for E2E connection-mode env parsing (no GPU)."""

from __future__ import annotations

import pytest
from infra.constants import (
    ENV_CONNECTION_MODE,
    ConnectionMode,
    get_connection_mode_override,
)


def test_unset_returns_none(monkeypatch):
    monkeypatch.delenv(ENV_CONNECTION_MODE, raising=False)
    assert get_connection_mode_override() is None


@pytest.mark.parametrize(
    "value,expected",
    [
        ("zmq", ConnectionMode.ZMQ),
        ("ZMQ", ConnectionMode.ZMQ),
        ("Grpc", ConnectionMode.GRPC),
        ("  http  ", ConnectionMode.HTTP),
    ],
)
def test_valid_values_are_case_insensitive(monkeypatch, value, expected):
    monkeypatch.setenv(ENV_CONNECTION_MODE, value)
    assert get_connection_mode_override() == expected


@pytest.mark.parametrize("value", ["", "   "])
def test_set_but_blank_returns_none(monkeypatch, value):
    # The workflow always exports E2E_CONNECTION_MODE and leaves it empty for
    # non-override lanes, so a blank value must mean "no override", not an error.
    monkeypatch.setenv(ENV_CONNECTION_MODE, value)
    assert get_connection_mode_override() is None


def test_invalid_value_raises(monkeypatch):
    monkeypatch.setenv(ENV_CONNECTION_MODE, "bogus")
    with pytest.raises(ValueError, match="not a valid connection mode"):
        get_connection_mode_override()
