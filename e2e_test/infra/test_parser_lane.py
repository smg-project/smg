"""Unit tests for the E2E_PARSER lane switch: the gateway's --parser flag."""

import pytest
from infra.constants import ENV_PARSER, get_parser
from infra.gateway import Gateway


def test_unset_or_blank_leaves_the_gateways_default(monkeypatch):
    monkeypatch.delenv(ENV_PARSER, raising=False)
    assert get_parser() is None
    assert "--parser" not in Gateway()._build_base_cmd()
    monkeypatch.setenv(ENV_PARSER, "  ")
    assert get_parser() is None


def test_symphony_reaches_the_command_whatever_its_case(monkeypatch):
    monkeypatch.setenv(ENV_PARSER, " Symphony ")
    assert get_parser() == "symphony"
    cmd = Gateway()._build_base_cmd()
    assert cmd[cmd.index("--parser") + 1] == "symphony"


def test_legacy_is_passed_as_spelled(monkeypatch):
    monkeypatch.setenv(ENV_PARSER, "legacy")
    cmd = Gateway()._build_base_cmd()
    assert cmd[cmd.index("--parser") + 1] == "legacy"


def test_unknown_value_raises(monkeypatch):
    monkeypatch.setenv(ENV_PARSER, "fast")
    with pytest.raises(ValueError, match="E2E_PARSER='fast'"):
        get_parser()
