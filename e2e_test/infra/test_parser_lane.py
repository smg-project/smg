"""Unit tests for the E2E_PARSER lane switch: the gateway's --parser flag, which keeps its one
value, symphony, for one release and is removed in 1.17."""

import pytest
from infra.constants import ENV_PARSER, get_parser
from infra.gateway import Gateway


def test_unset_or_blank_leaves_the_command_line_without_the_flag(monkeypatch):
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


def test_any_other_value_raises(monkeypatch):
    for value in ("legacy", "fast"):
        monkeypatch.setenv(ENV_PARSER, value)
        with pytest.raises(ValueError, match=f"E2E_PARSER='{value}'"):
            get_parser()
