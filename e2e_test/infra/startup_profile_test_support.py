"""Test-only profile setup; literal expected contracts stay in the tests."""

from pathlib import Path

import pytest
from fixtures import hooks as hooks_module
from infra import constants as constants_module


def set_profile_lane(monkeypatch, profile, args=None, floor=None):
    monkeypatch.setattr(constants_module, "_STARTUP_PROFILE_CONTEXT", None)
    monkeypatch.setattr(constants_module, "_STARTUP_PROFILE_SELECTION", None)
    monkeypatch.delenv(constants_module.ENV_STARTUP_TIMEOUT, raising=False)
    monkeypatch.delenv("TEST_DIRS", raising=False)
    monkeypatch.delenv("TEST_FILTER", raising=False)
    if floor is not None:
        monkeypatch.setenv(constants_module.ENV_STARTUP_TIMEOUT, floor)
    definition = constants_module._STARTUP_PROFILES[profile]
    monkeypatch.setenv(constants_module.ENV_STARTUP_PROFILE, profile)
    for key, value in definition.lane:
        monkeypatch.setenv(key, value)
    chat = Path(__file__).resolve().parents[2] / definition.test_dir
    selected = [str(chat)] if args is None else args
    config = pytest.Config.fromdictargs({}, ["--noconftest", *selected])
    hooks_module.pytest_configure(config)
    return config
