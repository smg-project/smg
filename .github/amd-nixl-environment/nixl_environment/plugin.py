"""Observe environment continuity; never replace PD process/transfer proof."""

import importlib.util
import json
import os
from pathlib import Path

import pytest


class EnvironmentProof:
    def __init__(self, stage):
        self.stage = stage

    def check(self, suffix):
        root = Path(os.environ["GITHUB_WORKSPACE"])
        directory = Path(os.environ["SMG_CI_RUNTIME_STATE_DIR"])
        spec = importlib.util.spec_from_file_location(
            "p4_runtime_constraints", root / "scripts/ci_runtime_constraints.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        try:
            module.verify(directory, self.stage + "-" + suffix)
        except Exception as error:
            (directory / (self.stage + "-" + suffix + "-error.json")).write_text(
                json.dumps({"error": str(error)}) + "\n"
            )
            raise

    def pytest_sessionstart(self, session):
        self.check("before")

    @pytest.hookimpl(trylast=True)
    def pytest_sessionfinish(self, session, exitstatus):
        try:
            self.check("after")
        except Exception:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_configure(config):
    if not os.environ.get("SMG_CI_RUNTIME_STATE_DIR"):
        return
    e2e = any("e2e_test/router" in str(arg) for arg in config.args)
    units = any("grpc_servicer/tests" in str(arg) for arg in config.args)
    if e2e or units:
        config.pluginmanager.register(
            EnvironmentProof("pytest" if e2e else "unit"), "smg-nixl-environment"
        )

    if e2e:
        from .runtime import RuntimeProof

        config.pluginmanager.register(RuntimeProof(), "smg-nixl-runtime")
