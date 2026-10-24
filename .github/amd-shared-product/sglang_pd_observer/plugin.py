"""Explicit AMD SG2 observer; inactive for every other pytest invocation."""

import importlib.metadata
import json
import os
from pathlib import Path
import sys

import pytest


class EnvironmentProof:
    def snapshot(self):
        return {"python": sys.executable, "prefix": sys.prefix,
                "packages": sorted((d.metadata["Name"], d.version)
                                   for d in importlib.metadata.distributions())}

    def pytest_sessionstart(self, session):
        self.before = self.snapshot()
        directory = Path(os.environ["AMD_DEPENDENCY_DIRECTORY"])
        (directory / "sglang-pd-environment-before.json").write_text(json.dumps(self.before, indent=2) + "\n")

    @pytest.hookimpl(trylast=True)
    def pytest_sessionfinish(self, session, exitstatus):
        after = self.snapshot()
        (Path(os.environ["AMD_DEPENDENCY_DIRECTORY"]) / "sglang-pd-environment-after.json").write_text(
            json.dumps(after, indent=2) + "\n")
        if after != self.before:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_configure(config):
    if os.environ.get("SMG_SGLANG_PD_PROOF") != "1":
        return
    if not any("e2e_test/router" in str(arg) for arg in config.args):
        return
    if os.environ.get("E2E_ENGINE") != "sglang" or os.environ.get("E2E_GPU_TIER") != "2":
        raise pytest.UsageError("SG2 observer is scoped to the one original SG2 row")
    if not os.environ.get("AMD_DEPENDENCY_DIRECTORY"):
        raise pytest.UsageError("same-job AMD dependency evidence directory missing")
    from .runtime import RuntimeProof

    config.pluginmanager.register(EnvironmentProof(), "smg-sglang-pd-environment")
    config.pluginmanager.register(RuntimeProof(), "smg-sglang-pd-runtime")
