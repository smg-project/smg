"""Observe environment continuity; never replace PD process/transfer proof."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest


def error_context(error, stage):
    environment = {}
    for name in ("PYTHONPATH", "LD_LIBRARY_PATH"):
        value = os.environ.get(name)
        raw = (value or "").encode(errors="backslashreplace")
        environment[name] = {
            "present": value is not None,
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "value_prefix": raw[:8192].decode(errors="backslashreplace"),
            "truncated": len(raw) > 8192,
        }
    return {"error": str(error), "stage": stage, "private_environment": environment}


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
            from .private_wheel import verify

            verify(self.stage + "-" + suffix)
        except Exception as error:
            (directory / (self.stage + "-" + suffix + "-error.json")).write_text(
                json.dumps(error_context(error, self.stage + "-" + suffix)) + "\n"
            )
            raise

    def pytest_sessionstart(self, session):
        self.check("before")

    @pytest.hookimpl(trylast=True)
    def pytest_sessionfinish(self, session, exitstatus):
        try:
            self.check("after")
        except Exception:
            if not session.exitstatus:
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
