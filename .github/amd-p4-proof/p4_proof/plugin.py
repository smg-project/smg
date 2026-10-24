"""Observe original calls only; the shared workflow owns all test execution."""

import importlib.util
import json
import os
from pathlib import Path

import pytest

from .process import argument, gateway_identity, python_log, require, worker_identity
from .ranks import model_cache, ranks, worker_log


def canonical(nodeid):
    return nodeid.removeprefix("e2e_test/")


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


class Proof:
    def __init__(self):
        self.directory = Path(os.environ["SMG_CI_RUNTIME_STATE_DIR"])
        self.records, self.errors, self.reports = [], [], {}
        self.recipe = json.loads(Path(__file__).with_name("cases.json").read_text())
        root = Path(os.environ["GITHUB_WORKSPACE"])
        self.native = json.loads((root / ".github/amd-shared-product/inputs.json").read_text())[
            "gateway_native"
        ]

    def save(self):
        (self.directory / "p4-proof.json").write_text(
            json.dumps(
                {
                    "records": self.records,
                    "errors": self.errors,
                    "reports": self.reports,
                    "scope": (
                        "Original calls with owned live TP4 rank witnesses; "
                        "no physical GPU exclusivity or per-rank request-ID attribution"
                    ),
                },
                indent=2,
            )
            + "\n"
        )

    def pytest_sessionstart(self, session):
        require(os.environ.get("SPT_NOENV") == "1", "marker preservation missing")
        require(
            os.environ.get("E2E_ENGINE") == "vllm" and os.environ.get("E2E_GPU_TIER") == "4",
            "wrong P4 lane",
        )
        require(self.native is not None, "actual producer native has not been bound")
        require(
            (self.directory / "plan.json").is_file(), "install-before constraint record missing"
        )

    @pytest.hookimpl(trylast=True)
    def pytest_collection_finish(self, session):
        require(
            sorted(canonical(item.nodeid) for item in session.items)
            == sorted(self.recipe["cases"]),
            "original 39 case selection differs",
        )

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_call(self, item):
        record = None
        try:
            from fixtures.markers import model_id_for_engine, resolve_class_marker
            from fixtures.mock_answers import _CLASS_WORKERS

            backend = item.funcargs["setup_backend"]
            model = model_id_for_engine(resolve_class_marker(item, "model"), "vllm")
            require(model in self.recipe["models"], "unexpected original model")
            require(len(_CLASS_WORKERS) == 1, "original single TP4 worker topology changed")
            worker, gateway = _CLASS_WORKERS[0], backend[3]
            gw = gateway_identity(gateway, self.native)
            wr = worker_identity(worker, model, gw)
            record = {
                "nodeid": canonical(item.nodeid),
                "model": model,
                "gateway": gw,
                "worker": wr,
                "cache": model_cache(model, argument(wr["argv"], "--model")),
            }
            record["ranks_before"] = ranks(worker_log(wr["log"]), wr["pid"])
        except Exception as error:
            self.errors.append({"nodeid": item.nodeid, "stage": "before", "error": str(error)})
        yield
        if record is not None:
            try:
                require(gateway_identity(gateway, self.native) == gw, "gateway changed during case")
                require(worker_identity(worker, model, gw) == wr, "worker changed during case")
                require(
                    model_cache(model, argument(wr["argv"], "--model")) == record["cache"],
                    "model cache identity changed",
                )
                witness = python_log(wr["log"])
                record["python"] = witness
                record["ranks_after"] = ranks(worker_log(wr["log"]), wr["pid"])
                require(
                    record["ranks_before"] == record["ranks_after"],
                    "TP rank identity changed during case",
                )
                self.records.append(record)
            except Exception as error:
                self.errors.append({"nodeid": item.nodeid, "stage": "after", "error": str(error)})
        self.save()

    def pytest_runtest_logreport(self, report):
        self.reports.setdefault(canonical(report.nodeid), []).append(
            {
                "phase": report.when,
                "outcome": report.outcome,
                "xfail": hasattr(report, "wasxfail"),
                "skip_reason": str(report.longrepr) if report.skipped else None,
            }
        )
        self.save()

    @pytest.hookimpl(trylast=True)
    def pytest_sessionfinish(self, session, exitstatus):
        try:
            require(not self.errors, "runtime observations failed")
            require(
                {row["model"] for row in self.records} == set(self.recipe["models"]),
                "a model lacks runtime proof",
            )
            require(
                set(self.reports) == set(self.recipe["cases"]), "missing terminal original case"
            )
            for nodeid, reports in self.reports.items():
                require(not any(row["xfail"] for row in reports), "xfail cannot pass")
                last = {row["phase"]: row for row in reports}
                require(
                    set(last) == {"setup", "call", "teardown"}, "incomplete original case lifecycle"
                )
                require(
                    last["setup"]["outcome"] == last["teardown"]["outcome"] == "passed",
                    "setup or teardown failed",
                )
                reason = self.recipe["call_skips"].get(nodeid)
                if reason:
                    require(
                        last["call"]["outcome"] == "skipped"
                        and reason in last["call"]["skip_reason"],
                        "original call skip differs",
                    )
                else:
                    require(last["call"]["outcome"] == "passed", "original call failed")
        except Exception as error:
            self.errors.append({"stage": "finish", "error": str(error)})
            session.exitstatus = pytest.ExitCode.TESTS_FAILED
        warnings = {}
        for path in {record["worker"]["log"] for record in self.records}:
            try:
                warnings[path] = [
                    line
                    for line in worker_log(path).splitlines()
                    if any(
                        word in line.lower()
                        for word in ["warning", "shutdown", "not destroy", "not released"]
                    )
                ]
            except (OSError, ValueError) as error:
                warnings[path] = {"observation_unavailable": str(error)}
        (self.directory / "shutdown-warnings.json").write_text(
            json.dumps(warnings, indent=2) + "\n"
        )
        self.save()


def pytest_configure(config):
    if not os.environ.get("SMG_CI_RUNTIME_STATE_DIR"):
        return
    e2e = any("e2e_test/chat_completions" in str(arg) for arg in config.args)
    units = any("grpc_servicer/tests" in str(arg) for arg in config.args)
    if e2e or units:
        config.pluginmanager.register(
            EnvironmentProof("pytest" if e2e else "unit"), "smg-p4-environment"
        )
    if e2e:
        config.pluginmanager.register(Proof(), "smg-p4-owned-runtime")
