"""Default-off exact original167 selection and actual same-case process records."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest

from .process import process_identity, require


def canonical(nodeid):
    return nodeid.removeprefix("e2e_test/")


def argument(argv, name):
    require(argv.count(name) == 1 and argv.index(name) + 1 < len(argv), "missing/duplicate " + name)
    return argv[argv.index(name) + 1]


class Environment:
    def __init__(self, root, directory, stage):
        self.directory, self.stage = directory, stage
        spec = importlib.util.spec_from_file_location("ts167_constraints", root / "scripts/ci_runtime_constraints.py")
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def pytest_sessionstart(self, session):
        self.module.verify(self.directory, self.stage + "-before")

    @pytest.hookimpl(trylast=True)
    def pytest_sessionfinish(self, session, exitstatus):
        try:
            self.module.verify(self.directory, self.stage + "-after")
        except Exception as error:
            (self.directory / (self.stage + "-error.json")).write_text(json.dumps({"error": str(error)}) + "\n")
            session.exitstatus = pytest.ExitCode.TESTS_FAILED


class Original167:
    def __init__(self, root, directory, lane):
        self.root, self.directory, self.lane = root, directory, lane
        package = Path(__file__).parent
        self.contract = json.loads((package / "contract.json").read_text())
        raw = (package / "cases.json").read_bytes()
        require(hashlib.sha256(raw).hexdigest() == self.contract["cases_sha256"], "case contract differs")
        self.cases = json.loads(raw)
        self.records, self.reports, self.errors = [], {}, []
        self.native = json.loads((root / ".github/amd-shared-product/inputs.json").read_text())["gateway_native"]

    def save(self):
        (self.directory / "ts-chat167-proof.json").write_text(json.dumps({"lane": self.lane,
            "revision": self.contract["revision"], "records": self.records, "reports": self.reports,
            "errors": self.errors, "scope": "Original same-case owned TP1 worker/gateway identity; not loaded weight-file or per-token kernel proof"}, indent=2) + "\n")

    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(self, config, items):
        selected = {canonical(item.nodeid): item for item in items if canonical(item.nodeid) in self.cases}
        require(set(selected) == set(self.cases), "an original167 ID is absent after real runtime selection")
        require(sum(canonical(item.nodeid) in self.cases for item in items) == 167, "duplicate original case")
        deselected = [item for item in items if canonical(item.nodeid) not in selected]
        items[:] = [selected[name] for name in self.cases]
        config.hook.pytest_deselected(items=deselected)
        for name, expected in self.contract["original_modules_sha256"].items():
            require(hashlib.sha256((self.root / name).read_bytes()).hexdigest() == expected, "original test module changed")

    def snapshot(self, item):
        from fixtures.mock_answers import _CLASS_WORKERS
        require(len(_CLASS_WORKERS) == 1, "regular original167 requires one worker")
        worker, gateway = _CLASS_WORKERS[0], item.funcargs["setup_backend"][3]
        require(worker.engine == "vllm" and worker.model_id == self.cases[canonical(item.nodeid)], "case/model/runtime differs")
        require(len(worker.gpu_ids) == 1 and worker.mode.value == self.contract["lanes"][self.lane]["connection"], "TP1/wire differs")
        gw = process_identity(gateway.process, "smg::router")
        wr = process_identity(worker.process, "VLLM::EngineCore")
        launch = wr["launch_argv_intent"]
        require(argument(gw["launch_argv_intent"], "--worker-urls") == worker.base_url, "gateway/worker URL identity differs")
        if self.lane == "rust":
            require(argument(launch, "-m") == "vllm.entrypoints.grpc_server", "Rust gRPC launcher differs")
            require(argument(launch, "--tensor-parallel-size") == "1", "Rust TP differs")
            actual = (Path("/proc") / str(wr["pid"]) / "environ").read_bytes().split(b"\0")
            require(b"SMG_VLLM_SERVICER_IMPL=rust" in actual, "actual worker is not Rust servicer")
            require(argument(launch, "--model") == argument(gw["launch_argv_intent"], "--model-path"), "model path differs")
        else:
            require(argument(launch, "-m") == "vllm.entrypoints.cli.main" and "--headless" in launch, "ZMQ engine launcher differs")
            require(argument(launch, "--data-parallel-size") == argument(launch, "--data-parallel-size-local") == "1", "ZMQ DP1 differs")
            require(launch[launch.index("serve") + 1] == argument(gw["launch_argv_intent"], "--model-path"), "ZMQ model path differs")
            require(argument(gw["launch_argv_intent"], "--backend") == "vllm" and worker.base_url.startswith("ipc://"), "ZMQ backend differs")
        maps = {Path(line.split(maxsplit=5)[5]).resolve() for line in
                (Path("/proc") / str(gw["pid"]) / "maps").read_text().splitlines()
                if len(line.split(maxsplit=5)) == 6 and "/smg_rs" in line}
        require(len(maps) == 1, "actual gateway native is missing/ambiguous")
        native = maps.pop()
        require(native.stat().st_size == self.native["bytes"] and hashlib.sha256(native.read_bytes()).hexdigest() == self.native["sha256"], "loaded gateway native differs")
        return {"gateway": gw, "worker": wr, "model": worker.model_id, "native": str(native), "native_sha256": self.native["sha256"], "model_path_intent": argument(gw["launch_argv_intent"], "--model-path")}

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_call(self, item):
        before = None
        try:
            before = self.snapshot(item)
        except Exception as error:
            self.errors.append({"nodeid": canonical(item.nodeid), "stage": "before", "error": str(error)})
        yield
        if before is not None:
            try:
                after = self.snapshot(item)
                require(before == after, "owned worker/gateway changed inside original case")
                self.records.append({"nodeid": canonical(item.nodeid), "before": before, "after": after})
            except Exception as error:
                self.errors.append({"nodeid": canonical(item.nodeid), "stage": "after", "error": str(error)})
        self.save()

    def pytest_runtest_logreport(self, report):
        self.reports.setdefault(canonical(report.nodeid), []).append({"phase": report.when, "outcome": report.outcome,
            "xfail": hasattr(report, "wasxfail"), "skip_reason": str(report.longrepr) if report.skipped else None})
        self.save()

    @pytest.hookimpl(trylast=True)
    def pytest_sessionfinish(self, session, exitstatus):
        try:
            require(not self.errors and set(self.reports) == set(self.cases), "original167 observations incomplete")
            passed = set()
            for name, reports in self.reports.items():
                require(not any(row["xfail"] for row in reports), "xfail is not acceptance")
                require(sum(row["phase"] == "setup" for row in reports) <= 3, "original rerun2 exceeded")
                last = {row["phase"]: row for row in reports}
                reason = self.contract["common_setup_skips"].get(name)
                if reason:
                    require(set(last) == {"setup", "teardown"} and last["setup"]["outcome"] == "skipped" and reason in last["setup"]["skip_reason"] and last["teardown"]["outcome"] == "passed", "original setup skip differs")
                else:
                    require(set(last) == {"setup", "call", "teardown"} and all(row["outcome"] == "passed" for row in last.values()), "original case failed/incomplete")
                    passed.add(name)
            require(len(passed) == 161 and {r["nodeid"] for r in self.records} >= passed, "missing passed-case process evidence")
        except Exception as error:
            self.errors.append({"stage": "finish", "error": str(error)})
            session.exitstatus = pytest.ExitCode.TESTS_FAILED
        self.save()


def pytest_configure(config):
    lane = os.environ.get("SMG_TS_CHAT167_LANE")
    if lane is None:
        return
    require(lane in {"rust", "zmq"} and os.environ.get("E2E_ENGINE") == "vllm", "unsupported explicit equivalent lane")
    root, directory = Path(os.environ["GITHUB_WORKSPACE"]), Path(os.environ["SMG_CI_RUNTIME_STATE_DIR"])
    require(os.environ.get("SPT_NOENV") == "1", "job marker preservation required")
    e2e = any("e2e_test/chat_completions" in str(arg) for arg in config.args)
    units = any("grpc_servicer/tests" in str(arg) for arg in config.args)
    if e2e or units:
        config.pluginmanager.register(Environment(root, directory, "pytest" if e2e else "unit"), "ts167-environment")
    if e2e:
        require(os.environ.get("E2E_AMD_VLLM_ALWAYS_GRAMMAR_EQUIVALENT") == "1", "original always case must remain selected")
        config.pluginmanager.register(Original167(root, directory, lane), "ts167-original-owned")
