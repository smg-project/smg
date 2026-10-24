"""Default-off exact original173 selection and actual same-case process records."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest

from .process import process_identity, require
from .launch import EXTENSION, install, restore
from .extra_proof import Burst, trace_artifacts


def canonical(nodeid):
    return nodeid.removeprefix("e2e_test/")


def argument(argv, name):
    require(argv.count(name) == 1 and argv.index(name) + 1 < len(argv), "missing/duplicate " + name)
    return argv[argv.index(name) + 1]


class Environment:
    def __init__(self, root, directory, stage):
        self.directory, self.stage = directory, stage
        spec = importlib.util.spec_from_file_location("ts173_constraints", root / "scripts/ci_runtime_constraints.py")
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


NATIVE_HASHES = {}


def native_mapping(pid, expected):
    paths = {Path(line.split(maxsplit=5)[5]).resolve() for line in
             (Path("/proc") / str(pid) / "maps").read_text().splitlines()
             if len(line.split(maxsplit=5)) == 6 and "/smg_rs" in line}
    require(len(paths) == 1, "actual native mapping is missing/ambiguous")
    path = paths.pop()
    stat = path.stat()
    require(stat.st_size == expected["bytes"], "loaded native size differs")
    key = (str(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    if key not in NATIVE_HASHES:
        with path.open("rb") as stream:
            NATIVE_HASHES[key] = hashlib.file_digest(stream, "sha256").hexdigest()
    require(NATIVE_HASHES[key] == expected["sha256"], "loaded native hash differs")
    return str(path)


class Original173:
    def __init__(self, root, directory):
        self.root, self.directory = root, directory
        package = Path(__file__).parent
        self.contract = json.loads((package / "contract.json").read_text())
        raw = (package / "cases.json").read_bytes()
        require(hashlib.sha256(raw).hexdigest() == self.contract["cases_sha256"], "case contract differs")
        self.cases = json.loads(raw)
        self.records, self.reports, self.errors = [], {}, []
        self.native = json.loads((root / ".github/amd-shared-product/inputs.json").read_text())["gateway_native"]
        self.launch_original = install()

    def save(self):
        (self.directory / "ts-python173-proof.json").write_text(json.dumps({"lane": "python",
            "revision": self.contract["revision"], "records": self.records, "reports": self.reports,
            "errors": self.errors, "scope": "Original same-case owned Python TP1 worker/gateway; trace event files are checked only for the original profile case; loaded weights and per-token kernel identity remain outside scope"}, indent=2) + "\n")

    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(self, config, items):
        selected = {canonical(item.nodeid): item for item in items if canonical(item.nodeid) in self.cases}
        require(set(selected) == set(self.cases), "an original173 ID is absent after real runtime selection")
        require(sum(canonical(item.nodeid) in self.cases for item in items) == 173, "duplicate original case")
        deselected = [item for item in items if canonical(item.nodeid) not in selected]
        items[:] = [item for item in items if canonical(item.nodeid) in selected]
        config.hook.pytest_deselected(items=deselected)
        for name, expected in self.contract["original_modules_sha256"].items():
            require(hashlib.sha256((self.root / name).read_bytes()).hexdigest() == expected, "original test module changed")

    def snapshot(self, item):
        from fixtures.mock_answers import _CLASS_WORKERS
        require(len(_CLASS_WORKERS) == 1, "each selected original regular fixture requires one worker")
        worker, gateway = _CLASS_WORKERS[0], item.funcargs["setup_backend"][3]
        require(worker.engine == "vllm" and worker.model_id == self.cases[canonical(item.nodeid)], "case/model/runtime differs")
        require(len(worker.gpu_ids) == 1 and worker.mode.value == self.contract["connection"], "TP1/wire differs")
        gw = process_identity(gateway.process, "smg::router")
        wr = process_identity(worker.process, "VLLM::EngineCore")
        launch = wr["launch_argv_intent"]
        require(argument(gw["launch_argv_intent"], "--worker-urls") == worker.base_url, "gateway/worker URL identity differs")
        require("--backend" not in gw["launch_argv_intent"], "original Python auto-backend changed")
        require(argument(launch, "-m") == "vllm.entrypoints.grpc_server", "Python gRPC launcher differs")
        require(argument(launch, "--tensor-parallel-size") == "1", "Python TP differs")
        require(argument(launch, "--worker-extension-cls") == EXTENSION, "request-profile extension differs")
        require(argument(launch, "--servicer-impl") == "python", "Python servicer flag differs")
        actual = (Path("/proc") / str(wr["pid"]) / "environ").read_bytes().split(b"\0")
        require(b"SMG_VLLM_SERVICER_IMPL=rust" not in actual, "actual worker selected Rust")
        require(b"SMG_VLLM_REQUEST_PROFILE=1" in actual, "actual worker lacks profile opt-in")
        require(argument(launch, "--model") == argument(gw["launch_argv_intent"], "--model-path"), "model path differs")
        native = native_mapping(gw["pid"], self.native)
        worker_native = None
        return {"gateway": gw, "worker": wr, "model": worker.model_id, "native": native, "worker_native": worker_native, "native_sha256": self.native["sha256"], "model_path_intent": argument(gw["launch_argv_intent"], "--model-path")}

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_call(self, item):
        before, record, burst = None, None, None
        name = canonical(item.nodeid)
        try:
            before = self.snapshot(item)
            if "::test_start_and_stop_profile_produces_traces[" in name:
                require(not (item.funcargs["tmp_path"] / "traces").exists(), "profile directory was not fresh")
            if "::test_report_reflects_in_flight_requests[" in name:
                burst = Burst(item.funcargs["setup_backend"][2])
                burst.start()
        except Exception as error:
            self.errors.append({"nodeid": canonical(item.nodeid), "stage": "before", "error": str(error)})
        try:
            yield
        finally:
            if burst is not None:
                try:
                    burst_result = burst.finish()
                except Exception as error:
                    self.errors.append({"nodeid": name, "stage": "original-load-burst", "error": str(error)})
        if before is not None:
            try:
                after = self.snapshot(item)
                require(before == after, "owned worker/gateway changed inside original case")
                record = {"nodeid": canonical(item.nodeid), "before": before, "after": after}
                if "::test_start_and_stop_profile_produces_traces[" in name:
                    record["profile"] = trace_artifacts(item.funcargs["tmp_path"] / "traces")
                if burst is not None:
                    record["original_load_burst"] = burst_result
                self.records.append(record)
            except Exception as error:
                self.errors.append({"nodeid": canonical(item.nodeid), "stage": "after", "error": str(error)})
        if record is not None:
            try:
                from fixtures.mock_answers import _CLASS_WORKERS
                log = Path(_CLASS_WORKERS[0].log_path)
                with log.open("rb") as stream:
                    raw = stream.read(16 * 1024**2 + 1)
                require(0 < len(raw) <= 16 * 1024**2, "Python worker log cap")
                lines = [line for line in raw.decode(errors="replace").splitlines() if "VllmEngineServicer initialized (" in line and ("pid=" + str(record["after"]["worker"]["pid"]) + ")") in line]
                require(lines, "same-PID actual Python constructor log missing")
                record["Python_constructor"] = {"path": str(log), "sha256_observed": hashlib.sha256(raw).hexdigest(), "lines": lines[:8]}
            except Exception as error:
                self.errors.append({"nodeid": canonical(item.nodeid), "stage": "Python-constructor", "error": str(error)})
        self.save()

    def pytest_runtest_logreport(self, report):
        self.reports.setdefault(canonical(report.nodeid), []).append({"phase": report.when, "outcome": report.outcome,
            "xfail": hasattr(report, "wasxfail"), "skip_reason": str(report.longrepr) if report.skipped else None})
        self.save()

    @pytest.hookimpl(trylast=True)
    def pytest_sessionfinish(self, session, exitstatus):
        try:
            require(not self.errors and set(self.reports) == set(self.cases), "original173 observations incomplete")
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
            require(len(passed) == 167 and {r["nodeid"] for r in self.records} >= passed, "missing passed-case process evidence")
        except Exception as error:
            self.errors.append({"stage": "finish", "error": str(error)})
            session.exitstatus = pytest.ExitCode.TESTS_FAILED
        self.save()
        restore(self.launch_original)


def pytest_configure(config):
    lane = os.environ.get("SMG_TS_PYTHON173_LANE")
    if lane is None:
        return
    require(lane == "python" and os.environ.get("E2E_ENGINE") == "vllm", "unsupported explicit equivalent lane")
    root, directory = Path(os.environ["GITHUB_WORKSPACE"]), Path(os.environ["SMG_CI_RUNTIME_STATE_DIR"])
    require(os.environ.get("SPT_NOENV") == "1", "job marker preservation required")
    e2e = any("e2e_test/chat_completions" in str(arg) for arg in config.args)
    units = any("grpc_servicer/tests" in str(arg) for arg in config.args)
    if e2e or units:
        config.pluginmanager.register(Environment(root, directory, "pytest" if e2e else "unit"), "ts173-environment")
    if e2e:
        require(os.environ.get("E2E_AMD_VLLM_ALWAYS_GRAMMAR_EQUIVALENT") == "1", "original always case must remain selected")
        config.pluginmanager.register(Original173(root, directory), "ts173-original-owned")
