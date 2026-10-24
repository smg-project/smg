"""Observe original SG2 fixtures and requests; never issue replacement requests."""

import functools
import hashlib
import json
import os
import threading
import time
from pathlib import Path

import pytest

from . import owned

def canonical(nodeid):
    return nodeid.removeprefix("e2e_test/")


def report_row(report):
    reason = None
    if report.skipped and isinstance(report.longrepr, tuple) and len(report.longrepr) == 3:
        text = report.longrepr[2]
        if isinstance(text, str) and text.startswith("Skipped: "):
            reason = text.removeprefix("Skipped: ")
    return {
        "phase": report.when,
        "outcome": report.outcome,
        "xfail": hasattr(report, "wasxfail"),
        "reason": reason,
        "longrepr": str(report.longrepr) if report.skipped else None,
    }


def verify_reports(recipe, reports):
    owned.require(set(reports) == set(recipe["cases"]), "missing terminal original case")
    for nodeid, rows in reports.items():
        owned.require(not any(row["xfail"] for row in rows), "xfail cannot pass")
        last = {row["phase"]: row for row in rows}
        reason = recipe["setup_skips"].get(nodeid)
        if reason:
            owned.require(set(last) == {"setup", "teardown"}, "original setup skip phase changed")
            owned.require(
                last["setup"]["outcome"] == "skipped" and reason == last["setup"]["reason"],
                "original setup skip differs",
            )
        else:
            owned.require(set(last) == {"setup", "call", "teardown"}, "incomplete original case")
            owned.require(last["setup"]["outcome"] == "passed", "original setup failed")
            call_reason = recipe.get("call_skips", {}).get(nodeid)
            if call_reason and last["call"]["outcome"] == "skipped":
                owned.require(
                    last["call"]["outcome"] == "skipped" and last["call"]["reason"] == call_reason,
                    "original call skip differs",
                )
            else:
                owned.require(last["call"]["outcome"] == "passed", "original case did not pass")
        owned.require(last["teardown"]["outcome"] == "passed", "original teardown failed")


class RuntimeProof:
    def __init__(self):
        self.selection = json.loads(Path(__file__).with_name("selection.json").read_text())
        owned.require(self.selection["status"] == "CLOSED_REAL_ORIGINAL_AND_CURRENT_COLLECTION",
                      "HOLD: real original/current SG collection is still pending")
        owned.require(isinstance(self.selection["current_cases"], list), "real SG case IDs missing")
        self.recipe = {"cases": self.selection["current_cases"],
                       "setup_skips": self.selection["setup_skips"],
                       "call_skips": self.selection["allowed_call_skips"]}
        self.directory = Path(os.environ["AMD_DEPENDENCY_DIRECTORY"]) / "sglang-pd-runtime"
        self.directory.mkdir()
        root = Path(os.environ["GITHUB_WORKSPACE"])
        self.native = json.loads((root / ".github/amd-shared-product/inputs.json").read_text())["gateway_native"]
        self.objects, self.logs, self.offsets = {}, {}, {}
        self.events, self.errors, self.reports, self.patches = [], [], {}, []
        self.nodeid, self.total_log_bytes = None, 0
        self.lock = threading.RLock()

    def record(self, row):
        with self.lock:
            owned.require(len(self.events) < 20000, "runtime event cap")
            row = {"nodeid": self.nodeid, "monotonic": time.monotonic(), **row}
            self.events.append(row)
            with (self.directory / "events.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")

    def capture_logs(self):
        for path in self.logs.values():
            try:
                info = path.stat()
                prior = self.offsets.get(str(path), (None, 0))
                start = prior[1] if prior[0] == info.st_ino and prior[1] <= info.st_size else 0
                with path.open("rb") as stream:
                    stream.seek(start)
                    raw = stream.read(16 * 1024**2 + 1)
                owned.require(len(raw) <= 16 * 1024**2, "log increment cap")
                owned.require(self.total_log_bytes + len(raw) <= 128 * 1024**2, "total log cap")
                if raw:
                    name = hashlib.sha256(str(path).encode()).hexdigest() + ".log"
                    with (self.directory / name).open("ab") as stream:
                        stream.write(raw)
                    self.total_log_bytes += len(raw)
                    self.record({"kind": "log", "source": str(path), "inode": info.st_ino,
                                 "start": start, "bytes": len(raw),
                                 "sha256": hashlib.sha256(raw).hexdigest(), "artifact": name})
                    self.offsets[str(path)] = (info.st_ino, start + len(raw))
            except FileNotFoundError:
                continue  # The original fixture can remove its temporary log after shutdown.

    def capture(self, obj, kind, stage):
        with self.lock:
            try:
                process = obj.process
                row = {"object": id(obj), "kind": kind, "stage": stage,
                       "state": "absent" if process is None else
                       "exited" if process.poll() is not None else "live"}
                if process is not None:
                    row.update(pid=process.pid, exit_code=process.poll())
                if row["state"] == "live":
                    row["identity"] = (owned.worker(obj) if kind == "worker" else
                                       owned.gateway(obj, self.native))
                    if kind == "worker":
                        row["transport_native"] = owned.transport_mappings(process)
                paths = [Path(obj.log_path)] if obj.log_path else []
                if kind == "gateway" and obj.log_dir:
                    paths.extend(Path(obj.log_dir).glob("smg*"))
                for path in paths:
                    if path.is_file():
                        self.logs[str(path)] = path
                self.record(row)
                self.capture_logs()
            except Exception as error:
                self.errors.append({"nodeid": self.nodeid, "stage": stage, "error": str(error)})

    def wrap(self, cls, method, kind):
        original = getattr(cls, method)

        @functools.wraps(original)
        def call(obj, *args, **kwargs):
            self.objects[id(obj)] = (obj, kind)
            self.capture(obj, kind, method + "-before")
            try:
                result = original(obj, *args, **kwargs)
                if method in {"add_worker", "remove_worker"}:
                    self.record({"kind": "registry_api", "operation": method,
                                 "URL": args[0] if args else kwargs.get("worker_url"),
                                 "role": kwargs.get("worker_type"), "result": str(result)[:2048]})
                return result
            except BaseException as error:
                self.record({"kind": kind, "stage": method + "-exception", "error": repr(error)[:2048]})
                raise
            finally:
                self.capture(obj, kind, method + "-after")

        setattr(cls, method, call)
        self.patches.append((cls, method, original))

    def snapshot(self, stage):
        for obj, kind in list(self.objects.values()):
            self.capture(obj, kind, stage)

    def pytest_sessionstart(self, session):
        from infra.gateway import Gateway
        from infra.worker import Worker

        owned.require(self.native is not None, "actual producer native not bound")
        owned.require(os.environ.get("SPT_NOENV") == "1", "marker preservation missing")
        for cls, kind, methods in (
            (Worker, "worker", ("start", "wait_ready", "stop", "kill_engine")),
            (Gateway, "gateway", ("start", "shutdown", "add_worker", "remove_worker")),
        ):
            for method in methods:
                self.wrap(cls, method, kind)

    def pytest_collection_finish(self, session):
        owned.require(sorted(canonical(x.nodeid) for x in session.items) == sorted(self.recipe["cases"]),
                      "original SG selection changed")

    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_runtest_protocol(self, item, nextitem):
        self.nodeid = canonical(item.nodeid)
        self.snapshot("case-before-setup")
        yield
        self.snapshot("case-after-teardown")
        self.nodeid = None

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_call(self, item):
        self.snapshot("call-before")
        yield
        self.snapshot("call-after")

    def pytest_runtest_logreport(self, report):
        self.reports.setdefault(canonical(report.nodeid), []).append(report_row(report))

    def verify_case_ownership(self):
        for nodeid, expected in self.selection["case_witnesses"].items():
            if nodeid in self.recipe["setup_skips"]:
                continue
            rows = [e for e in self.events if e["nodeid"] == nodeid and e.get("identity")]
            owned.require(any(e["kind"] == "gateway" for e in rows), "case lacks owned native gateway")
            for want in expected:
                workers = [e for e in rows if e["kind"] == "worker" and all(
                    e["identity"].get(k) == want[k] for k in ("role", "mode", "model_id"))]
                identities = {(e["identity"]["pid"], e["identity"]["start_ticks"]) for e in workers}
                owned.require(len(identities) >= want["min_instances"], "case lacks original role ownership")
                if want["role"] in {"prefill", "decode"}:
                    owned.require(any("mooncake" in path.lower() for e in workers
                        for child in e.get("transport_native", []) for path in child["loaded_native"]),
                        "case role lacks actually mapped Mooncake native")

    @pytest.hookimpl(trylast=True)
    def pytest_sessionfinish(self, session, exitstatus):
        self.snapshot("session-finish")
        for cls, method, original in reversed(self.patches):
            setattr(cls, method, original)
        try:
            owned.require(not self.errors, "runtime observations failed")
            verify_reports(self.recipe, self.reports)
            self.verify_case_ownership()
        except Exception as error:
            self.errors.append({"stage": "finish", "error": str(error)})
            session.exitstatus = pytest.ExitCode.TESTS_FAILED
        (self.directory / "summary.json").write_text(json.dumps({
            "errors": self.errors, "reports": self.reports, "event_count": len(self.events),
            "accepted": None, "scope": "Original SG fixtures, role-native ownership and reports only; "
            "original request oracles remain authoritative. No transfer bytes/ACK, peer memory, "
            "physical exclusivity or HBM release proof is inferred."}, indent=2) + "\n")


