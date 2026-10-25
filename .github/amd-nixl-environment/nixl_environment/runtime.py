"""Observe existing role fixtures and original requests without issuing new ones."""

import functools
import hashlib
import json
import os
import time
from pathlib import Path

import pytest

from . import canary, mooncake, mooncake_topology, owned, passive_flush


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
            if call_reason:
                owned.require(
                    last["call"]["outcome"] == "skipped" and last["call"]["reason"] == call_reason,
                    "original call skip differs",
                )
            else:
                owned.require(last["call"]["outcome"] == "passed", "original case did not pass")
        owned.require(last["teardown"]["outcome"] == "passed", "original teardown failed")


def verify_mixed_native(recipe, events):
    mixed = [nodeid for nodeid in recipe["cases"] if "::TestPDMixedTransport::" in nodeid]
    for nodeid in mixed:
        seen = set()
        for event in events:
            identity = event.get("identity", {})
            config = identity.get("kv_config")
            if event["nodeid"] != nodeid or not config:
                continue
            word = "mooncake" if config["kv_connector"] == "MooncakeConnector" else "nixl"
            if any(
                word in path.lower()
                for child in event.get("transport_native", [])
                for path in child["loaded_native"]
            ):
                seen.add((identity["role"], word))
        owned.require(
            seen
            >= {
                (role, backend)
                for role in ("prefill", "decode")
                for backend in ("nixl", "mooncake")
            },
            "original mixed case lacks role-bound native connector mappings",
        )


class RuntimeProof:
    def __init__(self):
        self.directory = Path(os.environ["SMG_CI_RUNTIME_STATE_DIR"]) / "mooncake-runtime"
        self.directory.mkdir()
        self.recipe = json.loads(Path(__file__).with_name("cases.json").read_text())[
            os.environ["E2E_GPU_TIER"]
        ]
        owned.require(os.environ["E2E_GPU_TIER"] == "4", "original Mooncake4 tier required")
        owned.require(
            all(
                os.environ[name] == "mooncake"
                for name in ("E2E_VLLM_KV_BACKEND", "E2E_KV_BACKEND", "E2E_VLLM_EXTRA_KV_BACKENDS")
            ),
            "original Mooncake4 main/global/extra backend configuration differs",
        )
        root = Path(os.environ["GITHUB_WORKSPACE"])
        inputs = json.loads((root / ".github/amd-shared-product/inputs.json").read_text())
        self.native = inputs["gateway_native"]
        self.passive_flush = os.environ.get("SMG_CI_MC4_PERIODIC_FLUSH") == "1"
        if self.passive_flush:
            owned.require(os.environ.get("SMG_CI_PRIVATE_MOONCAKE") == "1" and all(
                inputs.get(key) is True for key in
                ("ready", "run_budget_admitted", "artifacts_pinned")),
                "passive flush requires admitted private AMD diagnostic")
        self.objects, self.logs, self.offsets = {}, {}, {}
        self.events, self.errors, self.reports, self.patches = [], [], {}, []
        self.nodeid = None

    def record(self, row):
        if len(self.events) >= 20000:
            if not self.errors or self.errors[-1].get("error") != "runtime event cap":
                self.errors.append({"nodeid": self.nodeid, "error": "runtime event cap"})
            return
        self.events.append({"nodeid": self.nodeid, "monotonic": time.monotonic(), **row})
        with (self.directory / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(self.events[-1]) + "\n")

    def capture(self, obj, kind, stage):
        try:
            process = obj.process
            row = {
                "object": id(obj),
                "kind": kind,
                "stage": stage,
                "state": "absent"
                if process is None
                else "exited"
                if process.poll() is not None
                else "live",
            }
            if process is not None:
                row.update(pid=process.pid, exit_code=process.poll())
            if row["state"] == "live":
                row["identity"] = (
                    owned.worker(obj) if kind == "worker" else owned.gateway(obj, self.native)
                )
                if kind == "worker":
                    row["transport_native"] = owned.transport_mappings(process)
                    config = row["identity"]["kv_config"]
                    if config and stage in {"wait_ready-after", "call-before", "call-after"}:
                        word = (
                            "mooncake" if config["kv_connector"] == "MooncakeConnector" else "nixl"
                        )
                        owned.require(
                            any(
                                word in path.lower()
                                for child in row["transport_native"]
                                for path in child["loaded_native"]
                            ),
                            "ready role lacks actual " + word + " native mapping",
                        )
                        if word == "mooncake":
                            mooncake.require_engine_mapped(row["transport_native"])
            paths = []
            if obj.log_path:
                paths.append(Path(obj.log_path))
            if kind == "gateway" and obj.log_dir:
                paths.extend(Path(obj.log_dir).glob("smg*"))
            for path in paths:
                if path.is_file():
                    self.logs[str(path)] = path
            row["log_sources"] = [str(path) for path in paths if path.is_file()]
            self.record(row)
        except Exception as error:
            self.errors.append({"nodeid": self.nodeid, "stage": stage, "error": str(error)})
        self.capture_logs()

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
                if raw:
                    name = hashlib.sha256(str(path).encode()).hexdigest() + ".log"
                    with (self.directory / name).open("ab") as stream:
                        stream.write(raw)
                    self.record(
                        {
                            "kind": "log",
                            "source": str(path),
                            "inode": info.st_ino,
                            "start": start,
                            "bytes": len(raw),
                            "sha256": hashlib.sha256(raw).hexdigest(),
                            "artifact": name,
                        }
                    )
                    self.offsets[str(path)] = (info.st_ino, start + len(raw))
            except FileNotFoundError:
                continue  # A fixture may remove its temporary stdout on shutdown.
            except Exception as error:
                self.errors.append({"nodeid": self.nodeid, "stage": "logs", "error": str(error)})

    def wrap(self, cls, method, kind):
        original = getattr(cls, method)

        @functools.wraps(original)
        def call(obj, *args, **kwargs):
            self.objects[id(obj)] = (obj, kind)
            self.capture(obj, kind, method + "-before")
            try:
                result = original(obj, *args, **kwargs)
                if method in {"add_worker", "remove_worker"}:
                    self.record(
                        {
                            "kind": "registry_api",
                            "operation": method,
                            "URL": args[0] if args else kwargs.get("worker_url"),
                            "role": kwargs.get("worker_type"),
                            "result": str(result)[:2048],
                        }
                    )
                return result
            except BaseException as error:
                self.record(
                    {"kind": kind, "stage": method + "-exception", "error": repr(error)[:2048]}
                )
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
        canary.capture(self.directory, os.environ["E2E_VLLM_KV_BACKEND"])
        nixl = self.directory / "mixed-nixl-canary"
        nixl.mkdir()
        canary.capture(nixl, "nixl")
        for cls, kind, names in [
            (Worker, "worker", ("start", "wait_ready", "stop", "kill_engine")),
            (Gateway, "gateway", ("start", "shutdown", "add_worker", "remove_worker")),
        ]:
            for method in names:
                self.wrap(cls, method, kind)

    def pytest_collection_finish(self, session):
        owned.require(
            sorted(canonical(x.nodeid) for x in session.items) == sorted(self.recipe["cases"]),
            "original Mooncake4 topology selection changed",
        )

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_protocol(self, item, nextitem):
        self.nodeid = canonical(item.nodeid)
        self.snapshot("case-before-setup")
        yield
        self.snapshot("case-after-teardown")
        self.nodeid = None

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_call(self, item):
        self.snapshot("call-before")
        outcome = yield
        self.snapshot("call-after")
        if self.passive_flush and canonical(item.nodeid) == mooncake_topology.CASE and outcome.excinfo is None:
            try:
                passive_flush.observe(self)
            except Exception as error:
                self.errors.append({"nodeid": self.nodeid, "stage": "passive-flush", "error": str(error)})

    def pytest_runtest_logreport(self, report):
        self.reports.setdefault(canonical(report.nodeid), []).append(report_row(report))

    @pytest.hookimpl(trylast=True)
    def pytest_sessionfinish(self, session, exitstatus):
        self.snapshot("session-finish")
        for cls, method, original in reversed(self.patches):
            setattr(cls, method, original)
        try:
            owned.require(not self.errors, "runtime observations failed")
            verify_reports(self.recipe, self.reports)
            verify_mixed_native(self.recipe, self.events)
            mooncake_topology.verify(self.recipe, self.events, self.directory)
            for nodeid in set(self.recipe["cases"]) - set(self.recipe["setup_skips"]):
                events = [e for e in self.events if e["nodeid"] == nodeid and e.get("identity")]
                owned.require(
                    {e["kind"] for e in events} >= {"worker", "gateway"},
                    "original case lacks owned worker/gateway proof",
                )
        except Exception as error:
            self.errors.append({"stage": "finish", "error": str(error)})
            if not session.exitstatus:
                session.exitstatus = pytest.ExitCode.TESTS_FAILED
        (self.directory / "summary.json").write_text(
            json.dumps(
                {
                    "errors": self.errors,
                    "reports": self.reports,
                    "event_count": len(self.events),
                    "accepted": None,
                    "scope": (
                        "Observed original fixtures/requests and terminal phases; "
                        "native imports and argv are not KV bytes or peer/HBM release proof"
                    ),
                },
                indent=2,
            )
            + "\n"
        )
