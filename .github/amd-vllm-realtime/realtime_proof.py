"""Observe the original local HTTP worker and IGW around every realtime case."""

import hashlib
import json
import os
import urllib.parse
import urllib.request
from dataclasses import asdict
from pathlib import Path

import pytest
from reuse import NoRedirect, require

NATIVE_HASHES = {}


def process_identity(process, title=None):
    require(process is not None and process.poll() is None, "owned process not alive")
    root = Path("/proc") / str(process.pid)
    fields = (root / "stat").read_text().rsplit(")", 1)[1].split()
    require(int(fields[1]) == os.getpid(), "process is not this pytest harness child")
    argv = [x.decode() for x in (root / "cmdline").read_bytes().split(b"\0") if x]
    launch = list(process.args)
    require(argv == launch or (title is not None and argv == [title]), "actual argv/title differs")
    marker = ("E2E_JOB_MARKER=" + os.environ["E2E_JOB_MARKER"]).encode()
    require(marker in (root / "environ").read_bytes().split(b"\0"), "process marker differs")
    return {
        "pid": process.pid,
        "parent_pid": int(fields[1]),
        "start_ticks": fields[19],
        "argv": argv,
        "launch_argv_intent": launch,
        "marker_visible_in_proc": True,
    }


def argument(argv, name):
    require(argv.count(name) == 1 and argv.index(name) + 1 < len(argv), "missing/duplicate " + name)
    return argv[argv.index(name) + 1]


def local_json(base, suffix):
    url = urllib.parse.urlsplit(base)
    require(
        url.scheme == "http"
        and url.hostname in ("127.0.0.1", "localhost", "::1")
        and url.port
        and not url.username
        and not url.password,
        "nonlocal HTTP endpoint",
    )
    request = urllib.request.Request(base + suffix)
    with urllib.request.build_opener(NoRedirect(), urllib.request.ProxyHandler({})).open(
        request, timeout=5
    ) as r:
        raw = r.read(1024 * 1024 + 1)
    require(len(raw) <= 1024 * 1024, "HTTP observation cap")
    return json.loads(raw)


def gateway_identity(gateway, native):
    value = process_identity(gateway.process, "smg::router")
    launch = value["launch_argv_intent"]
    require(argument(launch, "-m") == "smg.launch_router", "wrong gateway launcher")
    require("--enable-igw" in launch and "--cloud-backend" not in launch, "not local IGW")
    require("--backend" not in launch and not gateway.cloud_mode, "original HTTP backend changed")
    mapped = set()
    for line in (Path("/proc") / str(value["pid"]) / "maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and "x" in fields[1] and Path(fields[5]).name.startswith("smg_rs"):
            mapped.add(Path(fields[5]).resolve())
    require(len(mapped) == 1, "actual gateway native missing/ambiguous")
    path = mapped.pop()
    info = path.stat()
    key = (str(path), info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
    if key not in NATIVE_HASHES:
        with path.open("rb") as stream:
            NATIVE_HASHES[key] = hashlib.file_digest(stream, "sha256").hexdigest()
    require(NATIVE_HASHES[key] == native["sha256"], "loaded gateway native differs")
    return {**value, "native": str(path), "native_sha256": NATIVE_HASHES[key]}


def http_arguments(argv, manifest):
    require(argument(argv, "--model") == manifest["model_id"], "unpinned model alias")
    require(argument(argv, "-m") == "vllm.entrypoints.openai.api_server", "not HTTP API server")
    require(argument(argv, "--revision") == manifest["revision"], "ASR revision not pinned")
    require(argument(argv, "--served-model-name") == manifest["model_id"], "served model changed")
    require("Qwen3ASRRealtimeGeneration" in argument(argv, "--hf-overrides"), "wrong architecture")
    require(argument(argv, "--mm-encoder-attn-backend") == "TORCH_SDPA", "encoder backend changed")
    require("--enforce-eager" in argv, "original eager mode missing")


def worker_identity(worker, manifest):
    require(worker.engine == "vllm" and worker.mode.value == "http", "not local vLLM HTTP")
    require(worker.model_id == manifest["model_id"], "ASR model changed")
    value = process_identity(worker.process)
    argv = value["argv"]
    http_arguments(argv, manifest)
    models = local_json(worker.base_url, "/v1/models")
    require(worker.model_id in {m["id"] for m in models["data"]}, "worker not serving original ID")
    return {
        **value,
        "url": worker.base_url,
        "model_ids": sorted(m["id"] for m in models["data"]),
        "version": local_json(worker.base_url, "/version"),
        "log": worker.log_path,
    }


class RealtimeProof:
    def __init__(self, output, recipe):
        self.output, self.recipe = output, recipe
        self.records, self.errors, self.workers = [], [], []
        self.native = json.loads(Path(__file__).with_name("inputs.json").read_text())[
            "gateway_native"
        ]
        self.manifest = json.loads(Path(__file__).with_name("model-cache.json").read_text())

    def pytest_sessionstart(self):
        import infra

        self.original = infra.start_workers

        def start(model, *args, **kwargs):
            require(model == self.manifest["model_id"] and not args, "unexpected worker request")
            require(
                kwargs.get("engine") == "vllm" and kwargs.get("count") == 1,
                "original worker topology changed",
            )
            require(
                kwargs.get("mode").value == "http" and not kwargs.get("extra_engine_args"),
                "original HTTP arguments changed",
            )
            kwargs["extra_engine_args"] = ["--revision", self.manifest["revision"]]
            self.workers = self.original(model, **kwargs)
            return self.workers

        infra.start_workers = start

    def pytest_sessionfinish(self):
        import infra

        infra.start_workers = self.original

    def save(self):
        (self.output / "http-worker-proof.json").write_text(
            json.dumps({"records": self.records, "errors": self.errors}, indent=2) + "\n"
        )

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_call(self, item):
        record = None
        try:
            gateway = item.funcargs["realtime_gateway"]
            require(len(self.workers) == 1, "original single-worker topology changed")
            worker = self.workers[0]
            registry = gateway.list_workers(strict=True)
            require(
                len(registry) == 1 and registry[0].url == worker.base_url,
                "IGW is not bound to the owned HTTP worker",
            )
            record = {
                "nodeid": item.nodeid,
                "gateway": gateway_identity(gateway, self.native),
                "worker": worker_identity(worker, self.manifest),
                "registry": [asdict(w) for w in registry],
            }
        except Exception as error:
            self.errors.append({"nodeid": item.nodeid, "phase": "before", "error": str(error)})
        yield
        if record is not None:
            try:
                require(
                    gateway_identity(gateway, self.native) == record["gateway"], "gateway changed"
                )
                require(
                    worker_identity(worker, self.manifest) == record["worker"], "worker changed"
                )
                self.records.append(record)
            except Exception as error:
                self.errors.append({"nodeid": item.nodeid, "phase": "after", "error": str(error)})
        self.save()

    def verify(self):
        self.save()
        require(not self.errors, "local worker proof failed: " + repr(self.errors))
        require(
            len(self.records) == len(self.recipe["cases"])
            and {r["nodeid"] for r in self.records} == set(self.recipe["cases"]),
            "every original realtime case needs actual worker proof",
        )
