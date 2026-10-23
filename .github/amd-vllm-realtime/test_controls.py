"""CPU controls for new realtime boundaries; no engine, model, or provider is run."""

import copy
import hashlib
import importlib
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
preflight = importlib.import_module("preflight")
producer = importlib.import_module("producer")
proof = importlib.import_module("realtime_proof")
reuse = importlib.import_module("reuse")
Reports = importlib.import_module("run_tests").Reports


def test_actual_successful_build_and_every_step_fail_closed():
    evidence = Path(os.environ["REALTIME_PRODUCER_EVIDENCE"])
    inputs = json.loads((HERE / "inputs.json").read_text())
    run = json.loads((evidence / "run.json").read_text())
    jobs = json.loads((evidence / "jobs.json").read_text())["jobs"]
    job = next(j for j in jobs if j["id"] == inputs["build_job_id"])
    producer.verify(run, job, inputs)
    for index in range(len(job["steps"])):
        bad = copy.deepcopy(job)
        bad["steps"][index]["conclusion"] = "cancelled"
        with pytest.raises(ValueError):
            producer.verify(run, bad, inputs)
    for field, value in [("head_sha", "wrong"), ("run_attempt", 2), ("conclusion", "failure")]:
        with pytest.raises(ValueError):
            producer.verify(run, {**job, field: value}, inputs)


def test_actual_four_archives_and_tamper():
    directory = Path(os.environ["REALTIME_PRODUCER_EVIDENCE"])
    inputs = json.loads((HERE / "inputs.json").read_text())
    for artifact in inputs["artifacts"]:
        raw = (directory / f"{artifact['id']}.zip").read_bytes()
        assert set(reuse.unpack(raw, artifact)) == set(artifact["files"])
        with pytest.raises(ValueError):
            reuse.unpack(raw[:-1] + bytes([raw[-1] ^ 1]), artifact)


@pytest.mark.parametrize("ready,budget", [(False, False), (False, True), (True, False)])
def test_closed_flags_reject_before_reuse_api(tmp_path, monkeypatch, ready, budget):
    path = tmp_path / "inputs.json"
    path.write_text(json.dumps({"ready": ready, "run_budget_admitted": budget}))
    monkeypatch.setattr(reuse, "INPUTS", path)
    monkeypatch.setattr(reuse, "api_get", lambda *_: pytest.fail("unexpected API call"))
    with pytest.raises(ValueError, match="not admitted"):
        reuse.main()


def test_real_owned_child_requires_marker(monkeypatch):
    monkeypatch.setenv("E2E_JOB_MARKER", "realtime-control")
    for present in (True, False):
        environment = dict(os.environ)
        if not present:
            environment.pop("E2E_JOB_MARKER")
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; print('ready', flush=True); time.sleep(20)"],
            env=environment,
            stdout=subprocess.PIPE,
        )
        try:
            assert child.stdout.readline() == b"ready\n"
            if present:
                value = proof.process_identity(child)
                assert value["pid"] == child.pid and value["parent_pid"] == os.getpid()
            else:
                with pytest.raises(ValueError, match="marker"):
                    proof.process_identity(child)
        finally:
            child.terminate()
            child.wait(timeout=5)


def test_local_http_response_and_redirect_rejection():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/redirect":
                self.send_response(302)
                self.send_header("Location", "https://example.invalid/never-contact")
                self.end_headers()
            else:
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"cpu_control":true}')

        def log_message(self, *_):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        assert proof.local_json(base, "/models") == {"cpu_control": True}
        with pytest.raises(Exception, match="302"):
            proof.local_json(base, "/redirect")
        with pytest.raises(ValueError, match="nonlocal"):
            proof.local_json("https://example.invalid", "/models")
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_complete_pinned_snapshot_and_missing_shard(tmp_path):
    model, revision = "test/asr", "a" * 40
    snapshot = tmp_path / "models--test--asr" / "snapshots" / revision
    header = json.dumps({"x": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}}).encode()
    files = {
        "config.json": b"{}",
        "vocab.json": b"{}",
        "merges.txt": b"cpu",
        "model.safetensors": len(header).to_bytes(8, "little") + header + b"x",
    }
    manifest = {
        "model_id": model,
        "revision": revision,
        "config_sha256": hashlib.sha256(b"{}").hexdigest(),
        "runtime_files": [{"name": n, "bytes": len(v)} for n, v in files.items()],
    }
    calls = []

    def download(**kwargs):
        assert kwargs["revision"] == revision and set(kwargs["allow_patterns"]) == set(files)
        calls.append(kwargs)
        snapshot.mkdir(parents=True)
        for name, raw in files.items():
            (snapshot / name).write_bytes(raw)
        return str(snapshot)

    result = preflight.prepare(manifest, tmp_path, download)
    assert result["snapshot"] == revision and len(calls) == 1
    preflight.prepare(manifest, tmp_path, lambda **_: pytest.fail("complete cache downloaded"))
    (snapshot / "model.safetensors").unlink()
    with pytest.raises(ValueError, match="runtime file size"):
        preflight.prepare(manifest, tmp_path, lambda **_: str(snapshot))


def test_all_cases_need_ownership_and_complete_phases(tmp_path):
    recipe = json.loads((HERE / "cases.json").read_text())["realtime"]
    observer = proof.RealtimeProof(tmp_path, recipe)
    observer.records = [{"nodeid": node} for node in recipe["cases"]]
    observer.verify()
    observer.records.pop()
    with pytest.raises(ValueError, match="every original"):
        observer.verify()
    reports = Reports(tmp_path / "reports.jsonl")
    reports.items = {
        node: [{"phase": "setup", "outcome": "passed", "xfail": False, "detail": ""}]
        for node in recipe["cases"]
    }
    with pytest.raises(ValueError, match="incomplete case lifecycle"):
        reports.verify(recipe, tmp_path / "never-read.xml")


def test_real_pytest_fixture_entry_uses_revision_hook(tmp_path, monkeypatch):
    """The CPU lifecycle uses inert identities; it is not an engine execution proof."""
    from types import SimpleNamespace

    calls = []
    worker = SimpleNamespace(base_url="http://127.0.0.1:3333")

    def start(*args, **kwargs):
        calls.append((args, kwargs))
        return [worker]

    infra = SimpleNamespace(start_workers=start)
    monkeypatch.setitem(sys.modules, "infra", infra)
    monkeypatch.setattr(proof, "gateway_identity", lambda *_: {"cpu_inert_identity": True})
    monkeypatch.setattr(proof, "worker_identity", lambda *_: {"cpu_inert_identity": True})
    source = tmp_path / "test_fixture_entry.py"
    source.write_text("""import pytest
from types import SimpleNamespace
from dataclasses import dataclass
import infra
@dataclass
class Registry:
    url: str
class Gateway:
    def list_workers(self, strict=False):
        assert strict
        return [Registry("http://127.0.0.1:3333")]
@pytest.fixture(scope="module")
def realtime_gateway():
    infra.start_workers("Qwen/Qwen3-ASR-1.7B", engine="vllm",
                        mode=SimpleNamespace(value="http"), count=1)
    return Gateway()
def test_one(realtime_gateway):
    assert realtime_gateway
""")
    observer = proof.RealtimeProof(tmp_path, {"cases": ["test_fixture_entry.py::test_one"]})
    result = pytest.main(
        [
            str(source),
            "-q",
            "-p",
            "no:cacheprovider",
            "--rootdir=" + str(tmp_path),
            "--confcutdir=" + str(tmp_path),
        ],
        plugins=[observer],
    )
    assert result == 0
    observer.verify()
    assert calls[0][1]["extra_engine_args"] == ["--revision", observer.manifest["revision"]]
    assert infra.start_workers is start


def test_actual_http_builder_and_revision_resolution(monkeypatch):
    monkeypatch.syspath_prepend(str(HERE.resolve().parents[1] / "e2e_test"))
    monkeypatch.setenv("ROUTER_LOCAL_MODEL_PATH", "")
    from infra import ConnectionMode, Worker

    manifest = json.loads((HERE / "model-cache.json").read_text())
    worker = Worker(
        manifest["model_id"],
        "vllm",
        34567,
        [0],
        mode=ConnectionMode.HTTP,
        extra_engine_args=["--revision", manifest["revision"]],
    )
    argv = worker._build_cmd()
    proof.http_arguments(argv, manifest)
    bad = list(argv)
    bad[bad.index("--model") + 1] = "/unverified/model/alias"
    with pytest.raises(ValueError, match="alias"):
        proof.http_arguments(bad, manifest)
    bad = list(argv)
    bad[bad.index("--revision") + 1] = "main"
    with pytest.raises(ValueError, match="revision"):
        proof.http_arguments(bad, manifest)
    preflight.revision_source("get_model_path(self.model, self.revision)")
    with pytest.raises(ValueError, match="ignores pinned"):
        preflight.revision_source("get_model_path(self.model, 'main')")
