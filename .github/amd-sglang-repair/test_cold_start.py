"""CPU controls for source/lock evidence, bounded lifecycle and actual shell scope."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import cold_start_evidence as evidence  # noqa: E402

SPEC = importlib.util.spec_from_file_location("cold_prewarm", HERE / "prewarm.py")
prewarm = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prewarm)


def environment():
    return {
        "MAX_JOBS": "8",
        "CARGO_BUILD_JOBS": "8",
        "AITER_LOG_MORE": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "E2E_ENGINE": "sglang",
        "E2E_GPU_TIER": "1",
        "SGLANG_USE_AITER": "1",
    }


@pytest.mark.parametrize("key", sorted(environment()))
def test_prewarm_guard_rejects_changed_contract(key):
    env = environment()
    evidence.guard_environment(env)
    env[key] = "incorrect"
    with pytest.raises(ValueError):
        evidence.guard_environment(env)


@pytest.mark.parametrize("key", ["AITER_JIT_DIR", "AITER_REBUILD", "E2E_CONNECTION_MODE"])
def test_cache_or_wire_replacement_is_refused(key):
    env = environment()
    env[key] = "different"
    with pytest.raises(ValueError):
        evidence.guard_environment(env)


def test_actual_io_preserves_lock_bytes_and_rejects_source_mutation(tmp_path, monkeypatch):
    root = tmp_path / "image"
    source = root / "aiter/jit/core.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"fixed source\n")
    monkeypatch.setattr(
        evidence,
        "SOURCE_PINS",
        {"aiter/jit/core.py": hashlib.sha256(source.read_bytes()).hexdigest()},
    )
    collector = evidence.Collector(tmp_path / "proof", root)
    collector.sources()
    lock = source.parent / "build" / ("lock_" + evidence.TARGET)
    lock.parent.mkdir()
    lock.write_text("123456\nhost\n")
    before = lock.read_bytes()
    collector.snapshot("before", None)
    assert lock.read_bytes() == before
    source.write_bytes(b"changed source\n")
    with pytest.raises(ValueError, match="source mismatch"):
        collector.sources()
    assert any(p.read_bytes() == b"changed source\n" for p in collector.output.glob("*.bytes"))
    escaped = root / "escape"
    escaped.symlink_to(tmp_path.parent)
    with pytest.raises(ValueError, match="escape"):
        collector.read(escaped / "whatever")


@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.parametrize("optional_missing", [False, True])
def test_real_orchestrator_stops_on_health_success_or_failure(tmp_path, failure, optional_missing):
    calls = []
    clock = [0.0]

    class Worker:
        gpu_ids = [0]
        process = SimpleNamespace(pid=123)
        port = 12345

        def _build_cmd(self):
            return ["original-worker"]

        def wait_ready(self, seconds):
            calls.append(("health", seconds))
            assert seconds == 480
            clock[0] = 370
            if failure:
                raise TimeoutError("cold build did not finish")

    class Collector:
        def sources(self):
            calls.append("sources")

        def snapshot(self, label, worker):
            calls.append(label)
            if optional_missing:
                raise FileNotFoundError("optional build.ninja not yet present")

        def native_identity(self):
            return {"sha256": "target-native"}

    def start(*args, **kwargs):
        assert args == (prewarm.MODEL, "sglang")
        assert kwargs == {
            "mode": "grpc",
            "count": 1,
            "wait_ready": False,
            "log_dir": "e2e-logs/cold-start",
        }
        return [Worker()]

    def stop(workers):
        assert len(workers) == 1
        calls.append("stop")
        clock[0] += 20

    output = tmp_path / "result.json"
    if failure:
        with pytest.raises(TimeoutError):
            prewarm.execute(start, stop, Collector(), "grpc", output, lambda: clock[0])
    else:
        prewarm.execute(start, stop, Collector(), "grpc", output, lambda: clock[0])
    result = json.loads(output.read_text())
    assert result["ready"] is not failure and result["stop_returned"]
    assert calls[-2:] == ["stop", "after-stop"]
    assert result["total_seconds"] == 390
    assert bool(result["evidence_errors"]) is optional_missing


@pytest.mark.parametrize("fail_prewarm", [False, True])
def test_exact_shell_tail_limits_prewarm_and_restores_original_suite_environment(
    tmp_path, fail_prewarm
):
    fake = tmp_path / "python3"
    fake.write_text("#!/usr/bin/env python3\n")
    # Use the current interpreter explicitly: never recurse through the fake PATH entry.
    fake.write_text(
        f"#!{sys.executable}\nimport json,os,sys\n"
        "print(json.dumps({'args':sys.argv[1:],'jobs':os.getenv('MAX_JOBS'),'cargo':os.getenv('CARGO_BUILD_JOBS'),'verbose':os.getenv('AITER_LOG_MORE')}))\n"
        "sys.exit(7 if os.getenv('FAIL_PREWARM')=='1' and sys.argv[-1].endswith('prewarm.py') else 0)\n"
    )
    fake.chmod(0o755)
    tail = (HERE / "run.sh").read_text().split("timeout --signal=INT", 1)[1]
    env = {
        **os.environ,
        "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
        "MAX_JOBS": "2",
        "CARGO_BUILD_JOBS": "2",
        "FAIL_PREWARM": str(int(fail_prewarm)),
    }
    env.pop("AITER_LOG_MORE", None)
    run = subprocess.run(
        ["bash", "-c", "set -eu\ntimeout --signal=INT" + tail],
        env=env,
        capture_output=True,
        text=True,
    )
    calls = [json.loads(line) for line in run.stdout.splitlines()]
    assert calls[0]["jobs"] == calls[0]["cargo"] == "8" and calls[0]["verbose"] == "1"
    assert run.returncode == (7 if fail_prewarm else 0)
    if fail_prewarm:
        assert len(calls) == 1
    else:
        assert [r["args"][-1] for r in calls[1:]] == ["completions", "embeddings", "router"]
        assert all(r["jobs"] == r["cargo"] == "2" and r["verbose"] is None for r in calls[1:])


def test_optional_native_cache_absence_and_actual_hash(tmp_path):
    root = tmp_path / "image"
    jit = root / "aiter/jit"
    jit.mkdir(parents=True)
    collector = evidence.Collector(tmp_path / "proof", root)
    assert collector.native_identity()["exists"] is False
    native = jit / (evidence.TARGET + ".so")
    native.write_bytes(b"record bytes only; never load this file")
    assert collector.native_identity()["sha256"] == hashlib.sha256(native.read_bytes()).hexdigest()


def test_insufficient_remaining_core_budget_refuses_before_worker_import(tmp_path, monkeypatch):
    for key, value in environment().items():
        monkeypatch.setenv(key, value)
    for key in [
        "AITER_JIT_DIR",
        "AITER_REBUILD",
        "E2E_CONNECTION_MODE",
        "E2E_SGLANG_SERVICER_IMPL",
    ]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SMG_QA_IMAGE", prewarm.IMAGE)
    budget = tmp_path / "budget.json"
    budget.write_text(json.dumps({"diagnostic_deadline_epoch": 0}))
    monkeypatch.setenv("SMG_SGLANG_REPAIR_BUDGET_FILE", str(budget))
    with pytest.raises(ValueError, match="insufficient core budget"):
        prewarm.main()
