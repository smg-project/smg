"""Same-job consumer source capture privacy/contract fixtures; no SDK/model import."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def collector():
    path = Path(__file__).resolve().parents[2] / "scripts" / "ci_capture_qwenvl_source.py"
    spec = importlib.util.spec_from_file_location("qwenvl_source_capture_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def log_metadata(logs, files):
    record = {"event": "model_consumer_metadata", "metadata": {"source_files": files}}
    path = logs / "worker-Qwen__Qwen3-VL-8B-Instruct_sglang_grpc_fixture.log"
    path.write_text("SMG_QWENVL_REQUEST_DIAGNOSTIC " + json.dumps(record) + "\n")


def test_loaded_consumer_capture_requires_same_job_hash_and_keeps_public_bytes(tmp_path):
    module = collector()
    logs, output, sdk = [tmp_path / name for name in ("logs", "output", "sglang")]
    for path in (logs, output, sdk):
        path.mkdir()
    source = sdk / "consumer.py"
    body = b"# public fixture, never executed\n"
    source.write_bytes(body)
    digest = hashlib.sha256(body).hexdigest()
    log_metadata(
        logs, {"sglang.consumer": {"status": "captured", "path": str(source), "sha256": digest}}
    )
    result = module.collect_loaded_sources(logs, output, {"sglang": sdk})
    assert result["captured"] == 1 and result["source_bytes_read"] == len(body)
    assert (output / (digest + ".py")).read_bytes() == body
    assert source.read_bytes() == body


@pytest.mark.parametrize("case", ["wrong_hash", "outside_root", "symlink", "not_py"])
def test_loaded_consumer_capture_rejects_untrusted_provenance(tmp_path, case, monkeypatch):
    module = collector()
    logs, output, sdk = [tmp_path / name for name in ("logs", "output", "sglang")]
    for path in (logs, output, sdk):
        path.mkdir()
    source = sdk / "consumer.py"
    secret = tmp_path / "credential.py"
    secret.write_bytes(b"private fixture must not be exported")
    body = b"# public fixture\n"
    digest = hashlib.sha256(body).hexdigest()
    if case == "symlink":
        source.symlink_to(secret)
    else:
        source.write_bytes(body)
    if case == "outside_root":
        source = secret
    if case == "not_py":
        source = tmp_path / "credential.env"
    if case == "wrong_hash":
        digest = "0" * 64
    original_open = module.os.open

    def guarded_open(path, *args, **kwargs):
        assert Path(path) != secret, "outside-root private source must never be opened"
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(module.os, "open", guarded_open)
    log_metadata(
        logs, {"sglang.consumer": {"status": "captured", "path": str(source), "sha256": digest}}
    )
    result = module.collect_loaded_sources(logs, output, {"sglang": sdk})
    assert result["captured"] == 0 and not list(output.glob("*.py"))
    assert secret.read_bytes() == b"private fixture must not be exported"


def test_loaded_consumer_capture_enforces_global_source_budget(tmp_path):
    module = collector()
    logs, output, sdk = [tmp_path / name for name in ("logs", "output", "sglang")]
    for path in (logs, output, sdk):
        path.mkdir()
    files = {}
    for index in range(6):
        path = sdk / (f"consumer{index}.py")
        body = (str(index).encode() + b"#") * 400000
        path.write_bytes(body)
        files[f"sglang.consumer{index}"] = {
            "status": "captured",
            "path": str(path),
            "sha256": hashlib.sha256(body).hexdigest(),
        }
    log_metadata(logs, files)
    result = module.collect_loaded_sources(logs, output, {"sglang": sdk})
    assert result["source_bytes_read"] <= 4 * 1024**2
    assert sum(p.stat().st_size for p in output.glob("*.py")) <= 4 * 1024**2
    assert len(result["source"]) <= 6 and result["captured"] < 6
