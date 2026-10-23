"""CPU controls use synthetic archives and no network, engine, or GPU execution."""

import base64
import copy
import csv
import io
import json
import subprocess
import zipfile
from pathlib import Path
from unittest.mock import patch

import artifact_io
import producer
import pytest
import relay
import source
import wheel_record
import yaml

HERE = Path(__file__).parent
INPUTS = json.loads((HERE / "inputs.json").read_text())
ROOT = HERE.parents[1]


def zip_payload(files, extra=None):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name, value in files.items():
            archive.writestr(name, value)
        if extra:
            extra(archive)
    return stream.getvalue()


def pin(name, payload):
    data = zip_payload({name: payload})
    return data, {
        "bytes": len(data),
        "sha256": artifact_io.digest(data),
        "files": {name: {"bytes": len(payload), "sha256": artifact_io.digest(payload)}},
    }


def wheel(extra=False, bad_record=False):
    files = {
        "smg/smg_rs.abi3.so": b"NOT-AN-EXECUTED-NATIVE-FIXTURE",
        "smg-1.dist-info/METADATA": b"Name: smg\nVersion: 1\n",
    }
    output = io.StringIO()
    writer = csv.writer(output)
    for name, data in files.items():
        digest = (
            base64.urlsafe_b64encode(bytes.fromhex(artifact_io.digest(data))).decode().rstrip("=")
        )
        writer.writerow([name, "sha256=" + ("bad" if bad_record else digest), str(len(data))])
    writer.writerow(["smg-1.dist-info/RECORD", "", ""])
    files["smg-1.dist-info/RECORD"] = output.getvalue().encode()
    if extra:
        files["unrecorded"] = b"not-covered"
    native = {
        "name": "smg/smg_rs.abi3.so",
        "bytes": len(files["smg/smg_rs.abi3.so"]),
        "sha256": artifact_io.digest(files["smg/smg_rs.abi3.so"]),
    }
    return zip_payload(files), native


@pytest.mark.parametrize("flag", ["ready", "run_budget_admitted", "artifacts_pinned"])
def test_each_admission_flag_fails_before_any_api(flag, tmp_path):
    inputs = {
        **INPUTS,
        "ready": True,
        "run_budget_admitted": True,
        "artifacts_pinned": True,
        flag: False,
    }
    with patch.object(
        artifact_io, "api_get", side_effect=AssertionError("network forbidden")
    ) as api:
        with pytest.raises(ValueError, match="admitted"):
            relay.main(inputs, tmp_path / "never-created")
        api.assert_not_called()
        assert not (tmp_path / "never-created").exists()


def test_current_empty_pins_are_not_dispatchable(tmp_path):
    with pytest.raises(ValueError, match="admitted"):
        relay.main(INPUTS, tmp_path / "never-created")


def test_producer_requires_every_critical_step():
    inputs = {
        **INPUTS,
        "artifacts": [{"id": i, "name": name} for i, name in enumerate(sorted(relay.NAMES), 1)],
    }
    run = {
        "id": inputs["run_id"],
        "head_sha": inputs["producer_sha"],
        "run_attempt": 1,
        "event": "push",
        "path": ".github/workflows/pr-test-rust.yml",
        "head_branch": "main",
        "status": "in_progress",
        "conclusion": None,
    }
    job = {
        "id": inputs["build_job_id"],
        "run_id": run["id"],
        "run_attempt": 1,
        "head_sha": run["head_sha"],
        "name": "build-wheel",
        "status": "completed",
        "conclusion": "success",
        "steps": [
            {"name": n, "status": "completed", "conclusion": "success"}
            for n in [*producer.SUCCESS_STEPS, "Post Setup Rust"]
        ],
    }
    producer.verify(run, job, inputs)
    for name in (
        "Build Python wheel and Go FFI library",
        "Upload wheel artifact",
        "Test wheel install",
    ):
        bad = copy.deepcopy(job)
        next(s for s in bad["steps"] if s["name"] == name)["conclusion"] = "cancelled"
        with pytest.raises(ValueError, match="not successful"):
            producer.verify(run, bad, inputs)
    with pytest.raises(ValueError, match="identity"):
        producer.verify({**run, "head_sha": "0" * 40}, job, inputs)
    with pytest.raises(ValueError, match="identity"):
        producer.verify(run, {**job, "run_attempt": 2}, inputs)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "bad\\path"])
def test_safe_zip_rejects_escaping_members_even_with_pinned_hash(name):
    data, artifact = pin(name, b"payload")
    with pytest.raises(ValueError, match="unsafe archive"):
        artifact_io.unpack(data, artifact)


def test_safe_zip_rejects_symlink_duplicates_and_tampering():
    data, artifact = pin("payload", b"payload")
    assert artifact_io.unpack(data, artifact) == {"payload": b"payload"}
    with pytest.raises(ValueError, match="bytes/hash"):
        artifact_io.unpack(data + b"changed", artifact)
    info = zipfile.ZipInfo("payload")
    info.create_system, info.external_attr = 3, 0o120777 << 16
    data = zip_payload({}, lambda z: z.writestr(info, b"elsewhere"))
    artifact.update(bytes=len(data), sha256=artifact_io.digest(data))
    with pytest.raises(ValueError, match="unsafe archive"):
        artifact_io.unpack(data, artifact)
    data = zip_payload({"payload": b"payload"}, lambda z: z.writestr("payload", b"payload"))
    artifact.update(bytes=len(data), sha256=artifact_io.digest(data))
    with pytest.raises(ValueError, match="unexpected archive members"):
        artifact_io.unpack(data, artifact)


def test_wheel_record_and_native_are_checked_without_loading():
    data, native = wheel()
    assert wheel_record.verify(data, native)["imported_or_installed"] is False
    for data in (wheel(bad_record=True)[0], wheel(extra=True)[0]):
        with pytest.raises(ValueError, match="RECORD"):
            wheel_record.verify(data, native)
    with pytest.raises(ValueError, match="native member identity"):
        wheel_record.verify(wheel()[0], {**native, "sha256": "0" * 64})


def test_api_metadata_requires_size_hash_head_and_future_expiry():
    artifact = {
        "id": 7,
        "name": "smg-wheel",
        "bytes": 9,
        "sha256": "1" * 64,
        "expires_at": "2050-01-01T00:00:00Z",
    }
    meta = {
        "id": 7,
        "name": "smg-wheel",
        "size_in_bytes": 9,
        "digest": "sha256:" + "1" * 64,
        "expires_at": artifact["expires_at"],
        "expired": False,
        "workflow_run": {"id": INPUTS["run_id"], "head_sha": INPUTS["producer_sha"]},
    }
    relay.metadata(meta, artifact, INPUTS)
    for field, value in (
        ("size_in_bytes", 10),
        ("digest", "sha256:" + "0" * 64),
        ("expired", True),
        ("expires_at", "2000-01-01T00:00:00Z"),
    ):
        with pytest.raises(ValueError):
            relay.metadata({**meta, field: value}, artifact, INPUTS)
    with pytest.raises(ValueError, match="producer head"):
        relay.metadata(
            {**meta, "workflow_run": {"id": INPUTS["run_id"], "head_sha": "0" * 40}},
            artifact,
            INPUTS,
        )
    expired = "2000-01-01T00:00:00Z"
    with pytest.raises(ValueError, match="expiry elapsed"):
        relay.metadata({**meta, "expires_at": expired}, {**artifact, "expires_at": expired}, INPUTS)


def test_replaced_or_duplicated_producer_artifacts_are_rejected():
    inputs = {
        **INPUTS,
        "artifacts": [{"id": i, "name": name} for i, name in enumerate(sorted(relay.NAMES), 1)],
    }
    producer.unique_artifacts(inputs["artifacts"], inputs)
    with pytest.raises(ValueError, match="replaced or duplicated"):
        producer.unique_artifacts([*inputs["artifacts"], inputs["artifacts"][0]], inputs)
    with pytest.raises(ValueError, match="replaced or duplicated"):
        producer.unique_artifacts([{**a, "id": a["id"] + 10} for a in inputs["artifacts"]], inputs)


def test_real_git_source_drift_is_rejected(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "compiled").write_text("first")

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, text=True).strip()

    git("add", "compiled")
    git(
        "-c",
        "user.name=CPU Control",
        "-c",
        "user.email=cpu@example.invalid",
        "commit",
        "-qm",
        "first",
    )
    expected = {"compiled": git("rev-parse", "HEAD:compiled")}

    def read(revision, path):
        return git("rev-parse", f"{revision}:{path}")

    source.check_map(read, "HEAD", expected)
    (tmp_path / "compiled").write_text("drift")
    git(
        "-c",
        "user.name=CPU Control",
        "-c",
        "user.email=cpu@example.invalid",
        "commit",
        "-qam",
        "drift",
    )
    with pytest.raises(ValueError, match="identity differs"):
        source.check_map(read, "HEAD", expected)


def test_final_source_maps_and_true_shared_caller():
    source.verify(INPUTS)
    caller = yaml.load(
        (ROOT / ".github/workflows/pr-test-rust.yml").read_text(), Loader=yaml.BaseLoader
    )
    job = caller["jobs"]["amd-product-python-chat"]
    assert job["uses"] == "./.github/workflows/e2e-gpu-job.yml"
    assert job["needs"] == "artifact-relay"
    assert job["with"]["timeout"] == "60" and job["with"]["test_timeout"] == "45"
    assert job["with"]["test_dirs"] == "e2e_test/chat_completions"
    assert job["with"]["min_selected"] == "202" and job["with"]["servicer_unit_tests"] == "true"
    assert job["with"]["runner"] == "1-gpu-mi300x"
    options = job["with"]["container_options"]
    assert "--env SPT_NOENV=1" in options
    assert not any(value in options for value in ("--pid", "/proc", "--privileged"))
    assert set(caller["jobs"]) == {"artifact-relay", "amd-product-python-chat", "finish"}
