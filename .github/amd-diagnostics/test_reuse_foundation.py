"""Offline artifact rejection controls; no API requests or binary execution."""

import hashlib
import io
import json
import os
import shutil
import subprocess
import zipfile
from copy import deepcopy

import pytest
import reuse_foundation as reuse


def artifact():
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("payload", b"known")
    data = stream.getvalue()
    return data, {
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "files": {"payload": {"bytes": 5, "sha256": hashlib.sha256(b"known").hexdigest()}},
    }


@pytest.mark.parametrize("kind", ["archive", "payload", "unknown-member"])
def test_artifact_corruption_rejected(kind):
    data, expected = artifact()
    if kind == "archive":
        data = data[:-1] + bytes([data[-1] ^ 1])
    elif kind == "payload":
        expected["files"]["payload"]["sha256"] = "0" * 64
    else:
        expected["files"] = {"different": expected["files"]["payload"]}
    with pytest.raises(ValueError):
        reuse.unpack(data, expected)


def test_valid_artifact_and_expired_or_wrong_identity():
    data, expected = artifact()
    assert reuse.unpack(data, expected) == {"payload": b"known"}
    inputs = json.loads(reuse.INPUTS.read_text())
    expected = inputs["artifacts"][0]
    meta = {
        "id": expected["id"],
        "name": expected["name"],
        "expired": False,
        "size_in_bytes": expected["bytes"],
        "digest": "sha256:" + expected["sha256"],
        "workflow_run": {"id": inputs["run_id"], "head_sha": inputs["producer_sha"]},
    }
    reuse.check_metadata(meta, expected, inputs)
    for field, value in [("expired", True), ("digest", "sha256:" + "0" * 64), ("id", 0)]:
        bad = deepcopy(meta)
        bad[field] = value
        with pytest.raises(ValueError):
            reuse.check_metadata(bad, expected, inputs)
    meta["workflow_run"]["head_sha"] = "0" * 40
    with pytest.raises(ValueError):
        reuse.check_metadata(meta, expected, inputs)


def test_foreign_owned_checkout_trust_is_per_command(tmp_path, monkeypatch):
    for name in ("GIT_CONFIG_PARAMETERS", "GIT_CONFIG_COUNT", "SUDO_UID"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    uid = os.geteuid()
    if uid != 0 and (
        not shutil.which("sudo")
        or subprocess.run(["sudo", "-n", "true"], capture_output=True).returncode
    ):
        pytest.skip("changing real temporary repo ownership requires root or noninteractive sudo")

    repo, other = tmp_path / "repo", tmp_path / "other"
    for path in (repo, other):
        subprocess.check_call(["git", "init", "-q", str(path)])
    source = repo / ".cargo" / "config.toml"
    source.parent.mkdir()
    source.write_text("[build]\n")
    subprocess.check_call(["git", "add", ".cargo"], cwd=repo)
    subprocess.check_call(
        [
            "git",
            "-c",
            "user.name=Offline QA",
            "-c",
            "user.email=qa@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=repo,
    )
    oid = subprocess.check_output(["git", "rev-parse", "HEAD:.cargo"], cwd=repo, text=True).strip()
    config_before = (repo / ".git/config").read_bytes()
    paths = [p for r in (repo, other) for p in (r, r / ".git")]

    def owner(value):
        for path in paths:
            if uid == 0:
                os.chown(path, value, -1)
            else:
                subprocess.check_call(["sudo", "-n", "chown", str(value), str(path)])

    try:
        owner(65534 if uid == 0 else 0)
        assert repo.stat().st_uid != uid and (repo / ".git").stat().st_uid != uid
        before = subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True)
        assert before.returncode == 128 and b"dubious ownership" in before.stderr
        monkeypatch.setattr(reuse, "ROOT", repo)
        inputs = {"source_trees": {".cargo": oid}}
        reuse.check_source(inputs)
        source.write_text("[build]\n# dirty\n")
        with pytest.raises(ValueError, match="must be clean"):
            reuse.check_source(inputs)
        source.write_text("[build]\n")
        with pytest.raises(ValueError, match="source identity differs"):
            reuse.check_source({"source_trees": {".cargo": "0" * 40}})
        outside = subprocess.run(
            ["git", "-c", f"safe.directory={repo}", "status", "--porcelain"],
            cwd=other,
            capture_output=True,
        )
        assert outside.returncode == 128 and b"dubious ownership" in outside.stderr
        after = subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True)
        assert after.returncode == 128 and b"dubious ownership" in after.stderr
        assert (repo / ".git/config").read_bytes() == config_before
    finally:
        owner(uid)
