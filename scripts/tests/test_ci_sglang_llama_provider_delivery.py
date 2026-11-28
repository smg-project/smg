"""Ordinary packet transport fixtures, without SDK, pip or artifact API calls."""

import hashlib
import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
spec = importlib.util.spec_from_file_location(
    "ordinary_delivery", ROOT / "scripts/ci_sglang_llama_provider_delivery.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@pytest.fixture
def producer(tmp_path, monkeypatch):
    root = tmp_path / "producer"
    (root / "scripts/ci_data").mkdir(parents=True)
    (root / ".github/workflows").mkdir(parents=True)
    creator = root / "scripts/ci_sglang_llama_cohort.py"
    creator.write_text("SOURCES = ('ci_sglang_llama_cohort.py',)\n")
    workflow = root / ".github/workflows/e2e-gpu-job.yml"
    workflow.write_text("name: harmless fixture\n")
    packet = {
        "schema": "smg-ordinary-sg-provider-v1",
        "image": m.IMAGE,
        "workflow_sha256": m.digest(workflow),
        "source_sha256": {creator.name: m.digest(creator)},
        "provenance_fixture": "inert data only",
    }
    path = root / m.PACKET
    path.write_text(json.dumps(packet) + "\n")
    for args in (
        ["init", "-b", "main"],
        ["add", "."],
        [
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-m",
            "data",
        ],
    ):
        subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    monkeypatch.setattr(m, "ROOT", root)
    monkeypatch.chdir(root)
    output = root / "github-output"
    output.touch()
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    for key in (
        "SMG_GPTOSS_COHORT_MANIFEST",
        "E2E_SGLANG_GPTOSS_MXFP4_BF16",
        "SGLANG_GPTOSS_MXFP4_BF16_FALLBACK",
    ):
        monkeypatch.delenv(key, raising=False)
    return root, path, head, output


def test_real_git_producer_to_reader_preserves_exact_bytes_and_never_qualifies_runtime(producer):
    root, path, head, output = producer
    original = path.read_bytes()
    sha = hashlib.sha256(original).hexdigest()
    assert (
        m.main(["publish", "--input", str(path), "--checkout", head, "--output", str(m.DELIVERED)])
        == 0
    )
    assert output.read_text() == "provider_sha256=" + sha + "\n"
    assert m.DELIVERED.read_bytes() == original
    assert m.DELIVERED.stat().st_mode & 0o777 == 0o444
    args = [
        "--input",
        str(m.DELIVERED),
        "--checkout",
        head,
        "--sha256",
        sha,
        "--artifact-id",
        "123",
    ]
    assert m.main(["bindings", *args]) == 0
    report = root / "delivery-report.json"
    assert m.main(["read", *args, "--output", str(report)]) == 0
    observed = json.loads(report.read_text())
    assert observed["provider_sha256"] == sha
    assert observed["producer_checkout"] == head and observed["same_run_artifact_id"] == "123"
    assert observed["SDK_native_runtime_qualified"] is False


@pytest.mark.parametrize("mutation", ["checkout", "tracked_packet", "workflow", "source"])
def test_identity_drift_refuses_before_producer_output(producer, mutation):
    root, path, head, output = producer
    if mutation == "checkout":
        head = "0" * 40
    elif mutation == "tracked_packet":
        packet = json.loads(path.read_text())
        packet["provenance_fixture"] = "uncommitted replacement"
        path.write_text(json.dumps(packet))
    elif mutation == "workflow":
        (root / ".github/workflows/e2e-gpu-job.yml").write_text("name: changed\n")
    else:
        creator = root / "scripts/ci_sglang_llama_cohort.py"
        creator.write_text(creator.read_text() + "# changed\n")
    target = root / "never-created/provider.json"
    with pytest.raises(ValueError):
        m.main(["publish", "--input", str(path), "--checkout", head, "--output", str(target)])
    assert not target.exists() and output.read_text() == ""


@pytest.mark.parametrize(
    "field,value", [("schema", "smg-gptoss20-provider-v1"), ("image", "other")]
)
def test_other_provider_contract_is_not_an_ordinary_fallback(producer, field, value):
    _, path, _, _ = producer
    packet = json.loads(path.read_text())
    packet[field] = value
    with pytest.raises(ValueError, match="Ordinary fixed-image"):
        m.source_contract(packet)


@pytest.mark.parametrize(
    "path,sha,artifact",
    [
        (str(m.DELIVERED), "a" * 64, ""),
        (str(m.DELIVERED), "a" * 64, "0"),
        (str(m.DELIVERED), "a" * 64, "other-run"),
        (str(m.DELIVERED), "", "123"),
        (str(m.DELIVERED), "g" * 64, "123"),
        ("/g14/capture/provider.json", "a" * 64, "123"),
        ("../provider.json", "a" * 64, "123"),
    ],
)
def test_missing_or_foreign_transport_never_selects_a_local_path(path, sha, artifact):
    with pytest.raises(ValueError):
        m.bindings(Path(path), sha, artifact)


@pytest.mark.parametrize(
    "key",
    [
        "SMG_GPTOSS_COHORT_MANIFEST",
        "E2E_SGLANG_GPTOSS_MXFP4_BF16",
        "SGLANG_GPTOSS_MXFP4_BF16_FALLBACK",
    ],
)
def test_provider_selections_are_exclusive(monkeypatch, key):
    monkeypatch.setenv(key, "1")
    with pytest.raises(ValueError, match="exclusive"):
        m.bindings(m.DELIVERED, "a" * 64, "123")


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo", "oversize", "digest"])
def test_bounded_regular_packet_and_digest_are_checked_without_following_links(tmp_path, kind):
    path = tmp_path / "packet.json"
    path.write_text("{}")
    target = path
    expected = hashlib.sha256(b"{}").hexdigest()
    if kind == "symlink":
        target = tmp_path / "alias"
        target.symlink_to(path)
    elif kind == "hardlink":
        os.link(path, tmp_path / "alias")
    elif kind == "fifo":
        path.unlink()
        os.mkfifo(path)
    elif kind == "oversize":
        path.write_bytes(b"x" * (m.LIMIT + 1))
    else:
        expected = "0" * 64
    with pytest.raises(ValueError):
        m.read_packet(target, expected)


def test_reader_refuses_another_packet_even_when_its_supplied_digest_matches(producer):
    root, path, head, _ = producer
    packet = json.loads(path.read_text())
    packet["provenance_fixture"] = "another packet with the same Source contract"
    body = json.dumps(packet).encode()
    target = root / m.DELIVERED
    target.parent.mkdir(parents=True)
    target.write_bytes(body)
    report = root / "never-created.json"
    with pytest.raises(ValueError, match="tracked producer blob"):
        m.main(
            [
                "read",
                "--input",
                str(m.DELIVERED),
                "--checkout",
                head,
                "--sha256",
                hashlib.sha256(body).hexdigest(),
                "--artifact-id",
                "123",
                "--output",
                str(report),
            ]
        )
    assert not report.exists()


def test_existing_publication_evidence_is_never_overwritten(producer):
    root, path, head, output = producer
    target = root / m.DELIVERED
    target.parent.mkdir(parents=True)
    target.write_bytes(b"previous evidence")
    with pytest.raises(FileExistsError):
        m.main(["publish", "--input", str(path), "--checkout", head, "--output", str(target)])
    assert target.read_bytes() == b"previous evidence"
    assert output.read_text() == ""
