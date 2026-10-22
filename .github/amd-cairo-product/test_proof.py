"""Offline rejection controls; fixture metadata is not a real engine environment."""

import importlib.metadata as metadata
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import proof
import pytest


@pytest.fixture
def output(tmp_path, monkeypatch):
    directory = tmp_path / "evidence"
    directory.mkdir()
    monkeypatch.setattr(proof, "OUT", directory)
    return directory


@pytest.mark.parametrize("device", [None, "kfd", "dri/renderD128", "nvidia0"])
def test_real_filesystem_device_guard(output, tmp_path, device):
    devices = tmp_path / "dev"
    devices.mkdir()
    if device:
        path = devices / device
        path.parent.mkdir(exist_ok=True)
        path.touch()
        with pytest.raises(ValueError, match="GPU devices visible"):
            proof.devices("cpu", devices)
    else:
        proof.devices("cpu", devices)
    assert bool(json.loads((output / "devices-cpu.json").read_text())["visible"]) == bool(device)


def test_actual_product_identity_and_wrong_helper_bytes(output, tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(root)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    monkeypatch.setenv("GITHUB_SHA", head)
    proof.source_identity()
    assert json.loads((output / "source.json").read_text())["helper_sha256"] == proof.HELPER_SHA
    helper = tmp_path / "wrong.sh"
    helper.write_text("exit 0\n")
    monkeypatch.setattr(proof, "HELPER", helper)
    with pytest.raises(ValueError, match="product helper bytes changed"):
        proof.source_identity()


@pytest.mark.parametrize("change", [None, "record", "active", "constraint", "environment"])
def test_actual_metadata_files_and_constraint_changes(output, tmp_path, monkeypatch, change):
    distributions = {}
    for name in ["torch", "vllm", "PyGObject"]:
        root = tmp_path / (name + ".egg-info")
        root.mkdir()
        (root / "PKG-INFO").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: 1\n")
        distributions[name.lower()] = metadata.PathDistribution(root)
    monkeypatch.setattr(proof.metadata, "distributions", lambda: distributions.values())
    monkeypatch.setattr(proof.metadata, "distribution", lambda name: distributions[name])
    monkeypatch.setattr(proof.metadata, "version", lambda name: distributions[name].version)
    constraints = tmp_path / "constraints.txt"
    constraints.write_text("fixture==1\n")
    monkeypatch.setenv("PIP_CONSTRAINT", str(constraints))
    before = proof.snapshot("before")
    after = proof.snapshot("after")
    if change == "record":
        after["records"][0]["sha256"] = "changed"
    elif change == "active":
        after["active"] = deepcopy(after["active"])
        after["active"]["torch"]["version"] = "changed"
    elif change == "constraint":
        constraints.write_text("fixture==2\n")
        after = proof.snapshot("changed")
    elif change == "environment":
        after["constraint_environment"] = ""
    if change:
        with pytest.raises(ValueError, match="changed"):
            proof.unchanged(before, after)
    else:
        proof.unchanged(before, after)
    assert list((output / "metadata-before").glob("*.METADATA"))


def test_native_probe_rejects_initialized_gpu_before_cairo_import(output, monkeypatch):
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: True))
    )
    with pytest.raises(ValueError, match="GPU initialized before"):
        proof.native()


def test_native_archive_is_bounded(output, tmp_path):
    library = tmp_path / "library.so"
    library.write_bytes(b"too-large")
    with pytest.raises(ValueError, match="invalid evidence file"):
        proof.archive(library, "native/library.so", limit=2)
