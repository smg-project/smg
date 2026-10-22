"""Offline guard negatives and one real local Torch CPU DLPack control."""

import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "native_preflight", Path(__file__).with_name("probe_native_api.py")
)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


@pytest.mark.parametrize("device", ["/dev/kfd", "/dev/dri"])
def test_mapped_device_rejected(device, monkeypatch):
    monkeypatch.setattr(probe.Path, "exists", lambda path: str(path) == device)
    with pytest.raises(ValueError, match="GPU devices"):
        probe.device_guard()


@pytest.mark.parametrize("content", [None, b"wrong constraints"])
def test_missing_or_changed_constraints_prevent_imports(tmp_path, monkeypatch, content):
    path = tmp_path / "constraints"
    if content is not None:
        path.write_bytes(content)
    monkeypatch.setattr(probe, "CONSTRAINT", path)
    monkeypatch.setattr(probe, "device_guard", lambda: None)
    monkeypatch.setattr(
        probe.importlib, "import_module", lambda name: pytest.fail("import started")
    )
    assert probe.run(tmp_path / "output") == 1
    result = json.loads((tmp_path / "output/result.json").read_text())
    assert result["status"] == "failed"
    assert result["checks"]["constraints_before"]["status"] == "failed"
    assert "import_torch" not in result["checks"]


def test_missing_dependency_never_becomes_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "device_guard", lambda: None)
    monkeypatch.setattr(probe, "constraints", lambda: {"unit_guard_only": True})

    def missing(name):
        raise ModuleNotFoundError("unit negative: dependency unavailable")

    monkeypatch.setattr(probe.importlib, "import_module", missing)
    assert probe.run(tmp_path / "output") == 1
    result = json.loads((tmp_path / "output/result.json").read_text())
    assert result["status"] == "failed"
    assert result["checks"]["import_torch"]["error"] == "ModuleNotFoundError"
    assert result["checks"]["FFI_CPU_roundtrip"]["status"] == "not_executed"


def test_evidence_not_overwritten(tmp_path):
    result = tmp_path / "result.json"
    result.write_bytes(b"existing")
    with pytest.raises(ValueError, match="overwrite"):
        probe.run(tmp_path)
    assert result.read_bytes() == b"existing"


def test_actual_local_Torch_CPU_roundtrip():
    import torch

    before = torch.cuda.is_initialized()
    assert probe.roundtrip(torch, lambda tensor: tensor) == {
        "values": [17, 1, 2, 3, 4, 5],
        "shape": [6],
        "shared_storage": True,
    }
    assert torch.cuda.is_initialized() == before
