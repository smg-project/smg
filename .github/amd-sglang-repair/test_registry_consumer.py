"""Focused packet controls; no engine, model, GPU or extension compatibility claim."""

import json
import os
import sys
import zipfile
from pathlib import Path

import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import repack_ffi  # noqa: E402


def test_repack_real_official_bytes_and_requirements(tmp_path):
    original = Path(os.environ["SMG_TEST_FFI_WHEEL"])
    candidate = repack_ffi.repack(original, tmp_path)
    assert repack_ffi.sha(candidate.read_bytes()) == repack_ffi.CANDIDATE_SHA
    with zipfile.ZipFile(candidate) as archive:
        assert not any(i.is_dir() for i in archive.infolist())
        registry = archive.read("tvm_ffi/registry.py")
        assert b'if name == "__dict__":\n            continue' in registry
        assert b"if name not in type_cls.__dict__" in registry
    evidence = json.loads((tmp_path / "local-ffi-repack.json").read_text())
    assert len(evidence["native_and_header_sha256_unchanged"]) == 136
    assert evidence["requirements_unchanged"] == [
        "Requires-Python: >=3.8",
        "Requires-Dist: typing-extensions>=4.5",
        'Requires-Dist: ninja; extra == "cpp"',
    ]
    with pytest.raises(ValueError, match="overwrite"):
        repack_ffi.repack(original, tmp_path)


def test_repack_rejects_changed_original(tmp_path):
    original = tmp_path / "original.whl"
    original.write_bytes(b"not the pinned upstream wheel")
    with pytest.raises(ValueError, match="official wheel digest"):
        repack_ffi.repack(original, tmp_path)
