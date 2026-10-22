"""Source-guard controls only; genuine native slots evidence is separate."""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from tvm_slots import FIXED_SHA, ORIGINAL_SHA, SOURCE, apply, sha, verify_loaded  # noqa: E402


def test_exact_official_hunk_on_actual_old_source(tmp_path):
    original = Path(os.environ["SMG_TEST_TVM_SUPPORT"]).read_bytes()
    assert sha(original) == ORIGINAL_SHA
    source = tmp_path / "support.py"
    source.write_bytes(original)
    receipt = tmp_path / "receipt.json"
    apply(source, receipt)
    assert sha(source.read_bytes()) == FIXED_SHA
    assert (
        source.read_bytes().replace(b'        __slots__ = ("__dict__", "__weakref__",)\n\n', b"")
        == original
    )
    with pytest.raises(ValueError, match="differs from exact"):
        apply(source, tmp_path / "again.json")


def test_wrong_source_rejected_without_modification(tmp_path):
    source = tmp_path / "support.py"
    source.write_bytes(b"unrelated source")
    with pytest.raises(ValueError, match="differs from exact"):
        apply(source, tmp_path / "receipt.json")
    assert source.read_bytes() == b"unrelated source"


def test_existing_receipt_rejected_before_modification(tmp_path):
    original = Path(os.environ["SMG_TEST_TVM_SUPPORT"]).read_bytes()
    source = tmp_path / "support.py"
    source.write_bytes(original)
    receipt = tmp_path / "receipt.json"
    receipt.write_text("existing evidence")
    with pytest.raises(ValueError, match="receipt overwrite"):
        apply(source, receipt)
    assert source.read_bytes() == original and receipt.read_text() == "existing evidence"


@pytest.mark.parametrize("mutation", [None, "missing", "hash", "path"])
def test_actual_loaded_support_identity_required(mutation):
    record = {"sha256": FIXED_SHA, "path": str(SOURCE)}
    if mutation == "missing":
        record = {}
    elif mutation:
        record["sha256" if mutation == "hash" else "path"] = "wrong"
    if mutation is None:
        verify_loaded(record)
    else:
        with pytest.raises(ValueError, match="actual imported TVM support differs"):
            verify_loaded(record)
