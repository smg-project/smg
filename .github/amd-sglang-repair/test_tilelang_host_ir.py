"""Exact-source guards; these do not execute or validate a TileLang kernel."""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from tilelang_host_ir import FIXED_SHA, NEW, OLD, SOURCE, apply, sha, verify_loaded  # noqa: E402


def test_official_hunk_on_hash_bound_actual_wrapper(tmp_path):
    original = Path(os.environ["SMG_TEST_TILELANG_WRAPPER"]).read_bytes()
    source = tmp_path / "wrapper.py"
    source.write_bytes(original)
    receipt = tmp_path / "receipt.json"
    apply(source, receipt)
    assert sha(source.read_bytes()) == FIXED_SHA
    assert source.read_bytes().replace(NEW, OLD) == original
    assert json.loads(receipt.read_text())["native_binaries_changed"] is False
    with pytest.raises(ValueError, match="differs from exact"):
        apply(source, tmp_path / "again.json")


def test_unrelated_source_and_existing_receipt_refused(tmp_path):
    source = tmp_path / "wrapper.py"
    source.write_bytes(b"unrelated source")
    receipt = tmp_path / "receipt.json"
    with pytest.raises(ValueError, match="differs from exact"):
        apply(source, receipt)
    assert source.read_bytes() == b"unrelated source" and not receipt.exists()
    original = Path(os.environ["SMG_TEST_TILELANG_WRAPPER"]).read_bytes()
    source.write_bytes(original)
    receipt.write_text("previous evidence")
    with pytest.raises(ValueError, match="receipt overwrite"):
        apply(source, receipt)
    assert source.read_bytes() == original and receipt.read_text() == "previous evidence"


@pytest.mark.parametrize("mutation", [None, "missing", "sha256", "path"])
def test_imported_wrapper_requires_exact_source_identity(mutation):
    record = {"sha256": FIXED_SHA, "path": str(SOURCE)}
    if mutation == "missing":
        record = {}
    elif mutation:
        record[mutation] = "wrong"
    if mutation is None:
        verify_loaded(record)
    else:
        with pytest.raises(ValueError, match="actual imported TileLang wrapper differs"):
            verify_loaded(record)
