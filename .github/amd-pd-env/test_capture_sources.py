"""A source mismatch must leave bounded evidence and keep the original gate red."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from capture_sources import LIMIT, SOURCE_SHA256, capture  # noqa: E402
from infra.pd_scheduler_compat import require_source_contract  # noqa: E402


def package(tmp_path):
    root = tmp_path / "vllm"
    for relative in SOURCE_SHA256:
        source = root / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(("different actual source: " + relative).encode())
    output = tmp_path / "diagnostic-results"
    output.mkdir()
    return root, output


def test_mismatch_remains_red_after_all_five_actual_sources_are_preserved(tmp_path):
    root, output = package(tmp_path)
    receipt = capture(output, root)
    with pytest.raises(RuntimeError, match="unknown observer source contract"):
        require_source_contract(root, SOURCE_SHA256)
    assert len(receipt["sources"]) == 5 and not receipt["source_contract_verified"]
    for relative in SOURCE_SHA256:
        assert (output / "actual-local-kv-sources" / relative).read_bytes() == (
            root / relative
        ).read_bytes()
    with pytest.raises(FileExistsError):
        capture(output, root)


@pytest.mark.parametrize("mutation", ["missing", "oversized", "escape"])
def test_unknown_source_fails_before_copy_with_partial_receipt(tmp_path, mutation):
    root, output = package(tmp_path)
    relative = next(iter(SOURCE_SHA256))
    first = root / relative
    first.unlink()
    if mutation == "oversized":
        first.write_bytes(b"x" * (LIMIT + 1))
    elif mutation == "escape":
        outside = tmp_path / "outside.py"
        outside.write_bytes(b"SENSITIVE_SENTINEL")
        first.symlink_to(outside)
    with pytest.raises((OSError, ValueError)):
        capture(output, root)
    directory = output / "actual-local-kv-sources"
    assert not (directory / relative).exists()
    text = (directory / "capture.json").read_text()
    receipt = json.loads(text)
    assert receipt["error"] and not receipt["source_contract_verified"]
    assert "SENSITIVE_SENTINEL" not in text
