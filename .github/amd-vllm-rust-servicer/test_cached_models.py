"""Tiny real files test cache admission only, not real model or engine execution."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from cached_models import cached_model  # noqa: E402


def fixture(tmp_path):
    root = tmp_path / "hub"
    directory = root / "models--Org--Model"
    revision = "a" * 40
    snapshot = directory / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (directory / "refs").mkdir()
    (directory / "refs/main").write_text(revision)
    (snapshot / "config.json").write_text("{}")
    (snapshot / "tokenizer.json").write_text("{}")
    header = json.dumps({"x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    (snapshot / "model.safetensors").write_bytes(
        len(header).to_bytes(8, "little") + header + b"xxxx"
    )
    return root, snapshot, revision


def test_single_and_indexed_original_cache(tmp_path):
    root, snapshot, revision = fixture(tmp_path)
    result = cached_model("Org/Model", revision, root, "Org/Model")
    assert result["snapshot"] == revision and result["full_weight_hashes_verified"] is False
    assert result["weight_files"][0]["filename"] == "model.safetensors"
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"x": "model.safetensors"}})
    )
    assert cached_model("Org/Model", revision, root, str(snapshot))["index_sha256"]


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "truncated",
        "revision",
        "resolved",
        "index",
        "index-escape",
        "escape",
        "tokenizer",
    ],
)
def test_incomplete_changed_or_escaped_cache_refused(tmp_path, mutation):
    root, snapshot, revision = fixture(tmp_path)
    resolved = "Org/Model"
    path = snapshot / "model.safetensors"
    if mutation == "missing":
        path.unlink()
    elif mutation == "truncated":
        path.write_bytes(path.read_bytes()[:-1])
    elif mutation == "revision":
        revision = "b" * 40
    elif mutation == "resolved":
        resolved = str(tmp_path / "other")
    elif mutation == "index":
        (snapshot / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"x": "../outside"}})
        )
    elif mutation == "index-escape":
        outside = tmp_path / "outside-index"
        outside.write_text(json.dumps({"weight_map": {"x": "model.safetensors"}}))
        (snapshot / "model.safetensors.index.json").symlink_to(outside)
    elif mutation == "escape":
        outside = tmp_path / "outside"
        outside.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(outside)
    else:
        (snapshot / "tokenizer.json").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        cached_model("Org/Model", revision, root, resolved)
