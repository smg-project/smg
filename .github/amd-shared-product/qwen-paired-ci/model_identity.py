"""Recheck root-declared original Qwen snapshot metadata/header bytes, not tensors."""
import hashlib
import json
from pathlib import Path
import re
import struct


def validate(value):
    snapshot = Path(value["snapshot"])
    assert snapshot.is_absolute() and snapshot.is_dir() and re.fullmatch("[a-f0-9]{40}", value["revision"])
    assert snapshot.name == value["revision"] and snapshot.resolve().is_relative_to(Path("/models").resolve())
    count = 0

    def read(item, cap):
        nonlocal count
        path = Path(item["path"])
        assert path.is_absolute() and path.is_relative_to(snapshot)
        actual = path.resolve(strict=True)
        assert str(actual) == item["resolved_path"] and actual.is_relative_to(snapshot.parent.parent.resolve())
        assert actual.is_file() and actual.stat().st_size == item["bytes"] <= cap
        raw = actual.read_bytes()
        count += len(raw)
        assert count <= 128 * 1024**2 and hashlib.sha256(raw).hexdigest() == item["sha256"]
        return raw

    metadata = value["config_and_tokenizer_files"]
    assert 4 <= len(metadata) <= 16
    names = {Path(item["path"]).name for item in metadata}
    assert {"config.json", "tokenizer.json", "tokenizer_config.json", "preprocessor_config.json"} <= names
    for item in metadata:
        raw = read(item, 64 * 1024**2)
        if item["path"].endswith(".json"):
            json.loads(raw)
    index = json.loads(read(value["index"], 4 * 1024**2))
    mapping = index["weight_map"]
    assert isinstance(mapping, dict) and mapping
    assert all(isinstance(tensor, str) and tensor and tensor.strip() == tensor for tensor in mapping)
    assert all(isinstance(name, str) and name and Path(name).name == name and ".." not in name for name in mapping.values())
    declared = {item["name"]: item for item in value["shards"]}
    assert set(declared) == set(mapping.values()) and 0 < len(declared) <= 64 and len(declared) == len(value["shards"])
    total_headers = 0
    for name, item in declared.items():
        path = snapshot / name
        resolved = path.resolve(strict=True)
        assert resolved.is_relative_to(snapshot.parent.parent.resolve()) and resolved.is_file() and resolved.stat().st_size == item["bytes"]
        with resolved.open("rb") as stream:
            raw = stream.read(8)
            assert len(raw) == 8
            length = struct.unpack("<Q", raw)[0]
            assert 0 < length <= 1024**2 and 8 + length <= item["bytes"]
            header = stream.read(length)
        total_headers += length
        assert total_headers <= 16 * 1024**2 and hashlib.sha256(header).hexdigest() == item["header_sha256"]
        tensors = json.loads(header)
        names = set(tensors) - {"__metadata__"}
        assert names and names == {tensor for tensor, shard in mapping.items() if shard == name}
        for tensor in names:
            begin, end = tensors[tensor]["data_offsets"]
            assert 0 <= begin <= end <= item["bytes"] - 8 - length
    fixture_model = value["fixture_resolved_model_path"]
    assert isinstance(fixture_model, str)
    fixture_ref = value["fixture_HF_ref"]
    if fixture_model == "Qwen/Qwen3-VL-8B-Instruct":
        ref_path = Path(fixture_ref["path"])
        assert ref_path == snapshot.parent.parent / "refs/main"
        assert ref_path.is_file() and not ref_path.is_symlink()
        raw_ref = ref_path.read_bytes()
        assert len(raw_ref) == fixture_ref["bytes"] <= 128
        assert hashlib.sha256(raw_ref).hexdigest() == fixture_ref["sha256"]
        assert raw_ref.decode().strip() == value["revision"]
    else:
        assert Path(fixture_model).is_absolute() and Path(fixture_model).resolve() == snapshot.resolve()
    return {"model_id": "Qwen/Qwen3-VL-8B-Instruct", "snapshot": str(snapshot), "revision": value["revision"],
            "fixture_resolved_model_path": fixture_model, "fixture_HF_ref": fixture_ref,
            "config_and_tokenizer_files": metadata, "all_declared_weight_headers_valid": True,
            "complete_index_tensor_names_match": True, "full_tensor_hashes_verified": False,
            "header_bytes_read": total_headers, "config_bytes_read": count}
