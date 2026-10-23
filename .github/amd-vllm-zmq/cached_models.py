"""Check the original local HF snapshots without downloading or hashing full weights."""

import hashlib
import json
from pathlib import Path

from reuse import require

LIMIT = 4 * 1024 * 1024


def read(path, root):
    require(path.resolve().is_relative_to(root), "cache path escapes HF cache")
    with path.open("rb") as stream:
        raw = stream.read(LIMIT + 1)
    require(len(raw) <= LIMIT, "cache metadata too large")
    return raw


def weight(path, root):
    require(path.resolve().is_relative_to(root), "weight escapes HF cache")
    size = path.stat().st_size
    with path.open("rb") as stream:
        length = int.from_bytes(stream.read(8), "little")
        require(0 < length <= LIMIT, "invalid safetensors header size")
        raw = stream.read(length)
    require(len(raw) == length, "incomplete safetensors header")
    tensors = json.loads(raw)
    ends = []
    for name, entry in tensors.items():
        if name == "__metadata__":
            continue
        start, end = entry["data_offsets"]
        require(
            isinstance(start, int) and isinstance(end, int) and 0 <= start <= end, "bad offsets"
        )
        ends.append(end)
    require(ends and size == 8 + length + max(ends), "incomplete/changed safetensors payload")
    return {
        "path": str(path.resolve()),
        "filename": path.name,
        "bytes": size,
        "header_sha256": hashlib.sha256(raw).hexdigest(),
    }


def cached_model(model, revision, root, resolved):
    root = root.resolve()
    directory = root / ("models--" + model.replace("/", "--"))
    actual = read(directory / "refs/main", root).decode().strip()
    require(actual == revision, "cached original model revision changed")
    snapshot = directory / "snapshots" / revision
    require(snapshot.resolve().is_relative_to(root), "snapshot escapes HF cache")
    require(
        resolved == model or Path(resolved).resolve() == snapshot.resolve(), "model path changed"
    )
    config = read(snapshot / "config.json", root)
    require(isinstance(json.loads(config), dict), "invalid model config")
    index = snapshot / "model.safetensors.index.json"
    index_raw = read(index, root) if index.is_symlink() or index.exists() else None
    names = (
        set(json.loads(index_raw)["weight_map"].values()) if index_raw else {"model.safetensors"}
    )
    require(0 < len(names) <= 128, "missing/oversized shard index")
    require(
        all(Path(n).name == n and n.endswith(".safetensors") for n in names), "unsafe shard name"
    )
    weights = [weight(snapshot / name, root) for name in sorted(names)]
    tokenizer = [
        path
        for name in ("tokenizer.json", "tokenizer.model", "tokenizer.model.v3", "tekken.json")
        for path in (snapshot / name,)
        if path.is_file() and path.stat().st_size and path.resolve().is_relative_to(root)
    ]
    require(tokenizer, "original tokenizer assets missing")
    return {
        "model": model,
        "snapshot": actual,
        "snapshot_path": str(snapshot.resolve()),
        "resolved_model_spec": resolved,
        "config_sha256": hashlib.sha256(config).hexdigest(),
        "index_sha256": hashlib.sha256(index_raw).hexdigest() if index_raw else None,
        "weight_files": weights,
        "tokenizer_files": [str(p.resolve()) for p in sorted(tokenizer)],
        "full_weight_hashes_verified": False,
        "no_model_download": True,
    }
