"""Preserve the five allocator sources before the strict source comparison."""

import hashlib
import json
from importlib.util import find_spec
from pathlib import Path

from infra.pd_local_kv_compat import SOURCE_SHA256

LIMIT = 2 * 1024 * 1024


def capture(output, root=None):
    if root is None:
        spec = find_spec("vllm")
        locations = list(spec.submodule_search_locations or ()) if spec else []
        if len(locations) != 1:
            raise ValueError("unknown vLLM package location for source capture")
        root = Path(locations[0])
    root = root.resolve()
    directory = output / "actual-local-kv-sources"
    directory.mkdir()
    receipt = {"root": str(root), "sources": [], "source_contract_verified": False}
    try:
        for relative, expected in SOURCE_SHA256.items():
            source = root / relative
            if not source.resolve().is_relative_to(root):
                raise ValueError("allocator source escapes installed package: " + relative)
            with source.open("rb") as stream:
                raw = stream.read(LIMIT + 1)
            if len(raw) > LIMIT:
                raise ValueError("allocator source exceeds capture limit: " + relative)
            target = directory / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            receipt["sources"].append(
                {
                    "relative": relative,
                    "bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "expected_sha256": expected,
                }
            )
    except (OSError, ValueError) as exc:
        receipt["error"] = {"type": type(exc).__name__, "detail": str(exc)}
        raise
    finally:
        (directory / "capture.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt
