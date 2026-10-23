"""Backport only TileLang #2071's IR-string lookup to the exact image source."""

import hashlib
import json
from pathlib import Path

SOURCE = Path("/opt/tilelang/tilelang/jit/adapter/wrapper.py")
ORIGINAL_SHA = "0a1efc3ce996827456ae24d7efd1dffb39b07401cb7e1c170b5fc1826e663206"
FIXED_SHA = "e3f7c0a7eebee40038f61d67f3c237a2c1e9ea6af1f58d4a2bd0d6d47ff4e64d"
UPSTREAM = "https://github.com/tile-ai/tilelang/commit/046b1bdc698623efef4d45afd9b0ce4d41ff0982"
OLD = b"""                index = host_code.index(f'T.call_packed("{function_name}"')\n"""
NEW = b"""                try:
                    index = host_code.index(f'T.call_packed("{function_name}"')
                except ValueError:
                    index = host_code.index(f'value="{function_name}"')
"""


def sha(data):
    return hashlib.sha256(data).hexdigest()


def apply(source, output):
    with source.open("rb") as stream:
        original = stream.read(40155)  # Exact original size plus one; no unbounded source read.
    if sha(original) != ORIGINAL_SHA or original.count(OLD) != 1:
        raise ValueError("TileLang wrapper differs from exact image source")
    fixed = original.replace(OLD, NEW)
    if sha(fixed) != FIXED_SHA:
        raise ValueError("unexpected host-IR backport result")
    if output.exists():
        raise ValueError("refuse host-IR receipt overwrite")
    source.write_bytes(fixed)
    with source.open("rb") as stream:
        if sha(stream.read(40287)) != FIXED_SHA:
            raise ValueError("installed host-IR backport verification failed")
    output.write_text(
        json.dumps(
            {
                "path": str(source.resolve()),
                "original_sha256": ORIGINAL_SHA,
                "fixed_sha256": FIXED_SHA,
                "original_bytes": len(original),
                "fixed_bytes": len(fixed),
                "official_commit": UPSTREAM,
                "hunk_only": "wrapper.py parse_source_information IR string lookup",
                "full_upstream_commit_applied": False,
                "native_binaries_changed": False,
                "package_metadata_or_requirements_changed": False,
                "actual_host_IR_not_yet_observed": True,
            },
            indent=2,
        )
        + "\n"
    )


def verify_loaded(record):
    if record.get("sha256") != FIXED_SHA or record.get("path") != str(SOURCE):
        raise ValueError("actual imported TileLang wrapper differs from reviewed backport")


if __name__ == "__main__":
    apply(SOURCE, Path("diagnostic-results/tilelang-host-ir-backport.json"))
