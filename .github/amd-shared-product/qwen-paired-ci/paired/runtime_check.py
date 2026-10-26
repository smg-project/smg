"""Actual installed distribution metadata only; no engine/native import."""

import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    expected = json.loads(args.inputs.read_text())["runtime_versions"]
    observed = {}
    for name, version in expected.items():
        dist = metadata.distribution(name)
        observed[name] = {"version": dist.version, "root": str(dist.locate_file(""))}
        if dist.version != version:
            raise RuntimeError("runtime version changed: " + name)
    assert sys.version_info[:2] == (3, 12)
    args.out.write_text(json.dumps({"python": sys.executable, "distributions": observed,
                                    "engine_or_native_imported": False}, indent=2) + "\n")
