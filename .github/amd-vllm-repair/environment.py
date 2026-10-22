"""Preserve image package origins and enforce unchanged engine metadata."""

import argparse
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

from reuse import require

OUT = Path("diagnostic-results")
NAMES = {"torch", "vllm", "pygobject"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def package_records(distributions):
    records = []
    for dist in distributions:
        name = re.sub(r"[-_.]+", "-", dist.metadata["Name"]).lower()
        if name not in NAMES:
            continue
        origin = Path(dist._path) / "METADATA"
        raw = origin.read_bytes()
        records.append(
            {
                "name": name,
                "version": dist.version,
                "origin": str(origin.resolve()),
                "metadata_sha256": sha(raw),
                "raw": raw.decode(),
            }
        )
    return sorted(records, key=lambda item: (item["name"], item["version"], item["origin"]))


def snapshot(label):
    records = package_records(metadata.distributions())
    require({r["name"] for r in records} == NAMES, "missing image package metadata")
    active = {}
    raw_dir = OUT / ("raw-metadata-" + label)
    raw_dir.mkdir()
    for index, record in enumerate(records):
        (raw_dir / f"{index}-{record['name']}.METADATA").write_text(record.pop("raw"))
    for name in sorted(NAMES):
        dist = metadata.distribution(name)
        active[name] = {"version": dist.version, "origin": str(Path(dist._path).resolve())}
    value = {
        "python": sys.version,
        "executable": sys.executable,
        "records": records,
        "active": active,
    }
    (OUT / ("origins-" + label + ".json")).write_text(json.dumps(value, indent=2) + "\n")
    return value


def validate(before, after):
    require(before["records"] == after["records"], "engine/PyGObject versionbag or origin changed")
    require(before["active"] == after["active"], "active engine/PyGObject metadata changed")


def runtime_pins(before):
    constraints = []
    for name in ["torch", "vllm"]:
        versions = {r["version"] for r in before["records"] if r["name"] == name}
        require(versions == {before["active"][name]["version"]}, "ambiguous engine versionbag")
        constraints.append(name + "==" + versions.pop())
    original = []
    for value in os.environ.get("PIP_CONSTRAINT", "").split():
        path = Path(value)
        raw = path.read_bytes()
        original.append({"path": str(path.resolve()), "sha256": sha(raw), "text": raw.decode()})
    (OUT / "original-constraints.json").write_text(json.dumps(original, indent=2) + "\n")
    (OUT / "runtime-pins.txt").write_text("\n".join(constraints) + "\n")


def capture_generated():
    spec = importlib.util.find_spec("smg_grpc_proto")
    require(spec and spec.origin, "missing installed proto package")
    directory = Path(spec.origin).parent / "generated"
    files = sorted(directory.glob("*_pb2*.py"))
    require(files, "no actual installed generated proto files")
    records = []
    for path in files:
        raw = path.read_bytes()
        require(len(raw) <= 2 * 1024 * 1024, "oversized generated module")
        records.append(
            {
                "path": str(path.resolve()),
                "sha256": sha(raw),
                "guard_header": raw.decode().splitlines()[:40],
            }
        )
    (OUT / "installed-generated-guards.json").write_text(json.dumps(records, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["before", "installed", "final"])
    args = parser.parse_args()
    current = snapshot(args.mode)
    if args.mode == "before":
        runtime_pins(current)
    else:
        validate(json.loads((OUT / "origins-before.json").read_text()), current)
        capture_generated()


if __name__ == "__main__":
    main()
