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
        root = Path(dist._path)
        origin = root if root.is_file() else root / "METADATA"
        if not origin.is_file():
            origin = root / "PKG-INFO"
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
    original_environment = os.environ.get("PIP_CONSTRAINT")
    original = constraint_entries(original_environment)
    (OUT / "original-constraints.json").write_text(json.dumps(original, indent=2) + "\n")
    pins = OUT / "runtime-pins.txt"
    pins.write_text("\n".join(constraints) + "\n")
    runtime = constraint_entries(str(pins.resolve()))
    plan = {
        "original_environment": original_environment,
        "original": original,
        "runtime": runtime,
        "effective_environment": (original_environment + " " if original_environment else "")
        + str(pins.resolve()),
        "effective": original + runtime,
    }
    (OUT / "constraint-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    constraint_state("before")


def constraint_entries(environment):
    entries = []
    for value in (environment or "").split():
        path = Path(value).resolve()
        raw = path.read_bytes()
        entries.append({"path": str(path), "sha256": sha(raw), "text": raw.decode()})
    return entries


def constraint_state(label):
    environment = os.environ.get("PIP_CONSTRAINT")
    state = {"environment": environment, "entries": constraint_entries(environment)}
    (OUT / f"constraint-state-{label}.json").write_text(json.dumps(state, indent=2) + "\n")
    plan = json.loads((OUT / "constraint-plan.json").read_text())
    expected_environment = plan[
        "original_environment" if label == "before" else "effective_environment"
    ]
    expected = plan["original" if label == "before" else "effective"]
    require(environment == expected_environment, "constraint environment changed")
    require(state["entries"] == expected, "constraint path, bytes or order changed")
    name = "image-before.json" if label == "before" else f"{label}.json"
    snapshot_constraints = json.loads((OUT / name).read_text())["constraints"]
    normalized = [
        {**item, "path": str(Path(item["path"]).resolve())} for item in snapshot_constraints
    ]
    require(normalized == expected, "closure snapshot constraint identity differs")


def capture_generated(label):
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
    (OUT / ("generated-guards-" + label + ".json")).write_text(json.dumps(records, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["before", "installed", "final"])
    args = parser.parse_args()
    if args.mode != "before":
        constraint_state(args.mode)
    current = snapshot(args.mode)
    if args.mode == "before":
        runtime_pins(current)
    else:
        validate(json.loads((OUT / "origins-before.json").read_text()), current)
        capture_generated(args.mode)


if __name__ == "__main__":
    main()
