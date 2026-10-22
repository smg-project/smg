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

from repack_ffi import VERSION
from reuse import require

OUT = Path("diagnostic-results")
NAMES = {"torch", "torchaudio", "torchvision", "sglang", "tilelang", "xgrammar"}


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
    require(
        before["records"] == after["records"],
        "unchanged image package versionbag or origin changed",
    )
    require(before["active"] == after["active"], "active unchanged image package metadata changed")


def runtime_pins(before):
    constraints = []
    for name in sorted(NAMES):
        versions = {r["version"] for r in before["records"] if r["name"] == name}
        require(versions == {before["active"][name]["version"]}, "ambiguous engine versionbag")
        constraints.append(name + "==" + versions.pop())
    original = []
    for value in os.environ.get("PIP_CONSTRAINT", "").split():
        path = Path(value)
        raw = path.read_bytes()
        original.append({"path": str(path.resolve()), "sha256": sha(raw), "text": raw.decode()})
    (OUT / "original-constraints.json").write_text(json.dumps(original, indent=2) + "\n")
    constraints += ["apache-tvm-ffi==" + VERSION, "torch-c-dlpack-ext==0.1.5"]
    (OUT / "runtime-pins.txt").write_text("\n".join(constraints) + "\n")


def pin_state(label):
    path = OUT / "runtime-pins.txt"
    raw = path.read_bytes()
    actual = []
    for filename in os.environ.get("PIP_CONSTRAINT", "").split():
        file = Path(filename)
        data = file.read_bytes()
        actual.append({"path": str(file.resolve()), "sha256": sha(data), "text": data.decode()})
    value = {
        "path": str(path.resolve()),
        "sha256": sha(raw),
        "text": raw.decode(),
        "PIP_CONSTRAINT": os.environ.get("PIP_CONSTRAINT"),
        "constraints": actual,
    }
    (OUT / ("runtime-pins-state-" + label + ".json")).write_text(json.dumps(value, indent=2) + "\n")
    if label == "before":
        return
    previous = json.loads((OUT / "runtime-pins-state-before.json").read_text())
    require(sha(raw) == previous["sha256"], "runtime pins changed")
    expected = json.loads((OUT / "original-constraints.json").read_text()) + [
        {key: value[key] for key in ("path", "sha256", "text")}
    ]
    require(actual == expected, "constraint environment omitted/reordered/changed")
    snapshot = json.loads((OUT / (label + ".json")).read_text())["constraints"]
    snapshot = [dict(item, path=str(Path(item["path"]).resolve())) for item in snapshot]
    require(snapshot == actual, "closure snapshot constraint path/hash/text/order differs")


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


def repair_sources(label):
    expected = json.loads(Path(__file__).with_name("repair-sources.json").read_text())
    records = []
    for name, repair in expected.items():
        found = [
            d
            for d in metadata.distributions()
            if re.sub(r"[-_.]+", "-", d.metadata["Name"]).lower() == name
        ]
        require(len(found) == 1 and found[0].version == repair["version"], "repair versionbag")
        dist = metadata.distribution(name)
        for filename, spec in repair["files"].items():
            path = Path(dist.locate_file(filename))
            raw = path.read_bytes()
            require(
                len(raw) == spec["bytes"] and sha(raw) == spec["sha256"], "installed repair source"
            )
            records.append(
                {
                    "package": name,
                    "version": dist.version,
                    "origin": str(path.resolve()),
                    "sha256": sha(raw),
                }
            )
    (OUT / ("repair-sources-" + label + ".json")).write_text(json.dumps(records, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["before", "installed", "final"])
    args = parser.parse_args()
    current = snapshot(args.mode)
    if args.mode == "before":
        runtime_pins(current)
        pin_state("before")
    else:
        validate(json.loads((OUT / "origins-before.json").read_text()), current)
        for record in json.loads((OUT / "original-constraints.json").read_text()):
            require(
                sha(Path(record["path"]).read_bytes()) == record["sha256"],
                "image constraints changed",
            )
        require(metadata.version("apache-tvm-ffi") == VERSION, "FFI repair version")
        require(metadata.version("torch-c-dlpack-ext") == "0.1.5", "addon requirement version")
        pin_state(args.mode)
        repair_sources(args.mode)
        capture_generated(args.mode)


if __name__ == "__main__":
    main()
