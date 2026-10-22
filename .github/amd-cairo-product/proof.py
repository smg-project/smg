"""Device-free evidence for the exact ROCm setup helper; no model is launched."""

import argparse
import base64
import csv
import hashlib
import importlib.metadata as metadata
import io
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

OUT = Path("cairo-results")
HELPER = Path("scripts/ci_setup_vllm_rocm_image.sh")
HELPER_SHA = "e50baa665878846eb757949422a1db16cc600216337da367f9a3a089d9d6616e"
NAMES = {"torch", "vllm", "pygobject"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write(name, value):
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n")


def archive(path, name, limit=16 * 1024 * 1024):
    path = Path(path).resolve(strict=True)
    require(path.is_file() and path.stat().st_size <= limit, "invalid evidence file")
    target = OUT / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, target)
    return {
        "origin": str(path),
        "file": name,
        "bytes": target.stat().st_size,
        "sha256": sha(target.read_bytes()),
    }


def devices(label, root=Path("/dev")):
    visible = sorted(str(p) for pattern in ["kfd", "dri/*", "nvidia*"] for p in root.glob(pattern))
    write("devices-" + label + ".json", {"visible": visible})
    require(not visible, "GPU devices visible to device-free diagnostic")


def source_identity():
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    require(head == os.environ["GITHUB_SHA"], "checkout does not match dispatched source")
    require(sha(HELPER.read_bytes()) == HELPER_SHA, "product helper bytes changed")
    trees = {}
    for path, expected in {
        "scripts/ci_retry.sh": "f292dd730fde1b08e79e20948657cc2bba430441",
        "crates/grpc_client/python": "0422bed96aef11522260d87f86d51525c0ad00d0",
        "grpc_servicer": "8abb3d9a201fd9e955bddea291e1f724e8de6a08",
    }.items():
        trees[path] = subprocess.check_output(
            ["git", "rev-parse", "HEAD:" + path], text=True
        ).strip()
        require(trees[path] == expected, "product dependency source changed: " + path)
    require(
        not subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], text=True
        ).strip(),
        "tracked source is dirty",
    )
    write(
        "source.json",
        {
            "head": head,
            "product": "04acb7750bfdd329e396040b82494b7fa3c7366b",
            "helper_sha256": HELPER_SHA,
            "trees": trees,
        },
    )


def snapshot(label):
    records = []
    for dist in metadata.distributions():
        name = re.sub(r"[-_.]+", "-", dist.metadata["Name"]).lower()
        if name not in NAMES:
            continue
        root = Path(dist._path)
        path = root if root.is_file() else root / "METADATA"
        if not path.is_file():
            path = root / "PKG-INFO"
        record = archive(path, f"metadata-{label}/{len(records)}-{name}.METADATA", 1024 * 1024)
        records.append({"name": name, "version": dist.version, **record})
    require({item["name"] for item in records} == NAMES, "missing image package metadata")
    active = {
        name: {
            "version": metadata.version(name),
            "origin": str(Path(metadata.distribution(name)._path).resolve()),
        }
        for name in sorted(NAMES)
    }
    original = os.environ.get("PIP_CONSTRAINT")
    constraints = []
    require(len((original or "").split()) <= 8, "too many constraint files")
    for value in (original or "").split():
        constraints.append(
            archive(value, f"constraints-{label}/{len(constraints)}.txt", 1024 * 1024)
        )
    value = {
        "python": sys.version,
        "executable": sys.executable,
        "records": sorted(
            records, key=lambda item: (item["name"], item["version"], item["origin"])
        ),
        "active": active,
        "constraint_environment": original,
        "constraints": constraints,
    }
    write("packages-" + label + ".json", value)
    return value


def unchanged(before, after):
    for key in ["records", "constraints"]:
        normalized = [
            [{k: v for k, v in item.items() if k != "file"} for item in state[key]]
            for state in [before, after]
        ]
        require(normalized[0] == normalized[1], "image package or constraint bytes/origins changed")
    require(before["active"] == after["active"], "active image package changed")
    require(
        before["constraint_environment"] == after["constraint_environment"],
        "constraint environment changed",
    )


def native():
    import torch

    require(not torch.cuda.is_initialized(), "GPU initialized before native probe")
    import cairo
    import cairo._cairo

    image = cairo.ImageSurface(cairo.FORMAT_ARGB32, 2, 2)
    image.write_to_png(str(OUT / "native.png"))
    raw = (OUT / "native.png").read_bytes()
    require(
        raw[:8] == b"\x89PNG\r\n\x1a\n" and raw[16:24] == b"\0\0\0\2\0\0\0\2",
        "invalid 2x2 native PNG",
    )
    extension = Path(cairo._cairo.__file__).resolve(strict=True)
    dist = metadata.distribution("pycairo")
    record = dist.read_text("RECORD")
    require(record is not None, "missing installed Pycairo RECORD")
    matches = [
        row
        for row in csv.reader(io.StringIO(record))
        if Path(dist.locate_file(row[0])).resolve() == extension
    ]
    expected = "sha256=" + base64.urlsafe_b64encode(
        hashlib.sha256(extension.read_bytes()).digest()
    ).decode().rstrip("=")
    require(
        len(matches) == 1 and matches[0][1:] == [expected, str(extension.stat().st_size)],
        "installed extension differs from RECORD",
    )
    maps = Path("/proc/self/maps").read_text()
    (OUT / "native-maps.txt").write_text(maps)
    libraries = {
        line.split()[-1]
        for line in maps.splitlines()
        if line.split()[-1].startswith("/")
        and Path(line.split()[-1]).name.startswith("libcairo.so")
    }
    require(len(libraries) == 1, "ambiguous or missing actually loaded libcairo")
    library = Path(libraries.pop()).resolve(strict=True)
    files = {
        "extension": archive(extension, "native/" + extension.name),
        "libcairo": archive(library, "native/" + library.name),
        "metadata": archive(Path(dist._path) / "METADATA", "native/METADATA", 1024 * 1024),
        "record": archive(Path(dist._path) / "RECORD", "native/RECORD", 1024 * 1024),
    }
    for key in ["extension", "libcairo"]:
        for option in ["-h", "-V"]:
            result = subprocess.run(
                ["readelf", option, files[key]["origin"]],
                capture_output=True,
                text=True,
                timeout=10,
            )
            (OUT / f"native/{key}-{option[1:]}.txt").write_text(result.stdout + result.stderr)
            require(result.returncode == 0, "native ELF inspection failed")
    require(not torch.cuda.is_initialized(), "GPU initialized by native probe")
    devices("native-after")
    write(
        "native.json",
        {
            "pycairo": dist.version,
            "cairo": cairo.cairo_version_string(),
            "torch": torch.__version__,
            "cuda_initialized_before": False,
            "cuda_initialized_after": False,
            "png_sha256": sha(raw),
            "files": files,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["before", "after", "native", "devices-after"])
    mode = parser.parse_args().mode
    OUT.mkdir(exist_ok=True)
    if mode == "before":
        devices("before")
        source_identity()
        snapshot("before")
    elif mode == "after":
        devices("after")
        unchanged(json.loads((OUT / "packages-before.json").read_text()), snapshot("after"))
    elif mode == "native":
        native()
    else:
        devices("final")


if __name__ == "__main__":
    main()
