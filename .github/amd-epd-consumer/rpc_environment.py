"""Install the two audited local RPC wheels and retain the image runtime cohort."""

import argparse
import base64
import hashlib
import importlib
import importlib.metadata as md
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

from artifact_io import require
from packaging.version import Version
from rpc_wheels import verify_wheel
from wheel_record import verify as verify_record

PACKAGES = {
    "smg-grpc-proto": ("smg_grpc_proto", "0.4.29"),
    "smg-grpc-servicer": ("smg_grpc_servicer", "0.14.2"),
}
RUNTIME = ["torch", "sglang", "grpcio", "protobuf", "grpcio-health-checking", "grpcio-reflection"]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def constraints(value):
    return [
        {
            "path": str(Path(p).resolve()),
            "sha256": digest(Path(p).read_bytes()),
            "text": Path(p).read_text(),
        }
        for p in (value or "").split()
    ]


def runtime():
    result = {}
    for name in RUNTIME:
        dist = md.distribution(name)
        path = Path(dist._path) / "METADATA"
        result[name] = {
            "version": dist.version,
            "path": str(path.resolve()),
            "sha256": digest(path.read_bytes()),
        }
    return result


def prepare(directory):
    directory.mkdir(parents=True, exist_ok=False)
    current = runtime()
    require(Version(current["grpcio"]["version"]) >= Version("1.81.1"), "SGLang gRPC gencode floor")
    require(
        Version(current["protobuf"]["version"]) >= Version("6.33.5"),
        "SGLang protobuf gencode floor",
    )
    require(
        current["grpcio-reflection"]["version"] == current["grpcio-health-checking"]["version"],
        "image companions differ; the shared installer needs an explicit cohort",
    )
    original = os.environ.get("PIP_CONSTRAINT")
    pins = directory / "runtime-pins.txt"
    pins.write_text("".join(f"{name}=={current[name]['version']}\n" for name in RUNTIME))
    effective = ((original + " ") if original else "") + str(pins.resolve())
    state = {
        "python": sys.executable,
        "runtime": current,
        "original_environment": original,
        "original_constraints": constraints(original),
        "effective_environment": effective,
        "effective_constraints": constraints(effective),
    }
    (directory / "plan.json").write_text(json.dumps(state, indent=2) + "\n")
    with Path(os.environ["GITHUB_ENV"]).open("a") as output:
        output.write("PIP_CONSTRAINT=" + effective + "\n")
        output.write(
            "SMG_CI_GRPC_COMPANION_VERSION=" + current["grpcio-reflection"]["version"] + "\n"
        )


def check_runtime(directory):
    state = json.loads((directory / "plan.json").read_text())
    require(sys.executable == state["python"], "RPC audit interpreter changed")
    require(
        os.environ.get("PIP_CONSTRAINT") == state["effective_environment"],
        "runtime constraints not exported",
    )
    require(
        constraints(os.environ.get("PIP_CONSTRAINT")) == state["effective_constraints"],
        "runtime constraint path/hash/text/order changed",
    )
    require(runtime() == state["runtime"], "image runtime version/metadata origin changed")
    return state


def wheel_paths(directory):
    paths = sorted(directory.glob("*.whl"))
    expected = {f"{module}-{version}-py3-none-any.whl" for module, version in PACKAGES.values()}
    require(
        len(paths) == 2 and {p.name for p in paths} == expected,
        "exact two local RPC wheels required",
    )
    for path in paths:
        require(
            not path.is_symlink() and path.resolve().parent == directory.resolve(),
            "RPC wheel path escape",
        )
        verify_record(path.read_bytes())
        verify_wheel(path)
    return paths


def verify(directory, wheels, label):
    report = {"verified": False, "stage": label, "pid": os.getpid(), "python": sys.executable}
    try:
        report["runtime"] = check_runtime(directory)["runtime"]
        paths = wheel_paths(wheels)
        records = {}
        for name, (module, version) in PACKAGES.items():
            dist = md.distribution(name)
            require(dist.version == version, "installed RPC package version differs")
            path = next(p for p in paths if p.name.startswith(module + "-"))
            direct = json.loads(dist.read_text("direct_url.json") or "null")
            require(
                isinstance(direct, dict) and not direct.get("dir_info", {}).get("editable"),
                "RPC package is editable or has no wheel origin",
            )
            require(
                direct.get("url") == path.resolve().as_uri(), "installed RPC wheel origin differs"
            )
            installed = {str(f): f for f in dist.files or ()}
            with zipfile.ZipFile(path) as archive:
                expected = {}
                for member in archive.namelist():
                    if member.endswith(".dist-info/RECORD"):
                        continue
                    require(member in installed, "installed RPC RECORD member missing: " + member)
                    raw = archive.read(member)
                    entry = installed[member]
                    target = Path(dist.locate_file(entry)).resolve()
                    require(target.read_bytes() == raw, "installed RPC bytes differ: " + member)
                    require(
                        entry.hash and entry.hash.mode == "sha256" and entry.size == len(raw),
                        "installed RPC RECORD is unhashed",
                    )
                    require(
                        entry.hash.value
                        == base64.urlsafe_b64encode(hashlib.sha256(raw).digest())
                        .decode()
                        .rstrip("="),
                        "installed RPC RECORD differs",
                    )
                    expected[str(target)] = {"sha256": digest(raw), "bytes": len(raw)}
            loaded = importlib.import_module(module)
            origin = str(Path(loaded.__file__).resolve())
            require(
                origin in expected
                and digest(Path(origin).read_bytes()) == expected[origin]["sha256"],
                "RPC import shadowed by checkout or another distribution",
            )
            records[name] = {
                "version": version,
                "origin": origin,
                "wheel": str(path.resolve()),
                "wheel_sha256": digest(path.read_bytes()),
                "files": expected,
            }
        report.update(verified=True, packages=records)
        return report
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        (directory / f"{label}.json").write_text(json.dumps(report, indent=2) + "\n")


def install(directory, wheels):
    check_runtime(directory)
    paths = wheel_paths(wheels)
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", *PACKAGES], check=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-index", *map(str, paths)], check=True
    )
    verify(directory, wheels, "paired-install")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "install", "installed", "final"])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--wheels", type=Path, default=Path("epd-rpc-wheels"))
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.directory)
    elif args.mode == "install":
        install(args.directory, args.wheels)
    else:
        verify(args.directory, args.wheels, args.mode)
