"""Copy bounded public SGLang source evidence without importing the engine."""

import argparse
import errno
import hashlib
import json
import os
import stat
from contextlib import ExitStack
from datetime import UTC, datetime
from importlib import metadata, util
from pathlib import Path, PurePosixPath

FILE_LIMIT = 512 * 1024
TOTAL_LIMIT = 2 * 1024 * 1024
PACKAGE_PATHS = {
    "sglang": (
        "srt/disaggregation/__init__.py",
        "srt/disaggregation/encoder/__init__.py",
        "srt/disaggregation/encoder/grpc_server.py",
        "srt/disaggregation/encoder/runtime.py",
        "srt/disaggregation/encoder/server.py",
        "srt/disaggregation/encoder/receiver.py",
        "srt/managers/io_struct.py",
    ),
    "smg_grpc_proto": (
        "sglang_encoder_pb2.py",
        "sglang_encoder_pb2_grpc.py",
        "sglang_scheduler_pb2.py",
    ),
}
SOURCE_PATHS = tuple(path for paths in PACKAGE_PATHS.values() for path in paths)
OPTIONAL_PATHS = {"srt/disaggregation/encoder/__init__.py"}


def package_location(package="sglang"):
    spec = util.find_spec(package)
    locations = list(spec.submodule_search_locations or []) if spec else []
    if len(locations) != 1:
        raise ValueError("SGLang must resolve to exactly one package directory")
    root = Path(locations[0]).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("SGLang package location is not a directory")
    if spec.origin and not Path(spec.origin).resolve().is_relative_to(root):
        raise ValueError("SGLang package origin escapes its package directory")
    distribution = metadata.distribution(package)
    return root, distribution.version, str(distribution.locate_file("").resolve())


def read_public_source(root, relative, budget):
    parts = PurePosixPath(relative).parts
    if relative not in SOURCE_PATHS or not parts or any(p in {"..", "."} for p in parts):
        raise ValueError("Path is outside the fixed public source allowlist")
    if PurePosixPath(relative).is_absolute():
        raise ValueError("Absolute source paths are forbidden")
    # Descriptor-relative traversal refuses every interior symlink, including
    # parent-directory links; editable installation roots were resolved above.
    with ExitStack() as stack:
        directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        stack.callback(os.close, directory)
        for part in parts[:-1]:
            directory = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory
            )
            stack.callback(os.close, directory)
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        stream = stack.enter_context(os.fdopen(fd, "rb"))
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("Source is not a regular file")
        if before.st_size > min(FILE_LIMIT, TOTAL_LIMIT - budget["bytes_read"]):
            raise ValueError("Source exceeds the per-file or remaining total byte limit")
        data = stream.read(before.st_size)
        budget["bytes_read"] += len(data)
        after = os.fstat(stream.fileno())
        if len(data) != before.st_size or (before.st_size, before.st_mtime_ns) != (
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ValueError("Source changed during collection")
        return data


def collect(output, package="sglang", budget=None):
    if package not in PACKAGE_PATHS:
        raise ValueError("Package is outside the fixed public allowlist")
    output = Path(output).resolve()
    try:
        root, version, distribution_root = package_location(package)
    except (ImportError, OSError, ValueError, metadata.PackageNotFoundError) as error:
        output.mkdir(parents=True, exist_ok=False)
        (output / "source-index.json").write_text(
            json.dumps({"package": package, "status": "unavailable", "error": type(error).__name__})
            + "\n"
        )
        return 1
    if output.is_relative_to(root):
        raise ValueError("Output must be outside the SGLang package")
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "schema_version": 1,
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "package": package,
        "version": version,
        "package_root": str(root),
        "distribution_root": distribution_root,
        "per_file_limit_bytes": FILE_LIMIT,
        "total_limit_bytes": TOTAL_LIMIT,
        "bytes_read": 0,
        "files": [],
    }
    distribution = metadata.distribution(package)
    record["metadata"] = {
        key: distribution.metadata.get_all(key) or []
        for key in ("Name", "Version", "Requires-Python", "Requires-Dist")
    }
    budget = budget if budget is not None else {"bytes_read": 0}
    initial_bytes = budget["bytes_read"]
    rejected = False
    for relative in PACKAGE_PATHS[package]:
        entry = {"path": relative, "required": relative not in OPTIONAL_PATHS}
        try:
            data = read_public_source(root, relative, budget)
        except FileNotFoundError:
            entry["status"] = "missing"
            rejected = rejected or entry["required"]
        except (OSError, ValueError) as error:
            rejected = True
            entry.update(status="rejected", error=type(error).__name__)
            if isinstance(error, OSError):
                entry["errno"] = errno.errorcode.get(error.errno, str(error.errno))
            else:
                entry["reason"] = str(error)
        else:
            target = output / "sources" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            entry.update(
                status="copied",
                bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest(),
            )
        record["files"].append(entry)
    record["bytes_read"] = budget["bytes_read"] - initial_bytes
    record["shared_total_bytes_read"] = budget["bytes_read"]
    record["status"] = "rejected" if rejected else "recorded"
    (output / "source-index.json").write_text(json.dumps(record, indent=2) + "\n")
    return int(rejected)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    output = parser.parse_args().output
    import sys

    forbidden = {"torch", *PACKAGE_PATHS}
    if forbidden.intersection(sys.modules):
        raise RuntimeError("Source-only process already imported an engine or Torch")
    budget = {"bytes_read": 0}
    result = max(collect(output / package, package, budget) for package in PACKAGE_PATHS)
    if forbidden.intersection(sys.modules):
        raise RuntimeError("Source collection imported an engine or Torch")
    (output / "capture-summary.json").write_text(
        json.dumps(
            {
                "packages": list(PACKAGE_PATHS),
                "bytes_read": budget["bytes_read"],
                "status": "rejected" if result else "recorded",
                "forbidden_imports": [],
            },
            indent=2,
        )
        + "\n"
    )
    raise SystemExit(result)
