"""Observe loaded native libraries in the same process as all three hard gates."""

import hashlib
import importlib.metadata as metadata
import json
import os
import sys
from pathlib import Path

from tilelang_host_ir import verify_loaded as verify_tilelang_wrapper
from tvm_slots import verify_loaded as verify_tvm_support

ENV = (
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "TVM_LIBRARY_PATH",
    "TVM_FFI_LIBRARY_PATH",
    "TVM_IMPORT_PYTHON_PATH",
    "TVM_FFI",
    "TVM_FFI_DISABLE_TORCH_C_DLPACK",
    "TORCH_EXTENSIONS_DIR",
    "MAX_JOBS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "PIP_CONSTRAINT",
)
PACKAGES = ("apache-tvm-ffi", "torch-c-dlpack-ext", "tilelang", "xgrammar", "torch", "sglang")
MODULES = (
    "tvm_ffi",
    "tvm_ffi.core",
    "tvm_ffi.base",
    "tvm_ffi.registry",
    "tvm_ffi.libinfo",
    "tvm",
    "tvm._ffi.base",
    "tvm.runtime.support",
    "tilelang.jit.adapter.wrapper",
    "tilelang",
    "xgrammar",
)


def relevant(path):
    return any(part in path for part in ("/tvm_ffi/", "/tvm/", "/tilelang/", "/xgrammar/"))


def libraries(text):
    paths = set()
    for line in text.splitlines():
        fields = line.split(None, 5)
        if len(fields) == 6 and fields[5].startswith("/"):
            path = fields[5]
            if relevant(path) and ".so" in Path(path).name:
                paths.add(path)
    return sorted(paths)


class RuntimeIdentity:
    def __init__(self, directory):
        self.directory = directory
        self.cache = {}
        self.sequence = 0
        self.failure_sources = set()

    def file(self, filename):
        path = Path(filename)
        entry = {"path": str(path), "exists": path.is_file()}
        if not entry["exists"]:
            return entry
        path = path.resolve()
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns)
        if key not in self.cache:
            digest = hashlib.sha256()
            with path.open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
            self.cache[key] = digest.hexdigest()
        return {"path": str(path), "exists": True, "bytes": stat.st_size, "sha256": self.cache[key]}

    def capture(self, phase):
        self.directory.mkdir(exist_ok=True)
        maps = Path("/proc/self/maps").read_text()
        modules = {}
        for name in MODULES:
            module = sys.modules.get(name)
            filename = getattr(module, "__file__", None)
            modules[name] = self.file(filename) if filename else {"loaded": False}
        handles = {}
        for name in MODULES:
            module = sys.modules.get(name)
            library = getattr(module, "_LIB", None) or getattr(module, "LIB", None)
            path = getattr(library, "_name", None)
            if isinstance(path, str):
                handles[name] = self.file(path)
        packages = {}
        for name in PACKAGES:
            dist = metadata.distribution(name)
            packages[name] = {
                "version": dist.version,
                "metadata_root": str(Path(dist._path).resolve()),
            }
        installed = metadata.distribution("apache-tvm-ffi")
        registry = self.file(installed.locate_file("tvm_ffi/registry.py"))
        constraints = [self.file(value) for value in os.environ.get("PIP_CONSTRAINT", "").split()]
        value = {
            "phase": phase,
            "pid": os.getpid(),
            "executable": sys.executable,
            "environment_allowlist": {key: os.environ.get(key) for key in ENV},
            "installed_packages": packages,
            "installed_registry": registry,
            "loaded_modules": modules,
            "native_library_handles": handles,
            "loaded_libraries": [self.file(p) for p in libraries(maps)],
            "failure_sources": [self.file(p) for p in sorted(self.failure_sources)],
            "constraint_files": constraints,
        }
        target = self.directory / f"runtime-identity-{self.sequence:02d}.json"
        target.write_text(json.dumps(value, indent=2) + "\n")
        self.sequence += 1
        return value

    def pytest_sessionstart(self, session):
        self.capture("session-start")

    def pytest_exception_interact(self, node, call, report):
        if call.excinfo:
            for entry in call.excinfo.traceback:
                filename = str(entry.path)
                if relevant(filename) and filename.endswith(".py"):
                    self.failure_sources.add(filename)
        self.capture("exception:" + report.nodeid)

    def pytest_runtest_logreport(self, report):
        if report.when == "teardown":
            self.capture("teardown:" + report.nodeid)

    def pytest_sessionfinish(self, session, exitstatus):
        value = self.capture("session-finish:" + str(exitstatus))
        if exitstatus == 0:
            if not all(item.get("exists", False) for item in value["loaded_libraries"]):
                raise RuntimeError("loaded native library file missing")
            paths = [item["path"] for item in value["loaded_libraries"]]
            if not any("libtvm_ffi.so" in p for p in paths):
                raise RuntimeError("missing actual loaded FFI native library identity")
            if not any("libtilelang" in p for p in paths):
                raise RuntimeError("missing actual loaded TileLang native library identity")
            if not any("libtvm" in p and "tvm_ffi" not in p for p in paths):
                raise RuntimeError("missing actual loaded bundled TVM native library identity")

            expected = {
                "core.abi3.so": "48dbc71a92a0d134feaeb5f81d7e7760485e856404d1e26e776330fcc6cfea06",
                "libtvm_ffi.so": "4f3e1f951aa56d2047b3353e75d6cea112f3f4ecf29fb4910b7b6ca11ca93c90",
            }
            for name, digest in expected.items():
                found = [
                    item
                    for item in value["loaded_libraries"]
                    if "/tvm_ffi/" in item["path"] and Path(item["path"]).name == name
                ]
                if len(found) != 1 or found[0]["sha256"] != digest:
                    raise RuntimeError("loaded FFI native identity differs: " + name)

            fixed = "68e9cd4017191fb00bf9193e751b0d79b27047005dea5c01ff2babd7a527f902"
            loaded = value["loaded_modules"].get("tvm_ffi.registry", {})
            installed = value["installed_registry"]
            if loaded.get("sha256") != fixed or loaded != installed:
                raise RuntimeError("actual imported registry differs from installed candidate")

            verify_tvm_support(value["loaded_modules"].get("tvm.runtime.support", {}))
            verify_tilelang_wrapper(value["loaded_modules"].get("tilelang.jit.adapter.wrapper", {}))
