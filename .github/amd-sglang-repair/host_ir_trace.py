"""Record only the real parser's bounded host text and public source identities."""

import hashlib
import json
import sys
from pathlib import Path

from tilelang_host_ir import SOURCE, verify_loaded

LIMIT = 512 * 1024
SOURCES = (
    SOURCE,
    Path("/opt/tilelang/tilelang/jit/adapter/cython/adapter.py"),
    Path("/opt/tilelang/3rdparty/tvm/python/tvm/tir/function.py"),
    Path("/opt/tilelang/3rdparty/tvm/python/tvm/runtime/object.py"),
    Path("/opt/tilelang/3rdparty/tvm/python/tvm/runtime/script_printer.py"),
)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class ParserTrace:
    def __init__(self, code, directory):
        self.code, self.directory = code, directory
        self.calls, self.exceptions = [], []
        self.invalid, self.total = [], 0

    def trace(self, frame, event, argument):
        if frame.f_code is not self.code:
            return None
        if event == "exception":
            self.exceptions.append(argument[0].__name__)
        if event == "return":
            local = frame.f_locals
            host = local.get("host_code")
            names = local.get("function_names")
            if not isinstance(host, str) or len(host.encode()) > LIMIT:
                self.invalid.append("missing/oversized actual parser host_code")
                return self.trace
            if (
                not isinstance(names, list)
                or not 0 < len(names) <= 128
                or not all(isinstance(n, str) and 0 < len(n) <= 256 for n in names)
            ):
                self.invalid.append("missing/invalid actual device function names")
                return self.trace
            raw = host.encode()
            self.total += len(raw)
            if self.total > 2 * 1024 * 1024:
                self.invalid.append("host-IR total size exceeds cap")
                return self.trace
            target = self.directory / f"host-{len(self.calls):02d}.txt"
            target.write_bytes(raw)
            self.calls.append(
                {
                    "host_file": target.name,
                    "host_bytes": len(raw),
                    "host_sha256": sha(raw),
                    "function_names": list(names),
                    "function_names_index": {
                        name: index
                        for name, index in local.get("function_names_index", {}).items()
                        if name in names and isinstance(index, int)
                    },
                    "exception_types_observed": list(self.exceptions),
                    "parser_file": self.code.co_filename,
                    "parser_first_line": self.code.co_firstlineno,
                }
            )
            self.exceptions.clear()
        return self.trace

    def __enter__(self):
        self.previous = sys.gettrace()
        if self.previous is not None:
            raise ValueError("refuse to replace an existing Python trace hook")
        self.directory.mkdir()
        sys.settrace(self.trace)
        return self

    def __exit__(self, kind, value, traceback):
        sys.settrace(self.previous)
        (self.directory / "calls.json").write_text(
            json.dumps({"calls": self.calls, "invalid": self.invalid}, indent=2) + "\n"
        )
        if kind is None:
            if self.invalid:
                raise ValueError("; ".join(self.invalid))
            if not self.calls:
                raise ValueError("actual TileLang parser call was not observed")


def observe(directory):
    module = sys.modules["tilelang.jit.adapter.wrapper"]
    source = Path(module.__file__).resolve()
    verify_loaded({"path": str(source), "sha256": sha(source.read_bytes())})
    function = module.TLCUDASourceWrapper.parse_source_information
    if Path(function.__code__.co_filename).resolve() != SOURCE:
        raise ValueError("actual parser code filename differs")
    copy_sources(directory.parent / "host-ir-sources")
    return ParserTrace(function.__code__, directory)


def copy_sources(copied, paths=SOURCES):
    # Whole public files only; no arbitrary globals, environment or frame locals are serialized.
    records, total = [], 0
    copied.mkdir()
    for index, path in enumerate(paths):
        if path.resolve() != path:
            raise ValueError("public source path escapes allowlist")
        raw = path.read_bytes()
        total += len(raw)
        if len(raw) > LIMIT or total > 2 * 1024 * 1024:
            raise ValueError("public host-IR source size exceeds cap")
        name = f"{index:02d}-{path.name}"
        (copied / name).write_bytes(raw)
        records.append({"path": str(path), "file": name, "bytes": len(raw), "sha256": sha(raw)})
    (copied / "index.json").write_text(json.dumps(records, indent=2) + "\n")
