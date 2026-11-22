"""Finite source-only diagnostics for the original public Qwen3-VL CI case.

No engine imports, subprocesses, network, model loading or behavior changes.
Only main() selects inputs/outputs; helper arguments support isolated unit fixtures.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata as metadata
import json
import os
import re
import resource
import stat
from pathlib import Path

SG_ROOT = Path("/sgl-workspace/sglang/python/sglang/srt")
HF_ROOT = Path("/models/hub")
MODEL = "models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
VENDOR_FILES = (
    "managers/scheduler.py",
    "managers/scheduler_components/request_receiver.py",
    "managers/schedule_batch.py",
    "multimodal/processors/qwen_vl.py",
    "multimodal/processors/base_processor.py",
    "multimodal/media_artifacts/base.py",
    "multimodal/media_artifacts/cache.py",
)
MODEL_FILES = (
    "config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer_config.json",
    "generation_config.json",
    "chat_template.json",
    "chat_template.jinja",
)
SYMBOLS = {
    "_maybe_compute_mrope_positions",
    "from_processor_output",
    "mrope_from_unpadded",
    "get_rope_index",
}
SOURCE_LIMIT = 1024 * 1024
EXCERPT_LIMIT = 128 * 1024
TOTAL_EXCERPT_LIMIT = 512 * 1024


def read_fixed(
    path: Path, root: Path, *, internal_links: bool = False, limit: int = SOURCE_LIMIT
) -> tuple[dict, bytes | None]:
    """Read a bounded stable regular file beneath one fixed root; no outside links."""
    row = {"path": str(path), "status": "UNKNOWN"}
    try:
        real_root = root.resolve(strict=True)
        real = path.resolve(strict=True)
        if not real.is_relative_to(real_root):
            return dict(row, status="REJECTED_OUTSIDE_ROOT"), None
        if not internal_links:
            relative = path.relative_to(root)
            cursor = root
            for part in relative.parts:
                cursor = cursor / part
                if cursor.is_symlink():
                    return dict(row, status="REJECTED_SOURCE_LINK"), None
        fd = os.open(real, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
                return dict(row, status="REJECTED_TYPE_OR_SIZE"), None
            raw = stream.read(limit)
            row["bytes_read"] = len(raw)
            after = os.fstat(stream.fileno())

        def identity(s):
            return (
                s.st_dev,
                s.st_ino,
                s.st_size,
                s.st_mtime_ns,
                s.st_ctime_ns,
            )

        if len(raw) != before.st_size or identity(before) != identity(after):
            return dict(row, status="REJECTED_CHANGED_DURING_READ"), None
        return dict(
            row,
            status="CAPTURED",
            bytes=len(raw),
            sha256=hashlib.sha256(raw).hexdigest(),
            resolved_path=str(real),
        ), raw
    except OSError as exc:
        return dict(row, status="UNAVAILABLE", error_type=type(exc).__name__, errno=exc.errno), None
    except (RuntimeError, ValueError) as exc:
        return dict(row, status="REJECTED_PATH", error_type=type(exc).__name__), None


def source_summary(row: dict, raw: bytes | None) -> dict:
    """Public vendor/source text only; no execution or dynamic imports."""
    if raw is None:
        return row
    try:
        text = raw.decode("utf-8")
        tree = ast.parse(text)
        excerpts = []
        remaining = TOTAL_EXCERPT_LIMIT
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in SYMBOLS:
                code = ast.get_source_segment(text, node) or ""
                size = len(code.encode())
                item = {
                    "name": node.name,
                    "line": node.lineno,
                    "end_line": node.end_lineno,
                    "source_sha256": hashlib.sha256(code.encode()).hexdigest(),
                }
                if size <= min(EXCERPT_LIMIT, remaining):
                    item["public_source_body"] = code
                    remaining -= size
                else:
                    item["body_status"] = "OMITTED_BOUNDED_EXCERPT"
                excerpts.append(item)
        return dict(
            row,
            symbols=excerpts,
            scheduler_class_bases=[
                ast.unparse(base)
                for n in tree.body
                if isinstance(n, ast.ClassDef) and n.name == "Scheduler"
                for base in n.bases
            ],
        )
    except (UnicodeError, SyntaxError) as exc:
        return dict(row, parse_status="UNAVAILABLE", error_type=type(exc).__name__)


def collect(sg_root: Path, hf_root: Path, repo: Path) -> dict:
    result = {
        "status": "SOURCE_CAPTURE_ONLY_NOT_MODEL_OR_GPU_QUALIFICATION",
        "model": "Qwen/Qwen3-VL-8B-Instruct",
        "expected_cached_snapshot": MODEL.rsplit("/", 1)[-1],
        "source": [],
        "model_configs": [],
        "installed_versions": {},
        "limits": (
            "Fixed7 vendor sources<=1MiB each; fixed7 public model configs<=1MiB each; "
            "no weights/tokenizer body; <=512KiB source excerpts per file."
        ),
        "scope": (
            "Source compatibility/reference only. "
            "No proof which processor, hook or model revision executed in a request."
        ),
    }
    for relative in VENDOR_FILES:
        row, raw = read_fixed(sg_root / relative, sg_root)
        result["source"].append(source_summary(row, raw))
    plugin = repo / "grpc_servicer/smg_grpc_servicer/sglang/zmq_plugin.py"
    row, raw = read_fixed(plugin, repo)
    result["source"].append(source_summary(row, raw))
    for relative in MODEL_FILES:
        row, _ = read_fixed(hf_root / MODEL / relative, hf_root, internal_links=True)
        # Config bodies are deliberately not emitted.
        result["model_configs"].append(row)
    for name in (
        "sglang",
        "torch",
        "transformers",
        "triton",
        "smg-grpc-servicer",
        "smg-grpc-proto",
    ):
        try:
            result["installed_versions"][name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result["installed_versions"][name] = None
    result["missing_or_rejected"] = sum(
        row["status"] != "CAPTURED" for row in result["source"] + result["model_configs"]
    )
    return result


def collect_loaded_sources(logs: Path, output: Path, roots: dict[str, Path]) -> dict:
    """Only metadata-observed public SDK modules, same-job SHA, bounded source bodies."""
    result = {
        "status": "LOADED_PUBLIC_SOURCE_CAPTURE_ONLY",
        "logs": [],
        "source": [],
        "limits": {
            "source_files": 6,
            "source_each": SOURCE_LIMIT,
            "source_total": 4 * 1024**2,
            "worker_log_files": 8,
            "worker_log_prefix_each": 2 * 1024**2,
        },
        "scope": "No import/model execution. Prefix logs may omit metadata; unavailable is unknown.",
    }
    observed, records_seen = {}, 0
    # Exact existing Qwen worker-log filename contract; never scan other worker logs.
    for path in sorted(logs.glob("worker-Qwen__Qwen3-VL-8B-Instruct_sglang_*.log"))[:8]:
        fd = None
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode):
                continue
            with os.fdopen(fd, "rb", closefd=False) as stream:
                raw = stream.read(2 * 1024**2)
            result["logs"].append(
                {
                    "name": path.name,
                    "prefix_bytes": len(raw),
                    "prefix_sha256": hashlib.sha256(raw).hexdigest(),
                    "prefix_truncated": st.st_size > len(raw),
                }
            )
            for line in raw.splitlines():
                marker = b"SMG_QWENVL_REQUEST_DIAGNOSTIC "
                if marker not in line or len(line) > 16384:
                    continue
                try:
                    record = json.loads(line.split(marker, 1)[1])
                except (ValueError, UnicodeError):
                    continue
                if not isinstance(record, dict) or record.get("event") != "model_consumer_metadata":
                    continue
                fields = record.get("metadata")
                if not isinstance(fields, dict) or records_seen >= 16:
                    continue
                records_seen += 1
                files = fields.get("source_files", {})
                if not isinstance(files, dict) or len(files) > 6:
                    continue
                for module, info in files.items():
                    if (
                        not isinstance(module, str)
                        or len(module) > 128
                        or not re.fullmatch(
                            r"(?:sglang|transformers|sgl_kernel)(?:\.[A-Za-z_][A-Za-z_0-9]*)+",
                            module,
                        )
                        or not isinstance(info, dict)
                        or info.get("status") != "captured"
                    ):
                        continue
                    value = (info.get("path"), info.get("sha256"))
                    if (
                        not all(isinstance(v, str) for v in value)
                        or len(value[0]) > 256
                        or not re.fullmatch(r"[0-9a-f]{64}", value[1])
                    ):
                        continue
                    if module not in observed and len(observed) >= 6:
                        continue
                    values = observed.setdefault(module, set())
                    if len(values) < 2:
                        values.add(value)
        except OSError:
            result["logs"].append({"name": path.name, "status": "UNAVAILABLE"})
        finally:
            if fd is not None:
                os.close(fd)
    remaining, attempted = 4 * 1024**2, 0
    for module, values in sorted(observed.items()):
        if attempted >= 6:
            break
        row = {"module": module, "status": "UNAVAILABLE"}
        result["source"].append(row)
        if len(values) != 1:
            row["status"] = "REJECTED_INCONSISTENT_METADATA"
            continue
        name, expected = next(iter(values))
        namespace, *parts = module.split(".")
        root = roots.get(namespace)
        path = Path(name)
        if root is None or not path.is_absolute() or path.suffix != ".py":
            row["status"] = "REJECTED_MODULE_PATH"
            continue
        relative = Path(*parts)
        if path not in (root / relative.with_suffix(".py"), root / relative / "__init__.py"):
            row["status"] = "REJECTED_MODULE_PATH"
            continue
        attempted += 1
        info, body = read_fixed(path, root, limit=min(SOURCE_LIMIT, remaining))
        remaining -= info.get("bytes_read", 0)
        row.update(info, expected_sha256=expected)
        if body is None:
            continue
        if info.get("sha256") != expected:
            row["status"] = "REJECTED_SHA_MISMATCH"
            continue
        target = output / (expected + ".py")
        if not target.exists():
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, "wb") as stream:
                stream.write(body)
                stream.flush()
                os.fsync(stream.fileno())
        row["artifact_file"] = target.name
    result["metadata_records_seen"] = records_seen
    result["source_bytes_read"] = 4 * 1024**2 - remaining
    result["captured"] = sum(row.get("artifact_file") is not None for row in result["source"])
    if not result["captured"]:
        result["status"] = "LOADED_PUBLIC_SOURCE_UNAVAILABLE"
    result["unavailable_does_not_mean_model_or_tests_passed"] = True
    return result


def main() -> int:
    if os.environ.get("SMG_QWENVL_SOURCE_DIAGNOSTIC") != "1":
        return 0
    os.umask(0o077)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2, 512 * 1024**2))
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024**2, 8 * 1024**2))
    repo = Path(__file__).resolve().parents[1]
    try:
        logs = repo / "e2e-logs"
        logs.mkdir(mode=0o700, exist_ok=True)
        if logs.is_symlink() or not logs.is_dir() or logs.resolve() != repo / "e2e-logs":
            raise ValueError("unsafe log directory")
        loaded = os.environ.get("SMG_QWENVL_LOADED_SOURCE_DIAGNOSTIC") == "1"
        output = logs / (
            "qwen-vl-loaded-source-diagnostic" if loaded else "qwen-vl-source-diagnostic"
        )
        output.mkdir(mode=0o700, exist_ok=False)
        if loaded:
            roots = {"sglang": SG_ROOT.parent}
            for namespace, distribution in (
                ("transformers", "transformers"),
                ("sgl_kernel", "sgl-kernel"),
            ):
                try:
                    roots[namespace] = Path(
                        metadata.distribution(distribution).locate_file(namespace)
                    )
                except metadata.PackageNotFoundError:
                    pass
            result = collect_loaded_sources(logs, output, roots)
        else:
            result = collect(SG_ROOT, HF_ROOT, repo)
        payload = (json.dumps(result, indent=2) + "\n").encode()
        if len(payload) > 6 * 1024**2:
            raise ValueError("bounded diagnostic output exceeded")
        fd = os.open(
            output / "source-and-config.json",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        print("QwenVL source diagnostic captured; missing or unavailable fields remain explicit.")
    except Exception as exc:
        # Non-gating diagnostics never replace the original pytest status; no exception body/env.
        print("QwenVL source diagnostic unavailable: " + type(exc).__name__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
