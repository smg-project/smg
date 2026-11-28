"""Derive native verifier paths only from the active owned cohort bootstrap."""

import hashlib
import json
import sys
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def current_layout():
    boot = sys.modules.get("ci_sglang_cohort")
    if boot is None or sys.modules.get("sitecustomize") is not boot:
        raise ValueError("Active owned cohort bootstrap required")
    prefix = Path(sys.prefix).resolve()
    source = Path(boot.__file__).resolve()
    if not source.is_relative_to(prefix) or prefix.is_relative_to(Path("/opt/venv")):
        raise ValueError("Expected a separate owned interpreter")
    manifest = source.with_name("smg-cohort.json")
    if digest(manifest) != boot.SMG_MANIFEST_SHA256:
        raise ValueError("Cohort manifest changed after startup")
    spec = json.loads(manifest.read_text())
    if spec["interpreter_prefix"] != str(prefix) or digest(source) != spec["bootstrap_sha256"]:
        raise ValueError("Cohort interpreter or bootstrap differs")
    roots = set()
    for name in ("tvm_ffi", "tilelang", "torch_c_dlpack_ext"):
        entry = boot.SMG_COHORT[name]
        module = Path(entry["module_path"])
        meta = Path(entry["metadata_path"])
        root = module.parent.parent
        if (
            module != root / name / "__init__.py"
            or meta.parent.parent != root
            or module.is_symlink()
            or meta.is_symlink()
            or module.resolve() != module.absolute()
            or meta.resolve() != meta.absolute()
            or digest(module) != entry["source_sha256"]
            or digest(meta) != entry["metadata_sha256"]
        ):
            raise ValueError("Native source/metadata origin differs")
        roots.add(root)
    if len(roots) != 1:
        raise ValueError("Mixed native install roots")
    root = roots.pop()
    if (
        root.name != "site-packages"
        or root.parent.name != f"python{sys.version_info.major}.{sys.version_info.minor}"
        or root.parent.parent.name != "lib"
        or not root.is_relative_to(prefix.parent)
        or root.is_relative_to(prefix)
        or root.is_relative_to(Path("/opt/venv"))
    ):
        raise ValueError("Expected an owned native data scheme beside the interpreter")
    console = root.parents[2] / "bin"
    if not console.is_dir() or console.resolve() != console.absolute():
        raise ValueError("Expected owned native console directory")
    return {"prefix": prefix, "site": root, "bin": console, "scope": prefix.parent}
