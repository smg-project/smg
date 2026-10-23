"""Exact source contract for the test-only vLLM observer; no backend imports."""

import hashlib
import sys
from importlib.util import find_spec
from pathlib import Path

from . import pd_local_kv_compat as local_kv

# Read from the installed, pinned ROCm image; version strings are insufficient.
SOURCE_SHA256 = {
    "engine/arg_utils.py": "95be1adb96f55933f6a52260c8cb2181d04707401ca21d97c33108c5ab58f3ac",
    "v1/core/sched/scheduler.py": "f820bee4f42b878b262a30f3728a97d7413477b2a277748e20f2528ac2ec2f2a",
    "v1/core/sched/async_scheduler.py": "38d296e77e4d612e696241c67298eb1ca9b6a48b483cd33d35495bf31691032e",
    "config/scheduler.py": "82bc6805ecf30a9b3d1b678d3270dad919abed011782ad16f1e805a78aa6c241",
    "config/vllm.py": "057b0a7f80db28522ee46b41f9ae5bd7a0c1998a08e111c5a6ddcb8c08f07f8d",
    "platforms/rocm.py": "2715a421683973fc5b55a3e96299d36100541888c93d8ba671ef5aa27479b8cf",
    "v1/engine/core.py": "f116962c35d9d312b20b0b76c34a8cdaa6d3f1d31f2036db4e123957d2d1555b",
    "v1/engine/input_processor.py": "d79601e4a053a7bacb0d6a23e273348435e4ab70c6a315dcccb2d428ed57d96b",
    "v1/request.py": "dd7c30bf8b0894dc123a739c299c0133566f8cc075b081e45905f1c29506e69d",
    "entrypoints/openai/chat_completion/serving.py": "532dadea845c77a5a79011027b9989987c3e32467614e993f439f29c924de002",
    "entrypoints/serve/engine/serving.py": "e5239ca4e2119fd019ee0ffcbf6e68f642b7ce94bad1b3fabe22ccdd8cf9eb02",
}


def require_source_contract(root, expected_sources=None):
    if expected_sources is None:
        expected_sources = SOURCE_SHA256
    observed = {}
    for relative, expected in expected_sources.items():
        try:
            observed[relative] = hashlib.sha256((Path(root) / relative).read_bytes()).hexdigest()
        except OSError as exc:
            raise RuntimeError(f"unknown observer source contract: {relative}") from exc
        if observed[relative] != expected:
            raise RuntimeError(f"unknown observer source contract: {relative}")
    return observed


def require_installed_contract():
    spec = find_spec("vllm")
    locations = list(spec.submodule_search_locations or ()) if spec is not None else []
    if len(locations) != 1:
        raise RuntimeError("unknown vLLM package location")
    root = Path(locations[0]).resolve()
    contract = {"root": str(root), "sources": require_source_contract(root)}
    if local_kv.enabled():
        contract["local_kv_sources"] = require_source_contract(root, local_kv.SOURCE_SHA256)
    return contract


def require_loaded_contract(contract, scheduler, async_scheduler):
    root = Path(contract["root"])
    sources = dict(SOURCE_SHA256, **contract.get("local_kv_sources", {}))
    for relative in sources:
        name = "vllm." + relative[:-3].replace("/", ".")
        module = sys.modules.get(name)
        if module is None:
            continue
        paths = (
            getattr(module, "__file__", None),
            getattr(getattr(module, "__spec__", None), "origin", None),
        )
        if any(path is None or Path(path).resolve() != root / relative for path in paths):
            raise RuntimeError(f"unknown loaded observer source: {name}")
    for cls, suffix, expected_name in (
        (scheduler, "scheduler", "Scheduler"),
        (async_scheduler, "async_scheduler", "AsyncScheduler"),
    ):
        name = f"vllm.v1.core.sched.{suffix}"
        if cls.__module__ != name or cls.__name__ != expected_name or name not in sys.modules:
            raise RuntimeError(f"unknown native scheduler class origin: {name}")
