"""Check the installed vLLM ROCm connector through its own NIXL loader."""

import hashlib
import importlib
import inspect
import json
from importlib.metadata import distribution, version
from pathlib import Path

import smg_grpc_servicer
import torch
import vllm
from vllm.distributed import nixl_utils
from vllm.platforms import current_platform

assert torch.version.hip and current_platform.is_rocm(), "ROCm runtime required"
assert nixl_utils.is_nixl_available(), "vLLM ROCm NIXL backend is unavailable"
assert nixl_utils.NixlWrapper is not None, "NIXL native agent import failed"
assert nixl_utils.nixl_agent_config is not None, "NIXL native configuration import failed"
api = importlib.import_module(nixl_utils.NixlWrapper.__module__)
assert api.__name__.startswith("nixl_rocm."), "Unexpected NIXL backend selected"
out = Path("diagnostic-results")
files = {}
for name, module in [
    ("nixl_utils", nixl_utils),
    ("nixl_api", api),
    ("nixl_bindings", importlib.import_module("nixl_rocm._bindings")),
]:
    path = Path(inspect.getfile(module)).resolve()
    raw = path.read_bytes()
    if path.suffix == ".py":
        (out / (name + path.suffix)).write_bytes(raw)
    files[name] = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}
dist = distribution("nixl-rocm")
metadata = dist.read_text("METADATA")
(out / "nixl-rocm.METADATA").write_text(metadata)
files["distribution"] = {
    "origin": str(dist.locate_file("")),
    "metadata_sha256": hashlib.sha256(metadata.encode()).hexdigest(),
}
files["smg_grpc_servicer"] = {"path": smg_grpc_servicer.__file__}
loaded = {}
for line in Path("/proc/self/maps").read_text().splitlines():
    path = Path(line.split()[-1])
    if (
        path.is_absolute()
        and path.is_file()
        and any(key in str(path) for key in ("nixl", "ucx", "libucp", "libucs", "libuct"))
    ):
        loaded[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
(out / "nixl-canary.json").write_text(
    json.dumps(
        {
            "vllm": vllm.__version__,
            "HIP": torch.version.hip,
            "nixl_rocm": version("nixl-rocm"),
            "selected_module": api.__name__,
            "source": files,
            "actually_mapped_native_libraries": loaded,
            "PD_compute_proven": False,
        },
        indent=2,
    )
    + "\n"
)
print("vLLM ROCm NIXL native import passed", files)
