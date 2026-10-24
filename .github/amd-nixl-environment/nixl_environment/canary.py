"""Reuse the actual ROCm loader canary; this does not prove KV transfer."""


def capture(out):
    import hashlib
    import importlib
    import inspect
    import json
    import os
    from importlib.metadata import distribution, version
    from pathlib import Path

    import smg_grpc_servicer
    import torch
    import vllm
    from vllm.distributed import nixl_utils
    from vllm.platforms import current_platform

    files = {}
    utils_path = Path(inspect.getfile(nixl_utils)).resolve()
    utils_raw = utils_path.read_bytes()
    (out / "nixl_utils.py").write_bytes(utils_raw)
    files["nixl_utils"] = {"path": str(utils_path), "sha256": hashlib.sha256(utils_raw).hexdigest()}
    dist = distribution("nixl-rocm")
    metadata_path = Path(dist._path) / "METADATA"
    metadata = metadata_path.read_bytes()
    (out / "nixl-rocm.METADATA").write_bytes(metadata)
    files["distribution"] = {
        "origin": str(metadata_path.resolve()),
        "metadata_sha256": hashlib.sha256(metadata).hexdigest(),
    }
    files["smg_grpc_servicer"] = {"path": smg_grpc_servicer.__file__}
    (out / "nixl-source-before-native.json").write_text(json.dumps(files, indent=2) + "\n")
    assert torch.version.hip and current_platform.is_rocm(), "ROCm runtime required"
    assert nixl_utils.is_nixl_available(), "vLLM ROCm NIXL backend is unavailable"
    assert nixl_utils.NixlWrapper is not None, "NIXL native agent import failed"
    assert nixl_utils.nixl_agent_config is not None, "NIXL native configuration import failed"
    api = importlib.import_module(nixl_utils.NixlWrapper.__module__)
    assert api.__name__.startswith("nixl_rocm."), "Unexpected NIXL backend selected"
    for name, module in [
        ("nixl_api", api),
        ("nixl_bindings", importlib.import_module("nixl_rocm._bindings")),
    ]:
        path = Path(inspect.getfile(module)).resolve()
        raw = path.read_bytes()
        if path.suffix == ".py":
            (out / (name + path.suffix)).write_bytes(raw)
        files[name] = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}
    loaded = {}
    maps = Path("/proc/self/maps").read_text()
    (out / "nixl-native.maps").write_text(maps)
    for line in maps.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6:
            continue
        path = Path(fields[5])
        if (
            path.is_absolute()
            and path.is_file()
            and any(
                key in str(path) for key in ("nixl", "ucx", "libucp", "libucs", "libuct", "libucm")
            )
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
                "public_loader_environment": {
                    name: os.environ.get(name)
                    for name in (
                        "LD_LIBRARY_PATH",
                        "UCX_TLS",
                        "UCX_NET_DEVICES",
                        "UCX_MEMTYPE_CACHE",
                        "UCX_RCACHE_MAX_UNRELEASED",
                        "NIXL_PLUGIN_DIR",
                    )
                },
            },
            indent=2,
        )
        + "\n"
    )
