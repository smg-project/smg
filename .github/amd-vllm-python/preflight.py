"""Require all original cached models and the real installed Rust launcher binding."""

import hashlib
import importlib
import importlib.machinery
import inspect
import json
import os
import sys
from pathlib import Path

from cached_models import cached_model
from reuse import require

OUT = Path("diagnostic-results")
HERE = Path(__file__).parent


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def rust_binding():
    from smg import servicer, smg_rs
    from smg_grpc_servicer.vllm import launcher_switch, rust

    rust.require_upstream_hook()
    hook = launcher_switch.require_launcher_hook()
    require(hook.state == launcher_switch.BOUND, "actual Rust launcher is not bound")
    require(hook.module in launcher_switch.LAUNCHER_MODULES, "unknown launcher module")
    require(servicer.VllmGrpcServer is smg_rs.VllmGrpcServer, "Rust native server identity differs")
    native = Path(smg_rs.__file__).resolve()
    require(
        any(str(native).endswith(s) for s in importlib.machinery.EXTENSION_SUFFIXES), "not native"
    )
    digest = hashlib.sha256()
    with native.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    function = getattr(sys.modules[hook.module], launcher_switch.HOOK_ATTRIBUTE)
    original = inspect.unwrap(function)
    modules = [
        hook.module,
        "smg.servicer",
        "smg_grpc_servicer.vllm.rust",
        "smg_grpc_servicer.vllm.launcher_switch",
        "smg_grpc_servicer.vllm.plugin",
    ]
    directory = OUT / "rust-launcher-sources"
    directory.mkdir()
    records, total = [], 0
    for index, name in enumerate(modules):
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        with path.open("rb") as stream:
            raw = stream.read(2 * 1024 * 1024 + 1)
        total += len(raw)
        require(0 < len(raw) <= 2 * 1024 * 1024 and total <= 4 * 1024 * 1024, "launcher source cap")
        target = f"{index:02d}-{path.name}"
        (directory / target).write_bytes(raw)
        records.append({"module": name, "path": str(path), "file": target, "sha256": sha(raw)})
    require(
        Path(original.__code__.co_filename).resolve() == Path(records[0]["path"]),
        "bound launcher original code path differs",
    )
    return {
        "hook_state": hook.state,
        "hook_module": hook.module,
        "original_coroutine": original.__qualname__,
        "native_extension": str(native),
        "native_extension_sha256": digest.hexdigest(),
        "native_server_symbol_identity": True,
        "sources": records,
        "model_or_server_started_by_preflight": False,
    }


def main():
    require(
        os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1",
        "offline guards missing",
    )
    require(os.environ.get("E2E_VLLM_SERVICER_IMPL") == "rust", "wrong servicer lane")
    require(not os.environ.get("E2E_CONNECTION_MODE"), "wrong connection mode")
    from infra.model_specs import get_model_spec

    recipe = json.loads((HERE / "cases.json").read_text())["chat"]
    revisions = json.loads((HERE / "model-cache.json").read_text())
    require(set(revisions) == set(recipe["models"]), "original model union changed")
    qwen = json.loads((HERE / "qwen-cache.json").read_text())
    result = {"runner": os.environ["RUNNER_NAME"], "models": [], "preflight": "started"}
    try:
        for model in recipe["models"]:
            spec = get_model_spec(model)
            require(spec["tp"] == 1, "original model topology changed")
            cache = cached_model(
                model, revisions[model], Path(os.environ["HF_HOME"]) / "hub", spec["model"]
            )
            if model == qwen["model_id"]:
                require(
                    cache["snapshot"] == qwen["snapshot"]
                    and cache["config_sha256"] == qwen["config_sha256"],
                    "Qwen identity changed",
                )
                require(
                    {f["filename"]: f["bytes"] for f in cache["weight_files"]}
                    == qwen["weight_files"],
                    "Qwen original shards changed",
                )
            result["models"].append(cache)
        result["rust_binding"] = rust_binding()
        result["preflight"] = "passed"
    except BaseException as error:
        result["failure"] = {"type": type(error).__name__, "detail": str(error)}
        raise
    finally:
        (OUT / "rust-servicer-preflight.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
