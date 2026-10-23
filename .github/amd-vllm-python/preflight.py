"""Require original cached models and the real installed Python servicer binding."""

import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path

from cached_models import cached_model
from reuse import require

OUT = Path("diagnostic-results")
HERE = Path(__file__).parent


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def python_binding():
    from smg_grpc_servicer.vllm import launcher_switch, rust, servicer

    require(rust.resolve_servicer_impl() == "python", "Python implementation not selected")
    cls = servicer.VllmEngineServicer
    expected = Path("grpc_servicer/smg_grpc_servicer/vllm/servicer.py").resolve()
    require(
        Path(inspect.getfile(cls)).resolve() == expected, "installed Python class source differs"
    )
    require(
        inspect.isasyncgenfunction(cls.Generate)
        and Path(cls.Generate.__code__.co_filename).resolve() == expected,
        "Generate is not the installed Python async generator",
    )
    launcher = importlib.import_module(launcher_switch.LAUNCHER_MODULES[0])
    require(launcher.VllmEngineServicer is cls, "launcher uses another servicer class")
    original = inspect.unwrap(launcher.serve_grpc)
    require(
        Path(original.__code__.co_filename).resolve() == Path(launcher.__file__).resolve(),
        "Python launcher original coroutine source differs",
    )
    modules = [servicer, launcher, launcher_switch, rust]
    directory = OUT / "python-servicer-sources"
    directory.mkdir()
    records, total = [], 0
    for index, module in enumerate(modules):
        path = Path(module.__file__).resolve()
        with path.open("rb") as stream:
            raw = stream.read(2 * 1024 * 1024 + 1)
        total += len(raw)
        require(0 < len(raw) <= 2 * 1024 * 1024 and total <= 4 * 1024 * 1024, "source cap")
        target = f"{index:02d}-{path.name}"
        (directory / target).write_bytes(raw)
        records.append(
            {"module": module.__name__, "path": str(path), "file": target, "sha256": sha(raw)}
        )
    require(
        b"VllmEngineServicer initialized (mm_processor=" in expected.read_bytes(),
        "Python constructor log boundary changed",
    )
    require(
        b"vLLM gRPC server started on " in Path(launcher.__file__).read_bytes(),
        "Python server listener log boundary changed",
    )
    return {
        "class": cls.__module__ + "." + cls.__qualname__,
        "Generate_code_path": str(expected),
        "launcher_class_identity": True,
        "sources": records,
        "model_or_server_started_by_preflight": False,
    }


def main():
    require(
        os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1",
        "offline guards missing",
    )
    require(not os.environ.get("E2E_VLLM_SERVICER_IMPL"), "wrong servicer lane")
    require(os.environ.get("SPT_NOENV") == "1", "process marker preservation missing")
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
        result["python_binding"] = python_binding()
        result["preflight"] = "passed"
    except BaseException as error:
        result["failure"] = {"type": type(error).__name__, "detail": str(error)}
        raise
    finally:
        (OUT / "python-servicer-preflight.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
