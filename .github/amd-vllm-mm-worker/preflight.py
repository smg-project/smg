"""Require the recorded local Qwen snapshot and inspect real installed media interfaces."""

import hashlib
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path

from reuse import require

OUT = Path("diagnostic-results")
EXPECTED = Path(__file__).with_name("model-cache.json")
MODULES = (
    "vllm.multimodal.media.connector",
    "vllm.renderers.base",
    "vllm.transformers_utils.processor",
    "vllm.model_executor.models.qwen3_vl",
    "vllm.model_executor.models.qwen2_5_vl",
    "vllm.model_executor.models.vision",
    "vllm.model_executor.layers.attention.mm_encoder_attention",
    "vllm.platforms.rocm",
    "smg_grpc_servicer.vllm.mm_processor",
    "transformers",
    "torchvision",
    "PIL.Image",
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def check_cache(expected, cache, resolved):
    cache = cache.resolve()
    directory = cache / ("models--" + expected["model_id"].replace("/", "--"))
    revision = (directory / "refs/main").read_text().strip()
    require(revision == expected["snapshot"], "Qwen cache revision changed")
    snapshot = directory / "snapshots" / revision
    require(
        resolved == expected["model_id"] or Path(resolved).resolve() == snapshot.resolve(),
        "model spec points outside recorded snapshot",
    )
    config = snapshot / "config.json"
    require(config.resolve().is_relative_to(cache), "model config escapes cache")
    require(sha(config.read_bytes()) == expected["config_sha256"], "Qwen config hash changed")
    index = snapshot / "model.safetensors.index.json"
    names = set(json.loads(index.read_text())["weight_map"].values())
    require(names == set(expected["weight_files"]), "Qwen shard index changed")
    shards = []
    for name, size in expected["weight_files"].items():
        require(Path(name).name == name, "unsafe shard name")
        path = snapshot / name
        require(path.is_file() and path.resolve().is_relative_to(cache), "missing/escaped shard")
        require(path.stat().st_size == size, "incomplete/changed Qwen shard")
        shards.append({"path": str(path.resolve()), "bytes": size})
    return {
        "model": expected["model_id"],
        "snapshot": revision,
        "resolved_model_spec": resolved,
        "snapshot_path": str(snapshot.resolve()),
        "config_sha256": expected["config_sha256"],
        "index_sha256": sha(index.read_bytes()),
        "weight_files": shards,
        "full_weight_hashes_verified": False,
        "offline": True,
        "cache_on_current_runner_checked": os.environ.get("RUNNER_NAME"),
    }


def interfaces():
    from vllm.multimodal.media.connector import MEDIA_CONNECTOR_REGISTRY
    from vllm.renderers.base import BaseRenderer
    from vllm.transformers_utils.processor import get_video_processor_cls_name

    process = BaseRenderer.process_for_engine_async
    require(
        "skip_mm_cache" in inspect.signature(process).parameters, "renderer lacks skip_mm_cache"
    )
    require(callable(get_video_processor_cls_name), "video processor interface unavailable")
    require(callable(MEDIA_CONNECTOR_REGISTRY.load), "media connector interface unavailable")
    # Exercise the actual vision-package import; model construction remains with the original fixture.
    importlib.import_module("torchvision")
    records = []
    for name in MODULES:
        spec = importlib.util.find_spec(name)
        require(spec and spec.origin, "missing media source: " + name)
        path = Path(spec.origin)
        raw = path.read_bytes()
        require(0 < len(raw) <= 8 * 1024 * 1024, "media source size")
        records.append(
            {"module": name, "path": str(path.resolve()), "bytes": len(raw), "sha256": sha(raw)}
        )
    return {
        "renderer_signature": str(inspect.signature(process)),
        "connector_registry_type": type(MEDIA_CONNECTOR_REGISTRY).__name__,
        "video_processor_module": get_video_processor_cls_name.__module__,
        "sources": records,
        "model_constructor_and_selected_ViT_backend": "original worker fixture; retain actual worker logs",
        "backend_override_added": False,
    }


def main():
    require(os.environ.get("HF_HUB_OFFLINE") == "1", "HF offline guard missing")
    require(os.environ.get("TRANSFORMERS_OFFLINE") == "1", "Transformers offline guard missing")
    from infra.model_specs import get_model_spec

    expected = json.loads(EXPECTED.read_text())
    result = {"cache_check": "started", "runner": os.environ["RUNNER_NAME"]}
    path = OUT / "multimodal-preflight.json"
    try:
        spec = get_model_spec(expected["model_id"])
        require(spec["tp"] == 1, "original Qwen TP changed")
        result["cache"] = check_cache(expected, Path(os.environ["HF_HOME"]) / "hub", spec["model"])
        result["cache_check"] = "passed"
        result["interfaces"] = interfaces()
    except BaseException as exc:
        result["failure"] = {"type": type(exc).__name__, "detail": str(exc)}
        raise
    finally:
        path.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
