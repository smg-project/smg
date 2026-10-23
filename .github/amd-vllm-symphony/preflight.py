"""Require all original cached models and the real installed Rust launcher binding."""

import json
import os
from pathlib import Path

from cached_models import cached_model
from reuse import require

OUT = Path("diagnostic-results")
HERE = Path(__file__).parent


def main():
    require(
        os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1",
        "offline guards missing",
    )
    require(not os.environ.get("E2E_VLLM_SERVICER_IMPL"), "default servicer lane changed")
    require(os.environ.get("E2E_PARSER") == "symphony", "wrong parser lane")
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
        result["preflight"] = "passed"
    except BaseException as error:
        result["failure"] = {"type": type(error).__name__, "detail": str(error)}
        raise
    finally:
        (OUT / "symphony-preflight.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
