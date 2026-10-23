"""Prepare one pinned ASR snapshot inside the enclosing absolute job budget."""

import hashlib
import json
import os
import shutil
from pathlib import Path

from cached_models import cached_model
from reuse import require

HERE = Path(__file__).parent
OUT = Path("diagnostic-results")


def prepare(manifest, root, downloader):
    model, revision = manifest["model_id"], manifest["revision"]
    snapshot = root / ("models--" + model.replace("/", "--")) / "snapshots" / revision
    files = manifest["runtime_files"]
    missing = [f for f in files if not (snapshot / f["name"]).is_file()]
    if missing:
        root.mkdir(parents=True, exist_ok=True)
        require(
            shutil.disk_usage(root).free >= sum(f["bytes"] for f in missing) + 512 * 1024**2,
            "insufficient cache disk for complete pinned ASR files",
        )
        resolved = downloader(
            repo_id=model,
            revision=revision,
            cache_dir=str(root),
            allow_patterns=[f["name"] for f in files],
            max_workers=2,
            etag_timeout=10,
        )
        require(Path(resolved).resolve() == snapshot.resolve(), "download revision differs")
    for file in files:
        path = snapshot / file["name"]
        require(path.resolve().is_relative_to(root.resolve()), "runtime file escapes cache")
        require(path.is_file() and path.stat().st_size == file["bytes"], "ASR runtime file size")
    result = cached_model(model, revision, root, model, require_main=False)
    require(result["config_sha256"] == manifest["config_sha256"], "pinned ASR config differs")
    expected = {f["name"] for f in files if f["name"].endswith(".safetensors")}
    require({f["filename"] for f in result["weight_files"]} == expected, "incomplete ASR shards")
    return {**result, "prepared_missing_files": [f["name"] for f in missing]}


def main():
    from huggingface_hub import snapshot_download

    manifest = json.loads((HERE / "model-cache.json").read_text())
    value = {"model": manifest["model_id"], "revision": manifest["revision"], "ready": False}
    try:
        value["cache"] = prepare(manifest, Path(os.environ["HF_HOME"]) / "hub", snapshot_download)
        import vllm.entrypoints.openai.api_server as server

        path = Path(server.__file__).resolve()
        raw = path.read_bytes()
        require(0 < len(raw) <= 4 * 1024**2, "HTTP entrypoint source cap")
        (OUT / "actual-http-api-server.py").write_bytes(raw)
        value["http_entrypoint"] = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}
        value["ready"] = True
    except BaseException as error:
        value["failure"] = {"type": type(error).__name__, "detail": str(error)}
        raise
    finally:
        (OUT / "realtime-preflight.json").write_text(json.dumps(value, indent=2) + "\n")


if __name__ == "__main__":
    main()
