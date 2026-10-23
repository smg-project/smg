"""Keep existing probe outputs and reject skipped or incomplete diagnostic results."""

import argparse
import hashlib
import json
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

OUT = Path("diagnostic-results")


def preflight():
    from infra.model_specs import get_model_spec
    from infra.pd_scheduler_compat import require_installed_contract

    contract = require_installed_contract()
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    requested = get_model_spec(model_id)["model"]
    model = Path(requested)
    revision = None
    if not model.is_absolute():
        if requested != model_id:
            raise ValueError("Unknown cached model ID")
        cache = Path(os.environ["HF_HOME"]) / "hub"
        directory = cache / ("models--" + model_id.replace("/", "--"))
        revision = (directory / "refs/main").read_text().strip()
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Unknown cached model revision")
        model = directory / "snapshots" / revision
        if not model.resolve().is_relative_to(cache.resolve()):
            raise ValueError("Cached model path escapes cache")
    if not model.is_dir():
        raise ValueError("Existing local Llama8B model cache is required")
    # The normal loader still chooses model files. Never download or substitute weights.
    files = {}
    for name in ["config.json", "tokenizer_config.json"]:
        path = model / name
        if revision is not None and not path.resolve().is_relative_to(cache.resolve()):
            raise ValueError("Cached model metadata escapes cache")
        if not path.is_file() or path.stat().st_size > 8 * 1024 * 1024:
            raise ValueError(f"Missing or oversized cached model metadata: {name}")
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if any(os.environ.get(key) != "1" for key in ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"]):
        raise ValueError("Model access must remain offline")
    OUT.mkdir(exist_ok=True)
    (OUT / "model-and-observer.json").write_text(
        json.dumps(
            {
                "model_id": model_id,
                "requested_model": requested,
                "cached_revision": revision,
                "local_path": str(model.resolve()),
                "metadata_sha256": files,
                "weights_hashed": False,
                "installed_vllm_contract": contract,
                "accepted": False,
            },
            indent=2,
        )
        + "\n"
    )


def results():
    from infra.pd_local_kv_compat import enabled as local_kv_enabled

    cases = list(ET.parse(OUT / "pd-close.xml").getroot().iter("testcase"))
    if len(cases) != 1 or not cases[0].get("classname", "").endswith(".TestPDDecodeAbortProbe"):
        raise ValueError("Exactly the existing single close probe must execute")
    case = cases[0]
    if not case.get("name", "").startswith("test_decode_client_close[") or any(
        child.tag in {"skipped", "failure", "error", "rerunFailure", "rerunError"} for child in case
    ):
        raise ValueError("Close probe must pass without skip, error, failure or rerun")
    text = (OUT / "pd-close.log").read_text()
    if text.count("SMG_PD_PROBE_BEGIN\n") != 1 or text.count("SMG_PD_PROBE_END") != 1:
        raise ValueError("Exactly one complete probe export is required")
    exported = json.loads(text.split("SMG_PD_PROBE_BEGIN\n", 1)[1].splitlines()[0])
    abort = exported.get("decode_abort", {})
    if (
        exported.get("accepted") is not False
        or exported.get("error") is not None
        or exported.get("backend") != "nixl"
        or abort.get("client_closed") is not True
    ):
        raise ValueError("Probe did not complete its existing diagnostic contract")
    if local_kv_enabled():
        local = exported.get("local_kv", {})
        if (
            not isinstance(local.get("internal_request_id"), str)
            or not local["internal_request_id"]
            or local["internal_request_id"] != abort.get("internal_request_id")
            or local.get("request_references_released") is not True
            or not local.get("exclusive_pages_reusable")
            or local.get("observation") != "partial"
            or local.get("accepted") is not False
            or "transfer_ack" not in local
            or "peer_memory_released" not in local
            or local.get("transfer_ack") is not None
            or local.get("peer_memory_released") is not None
        ):
            raise ValueError("Missing local reusable-page evidence; peer release remains unknown")
    (OUT / "probe-export.json").write_text(json.dumps(exported, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["preflight", "results"])
    args = parser.parse_args()
    (preflight if args.mode == "preflight" else results)()
