"""Keep existing probe outputs and reject skipped or incomplete diagnostic results."""

import argparse
import hashlib
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

OUT = Path("diagnostic-results")


def preflight():
    from infra.model_specs import get_model_spec
    from infra.pd_scheduler_compat import require_installed_contract

    contract = require_installed_contract()
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    model = Path(get_model_spec(model_id)["model"])
    if not model.is_absolute() or not model.is_dir():
        raise ValueError("Existing local Llama8B model cache is required")
    # The normal loader still chooses model files. Never download or substitute weights.
    files = {}
    for name in ["config.json", "tokenizer_config.json"]:
        path = model / name
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
    (OUT / "probe-export.json").write_text(json.dumps(exported, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["preflight", "results"])
    args = parser.parse_args()
    (preflight if args.mode == "preflight" else results)()
