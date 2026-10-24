"""Select only the original Mooncake4 topology row before any relay API call."""

import argparse
import json
from pathlib import Path

RECIPES = {
    "4": {
        "reference_target_id": ".github/workflows/pr-test-rust.yml#e2e-4gpu-pd[3]/vllm/nvidia",
        "gpu_tier": "4",
        "timeout": 50,
        "test_timeout": 42,
        "test_dirs": "e2e_test/router/test_pd_topologies.py",
        "test_filter": (
            '-k "(2p2d and not http) or AssembledAtRuntime or SmallWindow '
            'or MismatchedTransport or MismatchedProtocol"'
        ),
        "min_selected": 14,
        "kv_backend": "mooncake",
        "vllm_kv_backend": "mooncake",
        "vllm_extra_kv_backends": "mooncake",
        "selected": 18,
        "original_skip_count": 1,
        "original_setup_skip_count": 0,
        "original_call_skip_count": 1,
    }
}


def auxiliary_staging(inputs):
    value = inputs["runtime_auxiliary_inputs"]
    if value["staging_accepted"] is not True:
        raise ValueError("separate auxiliary runtime inputs are not staged")
    receipt = value["staging_receipt"]
    if not isinstance(receipt, dict) or set(receipt) != {"manifest_sha256", "nodes"}:
        raise ValueError("auxiliary staging receipt schema")
    if receipt["manifest_sha256"] != value["manifest_sha256"]:
        raise ValueError("auxiliary staging manifest differs")
    if set(receipt["nodes"]) != {"amd-mi300x-1", "amd-mi300x-2"}:
        raise ValueError("both pooled runners require auxiliary staging")
    for node in receipt["nodes"].values():
        if (set(node) != {"accepted", "host_bundle", "manifest_sha256", "evidence_sha256"}
                or node["accepted"] is not True or node["host_bundle"] != value["host_bundle"]
                or node["manifest_sha256"] != value["manifest_sha256"]
                or not isinstance(node["evidence_sha256"], str)
                or len(node["evidence_sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in node["evidence_sha256"])):
            raise ValueError("auxiliary node staging evidence differs")


def select(inputs, tier):
    if any(inputs.get(key) is not True for key in ("ready", "run_budget_admitted")):
        raise ValueError("source and runner budget are not admitted")
    if inputs["private_mooncake_runtime"]["staging_accepted"] is not True:
        raise ValueError("exact private wheel and prefix staging is not accepted")
    auxiliary_staging(inputs)
    if tier not in RECIPES or inputs.get("admitted_tier") != tier:
        raise ValueError("dispatch tier differs from the single admitted tier")
    recipe = RECIPES[tier]
    if inputs["contracts"][tier] != recipe:
        raise ValueError("original row recipe changed")
    return recipe


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", required=True)
    parser.add_argument("--outputs", type=Path, required=True)
    args = parser.parse_args()
    inputs = json.loads(Path(__file__).with_name("inputs.json").read_text())
    recipe = select(inputs, args.tier)
    with args.outputs.open("a") as stream:
        for key, value in recipe.items():
            if "\n" in str(value) or "\r" in str(value):
                raise ValueError("multiline recipe output")
            stream.write(f"{key}={value}\n")
    print(json.dumps(recipe, sort_keys=True))
