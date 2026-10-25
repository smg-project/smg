"""Select only the original Mooncake4 topology row before any relay API call."""

import argparse
import hashlib
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


POLICY = 'historical_c345_CPU_plus_reviewed_checkout_env_delta_scheduled_full_validation'
BASIS_SHA = '15b22b08e0fce07e4c369011abf2490d32e556605414a4e3b117677c83ed92b4'


def historical_cpu_for_scheduled_validation(inputs):
    """An execution prerequisite only; never new-source CPU acceptance."""
    value = inputs["scheduled_validation"]
    expected = {"policy", "admitted", "candidate_CPU_accepted", "basis_sha256", "historical_node_receipts"}
    if (set(value) != expected or value["policy"] != POLICY
            or value["admitted"] is not True or value["candidate_CPU_accepted"] is not False
            or value["basis_sha256"] != BASIS_SHA):
        raise ValueError("exact reviewed scheduled-validation basis is not admitted")
    path = Path(__file__).with_name("scheduled-validation-basis.json")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != BASIS_SHA:
        raise ValueError("scheduled-validation basis bytes changed")
    basis = json.loads(raw)
    digest = lambda v: hashlib.sha256(json.dumps(v, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if (basis["policy"] != POLICY or basis["candidate_CPU_accepted"] is not False
            or inputs["same_source_CPU_prerequisite"]["accepted"] is not False
            or any(digest(inputs[key]) != expected for key, expected in basis["cohort_sha256"].items())
            or digest(value["historical_node_receipts"]) != basis["historical_CPU"]["node_receipts_sha256"]):
        raise ValueError("historical CPU/delta/cohort identity changed")
    root = Path(__file__).resolve().parents[2]
    expected = basis["checkout_environment_delta"]["three_files"]
    for name, key in [(".github/amd-nixl-environment/nixl_environment/runtime_inputs.py", "runtime_helper"),
                      (".github/amd-shared-product/private-runtime-overlay.json", "private_runtime_overlay")]:
        path = root / name
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected[key]["new"]:
            raise ValueError("reviewed environment helper/overlay differs")
    return {"accepted": True, "node_receipts": value["historical_node_receipts"]}


def select(inputs, tier):
    if any(inputs.get(key) is not True for key in ("ready", "run_budget_admitted")):
        raise ValueError("source and runner budget are not admitted")
    if inputs["private_mooncake_runtime"]["staging_accepted"] is not True:
        raise ValueError("exact private wheel and prefix staging is not accepted")
    auxiliary_staging(inputs)
    closure = inputs["normal_resolver_closure"]
    receipt = closure["staging_receipt"]
    if (closure["staging_accepted"] is not True or not isinstance(receipt, dict)
            or receipt.get("manifest_sha256") != closure["manifest_sha256"]
            or set(receipt.get("nodes", {})) != {"amd-mi300x-1", "amd-mi300x-2"}):
        raise ValueError("both pooled nodes require the exact normal-resolver data")
    for row in receipt["nodes"].values():
        if (row.get("accepted") is not True or row.get("host_bundle") != closure["host_bundle"]
                or row.get("manifest_sha256") != closure["manifest_sha256"]
                or row.get("regular_files") != 31 or row.get("wheel_count") != 30
                or not isinstance(row.get("evidence_sha256"), str)
                or len(row["evidence_sha256"]) != 64):
            raise ValueError("normal-resolver node data proof differs")
    cpu = historical_cpu_for_scheduled_validation(inputs)
    rows = cpu["node_receipts"]
    if (cpu["accepted"] is not True or not isinstance(rows, dict)
            or set(rows) != {"amd-mi300x-1", "amd-mi300x-2"}):
        raise ValueError("both pooled nodes require matching normal shared-CI CPU proof")
    for row in rows.values():
        if (row.get("accepted") is not True
                or row.get("gateway_native_sha256") != inputs["gateway_native"]["sha256"]
                or row.get("consumer19") != inputs["source_trees"]
                or row.get("compiled37") != inputs["compiled_source_trees"]
                or row.get("original_unit_cases") != 1602 or row.get("passed") != 1594
                or row.get("exact_original_skips") != 8 or row.get("canonical14_once") != 14
                or row.get("wholepip_passes") != 3 or row.get("normal_shared_install") is not True
                or row.get("full_SOURCE_RECORD_direct_url_postflight") is not True
                or not isinstance(row.get("root_acceptance_sha256"), str)
                or len(row["root_acceptance_sha256"]) != 64):
            raise ValueError("CPU proof is not this source/native complete normal cohort")
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
