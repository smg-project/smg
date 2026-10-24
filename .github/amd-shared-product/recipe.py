"""The one original SG2GPU row; unresolved collection closes the relay gate."""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONTRACT = {
    "engine": "sglang",
    "gpu_tier": "2",
    "test_dirs": "e2e_test/router",
    "test_filter": "",
    "min_selected": 15,
    "timeout": 40,
    "test_timeout": 34,
    "kv_backend": "",
    "vllm_kv_backend": "nixl",
    "vllm_extra_kv_backends": "",
    "parser": "",
    "connection_mode": "",
    "servicer_impl": "",
    "mm_processing": "",
    "servicer_unit_tests": False,
    "default_sglang_transfer_backend": "mooncake",
    "original_reruns": 2,
    "original_reruns_delay_seconds": 5,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(inputs, selection):
    require(all(inputs.get(k) is True for k in ("ready", "run_budget_admitted", "artifacts_pinned")),
            "source, budget and actual artifacts are not admitted")
    require(all(inputs["contract"].get(k) == v for k, v in CONTRACT.items()),
            "original SG2 recipe changed")
    require(inputs.get("selection_contract_verified") is True,
            "HOLD: real original/current collection has not closed")
    require(selection["status"] == "CLOSED_REAL_ORIGINAL_AND_CURRENT_COLLECTION",
            "HOLD: real collection evidence missing")
    original, current = selection["reference_cases"], selection["current_cases"]
    require(isinstance(original, list) and isinstance(current, list), "real case lists missing")
    require(len(set(current)) == len(current) >= CONTRACT["min_selected"], "case scope/floor")
    require(set(original) <= set(current) and original, "original case projection missing")
    require(inputs["contract"]["selected"] == len(current), "selected metadata differs")
    require(isinstance(selection["setup_skips"], dict), "exact setup skip contract missing")
    require(isinstance(selection["allowed_call_skips"], dict), "exact call skip contract missing")
    require(set(selection["case_witnesses"]) == set(current), "case role witnesses missing")
    require(set(selection["setup_skips"]) <= set(current), "unselected setup skip")
    require(set(selection["allowed_call_skips"]) <= set(current), "unselected call skip")
    for nodeid, witnesses in selection["case_witnesses"].items():
        if nodeid in selection["setup_skips"]:
            continue
        require(isinstance(witnesses, list) and len(witnesses) >= 2,
                "original two-worker witness contract missing")
        roles = {w["role"] for w in witnesses}
        require(roles == ({"regular"} if "::TestIGWMultiWorker::" in nodeid else
                          {"prefill", "decode"}), "original SG role set changed")
        for witness in witnesses:
            require(witness["mode"] in {"grpc", "http"}, "original SG wire changed")
            require(witness["model_id"] in inputs["contract"]["models"], "original model changed")
            require(isinstance(witness["min_instances"], int) and witness["min_instances"] >= 1,
                    "nonpositive worker witness floor")
    require(selection["actual_CPU_collection_result"] is not None, "raw collection receipt missing")
    return selection


if __name__ == "__main__":
    verify(json.loads((HERE / "inputs.json").read_text()),
           json.loads((HERE / "sglang_pd_observer/selection.json").read_text()))
