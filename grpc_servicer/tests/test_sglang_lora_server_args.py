"""The LoRA snapshot and scheduler agree on native SGLang argument handling."""

from __future__ import annotations

import copy
import json

import pytest
from smg_grpc_servicer.sglang.lora import PreloadedLoras
from smg_grpc_servicer.sglang.rust import server_facts

ServerArgs = pytest.importorskip("sglang.srt.server_args").ServerArgs
check_lora_server_args = pytest.importorskip(
    "sglang.srt.arg_groups.lora_hook"
).check_lora_server_args


@pytest.mark.parametrize("enable_lora", [None, True])
@pytest.mark.parametrize(
    "paths",
    [
        ["adapter=/weights"],
        ["/weights"],
        [{"lora_name": "adapter", "lora_path": "/weights", "pinned": True}],
        [{"lora_name": "adapter", "lora_path": "/weights", "lora_id": "ignored-id"}],
        {"adapter": "/weights"},
    ],
)
def test_facts_before_engine_validation_match_the_later_scheduler_ids(paths, enable_lora):
    args = ServerArgs(model_path="dummy", enable_lora=enable_lora, lora_paths=copy.deepcopy(paths))
    args.resolve_once()

    facts = json.loads(server_facts(args)["server_args_json"])
    snapshot = PreloadedLoras(args)
    # The engine still needs its original declarations: its validation hook
    # rejects an already-normalized LoRARef if invoked a second time.
    assert args.lora_paths == paths
    assert args.resolved_dict()["lora_paths"] == paths

    check_lora_server_args(args)
    expected = args.resolved_dict()["lora_paths"]
    assert facts["enable_lora"] is True
    assert facts["lora_paths"] == snapshot.adapters == expected
    for ref in expected:
        assert snapshot.resolve(ref["lora_name"]) == ref["lora_id"]
        assert snapshot.resolve(ref["lora_id"]) == ref["lora_id"]
    # The Python servicer reads the same already-validated args at startup.
    assert PreloadedLoras(args).adapters == expected


def test_native_lora_validation_errors_leave_the_engine_declarations_untouched():
    args = ServerArgs(
        model_path="dummy",
        enable_lora=True,
        lora_paths=["adapter=/weights"],
        max_loras_per_batch=2,
        max_loaded_loras=1,
    )
    args.resolve_once()
    before = args.resolved_dict()
    with pytest.raises(AssertionError, match="max_loaded_loras"):
        PreloadedLoras(args)
    assert args.resolved_dict() == before
