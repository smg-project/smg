"""Preloaded adapters reach SGLang by their resolved ID on either servicer."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from smg_grpc_servicer.sglang.lora import PreloadedLoras
from test_sglang_server_info import servicer_mod as _servicer_fixture

servicer_mod = _servicer_fixture


def _args(**overrides):
    return SimpleNamespace(
        **{
            "enable_lora": True,
            "lora_paths": [
                SimpleNamespace(
                    lora_id="engine-id", lora_name="adapter", lora_path="/weights", pinned=True
                )
            ],
            **overrides,
        }
    )


def test_preloaded_names_and_ids_resolve_without_rewriting_the_base_request():
    loras = PreloadedLoras(_args())
    assert loras.resolve("adapter") == "engine-id"
    assert loras.resolve("engine-id") == "engine-id"
    assert loras.resolve("") is None
    with pytest.raises(ValueError, match="Unknown preloaded"):
        loras.resolve("not-loaded")
    disabled = PreloadedLoras(_args(enable_lora=False, lora_paths=None))
    assert disabled.resolve("") is None
    with pytest.raises(ValueError, match="not enabled"):
        disabled.resolve("adapter")


@pytest.mark.parametrize(
    "paths",
    [
        ["unresolved-cli-argument"],
        [{"lora_name": "adapter", "lora_path": "/weights"}],
        [{"lora_id": "", "lora_name": "adapter", "lora_path": "/weights"}],
        [
            {"lora_id": "id-a", "lora_name": "adapter", "lora_path": "/a"},
            {"lora_id": "id-b", "lora_name": "id-a", "lora_path": "/b"},
        ],
    ],
)
def test_malformed_or_ambiguous_adapter_config_fails_startup(paths):
    with pytest.raises(ValueError):
        PreloadedLoras(_args(lora_paths=paths))


def test_rust_launcher_preserves_resolved_array_like_lora_refs():
    msgspec = pytest.importorskip("msgspec")
    from smg_grpc_servicer.sglang.rust import server_facts

    class LoRARef(msgspec.Struct, frozen=True, array_like=True):
        lora_id: str
        lora_name: str
        lora_path: str
        pinned: bool | None = None

    args = _args(lora_paths=[LoRARef("engine-id", "adapter", "/weights")])
    facts = json.loads(server_facts(args)["server_args_json"])
    assert facts["lora_paths"] == [
        {"lora_id": "engine-id", "lora_name": "adapter", "lora_path": "/weights", "pinned": False}
    ]


def test_cli_lora_resolution_precedes_the_rust_snapshot_and_preserves_scheduler_ids():
    from smg_grpc_servicer.sglang.rust import server_facts

    class ServerArgs:
        # SGLang retains these declarations; resolution stores its output
        # separately and enables LoRA implicitly for --lora-paths.
        lora_paths = ["adapter=/weights"]
        enable_lora = None
        resolved = None

        def resolve_once(self):
            if self.resolved is None:
                self.resolved = {
                    "enable_lora": True,
                    "lora_paths": [
                        {
                            "lora_id": "scheduler-id",
                            "lora_name": "adapter",
                            "lora_path": "/weights",
                            "pinned": False,
                        }
                    ],
                }

        def resolved_dict(self):
            assert self.resolved is not None
            return self.resolved

    args = ServerArgs()
    facts = json.loads(server_facts(args)["server_args_json"])
    assert facts["enable_lora"] is True
    assert facts["lora_paths"][0]["lora_id"] == "scheduler-id"
    assert PreloadedLoras(args).resolve("adapter") == "scheduler-id"
    assert args.lora_paths == ["adapter=/weights"]
    assert args.enable_lora is None


@pytest.mark.parametrize("selector", ["adapter", "engine-id", ""])
def test_python_servicer_forwards_the_resolved_lora_and_lists_loaded_adapters(
    monkeypatch, servicer_mod, selector
):
    pb = servicer_mod.sglang_scheduler_pb2
    servicer = servicer_mod.SGLangSchedulerServicer.__new__(servicer_mod.SGLangSchedulerServicer)
    servicer.loras = PreloadedLoras(_args())
    servicer.tokenizer = None
    monkeypatch.setattr(servicer_mod, "TokenizedGenerateReqInput", SimpleNamespace)
    monkeypatch.setattr(
        servicer,
        "_convert_sampling_params",
        lambda _: SimpleNamespace(normalize=lambda **_: None),
    )
    req = pb.GenerateRequest(
        request_id="r", tokenized=pb.TokenizedInput(input_ids=[1]), lora_id=selector
    )
    forwarded = servicer._convert_generate_request(req)
    assert forwarded.lora_id == ("engine-id" if selector else None)
    listed = asyncio.run(servicer.ListLoadedLoRAAdapters(pb.ListLoadedLoRAAdaptersRequest(), None))
    assert [(a.lora_name, a.lora_id, a.pinned) for a in listed.loaded_adapters] == [
        ("adapter", "engine-id", True)
    ]
    req.lora_id = "not-loaded"
    with pytest.raises(ValueError, match="Unknown preloaded"):
        servicer._convert_generate_request(req)
