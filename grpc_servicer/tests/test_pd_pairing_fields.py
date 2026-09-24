"""The KV-layout facts the vLLM servicer reports for PD pairing (engine-free, no vLLM required).

Run with: pytest grpc_servicer/tests/test_pd_pairing_fields.py
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("smg_grpc_proto")

# Import the module directly to avoid pulling vllm via the package __init__
_MODULE_PATH = Path(__file__).parents[1] / "smg_grpc_servicer" / "vllm" / "kv_transfer.py"
_spec = importlib.util.spec_from_file_location("kv_transfer", _MODULE_PATH)
kv_transfer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kv_transfer)
pairing_fields = kv_transfer.pairing_fields

_PAIRING_PATH = Path(__file__).parents[1] / "smg_grpc_servicer" / "pd_pairing.py"
_pspec = importlib.util.spec_from_file_location("pd_pairing", _PAIRING_PATH)
pd_pairing = importlib.util.module_from_spec(_pspec)
_pspec.loader.exec_module(pd_pairing)


def _config(**overrides):
    base = {
        "cache_config": SimpleNamespace(cache_dtype="fp8", block_size=16),
        "attention_config": SimpleNamespace(backend=SimpleNamespace(name="FLASH_ATTN")),
        "model_config": SimpleNamespace(dtype="torch.bfloat16"),
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_pairing_fields_reads_every_layout_fact():
    assert pairing_fields(_config()) == {
        "kv_cache_dtype": "fp8",
        "block_size": 16,
        "attention_backend": "FLASH_ATTN",
        "model_dtype": "torch.bfloat16",
    }


def test_pairing_fields_omits_what_the_config_does_not_carry():
    # An auto-selected attention backend is None until the worker resolves
    # it; an unresolved block size is None too. Neither becomes a field.
    config = _config(
        cache_config=SimpleNamespace(cache_dtype="auto", block_size=None),
        attention_config=SimpleNamespace(backend=None),
    )
    assert pairing_fields(config) == {"kv_cache_dtype": "auto", "model_dtype": "torch.bfloat16"}
    assert pairing_fields(SimpleNamespace()) == {}


def test_pairing_fields_round_trip_the_proto():
    from smg_grpc_proto import vllm_engine_pb2 as pb2

    # Field presence is a descriptor fact; `hasattr` on the generated class
    # only holds under the pure-Python protobuf runtime.
    fields = pb2.GetServerInfoResponse.DESCRIPTOR.fields_by_name
    if "kv_cache_dtype" not in fields:
        pytest.skip(
            "smg_grpc_proto stubs predate the GetServerInfoResponse pairing fields; "
            "regenerate from crates/grpc_client/proto"
        )
    assert {"kv_cache_dtype", "block_size", "attention_backend", "model_dtype"} <= set(fields)
    info = pb2.GetServerInfoResponse(**pairing_fields(_config()))
    parsed = pb2.GetServerInfoResponse.FromString(info.SerializeToString())
    assert parsed.kv_cache_dtype == "fp8"
    assert parsed.block_size == 16
    assert parsed.attention_backend == "FLASH_ATTN"
    assert parsed.model_dtype == "torch.bfloat16"


def test_pairing_protocol_comes_from_the_engine_environment():
    assert pd_pairing.pairing_protocol_from_env({}) == ""
    assert pd_pairing.pairing_protocol_from_env({"SMG_PAIRING_PROTOCOL": "  "}) == ""
    assert pd_pairing.pairing_protocol_from_env({"SMG_PAIRING_PROTOCOL": " kv-v1 "}) == "kv-v1"
    assert pd_pairing.PAIRING_PROTOCOL_ENV == "SMG_PAIRING_PROTOCOL"


def test_pairing_protocol_round_trips_the_vllm_proto():
    from smg_grpc_proto import vllm_engine_pb2 as pb2

    fields = pb2.GetServerInfoResponse.DESCRIPTOR.fields_by_name
    if "pairing_protocol" not in fields:
        pytest.skip(
            "smg_grpc_proto stubs predate GetServerInfoResponse.pairing_protocol; "
            "regenerate from crates/grpc_client/proto"
        )
    assert fields["pairing_protocol"].number == 15
    info = pb2.GetServerInfoResponse(
        pairing_protocol=pd_pairing.pairing_protocol_from_env({"SMG_PAIRING_PROTOCOL": "kv-v1"})
    )
    assert (
        pb2.GetServerInfoResponse.FromString(info.SerializeToString()).pairing_protocol == "kv-v1"
    )
