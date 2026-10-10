"""The engine facts both vLLM servicers advertise, from one reader of vLLM's
config: the dicts build the protos as they are, so a renamed or retyped field
fails here before either servicer answers it wrong."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest
from smg_grpc_proto import vllm_engine_pb2
from smg_grpc_servicer.vllm import model_info


def _model_config(**overrides):
    config = SimpleNamespace(
        model="org/m",
        served_model_name="served",
        tokenizer="org/m-tok",
        runner_type="generate",
        is_multimodal_model=False,
        max_model_len=4096,
        get_vocab_size=lambda: 1024,
        hf_config=SimpleNamespace(
            model_type="qwen3", eos_token_id=[151645, 151643], pad_token_id=None, bos_token_id=1
        ),
        architectures=["Qwen3ForCausalLM"],
        try_get_generation_config=lambda: {"eos_token_id": [151643, 7], "temperature": 0.6},
        get_diff_sampling_param=lambda: {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": None,
            "max_new_tokens": 10,
        },
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def test_mm_item_limits_follow_the_engines_per_prompt_limits():
    # A text model, and a multimodal config without vLLM's accessor, advertise
    # nothing: the Router keeps its own caps.
    assert model_info.mm_item_limits(SimpleNamespace(model_config=_model_config())) == ""
    bare = _model_config(is_multimodal_model=True, multimodal_config=SimpleNamespace())
    assert model_info.mm_item_limits(SimpleNamespace(model_config=bare)) == ""
    # The resolved --limit-mm-per-prompt of the modalities a vLLM worker takes,
    # as sorted pairs the Router parses.
    limits = {"image": 8, "video": 2, "audio": 1}
    config = _model_config(
        is_multimodal_model=True,
        multimodal_config=SimpleNamespace(get_limit_per_prompt=limits.__getitem__),
    )
    label = model_info.mm_item_limits(SimpleNamespace(model_config=config))
    assert label == "image=8,video=2"
    info = vllm_engine_pb2.GetServerInfoResponse(mm_item_limits=label)
    parsed = vllm_engine_pb2.GetServerInfoResponse.FromString(info.SerializeToString())
    assert parsed.mm_item_limits == "image=8,video=2"


def test_model_facts_build_the_proto_as_they_are(caplog):
    with caplog.at_level(logging.WARNING, logger=model_info.__name__):
        facts = model_info.model_facts(_model_config())
    # A text model: vLLM's own check said no, not the fallback below.
    assert not [r for r in caplog.records if "supports_vision" in r.getMessage()]
    response = vllm_engine_pb2.GetModelInfoResponse(
        max_req_input_len=facts["max_context_length"], **facts
    )
    assert response.model_path == "org/m"
    assert response.served_model_name == "served"
    assert response.tokenizer_path == "org/m-tok"
    assert response.is_generation is True
    assert response.max_context_length == 4096 == response.max_req_input_len
    assert response.vocab_size == 1024
    assert response.supports_vision is False
    assert response.model_type == "qwen3"
    assert list(response.architectures) == ["Qwen3ForCausalLM"]
    # What the HF config declares; the generation config's extras are the
    # engine's stop set, not the advertised ids.
    assert list(response.eos_token_ids) == [151645, 151643]
    assert response.pad_token_id == 0 and response.bos_token_id == 1
    # Compact JSON of the advertised keys only, None values dropped.
    assert response.default_sampling_params_json == '{"temperature":0.6,"top_p":0.95}'
    assert json.loads(response.default_sampling_params_json) == {"temperature": 0.6, "top_p": 0.95}


def test_model_facts_degrade_like_the_python_servicer_did():
    config = _model_config(
        served_model_name=["first", "second"],
        tokenizer=None,
        runner_type="pooling",
        hf_config=SimpleNamespace(
            model_type=None, eos_token_id=2, pad_token_id=3, bos_token_id=None
        ),
        architectures=None,
        get_diff_sampling_param=lambda: None,
    )
    facts = model_info.model_facts(config)
    assert facts["served_model_name"] == "first"
    assert facts["tokenizer_path"] == "org/m"
    assert facts["is_generation"] is False
    assert facts["model_type"] == "" and facts["architectures"] == []
    assert facts["eos_token_ids"] == [2]
    assert facts["pad_token_id"] == 3 and facts["bos_token_id"] == 0
    assert facts["default_sampling_params_json"] == ""
    vllm_engine_pb2.GetModelInfoResponse(**facts)


def test_supports_vision_records_why_it_fell_back(caplog):
    # A config shape the check cannot read: vision is reported off (the Router
    # then sends this worker no mm payloads), and the cause is in the log.
    config = _model_config()
    del config.is_multimodal_model
    with caplog.at_level(logging.WARNING, logger=model_info.__name__):
        assert model_info.supports_vision(config) is False
    record = next(r for r in caplog.records if "supports_vision=false" in r.getMessage())
    assert record.exc_info is not None and record.exc_info[0] is AttributeError


def test_eos_sets_distinguish_advertised_from_stopped_on():
    config = _model_config()
    assert model_info.hf_eos_token_ids(config) == [151645, 151643]
    assert model_info.eos_token_ids_with_generation_config(config) == [151645, 151643, 7]
    # Negative and boolean ids are not ids; a failing generation config adds nothing.
    config = _model_config(
        hf_config=SimpleNamespace(eos_token_id=[-1, True, 5]),
        try_get_generation_config=lambda: (_ for _ in ()).throw(RuntimeError("no config")),
    )
    assert model_info.eos_token_ids_with_generation_config(config) == [5]


def test_server_facts_build_the_proto_as_they_are(monkeypatch):
    monkeypatch.setenv("SMG_PAIRING_PROTOCOL", "nixl")
    config = SimpleNamespace(
        model_config=_model_config(dtype="torch.bfloat16"),
        parallel_config=SimpleNamespace(data_parallel_size=2),
        kv_transfer_config=SimpleNamespace(
            kv_connector="NixlConnector", kv_role="kv_producer", engine_id="eng-a"
        ),
        cache_config=SimpleNamespace(cache_dtype="auto", block_size=16),
        attention_config=SimpleNamespace(backend=SimpleNamespace(name="FLASH_ATTN")),
    )
    facts = model_info.server_facts(config)
    response = vllm_engine_pb2.GetServerInfoResponse(**facts)
    assert response.kv_connector == "NixlConnector"
    assert response.kv_role == "kv_producer"
    assert response.kv_engine_id == "eng-a"
    assert response.data_parallel_size == 2
    assert response.pairing_protocol == "nixl"
    assert response.kv_cache_dtype == "auto" and response.block_size == 16
    assert response.attention_backend == "FLASH_ATTN"
    assert response.model_dtype == "torch.bfloat16"
    assert isinstance(response.shm_namespace_id, str)
    # No connector and no pairing facts: the proto defaults, nothing invented.
    bare = SimpleNamespace(
        model_config=_model_config(), parallel_config=SimpleNamespace(data_parallel_size=1)
    )
    facts = model_info.server_facts(bare)
    assert (facts["kv_connector"], facts["kv_role"], facts["kv_engine_id"]) == ("", "", "")
    assert "block_size" not in facts and "model_dtype" not in facts
    vllm_engine_pb2.GetServerInfoResponse(**facts)


def test_server_facts_carry_the_running_window():
    config = SimpleNamespace(
        model_config=_model_config(),
        parallel_config=SimpleNamespace(data_parallel_size=1),
        scheduler_config=SimpleNamespace(max_num_seqs=64),
    )
    assert model_info.running_window(config) == 64
    facts = model_info.server_facts(config)
    assert facts["max_num_seqs"] == 64
    response = vllm_engine_pb2.GetServerInfoResponse(**facts)
    assert response.max_num_seqs == 64
    parsed = vllm_engine_pb2.GetServerInfoResponse.FromString(response.SerializeToString())
    assert parsed.max_num_seqs == 64
    # A config without a resolved window reports the proto default, which the
    # router reads as "no window" (and says so), never an invented figure.
    assert model_info.running_window(SimpleNamespace()) == 0
    for window in (None, 0, -1, True, "64"):
        config = SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=window))
        assert model_info.running_window(config) == 0, window


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        ("torch.bfloat16", "bfloat16"),
        ("torch.float16", "float16"),
        ("float32", "float32"),
        ("torch.float32", "float32"),
        ("bfloat16", "bfloat16"),
        ("float16", "float16"),
        ("torch.float64", ""),
        ("float64", ""),
        ("torch.uint8", ""),
        ("unknown", ""),
        ("", ""),
        (None, ""),
    ],
)
def test_server_facts_report_encoder_dtype_without_changing_model_dtype(dtype, expected):
    config = SimpleNamespace(
        model_config=_model_config(dtype=dtype),
        parallel_config=SimpleNamespace(data_parallel_size=1),
    )
    response = vllm_engine_pb2.GetServerInfoResponse(**model_info.server_facts(config))
    parsed = vllm_engine_pb2.GetServerInfoResponse.FromString(response.SerializeToString())
    assert parsed.multimodal_encoder_dtype == expected
    assert parsed.model_dtype == (dtype or "")
