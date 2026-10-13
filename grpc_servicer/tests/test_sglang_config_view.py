"""The SGLang engine config view: one reader of the server args' and model
config's attribute names, with the defaults an absent attribute reads."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from smg_grpc_servicer.sglang.config_view import EngineConfigView, ModelConfigSection


def _server_args(**overrides):
    args = SimpleNamespace(
        model_path="org/model",
        tokenizer_path="org/tok",
        served_model_name="served",
        weight_version="v2",
        preferred_sampling_params='{"top_k": 20}',
        revision="rev",
        max_running_requests=64,
        dp_size=2,
        enable_dp_attention=True,
        kv_events_config='{"publisher": "zmq"}',
        disaggregation_mode="prefill",
        language_only=False,
        encoder_only=True,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _model_config(**overrides):
    config = SimpleNamespace(
        context_len=4096,
        vocab_size=1024,
        is_generation=False,
        is_multimodal=True,
        hf_eos_token_id={7, 2},
        hf_config=SimpleNamespace(
            model_type="qwen3",
            architectures=["Qwen3ForCausalLM"],
            pad_token_id=0,
            bos_token_id=1,
            id2label=None,
            num_labels=0,
        ),
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def test_the_server_args_section_reads_every_knob():
    args = EngineConfigView(_server_args()).server_args
    assert (args.model_path, args.tokenizer_path, args.served_model_name) == (
        "org/model",
        "org/tok",
        "served",
    )
    assert (args.weight_version, args.preferred_sampling_params, args.revision) == (
        "v2",
        '{"top_k": 20}',
        "rev",
    )
    assert (args.max_running_requests, args.dp_size, args.enable_dp_attention) == (64, 2, True)
    assert args.kv_events_config == '{"publisher": "zmq"}'
    assert (args.disaggregation_mode, args.language_only, args.encoder_only) == (
        "prefill",
        False,
        True,
    )


def test_bare_server_args_read_as_the_defaults():
    args = EngineConfigView(SimpleNamespace()).server_args
    assert (args.model_path, args.tokenizer_path, args.served_model_name) == ("", None, None)
    assert (args.weight_version, args.preferred_sampling_params, args.revision) == ("", None, None)
    assert (args.max_running_requests, args.dp_size, args.enable_dp_attention) == (
        None,
        None,
        False,
    )
    assert args.kv_events_config is None
    assert (args.disaggregation_mode, args.language_only, args.encoder_only) == (
        "null",
        False,
        False,
    )


@pytest.mark.parametrize("value", [None, 0, -1, True, "64"])
def test_a_window_or_width_that_is_not_a_count_reads_as_none(value):
    args = EngineConfigView(SimpleNamespace(max_running_requests=value, dp_size=value)).server_args
    assert (args.max_running_requests, args.dp_size) == (None, None)


def test_the_model_section_reads_the_hf_config_and_the_shape():
    model = EngineConfigView(_server_args(), _model_config()).model
    assert model.hf_config is not None
    # The EOS set is advertised sorted; the HF facts as declared.
    assert (model.eos_token_ids, model.model_type) == ([2, 7], "qwen3")
    assert (model.architectures, model.pad_token_id, model.bos_token_id) == (
        ["Qwen3ForCausalLM"],
        0,
        1,
    )
    assert (model.id2label, model.num_labels) == (None, 0)
    assert (model.context_len, model.vocab_size) == (4096, 1024)
    assert (model.is_generation, model.is_multimodal) == (False, True)


def test_the_classification_heads_are_derived_like_the_python_servicer():
    labels_only = _model_config(
        hf_config=SimpleNamespace(id2label={0: "negative", 1: "positive"}, num_labels=0)
    )
    model = ModelConfigSection.read(labels_only)
    assert (model.id2label, model.num_labels) == ({0: "negative", 1: "positive"}, 2)
    count_only = _model_config(hf_config=SimpleNamespace(num_labels=3))
    model = ModelConfigSection.read(count_only)
    assert (model.id2label, model.num_labels) == ({0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}, 3)


def test_the_eos_field_is_one_id_a_sorted_set_or_nothing():
    assert ModelConfigSection.read(_model_config(hf_eos_token_id=5)).eos_token_ids == [5]
    assert ModelConfigSection.read(_model_config(hf_eos_token_id=[9, 3])).eos_token_ids == [3, 9]
    assert ModelConfigSection.read(_model_config(hf_eos_token_id=(4,))).eos_token_ids == [4]
    assert ModelConfigSection.read(_model_config(hf_eos_token_id=None)).eos_token_ids == []
    assert ModelConfigSection.read(_model_config(hf_eos_token_id="5")).eos_token_ids == []


def test_the_model_section_degrades_to_the_proto_defaults():
    bare = ModelConfigSection.read(SimpleNamespace())
    assert (bare.hf_config, bare.eos_token_ids, bare.model_type, bare.architectures) == (
        None,
        [],
        "",
        [],
    )
    assert (bare.pad_token_id, bare.bos_token_id, bare.id2label, bare.num_labels) == (0, 0, None, 0)
    assert (bare.context_len, bare.vocab_size) == (0, 0)
    assert (bare.is_generation, bare.is_multimodal) == (True, False)


def test_a_section_is_read_once_per_view():
    reads = []

    class Args:
        @property
        def dp_size(self):
            reads.append(1)
            return 2

    view = EngineConfigView(Args())
    assert view.server_args.dp_size == 2 and view.server_args.dp_size == 2
    assert reads == [1]
