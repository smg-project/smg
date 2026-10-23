"""The TokenSpeed engine config view: one reader of the server args' and
model config's attribute names, with the defaults an absent attribute reads."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from smg_grpc_servicer.tokenspeed.config_view import (
    EngineConfigView,
    ModelConfigSection,
    constrained_decoding_policy,
)


def _server_args(**overrides):
    args = SimpleNamespace(
        model="org/m",
        tokenizer="org/m-tok",
        served_model_name="served",
        preferred_sampling_params='{"temperature": 0.6}',
        weight_version="v3",
        revision="rev",
        reasoning_parser="qwen3",
        mapping=SimpleNamespace(attn=SimpleNamespace(dp_size=4)),
        max_num_seqs=64,
        page_size=16,
        max_total_num_tokens=65536,
        kv_events_config='{"enable_kv_cache_events": true, "publisher": "zmq"}',
        disaggregation_mode="prefill",
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _model_config(**overrides):
    config = SimpleNamespace(
        hf_config=SimpleNamespace(
            eos_token_id=[151645, 151643],
            model_type="qwen3",
            architectures=["Qwen3ForCausalLM"],
            pad_token_id=None,
            bos_token_id=1,
        ),
        context_len=8192,
        vocab_size=151936,
        dtype="torch.bfloat16",
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def test_the_server_args_section_reads_every_knob():
    args = EngineConfigView(_server_args()).server_args
    assert (args.model, args.tokenizer, args.served_model_name) == ("org/m", "org/m-tok", "served")
    assert args.preferred_sampling_params == '{"temperature": 0.6}'
    assert (args.weight_version, args.revision, args.reasoning_parser) == ("v3", "rev", "qwen3")
    assert args.attention_dp_size == 4
    assert (args.running_window, args.page_size, args.max_total_num_tokens) == (64, 16, 65536)
    assert args.kv_events_config == '{"enable_kv_cache_events": true, "publisher": "zmq"}'
    assert args.disaggregation_mode == "prefill"


def test_the_model_section_reads_the_hf_config_and_the_shape():
    model = EngineConfigView(_server_args(), _model_config()).model
    assert model.hf_config is not None
    assert (model.eos_token_ids, model.model_type) == ([151645, 151643], "qwen3")
    assert (model.architectures, model.pad_token_id, model.bos_token_id) == (
        ["Qwen3ForCausalLM"],
        0,
        1,
    )
    assert (model.context_len, model.vocab_size, model.dtype) == (8192, 151936, "torch.bfloat16")


def test_the_upstream_spellings_of_the_model_and_tokenizer_are_both_read():
    """Upstream renamed ``model_path`` to ``model`` and ``tokenizer_path`` to
    ``tokenizer``; either spelling serves, the newer one first."""
    older = EngineConfigView(
        SimpleNamespace(model_path="org/old", tokenizer_path="org/old-tok")
    ).server_args
    assert (older.model, older.tokenizer) == ("org/old", "org/old-tok")
    both = EngineConfigView(
        SimpleNamespace(model="org/new", model_path="org/old", tokenizer=None, tokenizer_path="t")
    ).server_args
    assert (both.model, both.tokenizer) == ("org/new", "t")
    bare = EngineConfigView(SimpleNamespace()).server_args
    assert (bare.model, bare.tokenizer) == ("", "")


def test_bare_server_args_read_as_the_defaults():
    args = EngineConfigView(SimpleNamespace()).server_args
    assert (args.model, args.tokenizer, args.served_model_name) == ("", "", None)
    assert (args.preferred_sampling_params, args.weight_version, args.revision) == (None, "", None)
    assert args.reasoning_parser is None and args.attention_dp_size is None
    assert (args.running_window, args.page_size, args.max_total_num_tokens) == (0, 0, 0)
    assert args.kv_events_config is None and args.disaggregation_mode == "null"
    # A dtype read without a model config (the request path's lenient read).
    assert EngineConfigView(SimpleNamespace()).model_dtype is None


@pytest.mark.parametrize(
    ("mapping", "expected"),
    [
        (None, None),
        (SimpleNamespace(), None),
        (SimpleNamespace(attn=SimpleNamespace()), None),
        (SimpleNamespace(attn=SimpleNamespace(dp_size=1)), 1),
        (SimpleNamespace(attn=SimpleNamespace(dp_size=0)), None),
        (SimpleNamespace(attn=SimpleNamespace(dp_size=True)), None),
    ],
)
def test_the_attention_dp_width_is_a_count_or_none(mapping, expected):
    args = _server_args()
    args.mapping = mapping
    assert EngineConfigView(args).server_args.attention_dp_size == expected


def test_the_running_window_takes_either_spelling_and_reads_zero_when_unset():
    assert EngineConfigView(SimpleNamespace(max_num_seqs=16)).server_args.running_window == 16
    assert (
        EngineConfigView(SimpleNamespace(max_running_requests=64)).server_args.running_window == 64
    )
    assert EngineConfigView(SimpleNamespace(max_num_seqs=None)).server_args.running_window == 0


def test_the_model_section_degrades_without_an_hf_config_and_requires_the_vocabulary():
    bare = ModelConfigSection.read(SimpleNamespace(vocab_size=100))
    assert (bare.hf_config, bare.eos_token_ids, bare.model_type, bare.architectures) == (
        None,
        [],
        "",
        [],
    )
    assert (bare.pad_token_id, bare.bos_token_id, bare.context_len, bare.dtype) == (0, 0, 0, None)
    # An int EOS is one id; a list is read in order; anything else is none.
    one = ModelConfigSection.read(_model_config(hf_config=SimpleNamespace(eos_token_id=2)))
    assert one.eos_token_ids == [2]
    none = ModelConfigSection.read(_model_config(hf_config=SimpleNamespace(eos_token_id="2")))
    assert none.eos_token_ids == []
    with pytest.raises(AttributeError, match="vocab_size"):
        ModelConfigSection.read(SimpleNamespace(hf_config=None))


def test_a_section_is_read_once_per_view():
    reads = []

    class Args:
        @property
        def page_size(self):
            reads.append(1)
            return 8

    view = EngineConfigView(Args())
    assert view.server_args.page_size == 8 and view.server_args.page_size == 8
    assert reads == [1]


def test_a_fresh_view_reads_a_weight_version_written_back_onto_the_args():
    """An RL weight update writes the new version onto the launcher's args while
    the servicer runs; a view built earlier keeps what it read, so the live
    answer of ``GetModelInfo`` comes from a fresh view over the same args."""
    args = _server_args(weight_version="v7")
    earlier = EngineConfigView(args)
    assert earlier.server_args.weight_version == "v7"
    args.weight_version = "v8"
    assert earlier.server_args.weight_version == "v7"
    assert EngineConfigView(args).server_args.weight_version == "v8"


def test_constrained_decoding_policy_is_spelled_as_the_gateway_reads_it():
    assert constrained_decoding_policy("always") == "always"
    assert constrained_decoding_policy(" Always ") == "always"
    assert constrained_decoding_policy("requested") == "requested"
    assert constrained_decoding_policy(None) == ""
    assert constrained_decoding_policy("") == ""
    assert constrained_decoding_policy("sometimes") == ""


def test_constrained_decoding_comes_off_the_server_args_then_the_model_config():
    args = SimpleNamespace(model="org/m", constrained_decoding="always")
    model = SimpleNamespace(constrained_decoding="requested")
    assert EngineConfigView(args, model).constrained_decoding == "always"
    assert EngineConfigView(SimpleNamespace(model="org/m"), model).constrained_decoding == (
        "requested"
    )
    assert EngineConfigView(SimpleNamespace(model="org/m")).constrained_decoding == ""
