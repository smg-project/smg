"""The engine config view: one reader of vLLM's attribute names, each section
with the defaults it reads when a version does not carry it."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest
from smg_grpc_servicer.vllm import config_view
from smg_grpc_servicer.vllm.config_view import EngineConfigView, ModelSection


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
        dtype="torch.bfloat16",
        revision="model-rev",
        tokenizer_revision="tok-rev",
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _full_config():
    return SimpleNamespace(
        model_config=_model_config(
            is_multimodal_model=True,
            multimodal_config=SimpleNamespace(
                mm_device_do_normalize=True, get_limit_per_prompt={"image": 8}.__getitem__
            ),
            uses_mrope=True,
            pooler_config=SimpleNamespace(use_activation=0, dimensions=512),
        ),
        parallel_config=SimpleNamespace(
            data_parallel_size=2,
            data_parallel_rank=1,
            data_parallel_size_local=2,
            data_parallel_master_ip="127.0.0.1",
            data_parallel_rpc_port=24321,
            node_rank_within_dp=0,
            data_parallel_backend="mp",
        ),
        shutdown_timeout=30,
        scheduler_config=SimpleNamespace(max_num_seqs=64),
        cache_config=SimpleNamespace(cache_dtype="auto", block_size=16, num_gpu_blocks=1000),
        attention_config=SimpleNamespace(backend=SimpleNamespace(name="FLASH_ATTN")),
        structured_outputs_config=SimpleNamespace(
            backend="xgrammar", reasoning_parser="glm45", enable_in_reasoning=False
        ),
        kv_events_config=SimpleNamespace(
            enable_kv_cache_events=True,
            publisher="zmq",
            endpoint="tcp://*:5557",
            replay_endpoint="tcp://*:5558",
            topic="kv",
        ),
        kv_transfer_config=SimpleNamespace(
            kv_connector="NixlConnector", kv_role="kv_producer", engine_id="eng-a"
        ),
    )


def test_every_section_reads_the_config_it_was_given():
    view = EngineConfigView(_full_config())
    assert view.model == ModelSection(
        model="org/m",
        served_model_name="served",
        tokenizer="org/m-tok",
        runner_type="generate",
        max_model_len=4096,
        vocab_size=1024,
        architectures=["Qwen3ForCausalLM"],
        model_type="qwen3",
        hf_eos_token_ids=[151645, 151643],
        pad_token_id=0,
        bos_token_id=1,
        tokenizer_revision="tok-rev",
    )
    assert view.multimodal.is_multimodal and view.multimodal.device_do_normalize
    assert view.multimodal.limit_per_prompt("image") == 8
    assert view.model_dtype == "torch.bfloat16"
    assert view.parallel.data_parallel_size == 2 and view.parallel.data_parallel_rank == 1
    assert (view.parallel.node_rank_within_dp, view.parallel.data_parallel_backend) == (0, "mp")
    assert view.data_parallel_size == 2
    launch = view.launch
    assert (launch.local_engine_count, launch.start_index) == (2, 1)
    assert (launch.master_ip, launch.rpc_port, launch.shutdown_timeout) == ("127.0.0.1", 24321, 30)
    assert view.max_num_seqs == 64
    assert (view.cache.cache_dtype, view.cache.block_size, view.cache.num_gpu_blocks) == (
        "auto",
        16,
        1000,
    )
    assert view.attention_backend == "FLASH_ATTN"
    structured = view.structured_outputs
    assert (structured.backend, structured.reasoning_parser) == ("xgrammar", "glm45")
    assert structured.reasoning_parser_known and structured.enable_in_reasoning is False
    events = view.kv_events
    assert events.config is not None and events.enable_kv_cache_events and events.relayed
    assert (events.publisher, events.endpoint) == ("zmq", "tcp://*:5557")
    assert (events.replay_endpoint, events.topic) == ("tcp://*:5558", "kv")
    assert (view.pooler.use_activation, view.pooler.dimensions) == (False, 512)
    assert view.kv_transfer.role == "kv_producer"
    assert view.kv_transfer.config.kv_connector == "NixlConnector"
    assert config_view.uses_mrope(view._config.model_config) is True


@pytest.mark.parametrize("config", [None, SimpleNamespace()], ids=["none", "bare"])
def test_a_config_without_a_section_reads_as_the_defaults(config):
    """An older engine, or an engine handle without a config: every lenient
    section is its defaults, and nothing is invented."""
    view = EngineConfigView(config)
    assert view.model_dtype is None
    multimodal = view.multimodal
    assert (multimodal.is_multimodal, multimodal.device_do_normalize) == (False, False)
    assert multimodal.limit_per_prompt is None
    assert view.parallel.data_parallel_size is None and view.parallel.data_parallel_rank == 0
    assert (view.parallel.node_rank_within_dp, view.parallel.data_parallel_backend) == (0, "mp")
    assert view.max_num_seqs == 0
    assert (view.cache.cache_dtype, view.cache.block_size, view.cache.num_gpu_blocks) == (
        None,
        None,
        None,
    )
    assert view.attention_backend is None
    structured = view.structured_outputs
    assert (structured.backend, structured.reasoning_parser) == (None, "")
    assert structured.reasoning_parser_known is False
    events = view.kv_events
    assert events.config is None and not events.enable_kv_cache_events and not events.relayed
    assert (events.publisher, events.endpoint, events.replay_endpoint, events.topic) == (
        None,
        "",
        "",
        "",
    )
    assert (view.pooler.use_activation, view.pooler.dimensions) == (None, None)
    assert view.kv_transfer.config is None and view.kv_transfer.role == ""
    assert config_view.uses_mrope(None) is False


def test_the_model_section_requires_the_facts_a_model_must_have():
    """The name, the context length and the vocabulary are never invented: a
    config without them fails here, before either servicer answers wrong."""
    for missing in ("model", "max_model_len", "get_vocab_size"):
        config = _model_config()
        delattr(config, missing)
        with pytest.raises(AttributeError):
            ModelSection.read(config)
    with pytest.raises(AttributeError):
        EngineConfigView(SimpleNamespace()).model  # noqa: B018 -- the read is the point
    # The data-parallel width the server is told is required too, by name.
    with pytest.raises(AttributeError, match="parallel_config"):
        EngineConfigView(SimpleNamespace()).data_parallel_size  # noqa: B018
    with pytest.raises(AttributeError, match="data_parallel_size"):
        EngineConfigView(SimpleNamespace(parallel_config=SimpleNamespace())).data_parallel_size  # noqa: B018
    two = SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_size=2))
    assert EngineConfigView(two).data_parallel_size == 2
    # So are the process manager's inputs: a config an engine cannot be
    # started from this way fails by name.
    with pytest.raises(AttributeError, match="data_parallel_size_local"):
        EngineConfigView(two).launch  # noqa: B018


def test_the_model_section_degrades_the_rest_to_the_proto_defaults():
    model = ModelSection.read(
        _model_config(
            served_model_name=["first", "second"],
            tokenizer=None,
            runner_type="pooling",
            hf_config=SimpleNamespace(
                model_type=None, eos_token_id=2, pad_token_id=3, bos_token_id=None
            ),
            architectures=None,
        )
    )
    assert (model.served_model_name, model.tokenizer, model.runner_type) == (
        "first",
        "org/m",
        "pooling",
    )
    # The tokenizer's own revision first, else the model's, else none.
    assert model.tokenizer_revision == "tok-rev"
    assert (
        ModelSection.read(_model_config(tokenizer_revision=None)).tokenizer_revision == "model-rev"
    )
    bare_revisions = _model_config(tokenizer_revision=None, revision=None)
    assert ModelSection.read(bare_revisions).tokenizer_revision is None
    assert (model.model_type, model.architectures, model.hf_eos_token_ids) == ("", [], [2])
    assert (model.pad_token_id, model.bos_token_id) == (3, 0)
    # An empty served-name list and a config without an HF config.
    bare = ModelSection.read(_model_config(served_model_name=[], hf_config=None))
    assert bare.served_model_name == "org/m" and bare.model_type == ""
    assert (bare.hf_eos_token_ids, bare.pad_token_id, bare.bos_token_id) == ([], 0, 0)


def test_token_ids_keep_non_negative_ints_only():
    assert config_view.token_ids(None) == []
    assert config_view.token_ids(True) == []
    assert config_view.token_ids(7) == [7]
    assert config_view.token_ids(-1) == []
    assert config_view.token_ids([-1, True, 5, "6", 0]) == [5, 0]
    assert config_view.token_ids("7") == []


def test_the_multimodal_section_is_off_on_a_text_model_or_without_the_accessor():
    text = EngineConfigView(SimpleNamespace(model_config=_model_config())).multimodal
    assert (text.is_multimodal, text.device_do_normalize, text.limit_per_prompt) == (
        False,
        False,
        None,
    )
    no_accessor = EngineConfigView(
        SimpleNamespace(
            model_config=_model_config(
                is_multimodal_model=True,
                multimodal_config=SimpleNamespace(mm_device_do_normalize=True),
            )
        )
    ).multimodal
    assert no_accessor.is_multimodal and no_accessor.device_do_normalize
    assert no_accessor.limit_per_prompt is None
    # Without the multimodal section at all the flags stay off.
    bare = EngineConfigView(
        SimpleNamespace(model_config=_model_config(is_multimodal_model=True))
    ).multimodal
    assert bare.is_multimodal and not bare.device_do_normalize and bare.limit_per_prompt is None


@pytest.mark.parametrize("window", [None, 0, -1, True, "64"])
def test_a_running_window_that_is_not_a_count_reads_as_zero(window):
    config = SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=window))
    assert EngineConfigView(config).max_num_seqs == 0


def test_the_cache_section_keeps_counts_and_a_named_dtype_only():
    cache = EngineConfigView(
        SimpleNamespace(
            cache_config=SimpleNamespace(cache_dtype="", block_size=0, num_gpu_blocks=-1)
        )
    ).cache
    assert (cache.cache_dtype, cache.block_size, cache.num_gpu_blocks) == (None, None, None)
    cache = EngineConfigView(
        SimpleNamespace(cache_config=SimpleNamespace(cache_dtype="fp8", block_size=32))
    ).cache
    assert (cache.cache_dtype, cache.block_size, cache.num_gpu_blocks) == ("fp8", 32, None)


def test_the_attention_backend_is_the_requested_one_by_name():
    by_enum = SimpleNamespace(
        attention_config=SimpleNamespace(backend=SimpleNamespace(name="FLASHINFER"))
    )
    assert EngineConfigView(by_enum).attention_backend == "FLASHINFER"
    by_str = SimpleNamespace(attention_config=SimpleNamespace(backend="TRITON_ATTN"))
    assert EngineConfigView(by_str).attention_backend == "TRITON_ATTN"
    # The auto path: the worker picks the backend, the config says nothing.
    auto = SimpleNamespace(attention_config=SimpleNamespace(backend=None))
    assert EngineConfigView(auto).attention_backend is None
    assert (
        EngineConfigView(SimpleNamespace(attention_config=SimpleNamespace())).attention_backend
        is None
    )


def test_the_structured_outputs_section_tells_an_absent_parser_from_none():
    older = SimpleNamespace(structured_outputs_config=SimpleNamespace(backend="auto"))
    structured = EngineConfigView(older).structured_outputs
    assert (structured.reasoning_parser, structured.reasoning_parser_known) == ("", False)
    for none in ("", None):
        structured = EngineConfigView(
            SimpleNamespace(
                structured_outputs_config=SimpleNamespace(
                    reasoning_parser=none, enable_in_reasoning=True
                )
            )
        ).structured_outputs
        assert (structured.reasoning_parser, structured.reasoning_parser_known) == ("", True)
        assert structured.enable_in_reasoning is True


def test_the_kv_events_section_relays_a_zmq_publisher_only():
    def config(**fields):
        return SimpleNamespace(kv_events_config=SimpleNamespace(**fields))

    events = EngineConfigView(config(enable_kv_cache_events=False, publisher="zmq")).kv_events
    assert (events.enable_kv_cache_events, events.publisher, events.endpoint) == (False, "zmq", "")
    assert not events.relayed
    events = EngineConfigView(config(enable_kv_cache_events=True, publisher="null")).kv_events
    assert (events.enable_kv_cache_events, events.publisher) == (True, "null")
    assert not events.relayed
    events = EngineConfigView(config(enable_kv_cache_events=True, publisher="zmq")).kv_events
    assert events.relayed and (events.endpoint, events.replay_endpoint, events.topic) == (
        "",
        "",
        "",
    )


def test_a_section_is_read_once_per_view():
    reads = []

    class Scheduler:
        @property
        def max_num_seqs(self):
            reads.append(1)
            return 8

    view = EngineConfigView(SimpleNamespace(scheduler_config=Scheduler()))
    assert view.max_num_seqs == 8 and view.max_num_seqs == 8
    assert reads == [1]


@dataclasses.dataclass
class _EngineModelConfig:
    """The engine's layout of the processor kwargs: ``mm_processor_kwargs``
    is an init-only field whose value ``__post_init__`` moves into the
    multimodal section (built for a multimodal model only); the attribute
    on the instance reads as the class default afterwards, and the accessor
    raises on a text model, as the engine's does."""

    model: str = "org/m"
    is_multimodal_model: bool = True
    mm_processor_kwargs: dataclasses.InitVar[dict | None] = None
    multimodal_config: SimpleNamespace | None = None

    def __post_init__(self, mm_processor_kwargs):
        if self.is_multimodal_model:
            self.multimodal_config = SimpleNamespace(
                mm_device_do_normalize=True, mm_processor_kwargs=mm_processor_kwargs
            )

    def get_multimodal_config(self):
        if self.multimodal_config is None:
            raise ValueError("The model is not multimodal.")
        return self.multimodal_config


def test_the_processor_kwargs_are_read_from_the_section_the_engine_moves_them_to():
    kwargs = {"max_pixels": 100_000, "do_resize": False}
    config = _EngineModelConfig(mm_processor_kwargs=kwargs)
    # What a reader of the model config's own attribute sees on every engine.
    assert config.mm_processor_kwargs is None
    assert getattr(config, "mm_processor_kwargs", None) is None
    assert config_view.multimodal_config(config) is config.multimodal_config
    assert config_view.mm_processor_kwargs(config) == kwargs
    # A copy: the pipeline's own filtering never edits the engine's config.
    read = config_view.mm_processor_kwargs(config)
    read.pop("do_resize")
    assert config.multimodal_config.mm_processor_kwargs == kwargs


def test_the_processor_kwargs_read_the_same_where_the_field_is_a_plain_attribute():
    # A config that keeps the field as a plain attribute carries the same value
    # in its multimodal section; the section is what is read either way.
    kwargs = {"min_pixels": 784}
    both = SimpleNamespace(
        mm_processor_kwargs=kwargs, multimodal_config=SimpleNamespace(mm_processor_kwargs=kwargs)
    )
    assert config_view.mm_processor_kwargs(both) == kwargs
    # The attribute alone is no source: on the engine it is the init-only
    # field, always None, so a stub that sets it describes no engine.
    attribute_only = SimpleNamespace(
        mm_processor_kwargs=kwargs, multimodal_config=SimpleNamespace(mm_device_do_normalize=True)
    )
    assert config_view.mm_processor_kwargs(attribute_only) == {}


def test_the_processor_kwargs_follow_the_accessor_when_the_section_is_behind_it():
    section = SimpleNamespace(mm_processor_kwargs={"max_pixels": 1})
    behind_the_accessor = SimpleNamespace(get_multimodal_config=lambda: section)
    assert config_view.multimodal_config(behind_the_accessor) is section
    assert config_view.mm_processor_kwargs(behind_the_accessor) == {"max_pixels": 1}
    # The attribute wins over the accessor when both are there (one object on the engine).
    config = _EngineModelConfig(mm_processor_kwargs={"max_pixels": 2})
    config.get_multimodal_config = lambda: section
    assert config_view.mm_processor_kwargs(config) == {"max_pixels": 2}
    # The section itself resolves the same way: one lookup for every reader.
    section = SimpleNamespace(
        mm_device_do_normalize=True,
        get_limit_per_prompt={"image": 4}.__getitem__,
        mm_processor_kwargs={"max_pixels": 1},
    )
    view = EngineConfigView(
        SimpleNamespace(
            model_config=SimpleNamespace(
                is_multimodal_model=True, get_multimodal_config=lambda: section
            )
        )
    )
    assert view.multimodal.device_do_normalize is True
    assert view.multimodal.limit_per_prompt("image") == 4


def test_no_processor_kwargs_read_as_an_empty_mapping():
    text = _EngineModelConfig(is_multimodal_model=False)
    assert config_view.multimodal_config(text) is None
    assert config_view.mm_processor_kwargs(text) == {}
    for config in (
        SimpleNamespace(),
        SimpleNamespace(multimodal_config=None),
        SimpleNamespace(multimodal_config=SimpleNamespace()),
        _EngineModelConfig(),
        _EngineModelConfig(mm_processor_kwargs={}),
        _EngineModelConfig(mm_processor_kwargs="max_pixels=1"),
        _EngineModelConfig(mm_processor_kwargs=["max_pixels"]),
    ):
        assert config_view.mm_processor_kwargs(config) == {}, config
