"""Unit tests for the Rust request path of the TokenSpeed servicer
(``smg_grpc_servicer.tokenspeed.rust``).

Everything TokenSpeed-shaped is lazy in the module, so these run without
TokenSpeed: the facts get a fake model config and a stub servicer module,
``serve_rust`` gets a fake Rust server class and a fake scheduler launch.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import importlib.util
import inspect
import json
import logging
import os
import signal
import sys
import threading
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
from smg_grpc_servicer import rust_lifecycle
from smg_grpc_servicer.tokenspeed import rust


def _install(monkeypatch, name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    parent, _, child = name.rpartition(".")
    if parent:
        parent_module = sys.modules.get(parent) or _install(monkeypatch, parent)
        # Undone with the test: the parent is the real package where an engine is installed.
        monkeypatch.setattr(parent_module, child, module, raising=False)
    return module


def _install_fake_servicer(monkeypatch, server, init_servicer_tracing=None):
    """A fake ``smg.servicer`` binding under a fake ``smg`` package: the
    installed package, where there is one, keeps its own ``servicer`` module."""
    _install(monkeypatch, "smg")
    return _install(
        monkeypatch,
        "smg.servicer",
        TokenSpeedGrpcServer=server,
        init_servicer_tracing=init_servicer_tracing or (lambda level=None: None),
    )


def _stub_servicer_module(monkeypatch, modalities=(1,)):
    """The pieces of ``smg_grpc_servicer.tokenspeed.servicer`` the facts use,
    without importing TokenSpeed."""

    class Servicer:
        @staticmethod
        def _static_supported_modalities(model_config, hf_config):
            return list(modalities) if getattr(model_config, "is_multimodal", False) else []

    def make_json_serializable(obj):
        if obj is None or isinstance(obj, str | int | float | bool):
            return obj
        if isinstance(obj, list | tuple | set):
            return [make_json_serializable(x) for x in obj]
        if isinstance(obj, dict):
            return {str(k): make_json_serializable(v) for k, v in obj.items()}
        return str(obj)

    _install(
        monkeypatch,
        "smg_grpc_servicer.tokenspeed.servicer",
        TokenSpeedSchedulerServicer=Servicer,
        _make_json_serializable=make_json_serializable,
        _shm_namespace_id=lambda: "boot:7",
    )


@dataclasses.dataclass
class FakeServerArgs:
    model: str = "org/m"
    tokenizer: str | None = None
    served_model_name: str | None = "served"
    host: str = "127.0.0.1"
    port: int = 50051
    max_num_seqs: int | None = 64
    preferred_sampling_params: str | None = '{"temperature": 0.6}'
    kv_events_config: str | None = None
    mapping: SimpleNamespace = dataclasses.field(
        default_factory=lambda: SimpleNamespace(
            attn=SimpleNamespace(dp_size=1, has_dp=False), rank=None
        )
    )
    enable_memory_saver: bool = False
    zmq_msgpack: bool = False
    skip_tokenizer_init: bool = False
    rl_control_api_key: str | None = None
    hf_token: str | None = None
    data_parallel_address: str = "127.0.0.1"
    data_parallel_rpc_port: int = 30500
    zmq_engine_index: int = 0
    disaggregation_mode: str = "null"
    disaggregation_bootstrap_port: int = 8998
    disaggregation_transfer_backend: str = "mooncake"


def test_package_exposes_the_flag_and_the_rust_entry_points():
    import smg_grpc_servicer.tokenspeed as pkg

    assert pkg.SERVICER_IMPL_ENV == "SMG_TOKENSPEED_SERVICER_IMPL"
    assert pkg.resolve_servicer_impl is rust.resolve_servicer_impl
    assert pkg.serve_rust is rust.serve_rust


def test_resolve_servicer_impl_prefers_the_launcher_flag_then_the_env():
    assert rust.resolve_servicer_impl(environ={}) == "python"
    assert rust.resolve_servicer_impl(environ={"SMG_TOKENSPEED_SERVICER_IMPL": " Rust "}) == "rust"
    args = argparse.Namespace(servicer_impl="rust")
    assert rust.resolve_servicer_impl(args, environ={}) == "rust"
    # The flag wins over the environment, in both directions.
    assert (
        rust.resolve_servicer_impl(args, environ={"SMG_TOKENSPEED_SERVICER_IMPL": "python"})
        == "rust"
    )
    args = argparse.Namespace(servicer_impl="python")
    assert (
        rust.resolve_servicer_impl(args, environ={"SMG_TOKENSPEED_SERVICER_IMPL": "rust"})
        == "python"
    )
    # An unset launcher flag falls through to the env.
    args = argparse.Namespace(servicer_impl=None)
    assert (
        rust.resolve_servicer_impl(args, environ={"SMG_TOKENSPEED_SERVICER_IMPL": "rust"}) == "rust"
    )
    with pytest.raises(ValueError, match="SMG_TOKENSPEED_SERVICER_IMPL must be one of"):
        rust.resolve_servicer_impl(environ={"SMG_TOKENSPEED_SERVICER_IMPL": "go"})
    with pytest.raises(ValueError, match="SMG_TOKENSPEED_SERVICER_IMPL must be one of"):
        rust.resolve_servicer_impl(argparse.Namespace(servicer_impl="go"), environ={})


def test_servicer_impl_source_names_the_origin_and_writes_a_flag_back():
    assert rust.servicer_impl_source(environ={}) == ("python", "default")
    assert rust.servicer_impl_source(environ={"SMG_TOKENSPEED_SERVICER_IMPL": "rust"}) == (
        "rust",
        "env",
    )
    # `--servicer-impl python` with `rust` still exported: the flag decides,
    # and the environment the headless child inherits says the same.
    environ = {"SMG_TOKENSPEED_SERVICER_IMPL": "rust"}
    args = argparse.Namespace(servicer_impl="python")
    assert rust.servicer_impl_source(args, environ=environ) == ("python", "flag")
    assert environ == {"SMG_TOKENSPEED_SERVICER_IMPL": "python"}
    # Without a flag in hand nothing is written.
    environ = {}
    assert rust.servicer_impl_source(SimpleNamespace(), environ=environ) == ("python", "default")
    assert environ == {}


def test_the_flag_joins_a_parser_once_with_argparse_validation():
    parser = argparse.ArgumentParser(prog="launcher")
    assert rust.add_servicer_impl_argument(parser) is True
    assert rust.add_servicer_impl_argument(parser) is False
    assert "--servicer-impl {python,rust}" in parser.format_help()
    assert "SMG_TOKENSPEED_SERVICER_IMPL" in parser.format_help()
    assert parser.parse_args([]).servicer_impl is None
    assert parser.parse_args(["--servicer-impl", "rust"]).servicer_impl == "rust"
    with pytest.raises(SystemExit):
        parser.parse_args(["--servicer-impl", "go"])


@pytest.fixture
def launcher(monkeypatch):
    """``python -m smg_grpc_servicer.tokenspeed`` with TokenSpeed's argument
    parsing and both serving paths stood in for: ``prepare_server_args``
    records the argv it is handed (and prints a help line on --help, as
    TokenSpeed's parser would, then exits), the two servers record which ran."""
    calls = {"argv": None, "served": None}

    def prepare_server_args(argv):
        calls["argv"] = list(argv)
        if any(arg in ("-h", "--help") for arg in argv):
            print("usage: tokenspeed [--model MODEL] (TokenSpeed's own help)")
            raise SystemExit(0)
        mode = "null"
        if "--disaggregation-mode" in argv:
            mode = argv[argv.index("--disaggregation-mode") + 1]
        return FakeServerArgs(model=argv[argv.index("--model") + 1], disaggregation_mode=mode)

    _install(
        monkeypatch,
        "tokenspeed.runtime.utils.server_args",
        prepare_server_args=prepare_server_args,
        ServerArgs=FakeServerArgs,
    )

    async def serve_grpc(server_args):
        calls["served"] = ("python", server_args)

    _install(monkeypatch, "smg_grpc_servicer.tokenspeed.server", serve_grpc=serve_grpc)
    path = Path(rust.__file__).with_name("__main__.py")
    spec = importlib.util.spec_from_file_location("test_tokenspeed_launcher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    async def serve_rust(server_args):
        calls["served"] = ("rust", server_args)
        return 3

    monkeypatch.setattr(module, "serve_rust", serve_rust)
    monkeypatch.setattr(module, "uvloop", None)
    return module, calls


def test_the_launcher_takes_the_flag_ahead_of_tokenspeeds_args(launcher, monkeypatch, caplog):
    module, calls = launcher
    monkeypatch.setenv(rust.SERVICER_IMPL_ENV, "python")
    with caplog.at_level("INFO", logger=module.logger.name):
        with pytest.raises(SystemExit) as raised:
            module.main(["--model", "org/m", "--servicer-impl", "rust", "--port", "50051"])
    assert raised.value.code == 3
    # TokenSpeed's parser never sees this package's flag; the rest is verbatim.
    assert calls["argv"] == ["--model", "org/m", "--port", "50051"]
    assert calls["served"][0] == "rust" and calls["served"][1].model == "org/m"
    assert "Servicer implementation: rust (source=flag)" in caplog.text
    # The decision reaches the environment the headless child inherits.
    assert os.environ[rust.SERVICER_IMPL_ENV] == "rust"


def test_the_launcher_falls_back_to_the_environment_then_python(launcher, monkeypatch, caplog):
    module, calls = launcher
    monkeypatch.setenv(rust.SERVICER_IMPL_ENV, "rust")
    with caplog.at_level("INFO", logger=module.logger.name):
        with pytest.raises(SystemExit) as raised:
            module.main(["--model", "org/m"])
    assert raised.value.code == 3 and calls["served"][0] == "rust"
    assert calls["argv"] == ["--model", "org/m"]
    assert "Servicer implementation: rust (source=env)" in caplog.text
    # `--servicer-impl python` overrides the exported `rust`, environment included.
    caplog.clear()
    with caplog.at_level("INFO", logger=module.logger.name):
        module.main(["--servicer-impl=python", "--model", "org/m"])
    assert calls["served"][0] == "python" and calls["argv"] == ["--model", "org/m"]
    assert "Servicer implementation: python (source=flag)" in caplog.text
    assert os.environ[rust.SERVICER_IMPL_ENV] == "python"
    # Neither: the Python servicer, and the environment is left alone.
    monkeypatch.delenv(rust.SERVICER_IMPL_ENV)
    caplog.clear()
    with caplog.at_level("INFO", logger=module.logger.name):
        module.main(["--model", "org/m"])
    assert calls["served"][0] == "python"
    assert "Servicer implementation: python (source=default)" in caplog.text
    assert rust.SERVICER_IMPL_ENV not in os.environ


def test_the_launcher_lists_the_flag_ahead_of_tokenspeeds_help(launcher, capsys):
    module, calls = launcher
    with pytest.raises(SystemExit) as raised:
        module.main(["--help"])
    assert raised.value.code == 0
    out = capsys.readouterr().out
    assert "--servicer-impl {python,rust}" in out
    assert out.index("--servicer-impl") < out.index("TokenSpeed's own help")
    assert calls["argv"] == ["--help"]
    # An invalid choice is refused by this package's parser, before TokenSpeed's runs.
    calls["argv"] = None
    with pytest.raises(SystemExit) as raised:
        module.main(["--model", "org/m", "--servicer-impl", "go"])
    assert raised.value.code == 2 and calls["argv"] is None


def test_model_facts_mirror_the_python_servicer(monkeypatch):
    _stub_servicer_module(monkeypatch, modalities=(1, 3))
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            eos_token_id=[151645, 151643],
            model_type="qwen3_vl",
            architectures=["Qwen3VLForConditionalGeneration"],
            pad_token_id=None,
            bos_token_id=None,
        ),
        context_len=8192,
        vocab_size=151936,
        dtype="torch.bfloat16",
        is_multimodal=True,
    )
    facts = rust.model_facts(FakeServerArgs(), model_config=model_config)
    assert facts == {
        "model_path": "org/m",
        "tokenizer_path": "org/m",
        "served_model_name": "served",
        "model_type": "qwen3_vl",
        "architectures": ["Qwen3VLForConditionalGeneration"],
        "max_context_length": 8192,
        "vocab_size": 151936,
        "eos_token_ids": [151645, 151643],
        "pad_token_id": 0,
        "bos_token_id": 0,
        "default_sampling_params_json": '{"temperature": 0.6}',
        "supports_vision": True,
        "supports_multimodal": True,
        "supported_modalities": [1, 3],
        "mrope_positions": None,
        "spatial_merge_size": None,
        "image_token_id": None,
        "video_token_id": None,
        "model_dtype": "bfloat16",
        "multimodal_encoder_dtype": "bfloat16",
        "constrained_decoding": "",
    }
    # A text model: no modalities, an int EOS.
    text = SimpleNamespace(
        hf_config=SimpleNamespace(eos_token_id=2, model_type="qwen3", architectures=["Q"]),
        context_len=4096,
        vocab_size=100,
        dtype="torch.float16",
        is_multimodal=False,
    )
    facts = rust.model_facts(FakeServerArgs(served_model_name=None), model_config=text)
    assert facts["served_model_name"] == "org/m"
    assert facts["eos_token_ids"] == [2]
    assert facts["supports_vision"] is False
    assert facts["supported_modalities"] == []


def test_server_facts_carry_labels_window_and_kv_events(monkeypatch):
    _stub_servicer_module(monkeypatch)
    monkeypatch.setenv("SMG_PAIRING_PROTOCOL", " nixl ")
    args = FakeServerArgs(
        kv_events_config='{"enable_kv_cache_events": true, "publisher": "zmq", '
        '"endpoint": "tcp://*:5600", "replay_endpoint": "tcp://*:5601", "topic": "kv"}',
        mapping=SimpleNamespace(attn=SimpleNamespace(dp_size=2)),
    )
    facts = rust.server_facts(args)
    server_args = json.loads(facts["server_args_json"])
    assert server_args["model"] == "org/m"
    assert server_args["max_num_seqs"] == 64
    # The DP width (only when > 1) and the pairing protocol ride with the
    # server args, as the Python servicer adds them.
    assert server_args["dp_size"] == 2
    assert server_args["pairing_protocol"] == "nixl"
    assert json.loads(facts["scheduler_info_json"]) == {"shm_namespace_id": "boot:7"}
    assert facts["max_running_requests"] == 32
    assert facts["data_parallel_size"] == 2
    assert (facts["kv_events_endpoint"], facts["kv_events_topic"]) == ("tcp://*:5600", "kv")
    assert facts["kv_events_replay_endpoint"] == "tcp://*:5601"

    monkeypatch.delenv("SMG_PAIRING_PROTOCOL")
    plain = rust.server_facts(FakeServerArgs())
    server_args = json.loads(plain["server_args_json"])
    assert "dp_size" not in server_args and "pairing_protocol" not in server_args
    assert plain["data_parallel_size"] == 1
    assert plain["kv_events_endpoint"] == ""
    assert plain["kv_events_replay_endpoint"] == ""


@dataclasses.dataclass
class _ReasoningServerArgs(FakeServerArgs):
    reasoning_parser: str | None = None


def test_server_facts_carry_where_the_engine_starts_a_grammar(monkeypatch):
    # The engine's reasoning parser and where it starts a request's grammar on
    # a thinking prompt ride with the server args, as the Python servicer adds
    # them: this engine starts every grammar at the first token, parser or not.
    _stub_servicer_module(monkeypatch)
    monkeypatch.delenv("SMG_PAIRING_PROTOCOL", raising=False)
    gated = json.loads(
        rust.server_facts(_ReasoningServerArgs(reasoning_parser="kimi_k25"))["server_args_json"]
    )
    assert gated["engine_reasoning_parser"] == "kimi_k25"
    assert gated["constrained_decoding_start"] == "first_token"
    # The engine's own argument stays under its own name; the Router's label
    # readers never take it as the per-model parser override.
    assert gated["reasoning_parser"] == "kimi_k25"

    plain = json.loads(rust.server_facts(_ReasoningServerArgs())["server_args_json"])
    assert plain["constrained_decoding_start"] == "first_token"
    assert "engine_reasoning_parser" not in plain


@dataclasses.dataclass
class _GrammarServerArgs(FakeServerArgs):
    grammar_backend: str | None = "none"


def test_server_facts_carry_the_grammar_backend_the_scheduler_runs(monkeypatch):
    # The grammar backend, resolved, under the engine's own argument name: the
    # Rust servicer refuses a constraint that backend cannot enforce before
    # the scheduler. An unset flag is the checkpoint's default, which the
    # engine resolves once it has read the checkpoint; until then nothing is
    # written, so nothing is judged ahead of the engine.
    _stub_servicer_module(monkeypatch)
    monkeypatch.delenv("SMG_PAIRING_PROTOCOL", raising=False)
    pinned = json.loads(
        rust.server_facts(_GrammarServerArgs(grammar_backend="xgrammar"))["server_args_json"]
    )
    assert pinned["grammar_backend"] == "xgrammar"
    explicit_none = _GrammarServerArgs(grammar_backend="none")
    explicit_none._grammar_backend_unset = False
    assert json.loads(rust.server_facts(explicit_none)["server_args_json"])["grammar_backend"] == (
        "none"
    )
    unset = _GrammarServerArgs(grammar_backend="none")
    unset._grammar_backend_unset = True
    unset._beam_checkpoint = None
    assert "grammar_backend" not in json.loads(rust.server_facts(unset)["server_args_json"])
    unset._beam_checkpoint = True
    assert json.loads(rust.server_facts(unset)["server_args_json"])["grammar_backend"] == "xgrammar"


def test_server_facts_never_carry_credentials(monkeypatch):
    """The Rust server reports the same server args as the Python one, so the
    in-engine RL control key and tokens must not leave the engine here either.
    """
    _stub_servicer_module(monkeypatch)
    facts = rust.server_facts(FakeServerArgs(rl_control_api_key="s3cret", hf_token="hf_abc"))
    server_args = json.loads(facts["server_args_json"])
    assert "rl_control_api_key" not in server_args
    assert "hf_token" not in server_args
    assert server_args["model"] == "org/m", "non-secrets stay"


def test_headless_server_args_dial_the_servicer():
    args = FakeServerArgs()
    headless = rust.headless_server_args(args, handshake_port=24321)
    assert headless is not args
    assert headless.zmq_msgpack is True
    assert headless.skip_tokenizer_init is True
    assert headless.data_parallel_address == "127.0.0.1"
    assert headless.data_parallel_rpc_port == 24321
    assert headless.zmq_engine_index == 0
    # The caller's args are untouched.
    assert args.zmq_msgpack is False


def test_kv_event_publishing_is_on_by_default_under_the_rust_servicer():
    """A launcher without --kv-events-config left TokenSpeed publishing nothing
    and the router's cache-aware routing blind (smg-lab #1): the Rust path
    turns the ZMQ publisher on itself unless told otherwise, at a free port of
    this host (one per attention-DP rank) rather than TokenSpeed's fixed
    default."""

    def published_at(args) -> int:
        """The publisher's port, from the JSON the server args carry: on, at
        a free port of this host, never the fixed default; TokenSpeed's own
        defaults for the rest."""
        config = json.loads(args.kv_events_config)
        expected = {"enable_kv_cache_events": True, "publisher": "zmq"}
        assert config == {**expected, "endpoint": config["endpoint"]}
        endpoint = config["endpoint"]
        assert endpoint.startswith("tcp://*:") and endpoint != "tcp://*:5557"
        port = int(endpoint.rpartition(":")[2])
        assert rust_lifecycle._bindable(port)
        return port

    # Nothing given: the ZMQ publisher at a free port, and the relay is told
    # the same endpoint.
    args = FakeServerArgs()
    assert rust.default_kv_events_config(args, environ={}) == args.kv_events_config
    port = published_at(args)
    resolved = rust.resolve_kv_events_config(args)
    assert resolved is not None and resolved.endpoint == f"tcp://*:{port}"
    args = FakeServerArgs(kv_events_config="")
    assert rust.default_kv_events_config(args, environ={}) == args.kv_events_config
    published_at(args)
    # Two attention-DP ranks publish on two consecutive ports: both free.
    args = FakeServerArgs(
        mapping=SimpleNamespace(attn=SimpleNamespace(dp_size=2, has_dp=True), rank=None)
    )
    rust.default_kv_events_config(args, environ={})
    base = published_at(args)
    assert rust_lifecycle._bindable(base + 1)
    # The opt-out.
    for value in ("0", "false", "No", " off "):
        args = FakeServerArgs()
        assert rust.default_kv_events_config(args, environ={rust.KV_EVENTS_ENV: value}) is None
        assert args.kv_events_config is None
    args = FakeServerArgs()
    assert rust.default_kv_events_config(args, environ={rust.KV_EVENTS_ENV: "1"}) is not None
    published_at(args)
    # An explicit configuration is kept as given, its endpoint and off included.
    for given in (
        '{"enable_kv_cache_events": false}',
        '{"enable_kv_cache_events": true, "publisher": "zmq", "endpoint": "tcp://*:5600"}',
    ):
        args = FakeServerArgs(kv_events_config=given)
        assert rust.default_kv_events_config(args, environ={}) is None
        assert args.kv_events_config == given
    resolved = rust.resolve_kv_events_config(args)
    assert resolved is not None and resolved.endpoint == "tcp://*:5600"
    # Server args that cannot be written are left alone.

    class Frozen:
        __slots__ = ()
        kv_events_config = None

    assert rust.default_kv_events_config(Frozen(), environ={}) is None


def _install_cache_vocabulary(monkeypatch, state_labels=("linear_attention",)):
    """TokenSpeed's own word on which layer labels get a State group: the
    cache recipes' vocabulary module, with nothing else of the engine."""
    return _install(
        monkeypatch,
        "tokenspeed.runtime.layers.attention.kv_cache.recipes.spec",
        STATE_LAYER_TYPES=frozenset(state_labels),
    )


def _model_config_of(*labels, nested=False, attribute="layer_types"):
    """A model config whose checkpoint labels its layers ``labels`` (on the
    config itself, or on the text config nested in it), as TokenSpeed's
    ``ModelConfig`` holds it; the facts' fields as well, for a launcher."""
    text = SimpleNamespace(
        eos_token_id=2,
        model_type="m",
        architectures=["MForCausalLM"],
        pad_token_id=None,
        bos_token_id=None,
        **{attribute: list(labels)},
    )
    return SimpleNamespace(
        hf_config=SimpleNamespace(text_config=text) if nested else text,
        context_len=8192,
        vocab_size=151936,
        dtype="torch.bfloat16",
        is_multimodal=False,
    )


def test_kv_event_publishing_stays_off_for_a_model_whose_cache_keeps_recurrent_state(
    monkeypatch, caplog
):
    """TokenSpeed keeps a hybrid model's recurrent state in State groups,
    which span no tokens, and its KV-event drain aborts the engine on such a
    group: every prefill and decode worker of a Gated-DeltaNet model died at
    its first request once the default turned the publisher on. The default
    follows the engine's own vocabulary of which layer labels get a State
    group, read off the model config the engine builds: the publisher stays
    off for such a model, said once with what turns it on; a model whose
    layers are all history is served as before, as is one on an engine
    without that vocabulary; an explicit configuration and the opt-out are
    not asked, and the model config is not read for them."""
    _install_cache_vocabulary(monkeypatch)
    hybrid = _model_config_of(
        "linear_attention", "linear_attention", "linear_attention", "full_attention"
    )
    args = FakeServerArgs()
    with caplog.at_level(logging.INFO, logger=rust_lifecycle.logger.name):
        assert rust.default_kv_events_config(args, environ={}, model_config=hybrid) is None
    assert args.kv_events_config is None
    assert (
        "KV event publishing left off: org/m keeps recurrent state in 3 of its 4 layers "
        "(linear_attention): State groups of TokenSpeed's cache, which span no tokens, and "
        "TokenSpeed's KV-event drain aborts the engine on a group with no token span; pass "
        "--kv-events-config to turn the publisher on anyway"
    ) in caplog.text
    # The labels on the text config nested in a multimodal checkpoint's, and a
    # checkpoint that declares its cache labels outright, read the same.
    for config in (
        _model_config_of("full_attention", "linear_attention", nested=True),
        _model_config_of("linear_attention", attribute="cache_layer_types"),
    ):
        args = FakeServerArgs()
        assert rust.default_kv_events_config(args, environ={}, model_config=config) is None
        assert args.kv_events_config is None
    # Layers that are all history, under any label the recipes know, a
    # checkpoint that labels none (all full attention), and a config with no
    # checkpoint config at all: the publisher is on, as before.
    for config in (
        _model_config_of("full_attention", "full_attention"),
        _model_config_of("full_attention", "sliding_attention"),
        _model_config_of("deepseek_sparse_attention"),
        _model_config_of(),
        SimpleNamespace(hf_config=None),
    ):
        args = FakeServerArgs()
        assert rust.default_kv_events_config(args, environ={}, model_config=config) is not None
        assert json.loads(args.kv_events_config)["enable_kv_cache_events"] is True
    # Without a model config given (the Python servicer's entry point), the
    # launcher's own is built, once the vocabulary says there is something to
    # look for; an explicit configuration and the opt-out never build it.
    monkeypatch.setattr(rust, "_model_config", lambda args: hybrid)
    args = FakeServerArgs()
    assert rust.default_kv_events_config(args, environ={}) is None and args.kv_events_config is None

    def never(args):
        raise AssertionError("the model config is not read when no default would apply")

    monkeypatch.setattr(rust, "_model_config", never)
    given = '{"enable_kv_cache_events": true, "publisher": "zmq", "endpoint": "tcp://*:5600"}'
    args = FakeServerArgs(kv_events_config=given)
    assert rust.default_kv_events_config(args, environ={}) is None
    assert args.kv_events_config == given
    args = FakeServerArgs()
    assert rust.default_kv_events_config(args, environ={rust.KV_EVENTS_ENV: "0"}) is None
    assert args.kv_events_config is None
    # An engine without the vocabulary has no State groups: the publisher is
    # on, and no model config is read for the decision.
    monkeypatch.delitem(sys.modules, "tokenspeed.runtime.layers.attention.kv_cache.recipes.spec")
    args = FakeServerArgs()
    assert rust.default_kv_events_config(args, environ={}) is not None
    assert json.loads(args.kv_events_config)["enable_kv_cache_events"] is True


def test_the_launcher_builds_the_model_config_once_for_the_default_and_the_facts(monkeypatch):
    """The Rust launcher reads the KV-events decision and the model facts off
    one ``ModelConfig``, built on first use and kept on the instance."""
    _install_cache_vocabulary(monkeypatch)
    _stub_servicer_module(monkeypatch)
    builds: list[str] = []

    def build(args):
        builds.append(args.model)
        return _model_config_of("linear_attention", "full_attention")

    monkeypatch.setattr(rust, "_model_config", build)
    launcher = rust.TokenSpeedLauncher()
    args = FakeServerArgs()
    assert launcher.kv_events_default(args) is None and args.kv_events_config is None
    facts = launcher.facts(args)
    assert facts["model_path"] == "org/m" and facts["kv_events_endpoint"] == ""
    assert builds == ["org/m"]
    # A dense model under a fresh launcher: on, at a free port, and the facts
    # carry the endpoint the relay dials.
    monkeypatch.setattr(rust, "_model_config", lambda args: _model_config_of("full_attention"))
    launcher = rust.TokenSpeedLauncher()
    args = FakeServerArgs()
    assert launcher.kv_events_default(args) == args.kv_events_config
    assert (
        launcher.facts(args)["kv_events_endpoint"] == json.loads(args.kv_events_config)["endpoint"]
    )


@pytest.fixture
def restore_fault_signals():
    """The rank launch installs SIGUSR1/SIGQUIT handlers in this process; put
    the previous dispositions back."""
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGUSR1, signal.SIGQUIT)}
    yield
    for sig, handler in previous.items():
        signal.signal(sig, handler if handler is not None else signal.SIG_DFL)


class _FakeRankProcess:
    """`multiprocessing.Process` as `launch_scheduler_ranks` uses it: records
    the target, the arguments and the process that started it; reports ready
    on the pipe it was given, as the scheduler entry point does once the
    model is loaded."""

    started: list[_FakeRankProcess] = []

    def __init__(self, *, target, args, name):
        self.target, self.args, self.name = target, args, name
        self.pid = 2000 + len(_FakeRankProcess.started)
        self.exitcode: int | None = None
        self.events: list[str] = []
        self.started_in: int | None = None
        _FakeRankProcess.started.append(self)

    report: dict | None = None  # what `start` sends; the ready report by default

    def start(self) -> None:
        self.started_in = os.getpid()
        self.handlers_in_at_start = callable(signal.getsignal(signal.SIGUSR1)) and callable(
            signal.getsignal(signal.SIGQUIT)
        )
        self.args[2].send(self.report or {"status": "ready", "pid": self.pid})

    def is_alive(self) -> bool:
        return self.exitcode is None

    def terminate(self) -> None:
        self.events.append("terminate")
        self.exitcode = -signal.SIGTERM

    def kill(self) -> None:
        self.events.append("kill")
        self.exitcode = -signal.SIGKILL

    def join(self, timeout=None) -> None:
        self.events.append("join")


class _FakePipeEnd:
    def __init__(self, queue: list):
        self._queue = queue
        self.closed = False

    def close(self) -> None:
        self.closed = True

    def send(self, report) -> None:
        self._queue.append(report)

    def poll(self, timeout=None) -> bool:
        return bool(self._queue)

    def recv(self):
        return self._queue.pop(0)


def _fake_pipe(duplex=True):
    queue: list = []
    return _FakePipeEnd(queue), _FakePipeEnd(queue)


def _stub_rank_launch(monkeypatch, *, ranks: int = 2, missing: str | None = None) -> dict:
    """Stub what `launch_scheduler_ranks` takes from TokenSpeed's launcher
    module (the preparation calls recorded in order) and the `multiprocessing`
    pieces it spawns the ranks with. `missing` drops one entry point, as an
    engine without it would."""
    recorded: dict = {"calls": []}
    port_args = SimpleNamespace(name="port-args")

    def run_event_loop(*_args, **_kwargs):
        raise AssertionError("the rank entry point runs in the ranks, not here")

    def run_data_parallel_controller_process(*_args, **_kwargs):
        raise AssertionError("the controller entry point runs in the controller, not here")

    class PortArgs:
        @staticmethod
        def init_new(server_args):
            recorded["calls"].append("port_args")
            return port_args

    class MemorySaver:
        @classmethod
        def create(cls, enable):
            recorded["memory_saver_enable"] = enable
            return cls()

        @contextlib.contextmanager
        def configure_subprocess(self):
            recorded["calls"].append("memory_saver")
            yield

    def prepare_model_and_tokenizer(model, tokenizer):
        recorded["calls"].append("prepare_model_and_tokenizer")
        return f"/local/{model}", f"/local/{tokenizer or model}"

    attrs = dict(
        _set_envs_and_config=lambda args: recorded["calls"].append("set_envs_and_config"),
        _scheduler_ranks=lambda args: range(ranks),
        run_event_loop=run_event_loop,
        run_data_parallel_controller_process=run_data_parallel_controller_process,
        prepare_model_and_tokenizer=prepare_model_and_tokenizer,
        configure_logger=lambda args: recorded["calls"].append("configure_logger"),
        PortArgs=PortArgs,
        TorchMemorySaverAdapter=MemorySaver,
    )
    if missing:
        del attrs[missing]
    _install(monkeypatch, "tokenspeed.runtime.entrypoints.engine", **attrs)
    monkeypatch.setattr(_FakeRankProcess, "started", [])
    monkeypatch.setattr(rust.multiprocessing, "Process", _FakeRankProcess)
    monkeypatch.setattr(rust.multiprocessing, "Pipe", _fake_pipe)
    # TokenSpeed's own setup sets the spawn method; the launcher's guard
    # reads it and must not switch this test process over.
    recorded["start_method_calls"] = []
    monkeypatch.setattr(
        rust_lifecycle.multiprocessing, "get_start_method", lambda allow_none=False: "spawn"
    )
    monkeypatch.setattr(
        rust_lifecycle.multiprocessing,
        "set_start_method",
        lambda method, force=False: recorded["start_method_calls"].append((method, force)),
    )
    recorded["attrs"] = attrs
    recorded["port_args"] = port_args
    return recorded


def _wait_until(condition, what: str, deadline_secs: float = 5.0) -> None:
    deadline = time.monotonic() + deadline_secs
    while not condition():
        assert time.monotonic() < deadline, what
        time.sleep(0.01)


def test_launch_scheduler_ranks_starts_them_from_this_process_as_the_headless_launcher_does(
    monkeypatch, caplog, restore_fault_signals
):
    """smg-lab#181: one process per rank of this node, on TokenSpeed's own
    scheduler entry point with per-rank server args, started from the
    servicer process the way `launch_scheduler_headless` starts them; no
    `multiprocessing` child runs a launcher. The ready reports are drained on
    a thread, and a rank's failure signal ends the group rather than this
    process."""
    recorded = _stub_rank_launch(monkeypatch, ranks=2)
    monkeypatch.setattr(
        rust.multiprocessing,
        "get_context",
        lambda method: pytest.fail(f"a {method} launcher child was started"),
    )
    headless = rust.headless_server_args(FakeServerArgs(), handshake_port=24321)

    with caplog.at_level("INFO"):
        engine = rust.launch_scheduler_ranks(headless)

    assert isinstance(engine, rust_lifecycle.EngineProcessGroup)
    assert recorded["calls"] == [
        "configure_logger",
        "set_envs_and_config",
        "port_args",
        "prepare_model_and_tokenizer",
        "memory_saver",
        "memory_saver",
    ]
    assert recorded["memory_saver_enable"] is False
    assert headless.model == "/local/org/m" and headless.tokenizer == "/local/org/m"
    processes = _FakeRankProcess.started
    assert len(processes) == 2
    for rank, process in enumerate(processes):
        assert process.started_in == os.getpid()
        assert process.target is recorded["attrs"]["run_event_loop"]
        assert process.name == f"smg-scheduler-rank-{rank}"
        rank_args, port_args, _writer = process.args
        assert port_args is recorded["port_args"]
        assert rank_args is not headless and rank_args.mapping is not headless.mapping
        assert rank_args.mapping.rank == rank
        assert rank_args.data_parallel_rpc_port == 24321 and rank_args.zmq_msgpack is True
    assert headless.mapping.rank is None  # the servicer's own copy is untouched
    assert engine.pids == [2000, 2001] and engine.pid == 2000 and engine.poll() is None
    _wait_until(
        lambda: "TokenSpeed scheduler: 2 scheduler process(es) ready" in caplog.text,
        "the ready log line never appeared",
    )
    assert "Started 2 scheduler rank process(es) from the servicer process: pids 2000, 2001" in (
        caplog.text
    )
    assert callable(signal.getsignal(signal.SIGUSR1)) and callable(signal.getsignal(signal.SIGQUIT))


def test_launch_scheduler_ranks_starts_the_data_parallel_controller_for_attention_dp(
    monkeypatch, restore_fault_signals
):
    """With attention data parallelism the controller is the one process this
    node starts (it spawns the ranks), as in TokenSpeed's own launcher."""
    recorded = _stub_rank_launch(monkeypatch)
    args = FakeServerArgs(
        mapping=SimpleNamespace(attn=SimpleNamespace(dp_size=2, has_dp=True), rank=None)
    )
    headless = rust.headless_server_args(args, handshake_port=24321)

    engine = rust.launch_scheduler_ranks(headless)

    (controller,) = _FakeRankProcess.started
    assert controller.started_in == os.getpid()
    assert controller.target is recorded["attrs"]["run_data_parallel_controller_process"]
    assert controller.name == "smg-scheduler-dp-controller"
    assert controller.args[0] is headless and controller.args[1] is recorded["port_args"]
    assert "memory_saver" not in recorded["calls"]
    assert engine.pids == [2000] and engine.poll() is None


def test_a_failing_rank_ends_the_group_through_its_signal_not_the_servicer(
    monkeypatch, caplog, restore_fault_signals
):
    """A rank that hits an exception sends its parent SIGUSR1 and exits on its
    own terms (TokenSpeed's own launch-phase SIGQUIT handler would kill the
    whole tree): the group reports exit code 1 and the other rank is
    terminated; the servicer process goes on to drain through its lifecycle."""
    _stub_rank_launch(monkeypatch)
    engine = rust.launch_scheduler_ranks(
        rust.headless_server_args(FakeServerArgs(), handshake_port=1)
    )

    with caplog.at_level("ERROR"):
        signal.raise_signal(signal.SIGUSR1)

    assert engine.poll() == 1
    assert [process.events[0] for process in _FakeRankProcess.started] == ["terminate", "terminate"]
    assert "TokenSpeed scheduler: a scheduler process signalled a failure (signal 10)" in (
        caplog.text
    )


def test_launch_scheduler_ranks_keeps_the_spawned_launcher_without_the_engines_entry_points(
    monkeypatch, caplog
):
    """An engine whose launcher module lacks one of the internals this launch
    uses still runs: through its own headless launcher in a spawned process,
    with the reason logged."""
    recorded = _stub_rank_launch(monkeypatch, missing="_scheduler_ranks")
    fallback = object()
    launched: list = []

    def fake_launch(headless):
        launched.append(headless)
        return fallback

    monkeypatch.setattr(rust, "launch_headless_scheduler", fake_launch)
    headless = rust.headless_server_args(FakeServerArgs(), handshake_port=1)
    with caplog.at_level("WARNING", logger=rust.logger.name):
        assert rust.launch_scheduler_ranks(headless) is fallback
    assert launched == [headless] and _FakeRankProcess.started == [] and recorded["calls"] == []
    assert "left to TokenSpeed's headless launcher in a spawned process" in caplog.text
    assert "_scheduler_ranks" in caplog.text


def test_launch_scheduler_ranks_off_the_main_thread_keeps_the_spawned_launcher(monkeypatch):
    """Where no signal handler can be installed, the spawned headless launcher
    (with a supervisor of its own) still serves."""
    recorded = _stub_rank_launch(monkeypatch)
    fallback = object()
    launched: list = []
    monkeypatch.setattr(
        rust, "launch_headless_scheduler", lambda headless: launched.append(headless) or fallback
    )
    headless = rust.headless_server_args(FakeServerArgs(), handshake_port=1)
    result: dict = {}
    thread = threading.Thread(
        target=lambda: result.update(engine=rust.launch_scheduler_ranks(headless))
    )
    thread.start()
    thread.join(5)
    assert result["engine"] is fallback and launched == [headless]
    assert _FakeRankProcess.started == [] and recorded["calls"] == []


def test_serve_rust_starts_the_scheduler_ranks_from_this_process(
    monkeypatch, tmp_path, restore_fault_signals
):
    """smg-lab#181 end to end: `serve_rust` binds the server, then starts the
    ranks in this process on the headless server args aimed at the server's
    handshake port; the supervisor gets the ranks' handle."""
    recorded = _stub_rank_launch(monkeypatch)
    created: dict = {}

    class FakeServer:
        def __init__(self, **kwargs):
            created.update(kwargs)
            self.address = kwargs["bind_address"]

    _install(
        monkeypatch,
        "smg.servicer",
        TokenSpeedGrpcServer=FakeServer,
        init_servicer_tracing=lambda level=None: None,
    )
    monkeypatch.setattr(rust.multiprocessing, "get_context", lambda method: pytest.fail(method))
    monkeypatch.setattr(rust, "_model_config", lambda args: SimpleNamespace(hf_config=None))
    monkeypatch.setattr(
        rust, "model_facts", lambda args, model_config=None: {"model_path": args.model}
    )
    monkeypatch.setattr(
        rust, "server_facts", lambda args: {"server_args_json": "{}", "data_parallel_size": 1}
    )
    monkeypatch.setattr(rust, "tokenizer_dir_for", lambda args: str(tmp_path))
    monkeypatch.setattr(rust_lifecycle, "free_port", lambda: 24321)
    monkeypatch.setattr(rust_lifecycle, "default_socket_dir", lambda: str(tmp_path / "sockets"))
    monkeypatch.delenv("SMG_TOKENSPEED_SERVICER_HANDSHAKE_PORT", raising=False)
    supervised: dict = {}

    async def fake_supervise(server, engine, *, drain_secs, **_kwargs):
        supervised["server"], supervised["engine"] = server, engine
        return 0

    monkeypatch.setattr(rust_lifecycle, "supervise", fake_supervise)
    args = FakeServerArgs(port=50123)

    assert asyncio.run(rust.serve_rust(args)) == 0

    engine = supervised["engine"]
    assert isinstance(engine, rust_lifecycle.EngineProcessGroup) and engine.pids == [2000, 2001]
    assert created["handshake_address"] == "tcp://127.0.0.1:24321"
    for process in _FakeRankProcess.started:
        assert process.started_in == os.getpid()
        rank_args = process.args[0]
        assert rank_args.data_parallel_rpc_port == 24321 and rank_args.zmq_msgpack is True
        assert rank_args.skip_tokenizer_init is True
    assert args.zmq_msgpack is False and recorded["calls"][0] == "configure_logger"
    assert isinstance(supervised["server"], FakeServer)


def test_serve_rust_wires_the_server_the_scheduler_and_the_supervisor(monkeypatch, tmp_path):
    """The Rust server binds the launcher's host:port, the scheduler is
    launched headless against the handshake port, both are supervised."""
    recorded: dict = {}

    class FakeServer:
        def __init__(self, bind_address, ipc_base_url, handshake_address, **kwargs):
            recorded["server"] = dict(
                bind_address=bind_address,
                ipc_base_url=ipc_base_url,
                handshake_address=handshake_address,
                **kwargs,
            )
            self.address = bind_address

    _install_fake_servicer(monkeypatch, FakeServer)
    monkeypatch.setattr(rust, "_model_config", lambda args: SimpleNamespace(hf_config=None))
    monkeypatch.setattr(
        rust, "model_facts", lambda args, model_config=None: {"model_path": args.model}
    )
    monkeypatch.setattr(
        rust,
        "server_facts",
        lambda args: {"server_args_json": "{}", "data_parallel_size": 2},
    )
    monkeypatch.setattr(rust, "tokenizer_dir_for", lambda args: str(tmp_path))
    monkeypatch.setattr(rust_lifecycle, "free_port", lambda: 24321)
    monkeypatch.setattr(rust_lifecycle, "default_socket_dir", lambda: str(tmp_path / "sockets"))

    def fake_launch(headless):
        recorded["headless"] = headless
        return "engine-handle"

    async def fake_supervise(server, engine, *, drain_secs, **_kwargs):
        recorded["supervised"] = (server, engine, drain_secs)
        return 0

    monkeypatch.setattr(rust, "launch_scheduler_ranks", fake_launch)
    monkeypatch.setattr(rust_lifecycle, "supervise", fake_supervise)
    monkeypatch.delenv("SMG_TOKENSPEED_SERVICER_HANDSHAKE_PORT", raising=False)
    monkeypatch.setenv("SMG_TOKENSPEED_SERVICER_DRAIN_SECS", "1.5")
    monkeypatch.setenv("SMG_TOKENSPEED_SERVICER_STARTUP_TIMEOUT_SECS", "42")
    monkeypatch.setenv("SMG_TOKENSPEED_SERVICER_STARTUP_CEILING_SECS", "0")
    monkeypatch.setenv("SMG_TOKENSPEED_SERVICER_SILENCE_TIMEOUT_SECS", "900")
    monkeypatch.setenv("SMG_TOKENSPEED_SERVICER_PREFILL_TOKENS_PER_SEC", "1000")

    assert asyncio.run(rust.serve_rust(FakeServerArgs(port=50123))) == 0

    server = recorded["server"]
    assert server["bind_address"] == "127.0.0.1:50123"
    assert server["handshake_address"] == "tcp://127.0.0.1:24321"
    assert server["ipc_base_url"].startswith(f"ipc://{tmp_path / 'sockets'}/tokenspeed-servicer-")
    assert server["engine_count"] == 2
    assert server["tokenizer_dir"] == str(tmp_path)
    assert server["engine_startup_timeout_secs"] == 42.0
    assert server["engine_startup_ceiling_secs"] == 0.0  # 0 lifts the ceiling
    assert server["engine_silence_timeout_secs"] == 900.0
    assert server["engine_prefill_tokens_per_sec"] == 1000
    assert server["model_path"] == "org/m"
    headless = recorded["headless"]
    assert headless.zmq_msgpack is True and headless.skip_tokenizer_init is True
    # No --kv-events-config was given: the headless scheduler publishes, at a
    # free port of this host.
    published = json.loads(headless.kv_events_config)
    expected = {"enable_kv_cache_events": True, "publisher": "zmq"}
    assert published == {**expected, "endpoint": published["endpoint"]}
    assert published["endpoint"].startswith("tcp://*:") and published["endpoint"] != "tcp://*:5557"
    assert headless.data_parallel_rpc_port == 24321
    supervised_server, engine, drain_secs = recorded["supervised"]
    assert isinstance(supervised_server, FakeServer)
    assert engine == "engine-handle"
    assert drain_secs == 1.5


def _run_serve_rust(monkeypatch, tmp_path) -> dict:
    """``serve_rust`` against a recording fake server with the launcher's own
    fact producers (over a fake model config and the stub servicer module), the
    scheduler launch and the supervisor stubbed; the keywords the server was
    constructed with."""
    created: dict = {}
    recorded: dict = {}

    class FakeServer:
        def __init__(self, **kwargs):
            created.update(kwargs)
            self.address = kwargs["bind_address"]
            self.serving: list[bool] = []
            self.ready: list[bool] = []
            recorded["server"] = self

        def set_serving(self, serving: bool) -> None:
            self.serving.append(serving)

        def set_ready(self, ready: bool) -> None:
            self.ready.append(ready)

    _install_fake_servicer(monkeypatch, FakeServer)
    _stub_servicer_module(monkeypatch)
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            eos_token_id=2, model_type="qwen3", architectures=["Qwen3ForCausalLM"]
        ),
        context_len=8192,
        vocab_size=151936,
        dtype="torch.bfloat16",
        is_multimodal=False,
    )
    monkeypatch.setattr(rust, "_model_config", lambda args: model_config)
    monkeypatch.setattr(rust, "tokenizer_dir_for", lambda args: str(tmp_path))
    monkeypatch.setattr(rust_lifecycle, "free_port", lambda: 24321)
    monkeypatch.setattr(rust_lifecycle, "default_socket_dir", lambda: str(tmp_path / "sockets"))
    monkeypatch.setattr(rust, "launch_scheduler_ranks", lambda headless: "engine-handle")

    async def fake_supervise(server, engine, *, drain_secs, **kwargs):
        recorded["supervise_kwargs"] = kwargs
        return 0

    monkeypatch.setattr(rust_lifecycle, "supervise", fake_supervise)
    monkeypatch.delenv("SMG_TOKENSPEED_SERVICER_HANDSHAKE_PORT", raising=False)
    assert asyncio.run(rust.serve_rust(FakeServerArgs(port=50123))) == 0
    _run_serve_rust.recorded = recorded
    return created


class _FakeBootstrapServer:
    """The engine's KV bootstrap server as the launcher drives it: built on a
    port (which starts it), closed once."""

    built: list[_FakeBootstrapServer] = []

    def __init__(self, port: int) -> None:
        self.port = port
        self.closed = 0
        _FakeBootstrapServer.built.append(self)

    def close(self) -> None:
        self.closed += 1


def _install_fake_pd_utils(monkeypatch) -> dict:
    """``tokenspeed.runtime.pd.utils`` as the launcher reads it: the transfer
    backend and class-type enums and the registry that hands out the
    bootstrap server class; records what was asked for."""
    import enum

    asked: dict = {}

    class TransferBackend(enum.Enum):
        MOONCAKE = "mooncake"

    class KVClassType(enum.Enum):
        MANAGER_PREFILL = "manager_prefill"
        BOOTSTRAP_SERVER = "bootstrap_server"

    def get_kv_class(backend, class_type):
        asked["backend"], asked["class_type"] = backend, class_type
        assert class_type is KVClassType.BOOTSTRAP_SERVER
        return _FakeBootstrapServer

    _install(
        monkeypatch,
        "tokenspeed.runtime.pd.utils",
        TransferBackend=TransferBackend,
        KVClassType=KVClassType,
        get_kv_class=get_kv_class,
    )
    _FakeBootstrapServer.built = []
    return asked


def _serve_rust_role(monkeypatch, tmp_path, caplog, mode: str) -> dict:
    """``serve_rust`` for a launch in disaggregation ``mode`` against the
    recording fakes: when the ranks were launched relative to the bootstrap
    server, and the supervisor's end."""
    recorded: dict = {"launched_with_servers": None}

    class FakeServer:
        def __init__(self, **kwargs):
            self.address = kwargs["bind_address"]

        def set_ready(self, ready: bool) -> None:
            pass

    _install_fake_servicer(monkeypatch, FakeServer)
    monkeypatch.setattr(rust, "_model_config", lambda args: SimpleNamespace(hf_config=None))
    monkeypatch.setattr(
        rust, "model_facts", lambda args, model_config=None: {"model_path": args.model}
    )
    monkeypatch.setattr(
        rust, "server_facts", lambda args: {"server_args_json": "{}", "data_parallel_size": 1}
    )
    monkeypatch.setattr(rust, "tokenizer_dir_for", lambda args: str(tmp_path))
    monkeypatch.setattr(rust_lifecycle, "free_port", lambda: 24321)
    monkeypatch.setattr(rust_lifecycle, "default_socket_dir", lambda: str(tmp_path / "sockets"))
    monkeypatch.delenv("SMG_TOKENSPEED_SERVICER_HANDSHAKE_PORT", raising=False)

    def fake_launch(headless):
        # The ranks register with the bootstrap server when they start: it
        # must already be up.
        recorded["launched_with_servers"] = [
            (server.port, server.closed) for server in _FakeBootstrapServer.built
        ]
        return "engine-handle"

    async def fake_supervise(server, engine, *, drain_secs, **kwargs):
        recorded["supervised"] = engine
        return 0

    monkeypatch.setattr(rust, "launch_scheduler_ranks", fake_launch)
    monkeypatch.setattr(rust_lifecycle, "supervise", fake_supervise)
    with caplog.at_level("INFO", logger=rust.logger.name):
        recorded["exit"] = asyncio.run(
            rust.serve_rust(FakeServerArgs(port=50123, disaggregation_mode=mode))
        )
    return recorded


def test_a_prefill_launch_starts_the_engines_kv_bootstrap_server_before_the_ranks(
    monkeypatch, tmp_path, caplog
):
    """The prefill side hosts the rendezvous the decode side discovers its KV
    at. The engine's frontend starts it on the Python path; here the
    launcher starts the same class, through the engine's registry for the
    transfer backend, on the configured port, before the ranks (each
    registers with it once), logs it, and closes it when it ends."""
    asked = _install_fake_pd_utils(monkeypatch)
    recorded = _serve_rust_role(monkeypatch, tmp_path, caplog, "prefill")
    assert recorded["exit"] == 0 and recorded["supervised"] == "engine-handle"
    assert recorded["launched_with_servers"] == [(8998, 0)]
    assert asked["backend"].value == "mooncake"
    assert asked["class_type"].value == "bootstrap_server"
    (server,) = _FakeBootstrapServer.built
    assert server.closed == 1
    assert (
        "Bootstrap server started for disaggregation mode on port 8998 (mooncake transfer)"
        in caplog.text
    )


@pytest.mark.parametrize("mode", ["null", "decode"])
def test_a_decode_or_aggregated_launch_starts_no_bootstrap_server(
    monkeypatch, tmp_path, caplog, mode
):
    """Only the prefill side hosts the rendezvous: the other roles start
    nothing and never reach for the engine's PD registry."""
    _install_fake_pd_utils(monkeypatch)
    recorded = _serve_rust_role(monkeypatch, tmp_path, caplog, mode)
    assert recorded["exit"] == 0
    assert recorded["launched_with_servers"] == []
    assert _FakeBootstrapServer.built == []
    assert "Bootstrap server started" not in caplog.text


def test_an_encode_launch_runs_on_the_rust_path_with_its_rendezvous_fact(launcher, monkeypatch):
    """The EPD encode role runs on the Rust path: the launch is not refused,
    the facts carry the worker's own rendezvous address (the host every
    encode request names, read the engine's way, next to the bootstrap port
    the server args already carry) for the encode role only, and the
    KV-events default stays off for it (a vision tower has no KV cache to
    publish from), as the Python server leaves it."""
    _stub_servicer_module(monkeypatch)
    monkeypatch.setattr(rust, "encode_bootstrap_host", lambda args: "10.0.0.9")
    encode = FakeServerArgs(disaggregation_mode="encode", disaggregation_bootstrap_port=8998)
    facts = rust.server_facts(encode)
    args = json.loads(facts["server_args_json"])
    assert args["encode_bootstrap_host"] == "10.0.0.9"
    assert args["disaggregation_bootstrap_port"] == 8998
    for mode in ("null", "prefill", "decode"):
        other = json.loads(
            rust.server_facts(FakeServerArgs(disaggregation_mode=mode))["server_args_json"]
        )
        assert "encode_bootstrap_host" not in other
    assert rust.TokenSpeedLauncher().kv_events_default(encode) is None

    module, calls = launcher
    served = {}

    async def fake_serve_rust(server_args):
        served["mode"] = server_args.disaggregation_mode
        return 0

    monkeypatch.setattr(module, "serve_rust", fake_serve_rust)
    with pytest.raises(SystemExit) as raised:
        module.main(
            ["--servicer-impl", "rust", "--model", "org/m", "--disaggregation-mode", "encode"]
        )
    assert raised.value.code == 0 and served == {"mode": "encode"}


def test_the_rendezvous_host_is_the_engines_reading_or_empty(monkeypatch):
    """The host is the engine's own reading (``get_local_ip_by_remote``); a
    helper that yields nothing, or an engine that cannot be imported, gives
    an empty host, which the Rust servicer refuses by name (never the
    launch's ``--host``, an unspecified address no peer can reach)."""
    import sys
    import types

    network = types.ModuleType("tokenspeed.runtime.utils.network")
    network.get_local_ip_by_remote = lambda: "192.0.2.7"
    monkeypatch.setitem(sys.modules, "tokenspeed.runtime.utils.network", network)
    assert rust.encode_bootstrap_host(FakeServerArgs(host="0.0.0.0")) == "192.0.2.7"
    network.get_local_ip_by_remote = lambda: None
    assert rust.encode_bootstrap_host(FakeServerArgs(host="10.1.2.3")) == ""
    monkeypatch.delitem(sys.modules, "tokenspeed.runtime.utils.network")
    monkeypatch.setitem(sys.modules, "tokenspeed.runtime.utils", None)
    assert rust.encode_bootstrap_host(FakeServerArgs(host="10.1.2.3")) == ""


def test_serve_rust_holds_readiness_for_the_planned_warm_up(monkeypatch, tmp_path):
    """With a warm-up planned (the default knob, a generation model whose
    facts carry a vocabulary and a context), the server's readiness is held
    before the scheduler starts (the health-only setter: admission stays open
    for the pass's own requests, never the drain switch) and the supervisor is
    handed the pass, which flips SERVING when it ran; with the knob off,
    neither happens."""
    _run_serve_rust(monkeypatch, tmp_path)
    recorded = _run_serve_rust.recorded
    assert recorded["server"].ready == [False] and recorded["server"].serving == []
    assert callable(recorded["supervise_kwargs"]["warmup"])

    monkeypatch.setenv(rust.WARMUP_ENV, "off")
    _run_serve_rust(monkeypatch, tmp_path)
    recorded = _run_serve_rust.recorded
    assert recorded["server"].ready == [] and recorded["server"].serving == []
    assert recorded["supervise_kwargs"] == {}


def test_serve_rust_refuses_a_warm_up_knob_it_cannot_read_before_the_server_binds(
    monkeypatch, tmp_path
):
    created: dict = {}

    class FakeServer:
        def __init__(self, **kwargs):
            created.update(kwargs)

    _install_fake_servicer(monkeypatch, FakeServer)
    _stub_servicer_module(monkeypatch)
    monkeypatch.setattr(rust_lifecycle, "default_socket_dir", lambda: str(tmp_path / "sockets"))
    monkeypatch.setenv(rust.WARMUP_ENV, "warm")
    with pytest.raises(
        rust_lifecycle.ServicerConfigError, match="SMG_TOKENSPEED_SERVICER_WARMUP='warm'"
    ):
        asyncio.run(rust.serve_rust(FakeServerArgs(port=50123)))
    assert created == {}


def test_serve_rust_defaults_both_startup_bounds(monkeypatch, tmp_path):
    """Without the environment variables the server gets the lifecycle's
    defaults for the silence bound and the ceiling."""
    monkeypatch.delenv(rust.STARTUP_TIMEOUT_SECS_ENV, raising=False)
    monkeypatch.delenv(rust.STARTUP_CEILING_SECS_ENV, raising=False)
    monkeypatch.delenv(rust.SILENCE_TIMEOUT_SECS_ENV, raising=False)
    monkeypatch.delenv(rust.PREFILL_TOKENS_PER_SEC_ENV, raising=False)
    created = _run_serve_rust(monkeypatch, tmp_path)
    assert created["engine_startup_timeout_secs"] == rust.DEFAULT_STARTUP_TIMEOUT_SECS
    assert created["engine_startup_ceiling_secs"] == rust.DEFAULT_STARTUP_CEILING_SECS
    assert created["engine_silence_timeout_secs"] == rust.DEFAULT_SILENCE_TIMEOUT_SECS
    assert created["engine_prefill_tokens_per_sec"] == rust.DEFAULT_PREFILL_TOKENS_PER_SEC


def test_serve_rust_constructs_the_server_with_keywords_the_binding_takes(monkeypatch, tmp_path):
    """The keyword set `serve_rust` hands the Rust server is bound against the
    compiled binding's constructor: a fact or a knob the launcher gains without
    a parameter on the binding fails here, not at the scheduler's start.
    Skipped where the `smg` wheel is not installed."""
    servicer = pytest.importorskip("smg.servicer")
    signature = inspect.signature(servicer.TokenSpeedGrpcServer)
    created = _run_serve_rust(monkeypatch, tmp_path)
    unexpected = sorted(set(created) - set(signature.parameters))
    assert not unexpected, f"keywords the binding's constructor does not take: {unexpected}"
    # The required parameters are all given too; pyo3 refuses the call otherwise.
    signature.bind(**created)


def test_serve_rust_fakes_leave_the_installed_package_untouched(tmp_path):
    """The fakes `_run_serve_rust` installs go under a fake ``smg`` package, so
    the installed one keeps its own ``servicer`` module once the test is over.
    Skipped where the `smg` wheel is not installed."""
    servicer = pytest.importorskip("smg.servicer")
    package = sys.modules["smg"]
    with pytest.MonkeyPatch.context() as monkeypatch:
        _run_serve_rust(monkeypatch, tmp_path)
    assert sys.modules["smg"] is package
    assert package.servicer is servicer


def test_install_undoes_a_fake_under_a_real_package():
    """A fake installed under a real package is undone with the test, the
    attribute on the package like the ``sys.modules`` entry, so a later
    ``from smg_grpc_servicer.tokenspeed import servicer`` gets the real module.
    ``_stub_servicer_module`` and the launcher test's fake ``server`` take this path."""
    from smg_grpc_servicer import tokenspeed as package

    name = "smg_grpc_servicer.tokenspeed.servicer"
    before = (vars(package).get("servicer"), sys.modules.get(name))
    with pytest.MonkeyPatch.context() as monkeypatch:
        fake = _install(monkeypatch, name)
        assert package.servicer is fake and sys.modules[name] is fake
    assert (vars(package).get("servicer"), sys.modules.get(name)) == before


def test_the_fault_handlers_are_in_before_the_first_rank_starts_and_the_pipes_close(
    monkeypatch, restore_fault_signals
):
    """A rank that fails during its start signals its parent at once, so the
    SIGUSR1/SIGQUIT handlers are installed before any rank starts; the
    parent's end of each ready pipe is closed after the start (a rank that
    dies before reporting shows as EOF, and no descriptor pair outlives the
    launch); the start method set by the engine's setup is left alone."""
    recorded = _stub_rank_launch(monkeypatch, ranks=2)

    rust.launch_scheduler_ranks(rust.headless_server_args(FakeServerArgs(), handshake_port=1))

    processes = _FakeRankProcess.started
    assert len(processes) == 2
    assert all(process.handlers_in_at_start for process in processes)
    assert all(process.args[2].closed for process in processes)
    assert recorded["start_method_calls"] == []


def test_a_rank_that_reports_a_failed_start_ends_the_group(
    monkeypatch, caplog, restore_fault_signals
):
    """A report that is not `ready` (a rank that failed to load and stays up)
    is a fault of the group: its poll turns 1 and the other rank is
    terminated, so the servicer drains and exits instead of supervising a
    scheduler that never came up."""
    _stub_rank_launch(monkeypatch, ranks=2)
    monkeypatch.setattr(_FakeRankProcess, "report", {"status": "error", "message": "no model"})

    with caplog.at_level("ERROR"):
        engine = rust.launch_scheduler_ranks(
            rust.headless_server_args(FakeServerArgs(), handshake_port=1)
        )
        _wait_until(lambda: engine.poll() == 1, "the failed start never ended the group")

    assert [process.events[0] for process in _FakeRankProcess.started] == ["terminate", "terminate"]
    assert "TokenSpeed scheduler: the scheduler processes did not start" in caplog.text


def test_the_spawn_start_method_is_forced_when_the_engines_setup_left_it_unset(
    monkeypatch, restore_fault_signals
):
    recorded = _stub_rank_launch(monkeypatch, ranks=1)
    monkeypatch.setattr(
        rust_lifecycle.multiprocessing, "get_start_method", lambda allow_none=False: None
    )

    rust.launch_scheduler_ranks(rust.headless_server_args(FakeServerArgs(), handshake_port=1))

    assert recorded["start_method_calls"] == [("spawn", True)]


# ── the knobs are validated before the bind (smg-lab #292) ─────────────────────


@pytest.mark.parametrize(
    "name, value, text",
    [
        ("SMG_TOKENSPEED_SERVICER_DRAIN_SECS", "abc", "is not a number"),
        ("SMG_TOKENSPEED_SERVICER_DRAIN_SECS", "-5", "must be at least 0"),
        ("SMG_TOKENSPEED_SERVICER_STARTUP_TIMEOUT_SECS", "abc", "is not a number"),
        ("SMG_TOKENSPEED_SERVICER_STARTUP_TIMEOUT_SECS", "-1", "must be greater than 0"),
        ("SMG_TOKENSPEED_SERVICER_HANDSHAKE_PORT", "70000", "must be between 1 and 65535"),
        ("SMG_TOKENSPEED_SERVICER_HANDSHAKE_PORT", "abc", "is not a port number"),
    ],
)
def test_serve_rust_refuses_a_bad_knob_before_the_server_binds(
    monkeypatch, tmp_path, name, value, text
):
    """smg-lab #292: every knob is read and validated first; a value the
    servicer cannot use is a named error before any engine work, before the
    server is constructed (and so before the port is bound), and before the
    scheduler ranks start. The drain was read only once the engine ran."""
    created: dict = {}

    class FakeServer:
        def __init__(self, **kwargs):
            created.update(kwargs)

    _install_fake_servicer(monkeypatch, FakeServer)
    _stub_servicer_module(monkeypatch)

    def no_engine_work(*_args, **_kwargs):
        raise AssertionError("no engine work before the knobs are validated")

    monkeypatch.setattr(rust, "model_facts", no_engine_work)
    monkeypatch.setattr(rust, "server_facts", no_engine_work)
    monkeypatch.setattr(rust, "launch_scheduler_ranks", no_engine_work)
    monkeypatch.setattr(rust_lifecycle, "default_socket_dir", lambda: str(tmp_path / "sockets"))
    monkeypatch.setenv(name, value)

    with pytest.raises(rust_lifecycle.ServicerConfigError, match=f"{name}='{value}' {text}"):
        asyncio.run(rust.serve_rust(FakeServerArgs(port=50123)))
    assert created == {}


def test_serve_rust_refuses_a_socket_dir_that_is_a_file_before_the_server_binds(
    monkeypatch, tmp_path
):
    created: dict = {}

    class FakeServer:
        def __init__(self, **kwargs):
            created.update(kwargs)

    _install_fake_servicer(monkeypatch, FakeServer)
    _stub_servicer_module(monkeypatch)
    a_file = tmp_path / "a-file"
    a_file.write_text("not a directory")
    monkeypatch.setattr(rust_lifecycle, "default_socket_dir", lambda: str(a_file))

    with pytest.raises(rust_lifecycle.ServicerConfigError, match="SMG_ZMQ_SOCKET_DIR=.*not a dir"):
        asyncio.run(rust.serve_rust(FakeServerArgs(port=50123)))
    assert created == {}


def test_the_launcher_answers_a_configuration_error_with_exit_code_2(launcher, monkeypatch, caplog):
    """smg-lab #292: a knob the servicer cannot use ends the launcher with one
    ERROR line naming it and exit code 2, not a traceback."""
    module, calls = launcher
    caplog.set_level(logging.ERROR)
    monkeypatch.setenv("SMG_TOKENSPEED_SERVICER_IMPL", "bogus")
    with pytest.raises(SystemExit) as raised:
        module.main(["--model", "org/m"])
    assert raised.value.code == 2 and calls["served"] is None
    assert "SMG_TOKENSPEED_SERVICER_IMPL must be one of" in caplog.text

    async def failing_serve_rust(server_args):
        raise rust_lifecycle.ServicerConfigError(
            "SMG_TOKENSPEED_SERVICER_DRAIN_SECS='abc' is not a number"
        )

    monkeypatch.setattr(module, "serve_rust", failing_serve_rust)
    with pytest.raises(SystemExit) as raised:
        module.main(["--servicer-impl", "rust", "--model", "org/m"])
    assert raised.value.code == 2
    assert "SMG_TOKENSPEED_SERVICER_DRAIN_SECS='abc' is not a number" in caplog.text


def test_a_launch_that_fails_stops_the_already_bound_server(monkeypatch, tmp_path):
    """The server owns the port and a runtime thread: a scheduler launch that
    raises must not leave them to the process exit (what the vLLM launcher
    does), and nothing is supervised."""
    events: list[str] = []

    class FakeServer:
        def __init__(self, **kwargs):
            self.address = kwargs["bind_address"]

        def set_serving(self, serving: bool) -> None:
            events.append(f"serving:{serving}")

        def stop(self, timeout: float) -> None:
            events.append(f"stop:{timeout}")

    _install_fake_servicer(monkeypatch, FakeServer)
    monkeypatch.setattr(rust, "_model_config", lambda args: SimpleNamespace(hf_config=None))
    monkeypatch.setattr(
        rust, "model_facts", lambda args, model_config=None: {"model_path": args.model}
    )
    monkeypatch.setattr(
        rust, "server_facts", lambda args: {"server_args_json": "{}", "data_parallel_size": 1}
    )
    monkeypatch.setattr(rust, "tokenizer_dir_for", lambda args: str(tmp_path))
    monkeypatch.setattr(rust_lifecycle, "free_port", lambda: 24321)
    monkeypatch.setattr(rust_lifecycle, "default_socket_dir", lambda: str(tmp_path / "sockets"))
    monkeypatch.delenv("SMG_TOKENSPEED_SERVICER_HANDSHAKE_PORT", raising=False)

    def failing_launch(headless):
        raise RuntimeError("scheduler would not start")

    async def never_supervise(*_args, **_kwargs):
        pytest.fail("a failed launch was supervised")

    monkeypatch.setattr(rust, "launch_scheduler_ranks", failing_launch)
    monkeypatch.setattr(rust_lifecycle, "supervise", never_supervise)
    with pytest.raises(RuntimeError, match="scheduler would not start"):
        asyncio.run(rust.serve_rust(FakeServerArgs(port=50123)))
    assert events == ["serving:False", "stop:5.0"]


def test_model_facts_carry_the_engines_constrained_decoding_policy(monkeypatch):
    """The Rust servicer reports the engine's declared constrained-decoding
    policy as the Python servicer reads it: off the server args, else the
    model config, which the launcher's model facts hold; neither declaring
    one leaves the field empty, and the server facts never carry it."""
    _stub_servicer_module(monkeypatch)
    text = SimpleNamespace(
        hf_config=SimpleNamespace(eos_token_id=2, model_type="qwen3", architectures=["Q"]),
        context_len=4096,
        vocab_size=100,
        dtype="torch.float16",
        is_multimodal=False,
    )
    assert rust.model_facts(FakeServerArgs(), model_config=text)["constrained_decoding"] == ""
    declared = SimpleNamespace(constrained_decoding="ALWAYS", **vars(text))
    assert rust.model_facts(FakeServerArgs(), model_config=declared)["constrained_decoding"] == (
        "always"
    )
    args = SimpleNamespace(model="org/m", constrained_decoding="requested")
    assert rust.model_facts(args, model_config=declared)["constrained_decoding"] == "requested"
    assert "constrained_decoding" not in rust.server_facts(FakeServerArgs())


def test_mrope_layout_facts_are_read_as_the_engines_input_processor_reads_them():
    # A Qwen3.5 checkpoint: the merge size on the vision config, the
    # placeholder ids at the top level (the engine's compute_mrope_positions
    # reads hf_config.vision_config.spatial_merge_size and hf_config.image_token_id).
    qwen35 = SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=2),
        image_token_id=248056,
        video_token_id=248057,
    )
    assert rust.mrope_layout(qwen35) == {
        "spatial_merge_size": 2,
        "image_token_id": 248056,
        "video_token_id": 248057,
    }
    # A dict-shaped config reads the same; a checkpoint without a vision
    # config or the ids ships None for each (the Rust rule then refuses the
    # item instead of guessing).
    assert rust.mrope_layout({"vision_config": {"spatial_merge_size": 2}, "image_token_id": 5}) == {
        "spatial_merge_size": 2,
        "image_token_id": 5,
        "video_token_id": None,
    }
    assert rust.mrope_layout(SimpleNamespace(image_token_id="x")) == {
        "spatial_merge_size": None,
        "image_token_id": None,
        "video_token_id": None,
    }
    assert rust.mrope_layout(None) == {
        "spatial_merge_size": None,
        "image_token_id": None,
        "video_token_id": None,
    }


def test_mrope_positions_are_read_off_the_text_rope_config():
    # Qwen-VL style: the rope scaling carries an mrope_section.
    qwen_vl = SimpleNamespace(
        rope_scaling={"rope_type": "default", "mrope_section": [24, 20, 20]},
    )
    assert rust.mrope_positions(qwen_vl) is True
    # Qwen2-VL style: the rope type itself says mrope.
    assert rust.mrope_positions(SimpleNamespace(rope_scaling={"type": "mrope"})) is True
    # A nested text config with a plain rope: 1-D positions.
    plain = SimpleNamespace(
        text_config={"rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"}},
    )
    assert rust.mrope_positions(plain) is False
    # A plain rope on the text config does not hide an MRoPE section at the top level.
    both = SimpleNamespace(
        rope_scaling={"type": "mrope", "mrope_section": [16, 24, 24]},
        text_config={"rope_parameters": {"rope_type": "default", "rope_theta": 1e6}},
    )
    assert rust.mrope_positions(both) is True
    # Dict-shaped configs read the same way; no rope section at all says nothing.
    assert rust.mrope_positions({"text_config": {"rope_scaling": {"mrope_section": [1]}}}) is True
    assert rust.mrope_positions(SimpleNamespace(model_type="x")) is None
    assert rust.mrope_positions(None) is None
