"""Unit tests for the Rust request path of the SGLang servicer
(``smg_grpc_servicer.sglang.rust``).

Everything SGLang-shaped is lazy in the module, so these run without SGLang:
the facts get a fake model config, ``serve_rust`` gets a fake Rust server
class and a fake scheduler launch.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import types
from types import SimpleNamespace

import pytest
from smg_grpc_servicer.sglang import plugin, rust


def _install(monkeypatch, name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    parent, _, child = name.rpartition(".")
    if parent:
        parent_module = sys.modules.get(parent) or _install(monkeypatch, parent)
        setattr(parent_module, child, module)
    return module


@pytest.fixture
def no_parsed_flag(monkeypatch):
    """No ``--servicer-impl`` parsed in this process, whatever ran before."""
    monkeypatch.setattr(plugin._state, "parsed", None)


def _server_args(**overrides):
    args = dict(
        model_path="org/model",
        tokenizer_path=None,
        served_model_name="served",
        weight_version=None,
        preferred_sampling_params='{"top_k": 20}',
        max_running_requests=64,
        dp_size=1,
        host="127.0.0.1",
        port=30011,
        revision=None,
    )
    args.update(overrides)
    return SimpleNamespace(**args)


def _model_config(**overrides):
    config = dict(
        context_len=4096,
        vocab_size=1024,
        is_generation=True,
        is_multimodal=False,
        hf_eos_token_id={7, 2},
        hf_config=SimpleNamespace(
            model_type="qwen3",
            architectures=["Qwen3ForCausalLM"],
            pad_token_id=0,
            bos_token_id=1,
            id2label=None,
            num_labels=0,
        ),
        get_default_sampling_params=lambda: {"temperature": 0.7, "top_p": 0.8},
    )
    config.update(overrides)
    return SimpleNamespace(**config)


def test_package_exposes_the_flag_and_the_rust_entry_points():
    import smg_grpc_servicer.sglang as pkg

    assert pkg.SERVICER_IMPL_ENV == rust.SERVICER_IMPL_ENV == "SMG_SGLANG_SERVICER_IMPL"
    assert pkg.resolve_servicer_impl is rust.resolve_servicer_impl
    assert pkg.serve_rust is rust.serve_rust
    assert set(plugin.SERVICER_IMPL_CHOICES) == set(rust.IMPLS)


def test_resolve_servicer_impl_prefers_the_launcher_flag_then_the_env(no_parsed_flag):
    assert rust.resolve_servicer_impl(environ={}) == "python"
    assert rust.resolve_servicer_impl(environ={"SMG_SGLANG_SERVICER_IMPL": " Rust "}) == "rust"
    args = argparse.Namespace(servicer_impl="rust")
    assert rust.resolve_servicer_impl(args, environ={}) == "rust"
    # The flag wins over the environment, in both directions.
    assert (
        rust.resolve_servicer_impl(args, environ={"SMG_SGLANG_SERVICER_IMPL": "python"}) == "rust"
    )
    args = argparse.Namespace(servicer_impl="python")
    assert (
        rust.resolve_servicer_impl(args, environ={"SMG_SGLANG_SERVICER_IMPL": "rust"}) == "python"
    )
    # An unset launcher flag falls through to the env.
    args = argparse.Namespace(servicer_impl=None)
    assert rust.resolve_servicer_impl(args, environ={"SMG_SGLANG_SERVICER_IMPL": "rust"}) == "rust"
    with pytest.raises(ValueError, match="SMG_SGLANG_SERVICER_IMPL must be one of"):
        rust.resolve_servicer_impl(environ={"SMG_SGLANG_SERVICER_IMPL": "go"})
    with pytest.raises(ValueError, match="SMG_SGLANG_SERVICER_IMPL must be one of"):
        rust.resolve_servicer_impl(argparse.Namespace(servicer_impl="go"), environ={})


def test_servicer_impl_source_names_the_origin_and_writes_a_flag_back(no_parsed_flag):
    assert rust.servicer_impl_source(environ={}) == ("python", "default")
    assert rust.servicer_impl_source(environ={"SMG_SGLANG_SERVICER_IMPL": "rust"}) == (
        "rust",
        "env",
    )
    # `--servicer-impl python` with `rust` still exported: the flag decides,
    # and the environment the headless child inherits says the same.
    environ = {"SMG_SGLANG_SERVICER_IMPL": "rust"}
    args = argparse.Namespace(servicer_impl="python")
    assert rust.servicer_impl_source(args, environ=environ) == ("python", "flag")
    assert environ == {"SMG_SGLANG_SERVICER_IMPL": "python"}
    # Without a flag in hand nothing is written.
    environ = {}
    assert rust.servicer_impl_source(SimpleNamespace(), environ=environ) == ("python", "default")
    assert environ == {}
    # The flag the plugin parsed in this process counts as the launcher's:
    # SGLang's ServerArgs does not carry it.
    plugin._state.parsed = "rust"
    environ = {"SMG_SGLANG_SERVICER_IMPL": "python"}
    assert rust.servicer_impl_source(SimpleNamespace(), environ=environ) == ("rust", "flag")
    assert environ == {"SMG_SGLANG_SERVICER_IMPL": "rust"}


def test_plugin_adds_the_servicer_impl_flag_to_sglangs_parser(monkeypatch, no_parsed_flag):
    """The flag joins a parser once, is listed by --help, keeps argparse's
    validation, and a parsed value is carried as this process's launcher
    flag and into the environment (SGLang's ServerArgs would drop it)."""
    monkeypatch.setenv(rust.SERVICER_IMPL_ENV, "python")
    parser = argparse.ArgumentParser(prog="sglang serve")
    parser.add_argument("--grpc-mode", action="store_true")
    assert plugin.add_servicer_impl_argument(parser) is True
    assert plugin.add_servicer_impl_argument(parser) is False
    assert "--servicer-impl {python,rust}" in parser.format_help()
    assert "SMG_SGLANG_SERVICER_IMPL" in parser.format_help()
    assert parser.parse_args(["--grpc-mode"]).servicer_impl is None
    assert plugin.parsed_flag() is None and os.environ[rust.SERVICER_IMPL_ENV] == "python"
    args = parser.parse_args(["--grpc-mode", "--servicer-impl", "rust"])
    assert (args.grpc_mode, args.servicer_impl) == (True, "rust")
    assert plugin.parsed_flag() == "rust" and os.environ[rust.SERVICER_IMPL_ENV] == "rust"
    with pytest.raises(SystemExit):
        parser.parse_args(["--grpc-mode", "--servicer-impl", "go"])
    # The parsed flag decides ahead of the environment, with or without the namespace.
    assert rust.servicer_impl_source(args, environ={"SMG_SGLANG_SERVICER_IMPL": "python"}) == (
        "rust",
        "flag",
    )
    assert rust.servicer_impl_source(SimpleNamespace(), environ={}) == ("rust", "flag")


def test_the_flag_is_a_store_action_and_a_new_parser_forgets_the_last_choice(
    monkeypatch, no_parsed_flag
):
    """SGLang's --config merger takes only store and store_true options from
    the YAML file, so `servicer-impl: rust` there must find a store action;
    and a new parser is a new parse, which drops the choice an earlier parser
    in this process made."""
    monkeypatch.setenv(rust.SERVICER_IMPL_ENV, "python")
    parser = argparse.ArgumentParser(prog="sglang serve")
    plugin.add_servicer_impl_argument(parser)
    action = parser._option_string_actions[plugin.SERVICER_IMPL_FLAG]  # noqa: SLF001
    assert isinstance(action, argparse._StoreAction)  # noqa: SLF001
    assert parser.parse_args(["--servicer-impl", "rust"]).servicer_impl == "rust"
    assert plugin.parsed_flag() == "rust"
    # The next parser SGLang builds in this process starts without it: a parse
    # without the flag falls through to the environment.
    fresh = argparse.ArgumentParser(prog="sglang serve")
    assert plugin.add_servicer_impl_argument(fresh) is True
    assert plugin.parsed_flag() is None
    assert fresh.parse_args([]).servicer_impl is None
    assert rust.servicer_impl_source(
        SimpleNamespace(), environ={"SMG_SGLANG_SERVICER_IMPL": "python"}
    ) == ("python", "env")


def test_plugin_hooks_sglangs_parser_builder(monkeypatch, no_parsed_flag):
    """SGLang runs the plugin before it parses: the hook on
    ``ServerArgs.add_cli_args`` adds the flag to the parser it filled,
    whether the method is called as a staticmethod or with a class."""
    registered = []

    class HookType:
        BEFORE, AFTER, AROUND, REPLACE = "before", "after", "around", "replace"

    class HookRegistry:
        @classmethod
        def register(cls, target, hook, hook_type=HookType.AFTER):
            registered.append((target, hook, hook_type))

    _install(
        monkeypatch,
        "sglang.srt.plugins.hook_registry",
        HookRegistry=HookRegistry,
        HookType=HookType,
    )
    plugin.register()
    assert registered == [
        ("sglang.srt.server_args.ServerArgs.add_cli_args", plugin.after_add_cli_args, "after")
    ]
    parser = argparse.ArgumentParser(prog="sglang serve")
    assert plugin.after_add_cli_args(None, parser) is None
    assert "--servicer-impl" in parser.format_help()
    with_class = argparse.ArgumentParser(prog="sglang serve")
    plugin.after_add_cli_args(None, object, with_class)
    assert "--servicer-impl" in with_class.format_help()
    plugin.after_add_cli_args(None, parser=with_class)  # once per parser
    assert with_class.format_help().count("--servicer-impl") == 2  # usage + option


def test_model_facts_mirror_the_python_servicer():
    facts = rust.model_facts(_server_args(), _model_config())
    assert facts["model_path"] == "org/model"
    assert facts["tokenizer_path"] == "org/model", "falls back to the model path"
    assert facts["served_model_name"] == "served"
    assert facts["is_generation"] is True
    assert facts["model_type"] == "qwen3" and facts["architectures"] == ["Qwen3ForCausalLM"]
    assert facts["max_context_length"] == 4096 and facts["max_req_input_len"] == 4096
    assert facts["vocab_size"] == 1024
    assert facts["eos_token_ids"] == [2, 7], "the set comes out sorted"
    assert (facts["pad_token_id"], facts["bos_token_id"]) == (0, 1)
    assert facts["weight_version"] == ""
    assert json.loads(facts["preferred_sampling_params"]) == {"top_k": 20}
    # Generation-config defaults merged with the preferred ones.
    assert json.loads(facts["default_sampling_params_json"]) == {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
    }
    assert facts["supports_vision"] is False
    assert facts["id2label_json"] == "" and facts["num_labels"] == 0


def test_model_facts_derive_classification_labels_like_the_python_servicer():
    config = _model_config(
        hf_config=SimpleNamespace(
            model_type="bert",
            architectures=["BertForSequenceClassification"],
            pad_token_id=0,
            bos_token_id=0,
            id2label=None,
            num_labels=2,
        ),
        is_generation=False,
        get_default_sampling_params=lambda: {},
    )
    facts = rust.model_facts(_server_args(preferred_sampling_params=None), config)
    assert json.loads(facts["id2label_json"]) == {"0": "LABEL_0", "1": "LABEL_1"}
    assert facts["num_labels"] == 2 and facts["is_generation"] is False
    assert facts["default_sampling_params_json"] == "" and facts["preferred_sampling_params"] == ""


def test_server_facts_carry_the_router_labels_and_the_window(monkeypatch):
    monkeypatch.setattr(rust, "pairing_protocol_from_env", lambda: "p1")
    facts = rust.server_facts(_server_args(dp_size=2, max_running_requests=32))
    args = json.loads(facts["server_args_json"])
    assert args["model_path"] == "org/model" and args["dp_size"] == 2
    assert args["pairing_protocol"] == "p1"
    assert facts["max_running_requests"] == 32
    assert facts["data_parallel_size"] == 2
    assert facts["scheduler_info_json"] == "{}"
    # Without --limit-mm-data-per-request the engine has no per-request media
    # limit to advertise: the Router keeps its own caps.
    assert "mm_item_limits" not in args


def test_server_facts_carry_the_engines_media_limits_flat(monkeypatch):
    # SGLang's --limit-mm-data-per-request is a JSON object per modality; the
    # Router's label readers take flat values only, so it rides along flat.
    monkeypatch.setattr(rust, "pairing_protocol_from_env", lambda: "")
    facts = rust.server_facts(_server_args(limit_mm_data_per_request={"image": 1, "video": 1}))
    args = json.loads(facts["server_args_json"])
    assert args["mm_item_limits"] == "image=1,video=1"
    assert args["limit_mm_data_per_request"] == {"image": 1, "video": 1}


def test_server_facts_carry_the_kv_events_publisher(monkeypatch):
    monkeypatch.setattr(rust, "pairing_protocol_from_env", lambda: "")
    off = rust.server_facts(_server_args())
    assert (off["kv_events_endpoint"], off["kv_events_topic"]) == ("", "")
    assert off["kv_events_replay_endpoint"] == ""
    null_publisher = rust.server_facts(_server_args(kv_events_config='{"publisher": "null"}'))
    assert null_publisher["kv_events_endpoint"] == ""
    zmq_defaults = rust.server_facts(_server_args(kv_events_config='{"publisher": "zmq"}'))
    assert (zmq_defaults["kv_events_endpoint"], zmq_defaults["kv_events_topic"]) == (
        "tcp://*:5557",
        "",
    )
    explicit = rust.server_facts(
        _server_args(
            kv_events_config='{"publisher": "zmq", "endpoint": "tcp://*:6100", "topic": "kv"}'
        )
    )
    assert (explicit["kv_events_endpoint"], explicit["kv_events_topic"]) == ("tcp://*:6100", "kv")
    assert explicit["kv_events_replay_endpoint"] == "", "no replay socket configured"
    with_replay = rust.server_facts(
        _server_args(
            kv_events_config='{"publisher": "zmq", "endpoint": "tcp://*:6100", '
            '"replay_endpoint": "tcp://*:6101", "topic": "kv"}'
        )
    )
    assert with_replay["kv_events_replay_endpoint"] == "tcp://*:6101"
    assert rust.kv_events_publisher(_server_args(kv_events_config="not json")) == ("", "", "")


def test_disaggregated_workers_are_refused_up_front():
    rust.refuse_disaggregation(_server_args())
    rust.refuse_disaggregation(_server_args(disaggregation_mode="null"))
    with pytest.raises(ValueError, match="PD disaggregation"):
        rust.refuse_disaggregation(_server_args(disaggregation_mode="prefill"))
    with pytest.raises(ValueError, match="EPD"):
        rust.refuse_disaggregation(_server_args(language_only=True))
    with pytest.raises(ValueError, match="PD disaggregation"):
        asyncio.run(rust.serve_rust(_server_args(disaggregation_mode="decode")))


def test_server_facts_refuse_non_finite_floats():
    with pytest.raises(ValueError):
        rust.server_facts(_server_args(mem_fraction_static=float("nan")))


def test_kv_event_publishing_is_on_by_default_under_the_rust_servicer():
    """A launcher without --kv-events-config left SGLang publishing nothing and
    the router's cache-aware routing blind (smg-lab #1): the Rust path turns
    the ZMQ publisher on itself unless told otherwise."""
    expected = '{"publisher": "zmq"}'
    # Nothing given: SGLang's own defaults behind the ZMQ publisher.
    args = _server_args()
    assert rust.default_kv_events_config(args, environ={}) == expected
    assert args.kv_events_config == expected
    assert rust.kv_events_publisher(args) == ("tcp://*:5557", "", "")
    args = _server_args(kv_events_config=None)
    assert rust.default_kv_events_config(args, environ={}) == expected
    args = _server_args(kv_events_config="")
    assert rust.default_kv_events_config(args, environ={}) == expected
    # The opt-out.
    for value in ("0", "false", "No", " off "):
        args = _server_args(kv_events_config=None)
        assert rust.default_kv_events_config(args, environ={rust.KV_EVENTS_ENV: value}) is None
        assert args.kv_events_config is None
    args = _server_args()
    assert rust.default_kv_events_config(args, environ={rust.KV_EVENTS_ENV: "1"}) == expected
    # An explicit configuration is kept as given, off included.
    for given in ('{"publisher": "null"}', '{"publisher": "zmq", "endpoint": "tcp://*:6100"}'):
        args = _server_args(kv_events_config=given)
        assert rust.default_kv_events_config(args, environ={}) is None
        assert args.kv_events_config == given
    # Server args that cannot be written are left alone.

    class Frozen:
        __slots__ = ()
        kv_events_config = None

    assert rust.default_kv_events_config(Frozen(), environ={}) is None


def test_serve_rust_wires_the_server_the_scheduler_and_the_supervisor(monkeypatch, tmp_path):
    created = {}

    class FakeServer:
        def __init__(self, **kwargs):
            created.update(kwargs)
            self.address = kwargs["bind_address"]

    smg_servicer = types.ModuleType("smg.servicer")
    smg_servicer.SglangGrpcServer = FakeServer
    smg_servicer.init_servicer_tracing = lambda _filter: None
    smg_pkg = types.ModuleType("smg")
    smg_pkg.servicer = smg_servicer
    monkeypatch.setitem(sys.modules, "smg", smg_pkg)
    monkeypatch.setitem(sys.modules, "smg.servicer", smg_servicer)
    monkeypatch.setenv(rust.HANDSHAKE_PORT_ENV, "24321")
    monkeypatch.setenv("SMG_ZMQ_SOCKET_DIR", str(tmp_path))
    monkeypatch.setattr(rust, "model_facts", lambda args: {"model_path": args.model_path})
    monkeypatch.setattr(
        rust, "server_facts", lambda args: {"data_parallel_size": 1, "sglang_version": "x"}
    )
    monkeypatch.setattr(rust, "tokenizer_dir_for", lambda args: "/tok")
    launched = {}

    def fake_launch(server_args, *, handshake_port):
        launched["args"] = server_args
        launched["port"] = handshake_port
        return "engine"

    monkeypatch.setattr(rust, "launch_headless_scheduler", fake_launch)

    async def fake_supervise(server, engine, *, drain_secs):
        assert isinstance(server, FakeServer) and engine == "engine"
        return 7

    monkeypatch.setattr(rust, "supervise", fake_supervise)
    args = _server_args()
    assert asyncio.run(rust.serve_rust(args)) == 7
    assert created["bind_address"] == "127.0.0.1:30011"
    assert created["handshake_address"] == "tcp://127.0.0.1:24321"
    assert created["ipc_base_url"].startswith(f"ipc://{tmp_path}/sglang-servicer-")
    assert created["engine_count"] == 1 and created["tokenizer_dir"] == "/tok"
    assert created["model_path"] == "org/model" and created["sglang_version"] == "x"
    assert launched == {"args": args, "port": 24321}
    # No --kv-events-config was given: the headless scheduler publishes.
    assert args.kv_events_config == '{"publisher": "zmq"}'


def test_serve_grpc_hands_the_process_to_rust_when_the_flag_says_so(
    monkeypatch, caplog, no_parsed_flag
):
    pytest.importorskip("sglang")
    from smg_grpc_servicer.sglang import server

    monkeypatch.setenv(rust.SERVICER_IMPL_ENV, "rust")

    async def fake_serve_rust(server_args):
        return 3

    monkeypatch.setattr(server, "serve_rust", fake_serve_rust)
    with caplog.at_level("INFO", logger=server.logger.name):
        with pytest.raises(SystemExit) as raised:
            asyncio.run(server.serve_grpc(SimpleNamespace()))
    assert raised.value.code == 3
    assert "Servicer implementation: rust (source=env)" in caplog.text
