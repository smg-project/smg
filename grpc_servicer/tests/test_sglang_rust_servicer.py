"""Unit tests for the Rust request path of the SGLang servicer
(``smg_grpc_servicer.sglang.rust``).

Everything SGLang-shaped is lazy in the module, so these run without SGLang:
the facts get a fake model config, ``serve_rust`` gets a fake Rust server
class and a fake scheduler launch.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import os
import signal
import sys
import threading
import time
import types
from types import SimpleNamespace

import pytest
from smg_grpc_servicer import rust_lifecycle
from smg_grpc_servicer.sglang import plugin, rust, zmq_plugin


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
        SglangGrpcServer=server,
        init_servicer_tracing=init_servicer_tracing or (lambda *_args, **_kwargs: None),
    )


@pytest.fixture
def no_parsed_flag(monkeypatch):
    """No ``--servicer-impl`` parsed in this process, whatever ran before."""
    monkeypatch.setattr(plugin._state, "parsed", None)


@pytest.fixture
def restore_fault_signals():
    """The rank launch installs a SIGQUIT handler in this process; put the
    previous disposition back."""
    previous = signal.getsignal(signal.SIGQUIT)
    yield
    signal.signal(signal.SIGQUIT, previous if previous is not None else signal.SIG_DFL)


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


def test_server_facts_carry_where_the_engine_starts_a_grammar(monkeypatch):
    # The engine's reasoning parser and where it starts a request's grammar on
    # a thinking prompt ride with the server args, as the Python servicer adds
    # them: after the thought behind --reasoning-parser, from the first token
    # without one; neither key when the server args predate the parser.
    monkeypatch.setattr(rust, "pairing_protocol_from_env", lambda: "")
    gated = json.loads(
        rust.server_facts(_server_args(reasoning_parser="qwen3"))["server_args_json"]
    )
    assert gated["engine_reasoning_parser"] == "qwen3"
    assert gated["structured_outputs_start"] == "after_reasoning"
    # The engine's own argument stays under its own name; the Router's label
    # readers never take it as the per-model parser override.
    assert gated["reasoning_parser"] == "qwen3"

    plain = json.loads(rust.server_facts(_server_args(reasoning_parser=None))["server_args_json"])
    assert plain["structured_outputs_start"] == "first_token"
    assert "engine_reasoning_parser" not in plain

    older = json.loads(rust.server_facts(_server_args())["server_args_json"])
    assert "structured_outputs_start" not in older and "engine_reasoning_parser" not in older


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


class _FakeRankProcess:
    """A ``multiprocessing``-shaped scheduler rank SGLang's spawner started."""

    def __init__(self, pid: int):
        self.pid = pid
        self.exitcode: int | None = None
        self.events: list[str] = []

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


def _stub_rank_launch(monkeypatch, *, ranks: int = 2) -> dict:
    """Stub what `launch_scheduler_ranks` imports from SGLang: the preparation
    calls are recorded in order, the spawner records the process and thread
    it ran on and returns fake rank processes plus a ready wait."""
    recorded: dict = {"calls": [], "ready_waits": 0}
    recorded["processes"] = [_FakeRankProcess(1000 + index) for index in range(ranks)]
    port_args = SimpleNamespace(name="port-args")

    class Engine:
        @classmethod
        def _launch_scheduler_processes(cls, server_args, port_args, run_scheduler_process_func):
            recorded["calls"].append("launch")
            recorded["launch"] = dict(
                server_args=server_args,
                port_args=port_args,
                target=run_scheduler_process_func,
                started_in=os.getpid(),
                thread=threading.current_thread(),
                handler_in=callable(signal.getsignal(signal.SIGQUIT)),
            )

            def wait_for_ready():
                recorded["ready_waits"] += 1
                recorded["ready_thread"] = threading.current_thread()
                if recorded.get("ready_error") is not None:
                    raise recorded["ready_error"]

            return SimpleNamespace(wait_for_ready=wait_for_ready), list(recorded["processes"])

    def run_scheduler_process(*_args, **_kwargs):
        raise AssertionError("the rank entry point runs in the ranks, not here")

    class PortArgs:
        @staticmethod
        def init_new(server_args):
            recorded["calls"].append("port_args")
            return port_args

    _install(
        monkeypatch,
        "sglang.srt.entrypoints.engine",
        Engine=Engine,
        _set_envs_and_config=lambda args: recorded["calls"].append("set_envs_and_config"),
    )
    _install(
        monkeypatch, "sglang.srt.managers.scheduler", run_scheduler_process=run_scheduler_process
    )
    _install(
        monkeypatch,
        "sglang.srt.plugins",
        load_plugins=lambda: recorded["calls"].append("load_plugins"),
    )
    _install(
        monkeypatch,
        "sglang.srt.runtime_context",
        publish=lambda args, role: recorded["calls"].append(f"publish:{role}"),
    )
    _install(monkeypatch, "sglang.srt.server_args", PortArgs=PortArgs)
    # The launch writes the plugin's two variables straight into the
    # environment; registered here so the teardown puts their absence back.
    monkeypatch.setenv(zmq_plugin.HANDSHAKE_ENV, "")
    monkeypatch.setenv(zmq_plugin.ENGINE_INDEX_ENV, "")
    # SGLang's own setup sets the spawn method; the launcher's guard reads it
    # and must not switch this test process over.
    recorded["start_method_calls"] = []
    monkeypatch.setattr(
        rust_lifecycle.multiprocessing, "get_start_method", lambda allow_none=False: "spawn"
    )
    monkeypatch.setattr(
        rust_lifecycle.multiprocessing,
        "set_start_method",
        lambda method, force=False: recorded["start_method_calls"].append((method, force)),
    )
    _install(
        monkeypatch,
        "sglang.srt.utils",
        configure_logger=lambda args: recorded["calls"].append("configure_logger"),
    )
    recorded["target"] = run_scheduler_process
    recorded["port_args"] = port_args
    return recorded


def _preparable_args(recorded: dict, **overrides):
    """Server args with the two preparation methods SGLang's entrypoint calls."""
    args = _server_args(**overrides)
    args.resolve_once = lambda: recorded["calls"].append("resolve_once")
    args.check_server_args = lambda: recorded["calls"].append("check_server_args")
    return args


def _wait_until(condition, what: str, deadline_secs: float = 5.0) -> None:
    deadline = time.monotonic() + deadline_secs
    while not condition():
        assert time.monotonic() < deadline, what
        time.sleep(0.01)


def test_launch_scheduler_ranks_starts_them_from_this_process_as_the_headless_launcher_does(
    monkeypatch, caplog, restore_fault_signals
):
    """smg-lab#181: the ranks are started by SGLang's own spawner from the
    servicer process, on the server args prepared the way the headless
    launcher prepares them, with the plugin's handshake in the environment
    the ranks inherit; no `multiprocessing` child runs a launcher. The ready
    reports are drained on a thread, and a rank's failure signal ends the
    group rather than this process."""
    from smg_grpc_servicer.sglang.zmq_plugin import ENGINE_INDEX_ENV, HANDSHAKE_ENV

    recorded = _stub_rank_launch(monkeypatch)
    monkeypatch.delenv(HANDSHAKE_ENV, raising=False)
    monkeypatch.delenv(ENGINE_INDEX_ENV, raising=False)
    monkeypatch.setattr(
        rust.multiprocessing,
        "get_context",
        lambda method: pytest.fail(f"a {method} launcher child was started"),
    )
    args = _preparable_args(recorded)

    with caplog.at_level("INFO"):
        engine = rust.launch_scheduler_ranks(args, handshake_port=24321)

    assert isinstance(engine, rust_lifecycle.EngineProcessGroup)
    launch = recorded["launch"]
    assert launch["started_in"] == os.getpid() and launch["thread"] is threading.main_thread()
    assert launch["server_args"] is args and launch["port_args"] is recorded["port_args"]
    assert launch["target"] is recorded["target"]
    assert recorded["calls"] == [
        "configure_logger",
        "resolve_once",
        "set_envs_and_config",
        "load_plugins",
        "check_server_args",
        "publish:tokenizer",
        "port_args",
        "launch",
    ]
    assert os.environ[HANDSHAKE_ENV] == "tcp://127.0.0.1:24321"
    assert os.environ[ENGINE_INDEX_ENV] == "0"
    assert engine.pids == [1000, 1001] and engine.pid == 1000 and engine.poll() is None
    _wait_until(lambda: recorded["ready_waits"] == 1, "the ready reports were never waited for")
    assert recorded["ready_thread"] is not threading.main_thread()
    _wait_until(
        lambda: "SGLang scheduler: 2 scheduler process(es) ready" in caplog.text,
        "the ready log line never appeared",
    )
    assert "Started 2 scheduler rank process(es) from the servicer process: pids 1000, 1001" in (
        caplog.text
    )
    assert callable(signal.getsignal(signal.SIGQUIT))


def test_a_failing_rank_ends_the_group_through_its_signal_not_the_servicer(
    monkeypatch, caplog, restore_fault_signals
):
    """A rank that hits an exception sends its parent SIGQUIT (SGLang's own
    launch-phase handler would kill the whole tree): the group reports exit
    code 1 and the other rank is terminated; the servicer process goes on to
    drain through its lifecycle."""
    recorded = _stub_rank_launch(monkeypatch)
    engine = rust.launch_scheduler_ranks(_preparable_args(recorded), handshake_port=24321)

    with caplog.at_level("ERROR"):
        signal.raise_signal(signal.SIGQUIT)

    assert engine.poll() == 1
    assert [process.events[0] for process in recorded["processes"]] == ["terminate", "terminate"]
    assert "SGLang scheduler: a scheduler process signalled a failure (signal 3)" in caplog.text


def test_launch_scheduler_ranks_off_the_main_thread_keeps_the_spawned_launcher(monkeypatch):
    """Where no signal handler can be installed, the spawned headless launcher
    (with a supervisor of its own) still serves."""
    recorded = _stub_rank_launch(monkeypatch)
    fallback = object()
    launched: list = []

    def fake_launch(server_args, *, handshake_port):
        launched.append((server_args, handshake_port))
        return fallback

    monkeypatch.setattr(rust, "launch_headless_scheduler", fake_launch)
    args = _preparable_args(recorded)
    result: dict = {}
    thread = threading.Thread(
        target=lambda: result.update(engine=rust.launch_scheduler_ranks(args, handshake_port=1))
    )
    thread.start()
    thread.join(5)
    assert result["engine"] is fallback and launched == [(args, 1)]
    assert "launch" not in recorded and recorded["calls"] == []


def test_serve_rust_starts_the_scheduler_ranks_from_this_process(
    monkeypatch, tmp_path, restore_fault_signals
):
    """smg-lab#181 end to end: `serve_rust` binds the server, then starts the
    ranks in this process against the server's handshake port; the supervisor
    gets the ranks' handle."""
    from smg_grpc_servicer.sglang.zmq_plugin import HANDSHAKE_ENV

    recorded = _stub_rank_launch(monkeypatch)
    created: dict = {}

    class FakeServer:
        def __init__(self, **kwargs):
            created.update(kwargs)
            self.address = kwargs["bind_address"]

    _install(monkeypatch, "smg")
    _install(
        monkeypatch,
        "smg.servicer",
        SglangGrpcServer=FakeServer,
        init_servicer_tracing=lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(rust.multiprocessing, "get_context", lambda method: pytest.fail(method))
    monkeypatch.delenv(HANDSHAKE_ENV, raising=False)
    monkeypatch.setenv(rust.HANDSHAKE_PORT_ENV, "24321")
    monkeypatch.setenv("SMG_ZMQ_SOCKET_DIR", str(tmp_path))
    monkeypatch.setattr(rust, "model_facts", lambda args: {"model_path": args.model_path})
    monkeypatch.setattr(
        rust, "server_facts", lambda args: {"data_parallel_size": 1, "sglang_version": "x"}
    )
    monkeypatch.setattr(rust, "tokenizer_dir_for", lambda args: "/tok")
    supervised: dict = {}

    async def fake_supervise(server, engine, *, drain_secs, **_kwargs):
        supervised["server"], supervised["engine"] = server, engine
        return 0

    monkeypatch.setattr(rust, "supervise", fake_supervise)
    args = _preparable_args(recorded)

    assert asyncio.run(rust.serve_rust(args)) == 0

    engine = supervised["engine"]
    assert isinstance(engine, rust_lifecycle.EngineProcessGroup) and engine.pids == [1000, 1001]
    assert recorded["launch"]["started_in"] == os.getpid()
    assert recorded["launch"]["server_args"] is args
    assert os.environ[HANDSHAKE_ENV] == created["handshake_address"] == "tcp://127.0.0.1:24321"
    assert isinstance(supervised["server"], FakeServer)


def test_serve_rust_wires_the_server_the_scheduler_and_the_supervisor(monkeypatch, tmp_path):
    created = {}

    class FakeServer:
        def __init__(self, **kwargs):
            created.update(kwargs)
            self.address = kwargs["bind_address"]

    _install_fake_servicer(monkeypatch, FakeServer)
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

    monkeypatch.setattr(rust, "launch_scheduler_ranks", fake_launch)

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


def _run_serve_rust(monkeypatch, tmp_path, *, real_facts: bool = False) -> dict:
    """``serve_rust`` against a recording fake server, the scheduler launch and
    the supervisor stubbed; the keywords the server was constructed with. With
    ``real_facts`` the launcher's own fact producers run, over a fake model
    config, as at a real start."""
    created: dict = {}

    class FakeServer:
        def __init__(self, **kwargs):
            created.update(kwargs)
            self.address = kwargs["bind_address"]

    _install_fake_servicer(monkeypatch, FakeServer)
    monkeypatch.setenv(rust.HANDSHAKE_PORT_ENV, "24321")
    monkeypatch.setenv("SMG_ZMQ_SOCKET_DIR", str(tmp_path))
    if real_facts:
        monkeypatch.setattr(rust, "_model_config", lambda args: _model_config())
    else:
        monkeypatch.setattr(rust, "model_facts", lambda args: {"model_path": args.model_path})
        monkeypatch.setattr(
            rust, "server_facts", lambda args: {"data_parallel_size": 1, "sglang_version": "x"}
        )
    monkeypatch.setattr(rust, "tokenizer_dir_for", lambda args: "/tok")
    monkeypatch.setattr(
        rust, "launch_scheduler_ranks", lambda server_args, *, handshake_port: "engine"
    )

    async def fake_supervise(server, engine, *, drain_secs, **_kwargs):
        return 0

    monkeypatch.setattr(rust, "supervise", fake_supervise)
    assert asyncio.run(rust.serve_rust(_server_args())) == 0
    return created


def test_serve_rust_hands_the_server_both_startup_bounds(monkeypatch, tmp_path):
    """The silence bound and the ceiling reach the server from their
    environment variables, the lifecycle defaults otherwise; 0 lifts the
    ceiling."""
    monkeypatch.delenv(rust.STARTUP_TIMEOUT_SECS_ENV, raising=False)
    monkeypatch.delenv(rust.STARTUP_CEILING_SECS_ENV, raising=False)
    created = _run_serve_rust(monkeypatch, tmp_path)
    assert created["engine_startup_timeout_secs"] == rust.DEFAULT_STARTUP_TIMEOUT_SECS
    assert created["engine_startup_ceiling_secs"] == rust.DEFAULT_STARTUP_CEILING_SECS

    monkeypatch.setenv(rust.STARTUP_TIMEOUT_SECS_ENV, "42")
    monkeypatch.setenv(rust.STARTUP_CEILING_SECS_ENV, "0")
    created = _run_serve_rust(monkeypatch, tmp_path)
    assert created["engine_startup_timeout_secs"] == 42.0
    assert created["engine_startup_ceiling_secs"] == 0.0


def test_serve_rust_constructs_the_server_with_keywords_the_binding_takes(monkeypatch, tmp_path):
    """The keyword set `serve_rust` hands the Rust server is bound against the
    compiled binding's constructor: a fact or a knob the launcher gains without
    a parameter on the binding fails here, not at the scheduler's start.
    Skipped where the `smg` wheel is not installed."""
    servicer = pytest.importorskip("smg.servicer")
    signature = inspect.signature(servicer.SglangGrpcServer)
    created = _run_serve_rust(monkeypatch, tmp_path, real_facts=True)
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
    ``from smg_grpc_servicer.sglang import servicer`` gets the real module."""
    from smg_grpc_servicer import sglang as package

    name = "smg_grpc_servicer.sglang.servicer"
    before = (vars(package).get("servicer"), sys.modules.get(name))
    with pytest.MonkeyPatch.context() as monkeypatch:
        fake = _install(monkeypatch, name)
        assert package.servicer is fake and sys.modules[name] is fake
    assert (vars(package).get("servicer"), sys.modules.get(name)) == before


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


def test_the_fault_handler_is_in_before_the_spawner_starts_a_rank(
    monkeypatch, restore_fault_signals
):
    """A rank that fails during its start signals its parent at once, so the
    SIGQUIT handler is installed before SGLang's spawner starts any rank; the
    start method set by SGLang's own setup is left alone."""
    recorded = _stub_rank_launch(monkeypatch)

    rust.launch_scheduler_ranks(_preparable_args(recorded), handshake_port=24321)

    assert recorded["launch"]["handler_in"] is True
    assert recorded["start_method_calls"] == []


def test_a_failed_ready_wait_ends_the_group(monkeypatch, caplog, restore_fault_signals):
    """SGLang's ready wait raising (a rank that failed to load and stays up)
    is a fault of the group: its poll turns 1 and the ranks are terminated,
    so the servicer drains and exits instead of supervising a scheduler that
    never came up."""
    recorded = _stub_rank_launch(monkeypatch)
    recorded["ready_error"] = RuntimeError("rank 1 failed to load")

    with caplog.at_level("ERROR"):
        engine = rust.launch_scheduler_ranks(_preparable_args(recorded), handshake_port=24321)
        deadline = time.monotonic() + 5
        while engine.poll() != 1:
            assert time.monotonic() < deadline, "the failed start never ended the group"
            time.sleep(0.01)

    assert [process.events[0] for process in recorded["processes"]] == ["terminate", "terminate"]
    assert "SGLang scheduler: the scheduler processes did not start: rank 1 failed to load" in (
        caplog.text
    )


def test_the_spawn_start_method_is_forced_when_the_engines_setup_left_it_unset(
    monkeypatch, restore_fault_signals
):
    recorded = _stub_rank_launch(monkeypatch)
    monkeypatch.setattr(
        rust_lifecycle.multiprocessing, "get_start_method", lambda allow_none=False: None
    )

    rust.launch_scheduler_ranks(_preparable_args(recorded), handshake_port=24321)

    assert recorded["start_method_calls"] == [("spawn", True)]


# ── the knobs are validated before the bind (smg-lab #292) ─────────────────────


@pytest.mark.parametrize(
    "name, value, text",
    [
        ("SMG_SGLANG_SERVICER_DRAIN_SECS", "abc", "is not a number"),
        ("SMG_SGLANG_SERVICER_DRAIN_SECS", "-5", "must be at least 0"),
        ("SMG_SGLANG_SERVICER_STARTUP_TIMEOUT_SECS", "abc", "is not a number"),
        ("SMG_SGLANG_SERVICER_STARTUP_TIMEOUT_SECS", "-1", "must be greater than 0"),
        ("SMG_SGLANG_SERVICER_HANDSHAKE_PORT", "70000", "must be between 1 and 65535"),
        ("SMG_SGLANG_SERVICER_HANDSHAKE_PORT", "abc", "is not a port number"),
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

    def no_engine_work(*_args, **_kwargs):
        raise AssertionError("no engine work before the knobs are validated")

    monkeypatch.setattr(rust, "model_facts", no_engine_work)
    monkeypatch.setattr(rust, "server_facts", no_engine_work)
    monkeypatch.setattr(rust, "launch_scheduler_ranks", no_engine_work)
    monkeypatch.setenv("SMG_ZMQ_SOCKET_DIR", str(tmp_path))
    monkeypatch.setenv(name, value)

    with pytest.raises(rust_lifecycle.ServicerConfigError, match=f"{name}='{value}' {text}"):
        asyncio.run(rust.serve_rust(_server_args()))
    assert created == {}


def test_serve_rust_refuses_a_socket_dir_that_is_a_file_before_the_server_binds(
    monkeypatch, tmp_path
):
    created: dict = {}

    class FakeServer:
        def __init__(self, **kwargs):
            created.update(kwargs)

    _install_fake_servicer(monkeypatch, FakeServer)
    a_file = tmp_path / "a-file"
    a_file.write_text("not a directory")
    monkeypatch.setenv("SMG_ZMQ_SOCKET_DIR", str(a_file))

    with pytest.raises(rust_lifecycle.ServicerConfigError, match="SMG_ZMQ_SOCKET_DIR=.*not a dir"):
        asyncio.run(rust.serve_rust(_server_args()))
    assert created == {}


def test_a_disaggregated_worker_is_refused_as_a_configuration_error():
    with pytest.raises(rust_lifecycle.ServicerConfigError, match="PD disaggregation"):
        rust.refuse_disaggregation(_server_args(disaggregation_mode="prefill"))
