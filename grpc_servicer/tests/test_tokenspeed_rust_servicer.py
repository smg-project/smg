"""Unit tests for the Rust request path of the TokenSpeed servicer
(``smg_grpc_servicer.tokenspeed.rust``).

Everything TokenSpeed-shaped is lazy in the module, so these run without
TokenSpeed: the facts get a fake model config and a stub servicer module,
``serve_rust`` gets a fake Rust server class and a fake scheduler launch.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import sys
import types
from types import SimpleNamespace

import pytest
from smg_grpc_servicer.tokenspeed import rust


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
        default_factory=lambda: SimpleNamespace(attn=SimpleNamespace(dp_size=1))
    )
    zmq_msgpack: bool = False
    skip_tokenizer_init: bool = False
    rl_control_api_key: str | None = None
    hf_token: str | None = None
    data_parallel_address: str = "127.0.0.1"
    data_parallel_rpc_port: int = 30500
    zmq_engine_index: int = 0


def test_package_exposes_the_flag_and_the_rust_entry_points():
    import smg_grpc_servicer.tokenspeed as pkg

    assert pkg.SERVICER_IMPL_ENV == "SMG_TOKENSPEED_SERVICER_IMPL"
    assert pkg.resolve_servicer_impl is rust.resolve_servicer_impl
    assert pkg.serve_rust is rust.serve_rust


def test_resolve_servicer_impl_reads_the_env():
    assert rust.resolve_servicer_impl(environ={}) == "python"
    assert rust.resolve_servicer_impl(environ={"SMG_TOKENSPEED_SERVICER_IMPL": " Rust "}) == "rust"
    with pytest.raises(ValueError, match="SMG_TOKENSPEED_SERVICER_IMPL"):
        rust.resolve_servicer_impl(environ={"SMG_TOKENSPEED_SERVICER_IMPL": "go"})


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
        "model_dtype": "bfloat16",
        "multimodal_encoder_dtype": "bfloat16",
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
    assert facts["max_running_requests"] == 64
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

    _install(
        monkeypatch,
        "smg.servicer",
        TokenSpeedGrpcServer=FakeServer,
        init_servicer_tracing=lambda level=None: None,
    )
    monkeypatch.setattr(rust, "model_facts", lambda args: {"model_path": args.model})
    monkeypatch.setattr(
        rust,
        "server_facts",
        lambda args: {"server_args_json": "{}", "data_parallel_size": 2},
    )
    monkeypatch.setattr(rust, "tokenizer_dir_for", lambda args: str(tmp_path))
    monkeypatch.setattr(rust, "free_port", lambda: 24321)
    monkeypatch.setattr(rust, "default_socket_dir", lambda: str(tmp_path / "sockets"))

    def fake_launch(headless):
        recorded["headless"] = headless
        return "engine-handle"

    async def fake_supervise(server, engine, *, drain_secs, **_kwargs):
        recorded["supervised"] = (server, engine, drain_secs)
        return 0

    monkeypatch.setattr(rust, "launch_headless_scheduler", fake_launch)
    monkeypatch.setattr(rust, "supervise", fake_supervise)
    monkeypatch.delenv("SMG_TOKENSPEED_SERVICER_HANDSHAKE_PORT", raising=False)
    monkeypatch.setenv("SMG_TOKENSPEED_SERVICER_DRAIN_SECS", "1.5")
    monkeypatch.setenv("SMG_TOKENSPEED_SERVICER_STARTUP_TIMEOUT_SECS", "42")

    assert asyncio.run(rust.serve_rust(FakeServerArgs(port=50123))) == 0

    server = recorded["server"]
    assert server["bind_address"] == "127.0.0.1:50123"
    assert server["handshake_address"] == "tcp://127.0.0.1:24321"
    assert server["ipc_base_url"].startswith(f"ipc://{tmp_path / 'sockets'}/tokenspeed-servicer-")
    assert server["engine_count"] == 2
    assert server["tokenizer_dir"] == str(tmp_path)
    assert server["engine_startup_timeout_secs"] == 42.0
    assert server["model_path"] == "org/m"
    headless = recorded["headless"]
    assert headless.zmq_msgpack is True and headless.skip_tokenizer_init is True
    assert headless.data_parallel_rpc_port == 24321
    supervised_server, engine, drain_secs = recorded["supervised"]
    assert isinstance(supervised_server, FakeServer)
    assert engine == "engine-handle"
    assert drain_secs == 1.5
