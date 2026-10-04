"""Unit tests for the Rust request path of the SGLang servicer
(``smg_grpc_servicer.sglang.rust``).

Everything SGLang-shaped is lazy in the module, so these run without SGLang:
the facts get a fake model config, ``serve_rust`` gets a fake Rust server
class and a fake scheduler launch.
"""

from __future__ import annotations

import asyncio
import json
import sys
import types
from types import SimpleNamespace

import pytest
from smg_grpc_servicer.sglang import rust


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
    assert rust.SERVICER_IMPL_ENV == "SMG_SGLANG_SERVICER_IMPL"
    assert callable(rust.serve_rust) and callable(rust.resolve_servicer_impl)


def test_resolve_servicer_impl_reads_the_env():
    assert rust.resolve_servicer_impl({}) == "python"
    assert rust.resolve_servicer_impl({rust.SERVICER_IMPL_ENV: " Rust "}) == "rust"
    with pytest.raises(ValueError, match="SMG_SGLANG_SERVICER_IMPL"):
        rust.resolve_servicer_impl({rust.SERVICER_IMPL_ENV: "go"})


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


def test_serve_grpc_hands_the_process_to_rust_when_the_flag_says_so(monkeypatch):
    pytest.importorskip("sglang")
    from smg_grpc_servicer.sglang import server

    monkeypatch.setenv(rust.SERVICER_IMPL_ENV, "rust")

    async def fake_serve_rust(server_args):
        return 3

    monkeypatch.setattr(server, "serve_rust", fake_serve_rust)
    with pytest.raises(SystemExit) as raised:
        asyncio.run(server.serve_grpc(SimpleNamespace()))
    assert raised.value.code == 3
