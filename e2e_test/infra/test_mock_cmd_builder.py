"""The mock engine's worker command: the scripted mock worker on each wire."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from infra.constants import (
    DEFAULT_HOST,
    ENV_MOCK_WIRE,
    LOCAL_RUNTIMES,
    MOCK_WIRES,
    RUNTIME_LABELS,
    ConnectionMode,
    Runtime,
    WorkerType,
    get_mock_wire,
)
from infra.worker import Worker

MODEL = "meta-llama/Llama-3.2-1B-Instruct"


def _worker(mode: ConnectionMode, **kwargs) -> Worker:
    return Worker(model_id=MODEL, engine="mock", port=31001, gpu_ids=[], mode=mode, **kwargs)


def _flag(cmd: list[str], flag: str) -> str:
    return cmd[cmd.index(flag) + 1]


def test_the_mock_is_a_local_runtime_with_a_label():
    assert Runtime.MOCK == "mock"
    assert Runtime.MOCK in LOCAL_RUNTIMES
    assert RUNTIME_LABELS["mock"] == "Mock"


@pytest.mark.parametrize("mode", [ConnectionMode.GRPC, ConnectionMode.HTTP])
def test_the_grpc_and_http_commands_bind_the_worker_port_and_an_admin_port(mode):
    worker = _worker(mode)
    with patch.dict(os.environ, {"MOCK_WORKER_BIN": "/opt/bin/mock-worker"}):
        cmd = worker._build_cmd()
    assert cmd[0] == "/opt/bin/mock-worker"
    assert _flag(cmd, "--host") == DEFAULT_HOST, "the lane's loopback, as the engines get it"
    port_flag = "--grpc-base-port" if mode == ConnectionMode.GRPC else "--http-base-port"
    count_flag = "--grpc-count" if mode == ConnectionMode.GRPC else "--http-count"
    assert _flag(cmd, port_flag) == "31001"
    assert _flag(cmd, count_flag) == "1"
    # The model spec's path is the model and the tokenizer: the gateway and the
    # mock read the same files.
    assert _flag(cmd, "--model") == _flag(cmd, "--tokenizer")
    assert worker.admin_port is not None
    assert _flag(cmd, "--admin-port") == str(worker.admin_port)
    assert worker.admin_url == f"http://127.0.0.1:{worker.admin_port}"


def test_the_zmq_command_dials_the_handshake_the_gateway_derives():
    from smg.serve import _zmq_handshake_port

    worker = _worker(ConnectionMode.ZMQ)
    cmd = worker._build_cmd()
    assert cmd[0] == "mock-worker", "the binary on the PATH when MOCK_WORKER_BIN is unset"
    assert _flag(cmd, "--zmq-count") == "1"
    # The gateway binds the derived handshake on the IPv4 loopback whatever
    # the lane's host is.
    assert _flag(cmd, "--zmq-handshake").startswith("tcp://127.0.0.1:")
    assert (
        _flag(cmd, "--zmq-handshake") == f"tcp://127.0.0.1:{_zmq_handshake_port(worker.base_url)}"
    )
    assert "--grpc-base-port" not in cmd and "--http-base-port" not in cmd


def test_an_admin_port_given_to_the_worker_is_kept():
    worker = _worker(ConnectionMode.GRPC, admin_port=31999)
    cmd = worker._build_cmd()
    assert _flag(cmd, "--admin-port") == "31999"
    assert not worker._admin_port_auto, "a caller's port is the caller's to release"
    picked = _worker(ConnectionMode.GRPC)
    picked._build_cmd()
    assert picked._admin_port_auto, "a port the harness picked is released at stop"


def test_an_engine_without_an_admin_api_has_no_admin_url():
    worker = Worker(model_id=MODEL, engine="vllm", port=31001, gpu_ids=[0])
    assert worker.admin_url is None


def test_the_prefill_and_decode_roles_reach_the_grpc_command_with_the_bootstrap_port():
    prefill = _worker(ConnectionMode.GRPC, worker_type=WorkerType.PREFILL, bootstrap_port=31900)
    cmd = prefill._build_cmd()
    assert _flag(cmd, "--worker-type") == "prefill"
    assert _flag(cmd, "--bootstrap-base-port") == "31900", "the port the gateway is told of"
    decode = _worker(ConnectionMode.GRPC, worker_type=WorkerType.DECODE)
    cmd = decode._build_cmd()
    assert _flag(cmd, "--worker-type") == "decode"
    assert "--bootstrap-base-port" not in cmd, "a decode worker binds no bootstrap port"
    assert "--worker-type" not in _worker(ConnectionMode.GRPC)._build_cmd(), "a regular worker"


def _last_flag(cmd: list[str], flag: str) -> str:
    return cmd[max(i for i, arg in enumerate(cmd) if arg == flag) + 1]


def test_an_encode_worker_takes_the_vllm_wire_and_a_bootstrap_port():
    encode = _worker(ConnectionMode.GRPC, worker_type=WorkerType.ENCODE, bootstrap_port=31901)
    cmd = encode._build_cmd()
    assert _flag(cmd, "--worker-type") == "encode"
    assert _last_flag(cmd, "--grpc-wire") == "vllm", "the encode role's default wire"
    assert _flag(cmd, "--bootstrap-base-port") == "31901", "the port the gateway is told of"
    # A class's own wire comes after the role's, as for every pool: the mock
    # takes the last, and refuses the encode role on a wire without one.
    stated = _worker(
        ConnectionMode.GRPC,
        worker_type=WorkerType.ENCODE,
        extra_engine_args=["--grpc-wire", "sglang"],
    )._build_cmd()
    assert _last_flag(stated, "--grpc-wire") == "sglang"


def test_the_prefill_and_decode_roles_reach_the_http_command_with_the_bootstrap_port():
    prefill = _worker(ConnectionMode.HTTP, worker_type=WorkerType.PREFILL, bootstrap_port=31900)
    cmd = prefill._build_cmd()
    assert _flag(cmd, "--http-base-port") == "31001", "an HTTP worker on its own port"
    assert _flag(cmd, "--worker-type") == "prefill"
    assert _flag(cmd, "--bootstrap-base-port") == "31900", "the port the gateway is told of"
    assert "--grpc-wire" not in cmd, "an HTTP worker speaks no scheduler wire"
    decode = _worker(ConnectionMode.HTTP, worker_type=WorkerType.DECODE)
    cmd = decode._build_cmd()
    assert _flag(cmd, "--worker-type") == "decode"
    assert "--bootstrap-base-port" not in cmd, "a decode worker binds no bootstrap port"


def test_a_role_on_another_wire_is_refused():
    # The ZMQ workers take no role; the encode role is the gRPC vLLM wire's.
    with pytest.raises(ValueError, match="HTTP or gRPC workers"):
        _worker(ConnectionMode.ZMQ, worker_type=WorkerType.DECODE)._build_cmd()
    with pytest.raises(ValueError, match="gRPC worker"):
        _worker(ConnectionMode.HTTP, worker_type=WorkerType.ENCODE)._build_cmd()


def test_the_mock_is_paired_on_its_wires_one_transport():
    # The mock moves no KV and names Mooncake as its transport on every
    # scheduler wire it can speak; the gateway pairs it on that, whatever the
    # lane's setting for the engines that have a choice.
    with patch.dict(os.environ, {"E2E_VLLM_KV_BACKEND": "nixl"}):
        assert (
            _worker(ConnectionMode.GRPC, worker_type=WorkerType.PREFILL).effective_kv_backend()
            == "mooncake"
        )


def test_the_lane_leaves_the_mocks_wire_alone_unless_it_names_one(monkeypatch):
    # Unset or blank: the mock's own default, no flag on the command line.
    monkeypatch.delenv(ENV_MOCK_WIRE, raising=False)
    assert get_mock_wire() is None
    assert "--grpc-wire" not in _worker(ConnectionMode.GRPC)._build_cmd()
    monkeypatch.setenv(ENV_MOCK_WIRE, "  ")
    assert get_mock_wire() is None
    assert "--grpc-wire" not in _worker(ConnectionMode.GRPC)._build_cmd()


@pytest.mark.parametrize("wire", MOCK_WIRES)
def test_the_lanes_wire_reaches_the_grpc_command_whatever_its_case(monkeypatch, wire):
    monkeypatch.setenv(ENV_MOCK_WIRE, f" {wire.upper()} ")
    assert get_mock_wire() == wire
    cmd = _worker(ConnectionMode.GRPC)._build_cmd()
    assert _flag(cmd, "--grpc-wire") == wire
    # The HTTP worker speaks no scheduler proto, so the lane's wire is not its business.
    assert "--grpc-wire" not in _worker(ConnectionMode.HTTP)._build_cmd()


def test_a_class_s_own_wire_comes_after_the_lanes_so_the_mock_takes_it(monkeypatch):
    monkeypatch.setenv(ENV_MOCK_WIRE, "vllm")
    worker = _worker(ConnectionMode.GRPC, extra_engine_args=["--grpc-wire", "sglang"])
    cmd = worker._build_cmd()
    flags = [cmd[i + 1] for i, flag in enumerate(cmd) if flag == "--grpc-wire"]
    assert flags == ["vllm", "sglang"], "the lane's first, the class's last"


def test_an_unknown_wire_raises(monkeypatch):
    monkeypatch.setenv(ENV_MOCK_WIRE, "trtllm")
    with pytest.raises(ValueError, match="E2E_MOCK_WIRE='trtllm'"):
        get_mock_wire()
    with pytest.raises(ValueError, match="E2E_MOCK_WIRE='trtllm'"):
        _worker(ConnectionMode.GRPC)._build_cmd()
