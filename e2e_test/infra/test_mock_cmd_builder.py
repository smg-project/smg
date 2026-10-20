"""The mock engine's worker command: the scripted mock worker on each wire."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from infra.constants import DEFAULT_HOST, LOCAL_RUNTIMES, RUNTIME_LABELS, ConnectionMode, Runtime
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
