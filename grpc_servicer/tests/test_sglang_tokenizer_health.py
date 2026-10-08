"""Engine-free coverage for SGLang tokenizer-socket health signals."""

import ast
import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

_REQUEST_MANAGER = Path(__file__).parents[1] / "smg_grpc_servicer" / "sglang" / "request_manager.py"


@dataclass
class HealthCheckOutput:
    rid: str | None = None


def _tokenizer_loop():
    tree = ast.parse(_REQUEST_MANAGER.read_text())
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_handle_tokenizer_loop"
    )
    namespace = {
        "HealthCheckOutput": HealthCheckOutput,
        "get_exception_traceback": lambda: "",
        "logger": logging.getLogger(__name__),
        "real_time": Mock(return_value=42.0),
        "zmq": SimpleNamespace(
            error=SimpleNamespace(
                Again=type("Again", (Exception,), {}),
                ZMQError=type("ZMQError", (Exception,), {}),
            )
        ),
    }
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(_REQUEST_MANAGER), "exec"),
        namespace,
    )
    return namespace["_handle_tokenizer_loop"], namespace["real_time"]


@pytest.mark.parametrize(
    ("rid", "handled"),
    [(None, False), ("", False), ("HEALTH_CHECK_test", True)],
)
def test_tokenizer_health_signal_refreshes_liveness_without_spurious_request(rid, handled):
    loop, real_time = _tokenizer_loop()
    health_output = HealthCheckOutput(rid=rid)

    async def receive():
        manager.gracefully_exit = True
        return health_output

    manager = SimpleNamespace(
        gracefully_exit=False,
        last_receive_tstamp=0.0,
        recv_from_tokenizer=SimpleNamespace(recv_pyobj=receive),
        _dispatch_communicator_output=Mock(return_value=False),
        _handle_health_check_output=AsyncMock(),
    )

    asyncio.run(loop(manager))

    assert manager.last_receive_tstamp == 42.0
    real_time.assert_called_once_with()
    if handled:
        manager._handle_health_check_output.assert_awaited_once_with(health_output)
    else:
        manager._handle_health_check_output.assert_not_awaited()
    manager._dispatch_communicator_output.assert_not_called()
