"""Unit tests for connection-mode/engine validation (no GPU)."""

from __future__ import annotations

import pytest
from infra.constants import ConnectionMode, Runtime

# setup_backend pulls in the cloud SDKs; skip if the env lacks them.
setup_backend = pytest.importorskip("fixtures.setup_backend")


@pytest.mark.parametrize("engine", [Runtime.VLLM.value, Runtime.TOKENSPEED.value])
def test_zmq_allowed_for_capable_engines(engine):
    # Does not raise.
    setup_backend._validate_connection_mode(ConnectionMode.ZMQ, engine)


@pytest.mark.parametrize(
    "engine",
    [Runtime.SGLANG.value, Runtime.TRTLLM.value, Runtime.MLX.value],
)
def test_zmq_rejected_for_incapable_engines(engine):
    with pytest.raises(ValueError, match="ConnectionMode.ZMQ is only supported"):
        setup_backend._validate_connection_mode(ConnectionMode.ZMQ, engine)


@pytest.mark.parametrize("mode", [ConnectionMode.GRPC, ConnectionMode.HTTP])
@pytest.mark.parametrize(
    "engine",
    [Runtime.SGLANG.value, Runtime.VLLM.value, Runtime.TOKENSPEED.value],
)
def test_non_zmq_modes_accept_any_engine(mode, engine):
    # Non-ZMQ wires impose no engine restriction.
    setup_backend._validate_connection_mode(mode, engine)
