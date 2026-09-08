"""Argument validation for the standalone two-tier Worker sidecar."""

from unittest.mock import patch

import pytest
from smg import worker_sidecar

_BASE = [
    "--bind-address",
    "127.0.0.1:0",
    "--worker-id",
    "w0",
    "--engine-endpoint",
    "grpc://127.0.0.1:31000",
    "--model-id",
    "org/model",
]


@pytest.mark.parametrize("transport", ["grpc", "zmq"])
def test_sglang_is_rejected_for_every_transport(transport):
    """Same rule as `smg serve`: no launch path serves the service the SGLang
    adapter dials, and it has no health RPC the sidecar could verify."""
    with patch.object(worker_sidecar, "WorkerControlServer") as server:
        with pytest.raises(ValueError, match="sglang"):
            worker_sidecar.main(
                [*_BASE, "--engine-type", "sglang", "--engine-transport", transport]
            )
    server.assert_not_called()


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--drain-secs", "-1"], "drain-secs"),
        (["--engine-count", "0"], "engine-count"),
        (["--max-concurrent-requests", "-1"], "max-concurrent-requests"),
    ],
)
def test_invalid_numeric_arguments_fail_before_the_server_starts(extra, message):
    with patch.object(worker_sidecar, "WorkerControlServer") as server:
        with pytest.raises(ValueError, match=message):
            worker_sidecar.main([*_BASE, "--engine-type", "vllm", *extra])
    server.assert_not_called()
