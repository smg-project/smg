"""The three engine servicer classes take the servicer's own arguments the same way.

What every servicer starts with, whichever engine it serves (the listener, the
engines' IPC base and handshake endpoints, the engine count, the tokenizer
directory and the two start-up bounds), is one block in the Rust crate; each
class adds the model facts its engine advertises. These tests pin the three
constructors to that block: the same parameter names and defaults, the same
refusal with the same message for a bad value, before anything is bound or
dialled, and a start and stop on the block alone.
"""

from __future__ import annotations

import inspect

import pytest
from smg.servicer import SglangGrpcServer, TokenSpeedGrpcServer, VllmGrpcServer

SERVERS = [VllmGrpcServer, TokenSpeedGrpcServer, SglangGrpcServer]
# The servicer's own parameters and the default every constructor gives them
# (``inspect.Parameter.empty``: required).
SHARED_DEFAULTS = {
    "bind_address": inspect.Parameter.empty,
    "ipc_base_url": inspect.Parameter.empty,
    "handshake_address": inspect.Parameter.empty,
    "engine_count": 1,
    "tokenizer_dir": None,
    "engine_startup_timeout_secs": None,
    "engine_startup_ceiling_secs": None,
}
# A call that passes the servicer's block and a model path only. The bad values
# below are refused before anything is bound or dialled, so the endpoints need
# not exist.
GOOD = {
    "bind_address": "127.0.0.1:0",
    "ipc_base_url": "ipc:///nonexistent/engine",
    "handshake_address": "ipc:///nonexistent/handshake",
    "model_path": "/nonexistent/model",
}
REFUSED = [
    ("bind_address", "nope", "invalid servicer config: bind_address must be host:port"),
    (
        "ipc_base_url",
        "/nonexistent/engine",
        "invalid servicer config: ipc_base_url must be ipc://<path>",
    ),
    (
        "handshake_address",
        "udp://127.0.0.1:1",
        "invalid servicer config: handshake_address must be tcp://host:port or ipc://<path>",
    ),
    ("engine_count", 0, "invalid servicer config: engine_count must be positive"),
    (
        "engine_startup_timeout_secs",
        -1.0,
        "engine_startup_timeout_secs must be positive, got -1",
    ),
    (
        "engine_startup_ceiling_secs",
        -1.0,
        "engine_startup_ceiling_secs must be positive, or 0 for no ceiling, got -1",
    ),
]


def class_name(cls: type) -> str:
    return cls.__name__


@pytest.mark.parametrize("server", SERVERS, ids=class_name)
def test_every_class_takes_the_servicers_parameters_under_one_name_and_default(server):
    parameters = inspect.signature(server).parameters
    assert {name: parameters[name].default for name in SHARED_DEFAULTS} == SHARED_DEFAULTS
    # The positional ones come first, in one order.
    assert list(parameters)[:4] == [
        "bind_address",
        "ipc_base_url",
        "handshake_address",
        "model_path",
    ]


@pytest.mark.parametrize("server", SERVERS, ids=class_name)
@pytest.mark.parametrize(("name", "value", "message"), REFUSED, ids=[case[0] for case in REFUSED])
def test_a_bad_servicer_argument_is_refused_with_one_message_by_every_class(
    server, name, value, message
):
    with pytest.raises(ValueError) as refused:
        server(**{**GOOD, name: value})
    assert str(refused.value) == message


@pytest.mark.parametrize("server", SERVERS, ids=class_name)
def test_every_class_starts_and_stops_on_the_servicers_block_alone(server, tmp_path):
    # The listener is bound and the engine's handshake endpoint opened under
    # the test's own directory; no engine dials in, and ``stop`` ends the wait.
    started = server(
        "127.0.0.1:0",
        f"ipc://{tmp_path}/engine",
        f"ipc://{tmp_path}/handshake",
        "/nonexistent/model",
        engine_startup_timeout_secs=60.0,
    )
    try:
        host, port = started.address.rsplit(":", 1)
        assert host == "127.0.0.1"
        assert int(port) > 0
        assert started.running
        assert not started.engine_ready
    finally:
        started.stop(timeout_secs=5.0)
    assert not started.running
