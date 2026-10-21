"""A launch writes into the process environment for the engine processes to
inherit: the SGLang rank launch its ZMQ handshake address and engine index,
the vLLM engine-core launch the worker spawn method. The autouse fixture of
this directory's ``conftest`` keeps those writes from reaching the tests
after the one that drove the launch. Two tests, in file order: the first
writes what the launchers write, as they write it (straight into
``os.environ``, past ``monkeypatch``); the second finds the variables unset.
"""

import os

from smg_grpc_servicer import sglang, vllm

LAUNCHER_WRITES = {
    sglang.HANDSHAKE_ENV: "tcp://127.0.0.1:24321",
    sglang.ENGINE_INDEX_ENV: "0",
    vllm.MULTIPROC_METHOD_ENV: "spawn",
}


def test_a_test_that_drives_a_launch_writes_the_launchers_variables():
    for name, value in LAUNCHER_WRITES.items():
        assert name not in os.environ, f"{name} was set before this test ran"
        os.environ[name] = value


def test_the_test_after_it_finds_the_variables_unset():
    leaked = {name: os.environ[name] for name in LAUNCHER_WRITES if name in os.environ}
    assert not leaked, f"the launchers' variables outlived the test that wrote them: {leaked}"
