"""Rust engine servicers exposed to Python.

:class:`VllmGrpcServer` serves ``vllm.grpc.engine.VllmEngine`` (the contract
the Router already speaks to the Python servicer) over a same-host vLLM
EngineCore, from a Rust-owned thread. Python owns the lifecycle only: launch
the headless engine, construct the server, poll ``engine_ready`` /
``last_error``, announce draining with ``set_serving(False)``, then ``stop``.

The launcher that does all of that is ``python -m smg_grpc_servicer.vllm
--impl rust`` in the ``smg-grpc-servicer`` package.
"""

from smg.smg_rs import VllmGrpcServer, init_servicer_tracing

__all__ = ["VllmGrpcServer", "init_servicer_tracing"]
