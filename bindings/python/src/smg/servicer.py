"""Rust engine servicers exposed to Python, one class per engine servicer.

:class:`VllmGrpcServer`, the first, serves ``vllm.grpc.engine.VllmEngine`` (the contract
the Router already speaks to the Python servicer) over a same-host vLLM
EngineCore, from a Rust-owned thread. Python owns the lifecycle only: launch
the headless engine, construct the server, poll ``engine_ready`` /
``last_error``, close admission with ``set_serving(False)``, then wait for
``active_requests`` to reach zero (or the drain deadline) before ``stop``.
The count includes aborted requests until their response streams are released.
``set_ready(False)`` holds the health answer alone, admission open to the
launcher's own requests before SERVING (its planned warm-up's, ``smg-warmup-<n>``)
and to nothing else; ``set_ready(True)`` releases it. ``kv_relay_subscribed``
says whether the KV-event relay's subscription reached the engine's publisher
(or there is none to wait for), which the launcher reads before it starts the
warm-up.
Worker-side media processing is the one request-time crossing, when the
Python processors are chosen: a ``media_processor`` object
(``smg_grpc_servicer.vllm.rust_media``) runs them for a request's
``media_refs``. The alternative, ``smg_media_processor`` (a dict of settings),
runs smg's own media pipeline inside the server with no Python on the path.

The launcher that does all of that is ``serve_rust`` in
``smg_grpc_servicer.vllm.rust``, what upstream vLLM's gRPC entrypoint runs by
default (``SMG_VLLM_SERVICER_IMPL=python`` selects the deprecated Python
servicer instead).

:class:`TokenSpeedGrpcServer` serves ``tokenspeed.grpc.scheduler.TokenSpeedScheduler``
over same-host headless TokenSpeed scheduler(s) on their msgpack ZMQ wire, with the
same lifecycle shape; its launcher is ``serve_rust`` in
``smg_grpc_servicer.tokenspeed.rust``, what ``python -m smg_grpc_servicer.tokenspeed``
runs by default (``SMG_TOKENSPEED_SERVICER_IMPL=python`` selects the deprecated
Python servicer instead).
"""

from smg.smg_rs import (
    SglangGrpcServer,
    TokenSpeedGrpcServer,
    VllmGrpcServer,
    init_servicer_tracing,
)

__all__ = ["SglangGrpcServer", "TokenSpeedGrpcServer", "VllmGrpcServer", "init_servicer_tracing"]
