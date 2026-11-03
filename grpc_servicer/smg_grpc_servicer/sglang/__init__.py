"""SGLang gRPC servicer — implements SglangScheduler proto service.

The servicer class is imported lazily: SGLang loads this package's plugin
submodules (``zmq_plugin``, ``plugin``) in every scheduler process, which
must not pull the gRPC server and its dependencies along.

``--servicer-impl rust`` on SGLang's launcher (a flag this package adds through
its SGLang plugin) or ``SMG_SGLANG_SERVICER_IMPL=rust`` serves the same
contract from Rust; see :mod:`smg_grpc_servicer.sglang.rust`.
"""

__all__ = [
    "ENGINE_INDEX_ENV",
    "GRAMMAR_ATTRIBUTION_ENV",
    "GRAMMAR_REJECTED_ERR_TYPE",
    "GRAMMAR_REJECTED_MESSAGE",
    "HANDSHAKE_ENV",
    "METRICS_HOST_ENV",
    "METRICS_PORT_ENV",
    "SERVICER_IMPL_ENV",
    "SGLangSchedulerServicer",
    "resolve_servicer_impl",
    "serve_rust",
]

# The flag that selects the Rust request path; see `smg_grpc_servicer.sglang.rust`.
SERVICER_IMPL_ENV = "SMG_SGLANG_SERVICER_IMPL"
# The Rust servicer's Prometheus endpoint: the fallbacks of --metrics-port and
# --metrics-host (see `smg_grpc_servicer.rust_lifecycle.metrics_bind_address`).
METRICS_PORT_ENV = "SMG_SGLANG_SERVICER_METRICS_PORT"
METRICS_HOST_ENV = "SMG_SGLANG_SERVICER_METRICS_HOST"
# Whoever spawns the scheduler ranks (the Rust servicer, or the standalone
# headless launcher) sets these for them; the ZMQ plugin reads them in each
# rank (see `smg_grpc_servicer.sglang.zmq_plugin`).
HANDSHAKE_ENV = "SMG_SGLANG_ZMQ_HANDSHAKE"
ENGINE_INDEX_ENV = "SMG_SGLANG_ZMQ_ENGINE_INDEX"
# The Python servicer sets this for the scheduler ranks it spawns, so the
# plugin's grammar-attribution hooks register without the ZMQ handshake (the
# headless launcher's HANDSHAKE_ENV implies them).
GRAMMAR_ATTRIBUTION_ENV = "SMG_SGLANG_ATTRIBUTE_GRAMMAR_ABORTS"
# The `err_type` the plugin stamps on the scheduler's abort of a request whose
# grammar rejected a sampled token (the scheduler leaves that finish a bare
# `FINISH_ABORT()`, which nothing tells from an abort-all's), and the message
# it gives the bare one. The servicers read the name as the one abort the
# engine attributes to the request's constraint; the Rust ZMQ wire reads the
# same name (`GRAMMAR_REJECTED_ERR_TYPE` in `crates/engine_zmq_adapter`).
GRAMMAR_REJECTED_ERR_TYPE = "GrammarRejectedToken"
GRAMMAR_REJECTED_MESSAGE = "the grammar rejected a sampled token"


def __getattr__(name: str):
    if name in ("resolve_servicer_impl", "serve_rust"):
        from smg_grpc_servicer.sglang import rust

        return getattr(rust, name)
    if name == "SGLangSchedulerServicer":
        from smg_grpc_servicer.sglang.servicer import SGLangSchedulerServicer

        return SGLangSchedulerServicer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
