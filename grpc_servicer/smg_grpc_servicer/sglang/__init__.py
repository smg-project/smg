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
    "HANDSHAKE_ENV",
    "SERVICER_IMPL_ENV",
    "SGLangSchedulerServicer",
    "resolve_servicer_impl",
    "serve_rust",
]

# The flag that selects the Rust request path; see `smg_grpc_servicer.sglang.rust`.
SERVICER_IMPL_ENV = "SMG_SGLANG_SERVICER_IMPL"
# Whoever spawns the scheduler ranks (the Rust servicer, or the standalone
# headless launcher) sets these for them; the ZMQ plugin reads them in each
# rank (see `smg_grpc_servicer.sglang.zmq_plugin`).
HANDSHAKE_ENV = "SMG_SGLANG_ZMQ_HANDSHAKE"
ENGINE_INDEX_ENV = "SMG_SGLANG_ZMQ_ENGINE_INDEX"


def __getattr__(name: str):
    if name in ("resolve_servicer_impl", "serve_rust"):
        from smg_grpc_servicer.sglang import rust

        return getattr(rust, name)
    if name == "SGLangSchedulerServicer":
        from smg_grpc_servicer.sglang.servicer import SGLangSchedulerServicer

        return SGLangSchedulerServicer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
