"""TokenSpeed gRPC servicer — wraps :class:`AsyncLLM` behind the gRPC wire.

``--servicer-impl rust`` on the launcher (``python -m smg_grpc_servicer.tokenspeed``)
or ``SMG_TOKENSPEED_SERVICER_IMPL=rust`` serves the same contract from Rust; see
:mod:`smg_grpc_servicer.tokenspeed.rust`.
"""

__all__ = [
    "SERVICER_IMPL_ENV",
    "TokenSpeedSchedulerServicer",
    "resolve_servicer_impl",
    "serve_rust",
]

# The flag that selects the Rust request path; see `smg_grpc_servicer.tokenspeed.rust`.
SERVICER_IMPL_ENV = "SMG_TOKENSPEED_SERVICER_IMPL"


def __getattr__(name: str):
    if name in ("resolve_servicer_impl", "serve_rust"):
        from smg_grpc_servicer.tokenspeed import rust

        return getattr(rust, name)
    if name == "TokenSpeedSchedulerServicer":
        from smg_grpc_servicer.tokenspeed.servicer import TokenSpeedSchedulerServicer

        return TokenSpeedSchedulerServicer
    raise AttributeError(name)
