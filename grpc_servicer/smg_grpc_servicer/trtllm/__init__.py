"""TensorRT-LLM gRPC servicer: the Rust request path over the engine's LLM API.

The engine's own Python servicer lives in the TensorRT-LLM tree
(``tensorrt_llm.grpc.smg``, what ``trtllm-serve --grpc --grpc-protocol smg``
runs). ``--servicer-impl rust`` on the launcher (``python -m
smg_grpc_servicer.trtllm``) or ``SMG_TRTLLM_SERVICER_IMPL=rust`` on the
engine's own entrypoint serves the same contract from Rust; see
:mod:`smg_grpc_servicer.trtllm.rust`.
"""

__all__ = [
    "METRICS_HOST_ENV",
    "METRICS_PORT_ENV",
    "SERVICER_IMPL_ENV",
    "resolve_servicer_impl",
    "serve_rust",
    "serve_rust_from_engine",
]

# The flag that selects the Rust request path; see `smg_grpc_servicer.trtllm.rust`.
SERVICER_IMPL_ENV = "SMG_TRTLLM_SERVICER_IMPL"
# The Rust servicer's Prometheus endpoint: the fallbacks of --metrics-port and
# --metrics-host (see `smg_grpc_servicer.rust_lifecycle.metrics_bind_address`).
METRICS_PORT_ENV = "SMG_TRTLLM_SERVICER_METRICS_PORT"
METRICS_HOST_ENV = "SMG_TRTLLM_SERVICER_METRICS_HOST"


def __getattr__(name: str):
    if name in ("resolve_servicer_impl", "serve_rust", "serve_rust_from_engine"):
        from smg_grpc_servicer.trtllm import rust

        return getattr(rust, name)
    raise AttributeError(name)
