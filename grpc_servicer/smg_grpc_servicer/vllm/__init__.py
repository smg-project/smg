"""vLLM gRPC servicers -- VllmEngine proto service and standard health check.

The servicer classes import vLLM and load on first attribute access, so the
engine-free submodules (media_refs, mm_processor, mm_tensors) import without it.
"""

import logging

__all__ = [
    "MULTIPROC_METHOD_ENV",
    "SERVICER_IMPL_ENV",
    "VllmEngineServicer",
    "VllmHealthServicer",
    "attach_vllm_logging",
    "resolve_servicer_impl",
    "serve_rust",
]

# The flag that selects the Rust request path; see `smg_grpc_servicer.vllm.rust`.
SERVICER_IMPL_ENV = "SMG_VLLM_SERVICER_IMPL"
# vLLM's own knob for how it starts its processes; the engine cores launched
# from the servicer process are spawned unless the deployment chose otherwise
# (see `smg_grpc_servicer.vllm.rust.launch_engine_cores`).
MULTIPROC_METHOD_ENV = "VLLM_WORKER_MULTIPROC_METHOD"


def attach_vllm_logging() -> None:
    """Route this package's logs through vLLM's handlers, once vLLM configured them."""
    vllm_logger = logging.getLogger("vllm")
    if not vllm_logger.handlers:
        return
    pkg_logger = logging.getLogger("smg_grpc_servicer")
    pkg_logger.handlers = list(vllm_logger.handlers)
    pkg_logger.setLevel(vllm_logger.level)
    pkg_logger.propagate = False


def __getattr__(name: str):
    if name in ("resolve_servicer_impl", "serve_rust"):
        from smg_grpc_servicer.vllm import rust

        return getattr(rust, name)
    if name == "VllmEngineServicer":
        from smg_grpc_servicer.vllm.servicer import VllmEngineServicer

        return VllmEngineServicer
    if name == "VllmHealthServicer":
        from smg_grpc_servicer.vllm.health_servicer import VllmHealthServicer

        return VllmHealthServicer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
