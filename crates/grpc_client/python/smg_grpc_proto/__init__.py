"""SMG gRPC Proto - Protocol definitions for SGLang, TokenSpeed, vLLM, TRT-LLM, and MLX."""

from importlib import import_module
from importlib.metadata import version

__version__ = version("smg-grpc-proto")

_GENERATED_MODULES = {
    "sglang_scheduler_pb2",
    "sglang_scheduler_pb2_grpc",
    "sglang_encoder_pb2",
    "sglang_encoder_pb2_grpc",
    "tokenspeed_encoder_pb2",
    "tokenspeed_encoder_pb2_grpc",
    "tokenspeed_scheduler_pb2",
    "tokenspeed_scheduler_pb2_grpc",
    "vllm_engine_pb2",
    "vllm_engine_pb2_grpc",
    "trtllm_service_pb2",
    "trtllm_service_pb2_grpc",
    "mlx_engine_pb2",
    "mlx_engine_pb2_grpc",
    "worker_control_pb2",
    "worker_control_pb2_grpc",
    "worker_inference_pb2",
    "worker_inference_pb2_grpc",
}

__all__ = sorted(_GENERATED_MODULES)


def __getattr__(name: str):
    if name not in _GENERATED_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(f".generated.{name}", __name__)
    globals()[name] = module
    return module


def __dir__():
    return sorted(set(globals()) | _GENERATED_MODULES)
