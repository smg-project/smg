"""SGLang gRPC servicer — implements SglangScheduler proto service.

The servicer class is imported lazily: SGLang loads this package's plugin
submodule (``zmq_plugin``) in every scheduler process, which must not pull
the gRPC server and its dependencies along.
"""

__all__ = ["SGLangSchedulerServicer"]


def __getattr__(name: str):
    if name == "SGLangSchedulerServicer":
        from smg_grpc_servicer.sglang.servicer import SGLangSchedulerServicer

        return SGLangSchedulerServicer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
