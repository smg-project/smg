"""Engine-client admin operations, kept independent of vLLM imports."""

import asyncio
import logging
import math

import grpc
from smg_grpc_proto.generated import common_pb2

logger = logging.getLogger(__name__)

_RESET_RPC_TIMEOUT_S = 30.0
_RETRY_INTERVAL_S = 0.1


async def _reset_prefix_cache(engine) -> bool:
    core = getattr(engine, "engine_core", None)
    identities = getattr(core, "core_engines", None)
    if identities is None or len(identities) == 1:
        # Single-rank clients (including external DP load balancing) expose
        # the result of their only managed engine through the public API.
        return await engine.reset_prefix_cache(reset_running_requests=False, reset_connector=False)

    # vLLM 0.19–0.27 DPLBAsyncMPClient.call_utility_async broadcasts but
    # discards every result except rank 0's. Use its per-engine IPC helper
    # until the public API aggregates reset results. Never fall back to that
    # lossy API if this internal interface changes.
    call = getattr(core, "_call_utility_async", None)
    if not identities or not callable(call):
        raise RuntimeError("vLLM client cannot report per-rank cache reset results")
    results = await asyncio.gather(
        *(
            call("reset_prefix_cache", False, False, engine=identity)
            for identity in tuple(identities)
        ),
        return_exceptions=True,
    )
    for result in results:
        if isinstance(result, BaseException):
            raise result
    return all(result is True for result in results)


async def flush_cache(engine, request, context) -> common_pb2.FlushCacheResponse:
    """Invalidate local prefix entries; do not preempt or reset external caches.

    Every managed rank must accept the same attempt; this is not an atomic drain.
    """
    timeout_s = request.timeout_s
    if not math.isfinite(timeout_s) or timeout_s < 0:
        context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
        context.set_details("timeout_s must be finite and non-negative")
        return common_pb2.FlushCacheResponse(success=False, message="Invalid timeout_s")

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    try:
        while True:
            # Zero means one immediate attempt, but still bound an unresponsive
            # engine. Positive timeouts bound the entire wait, including IPC.
            budget = _RESET_RPC_TIMEOUT_S if timeout_s == 0 else deadline - loop.time()
            rpc_remaining = context.time_remaining()
            if rpc_remaining is not None:
                budget = min(budget, rpc_remaining)
            if budget <= 0:
                raise TimeoutError
            success = await asyncio.wait_for(
                _reset_prefix_cache(engine),
                timeout=budget,
            )
            if success:
                return common_pb2.FlushCacheResponse(
                    success=True, message="Local KV prefix cache flushed successfully"
                )
            if timeout_s == 0:
                return common_pb2.FlushCacheResponse(
                    success=False,
                    message="KV prefix cache reset refused; requests may be in flight",
                )
            await asyncio.sleep(min(_RETRY_INTERVAL_S, max(0.0, deadline - loop.time())))
    # asyncio.TimeoutError is a separate exception on Python 3.10.
    except (TimeoutError, asyncio.TimeoutError):  # noqa: UP041
        message = "Flush cache timed out; the engine may still complete an issued reset"
        context.set_code(grpc.StatusCode.DEADLINE_EXCEEDED)
        context.set_details(message)
        return common_pb2.FlushCacheResponse(success=False, message=message)
    except Exception as exc:
        logger.exception("FlushCache failed")
        message = f"Flush cache failed: {exc}"
        context.set_code(grpc.StatusCode.INTERNAL)
        context.set_details(message)
        return common_pb2.FlushCacheResponse(success=False, message=message)
