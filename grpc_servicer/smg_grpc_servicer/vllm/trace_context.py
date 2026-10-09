"""The Router's W3C trace context, from the call's metadata to the engine's tracer."""

from __future__ import annotations

from collections.abc import Iterable

#: The W3C Trace Context headers, as the Router injects them into gRPC metadata.
TRACE_CONTEXT_KEYS = ("traceparent", "tracestate")


def trace_headers(metadata: Iterable[tuple[str, str | bytes]]) -> dict[str, str] | None:
    """The trace context among the call's ``invocation_metadata()``, in the shape
    vLLM's ``generate(trace_headers=...)`` takes, or ``None`` when the call
    carries none (vLLM then starts no span for the request)."""
    headers = {
        key: value
        for key, value in metadata
        if key in TRACE_CONTEXT_KEYS and isinstance(value, str)
    }
    return headers or None
