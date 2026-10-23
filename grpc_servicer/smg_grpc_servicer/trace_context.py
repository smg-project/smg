"""The Router's W3C trace context, from a call's gRPC metadata to the engine's tracer.

Every servicer reads it the same way; each hands it to its engine the way the
engine's own frontend does, so the engine's spans for the request join the
Router's trace.
"""

from __future__ import annotations

from collections.abc import Iterable

#: The W3C Trace Context headers, as the Router injects them into gRPC metadata.
TRACE_CONTEXT_KEYS = ("traceparent", "tracestate")


def trace_headers(metadata: Iterable[tuple[str, str | bytes]]) -> dict[str, str] | None:
    """The trace context among the call's ``invocation_metadata()``, keyed by
    header name (the shape an engine's tracer takes as external headers), or
    ``None`` when the call carries none (the engine then starts no span, or a
    trace of its own, for the request)."""
    headers = {
        key: value
        for key, value in metadata
        if key in TRACE_CONTEXT_KEYS and isinstance(value, str)
    }
    return headers or None
