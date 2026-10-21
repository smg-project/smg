"""The Router's trace context, read off a call's metadata (engine-free)."""

from smg_grpc_servicer.trace_context import trace_headers

TRACEPARENT = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"


def test_trace_context_metadata_becomes_trace_headers():
    metadata = (
        ("user-agent", "grpc-rust/0.1"),
        ("traceparent", TRACEPARENT),
        ("tracestate", "vendor=a"),
    )
    assert trace_headers(metadata) == {"traceparent": TRACEPARENT, "tracestate": "vendor=a"}


def test_a_call_without_the_context_sets_none():
    assert trace_headers((("user-agent", "grpc-rust/0.1"),)) is None
    assert trace_headers(()) is None


def test_binary_metadata_is_not_a_trace_header():
    assert trace_headers((("traceparent", b"binary"),)) is None
