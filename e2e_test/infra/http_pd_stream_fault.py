"""Judge HTTP relay termination under a deliberately injected engine fault.

An interrupted HTTP stream is a failed request, including a bounded bare EOF.
This oracle never promotes it to a completed request or claims KV reclamation.
"""

from __future__ import annotations

import time


def classify_http_fault_outcome(outcome) -> str:
    """Require an established stream, and distinguish completion from interruption."""
    assert outcome.status == 200, f"stream was not established: {outcome}"
    if outcome.exception is not None:
        # A read timeout merely reaches the client's limit; it does not prove
        # that the relay terminated after the fault. Unrelated exceptions fail.
        assert outcome.exception.startswith(("RemoteProtocolError(", "ReadError(")), (
            f"not a transport interruption: {outcome.exception}"
        )
        return "transport_interruption"
    end = outcome.end
    if end.error is not None:
        assert isinstance(end.error, dict) and any(
            end.error.get(key) for key in ("message", "code", "type")
        ), f"unexplained error frame: {end.error}"
        return "error_frame"
    if end.finish_reason == "abort":
        return "engine_abort"
    if end.finish_reason in ("stop", "length"):
        assert end.done, f"completion missing [DONE]: {end}"
        return "completed"
    assert end.finish_reason is None and not end.done, f"invalid terminal framing: {end}"
    # The reader exhausted the HTTP body without any completion marker. This
    # is observed failure, not a successful completion and not a gRPC error frame.
    return "eof_interruption"


def assert_http_fault_outcomes(outcomes: list) -> list[str]:
    assert outcomes, "no stream outcomes"
    kinds = [classify_http_fault_outcome(outcome) for outcome in outcomes]
    assert any(kind != "completed" for kind in kinds), (
        "all streams completed; the fault was not observed by a live request"
    )
    return kinds


def wait_for_fault_streams(threads: list, *, timeout: float) -> None:
    """Use one absolute deadline for the entire concurrent fleet."""
    deadline = time.monotonic() + timeout
    for thread in threads:
        thread.join(timeout=max(0.0, deadline - time.monotonic()))
    assert not any(thread.is_alive() for thread in threads), (
        f"HTTP streams did not terminate within {timeout:g}s after the fault"
    )
