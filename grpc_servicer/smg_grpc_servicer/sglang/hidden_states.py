"""Hidden states and the Python SGLang servicer: a request's
``return_hidden_states`` is refused here, ahead of the scheduler, as
``INVALID_ARGUMENT``: first under the engine's launch-time cap, in its own
frontend's words, then by name: hidden states are served by the Rust servicer
(``crates/engine_servicer``; the worker runs it with ``--servicer-impl rust``
or ``SMG_SGLANG_SERVICER_IMPL=rust``), and this servicer, which once accepted
the flag and returned nothing, says so.

``new_hidden_rows`` belongs to the scheduler-side plugin (``zmq_msgpack.py``):
it turns the engine's cumulative per-request list into the rows a step added,
the column the Rust servicer reads off the wire.
"""

from __future__ import annotations

from typing import Any

RUST_SERVICER_WORDS = (
    "Hidden states are served by the Rust servicer: run the worker with "
    "`--servicer-impl rust` (or SMG_SGLANG_SERVICER_IMPL=rust). The Python "
    "servicer does not return them."
)


def refuse_hidden_states_above_server_mode() -> None:
    """Raise ``ValueError`` with the engine's own words when the server was not
    launched to capture hidden states for every token
    (``--return-hidden-states-mode full``): SGLang's frontend refuses the same
    request before the scheduler sees it, and so does this servicer. An engine
    release without the helper has no cap to keep."""
    try:
        from sglang.srt.model_executor.forward_batch_info import (
            CaptureHiddenMode,
            get_server_return_hidden_states_mode,
        )
    except ImportError:  # pragma: no cover - older engine releases
        return
    server_mode = get_server_return_hidden_states_mode()
    if server_mode >= CaptureHiddenMode.FULL:
        return
    if server_mode.need_capture():
        raise ValueError(
            "The requested return_hidden_states mode exceeds the server maximum "
            f"`{server_mode.name.lower()}`. Please launch with "
            "`--return-hidden-states-mode full` to allow return_hidden_states=True."
        )
    raise ValueError(
        "The server is not configured to return hidden states. Please launch with "
        "`--return-hidden-states-mode full` to allow return_hidden_states=True."
    )


def refuse_hidden_states_on_this_servicer() -> None:
    """Raise ``ValueError`` (the servicer answers ``INVALID_ARGUMENT``) for a
    request that asks for hidden states: this servicer does not return them,
    the Rust one does, and the message names it. Called after the launch-time
    cap, so an engine that cannot capture them says so first, in its own
    words."""
    raise ValueError(RUST_SERVICER_WORDS)


def new_hidden_rows(value: Any, consumed: int) -> tuple[list[list[float]], int]:
    """The rows held by the elements of ``value`` past the first ``consumed``,
    and the element count a caller remembers for the next step.

    The engine keeps a request's hidden states cumulative and appends one
    element per forward: the prefill's block (a list of rows, one per prompt
    position the forward computed: a prefix the radix cache served has none),
    then a decode step's rows, one per token it fed back (several at once
    under speculative decoding)
    (``batch_result_processor._append_prefill_hidden_states`` /
    ``_append_decode_hidden_states``): the last sampled token is never fed
    back, so a request has (prompt - cached prefix) + generated - 1 rows. A
    caller that remembers the elements it consumed gets the rest reshaped,
    O(new rows) per step (this runs on the scheduler's loop for every step of
    every request that asked); the values are the floats ``tolist`` produced,
    no element is converted. Every element may be a list or a tensor-like
    with ``tolist``; None or an empty value adds no row. The engine's
    ``"last"`` mode keeps a single row as a bare vector: one row, consumed
    whole."""
    if value is None:
        return [], consumed
    elements = _listed(value)
    if elements and isinstance(elements[0], (int, float)):
        # A bare vector given as its scalars: one row.
        return ([] if consumed else [list(elements)]), len(elements)
    rows: list[list[float]] = []
    for element in elements[consumed:]:
        element = _listed(element)
        if element and _is_vector(element[0]):
            # A block of rows: the prompt positions the prefill computed.
            rows.extend(_listed(row) for row in element)
        else:
            rows.append(element)
    return rows, len(elements)


def _listed(value: Any) -> Any:
    return value.tolist() if hasattr(value, "tolist") else value


def _is_vector(item: Any) -> bool:
    return isinstance(item, (list, tuple)) or hasattr(item, "tolist")
