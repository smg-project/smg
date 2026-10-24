"""Hidden states on the Python SGLang servicer: the request's flag goes to the
scheduler as it stands (full capture; the gRPC contract carries a bool, so the
engine's ``"last"`` mode is not asked for), the engine's launch-time cap is kept
the way its own frontend keeps it, and the scheduler's cumulative rows become
the proto's per-step delta (``GenerateStreamChunk.hidden_states``, the step's
new rows flattened) and the whole list on completion
(``GenerateComplete.all_hidden_states``, one ``HiddenStates`` per position a
forward ran on: the prompt positions the prefill computed, then each generated
token fed back, the last sampled token having none; ``layer = -1`` for the
final hidden layer; the positions absolute, anchored from the engine's own
counters because a prefix the radix cache served has no rows). The Rust path
maps the same wire the same way (``crates/engine_zmq_adapter/src/sglang.rs``);
the two agree row for row.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

FINAL_LAYER = -1


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


def hidden_state_rows(value: Any) -> list[list[float]]:
    """All rows of one request's hidden states (``new_hidden_rows`` from the
    first element)."""
    rows, _ = new_hidden_rows(value, 0)
    return rows


def new_hidden_rows(value: Any, consumed: int) -> tuple[list[list[float]], int]:
    """The rows held by the elements of ``value`` past the first ``consumed``,
    and the element count a caller remembers for the next step.

    The engine keeps a request's hidden states cumulative and appends one
    element per forward: the prefill's block (a list of rows, one per prompt
    position the forward computed: a prefix the radix cache served has none),
    then one row per decode step, the generated token it fed back
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


def first_hidden_position(prompt_tokens: int, completion_tokens: int, rows: int) -> int:
    """The absolute position of the first row a request carries, from the
    engine's own counters: the last row is always the last fed-back token
    (position prompt + generated - 2), whatever prefix the radix cache served
    without rows, so the first is that position less the rows after it. The
    Rust servicer anchors the same way."""
    return max(0, int(prompt_tokens) + int(completion_tokens) - 1 - int(rows))


def flat_hidden_states(rows: Iterable[Sequence[float]] | None) -> list[float]:
    """``GenerateStreamChunk.hidden_states``: the step's rows flattened (the
    client splits them by the hidden size)."""
    if not rows:
        return []
    return [x for row in rows for x in row]


def all_hidden_states_proto(
    rows: Sequence[Sequence[float]] | None, first_position: int = 0
) -> list[Any]:
    """``GenerateComplete.all_hidden_states``: one entry per position a forward
    ran on (the prompt positions the prefill computed, then each generated
    token fed back; the last sampled token has none), the final layer, the
    positions absolute from ``first_position`` (``first_hidden_position``)."""
    from smg_grpc_proto import sglang_scheduler_pb2

    if not rows:
        return []
    return [
        sglang_scheduler_pb2.HiddenStates(
            values=row, layer=FINAL_LAYER, position=first_position + i
        )
        for i, row in enumerate(rows)
    ]
