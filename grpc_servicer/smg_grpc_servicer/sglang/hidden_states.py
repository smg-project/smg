"""Hidden states on the Python SGLang servicer: the request's flag goes to the
scheduler as it stands (full capture; the gRPC contract carries a bool, so the
engine's ``"last"`` mode is not asked for), the engine's launch-time cap is kept
the way its own frontend keeps it, and the scheduler's cumulative rows become
the proto's per-step delta (``GenerateStreamChunk.hidden_states``, the step's
new rows flattened) and the whole list on completion
(``GenerateComplete.all_hidden_states``, one ``HiddenStates`` per position a
forward ran on: the prompt's, then each generated token fed back, the last
sampled token having none; ``layer = -1`` for the final hidden layer). The Rust path maps the
same wire the same way (``crates/engine_zmq_adapter/src/sglang.rs``); the two
agree row for row.
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
    """The scheduler's hidden states of one request as rows of floats, one per
    position a forward ran on. The engine appends the prefill's block as ONE
    element (a list of rows, one per prompt position) and then one row per
    decode step, the generated token it fed back
    (``batch_result_processor._append_prefill_hidden_states`` /
    ``_append_decode_hidden_states``): the last sampled token is never fed
    back, so a request has prompt + generated - 1 rows; the engine's
    ``"last"`` mode keeps a single row.
    Every element may be a list or a tensor-like with ``tolist``; None or an
    empty value is no row."""
    if value is None:
        return []
    elements = _listed(value)
    rows: list[list[float]] = []
    for element in elements:
        element = _listed(element)
        if isinstance(element, (int, float)):
            # A bare vector given as its scalars: one row.
            return [[float(x) for x in elements]]
        if element and _is_vector(element[0]):
            # A block of rows: the prefill's prompt positions.
            rows.extend([float(x) for x in _listed(row)] for row in element)
        else:
            rows.append([float(x) for x in element])
    return rows


def _listed(value: Any) -> Any:
    return value.tolist() if hasattr(value, "tolist") else value


def _is_vector(item: Any) -> bool:
    return isinstance(item, (list, tuple)) or hasattr(item, "tolist")


def new_rows(cumulative: Sequence[Sequence[float]], already_sent: int) -> list[list[float]]:
    """The rows a step adds: the scheduler ships the whole list so far, the
    frontend emits what the client has not seen."""
    return [list(row) for row in cumulative[already_sent:]]


def flat_hidden_states(rows: Iterable[Sequence[float]] | None) -> list[float]:
    """``GenerateStreamChunk.hidden_states``: the step's rows flattened (the
    client splits them by the hidden size)."""
    if not rows:
        return []
    return [float(x) for row in rows for x in row]


def all_hidden_states_proto(rows: Iterable[Sequence[float]] | None) -> list[Any]:
    """``GenerateComplete.all_hidden_states``: one entry per position a forward
    ran on (the prompt's positions first, as the engine returns them, then
    each generated token fed back; the last sampled token has none), the
    final layer, the position as its index."""
    from smg_grpc_proto import sglang_scheduler_pb2

    if not rows:
        return []
    return [
        sglang_scheduler_pb2.HiddenStates(
            values=[float(x) for x in row], layer=FINAL_LAYER, position=position
        )
        for position, row in enumerate(rows)
    ]
