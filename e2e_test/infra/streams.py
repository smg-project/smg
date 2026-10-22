"""Collectors for streamed responses.

The one place the suites turn a stream into the pieces they assert on: a chat
completion's text, its tool calls assembled from their deltas and its finish
reason; a Responses stream's events and the response its ``response.completed``
event carries; a chat completion's raw SSE lines, for the way a stream ended
(its finish reason, the gateway's error frame, the closing ``[DONE]``).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StreamedToolCall:
    """One tool call of the first choice, assembled from its deltas.

    ``name`` and ``id`` are the last values a delta carried, ``arguments`` the
    fragments joined in the order they came.
    """

    index: int
    id: str | None = None
    name: str = ""
    arguments: str = ""


@dataclass
class StreamedChat:
    """A chat completion stream read to its end."""

    chunks: list[Any]
    """Every chunk, in order."""
    content: str
    """Choice 0's text deltas, joined."""
    tool_calls: list[StreamedToolCall]
    """Choice 0's tool calls, in index order."""
    tool_call_deltas: list[Any] = field(default_factory=list)
    """Choice 0's tool-call deltas as they came."""
    finish_reason: str | None = None
    """The last finish reason choice 0 reported."""


def collect_streamed_chat(stream: Iterable[Any]) -> StreamedChat:
    """Read a chat completion stream to its end and assemble choice 0.

    A chunk without choice 0 (a usage chunk, another choice of an ``n > 1``
    request) is kept but contributes nothing.
    """
    chunks = list(stream)
    content_parts: list[str] = []
    calls: dict[int, StreamedToolCall] = {}
    deltas: list[Any] = []
    finish_reason = None
    for chunk in chunks:
        choice = next((c for c in chunk.choices if c.index == 0), None)
        if choice is None:
            continue
        if choice.finish_reason:
            finish_reason = choice.finish_reason
        if choice.delta.content:
            content_parts.append(choice.delta.content)
        for delta in choice.delta.tool_calls or []:
            deltas.append(delta)
            call = calls.setdefault(delta.index, StreamedToolCall(index=delta.index))
            if delta.id:
                call.id = delta.id
            if delta.function:
                if delta.function.name:
                    call.name = delta.function.name
                if delta.function.arguments:
                    call.arguments += delta.function.arguments
    return StreamedChat(
        chunks=chunks,
        content="".join(content_parts),
        tool_calls=[calls[index] for index in sorted(calls)],
        tool_call_deltas=deltas,
        finish_reason=finish_reason,
    )


@dataclass
class SseChatEnd:
    """How a chat completion's raw SSE stream ended.

    Read off the ``data:`` lines, so a stream the gateway failed after its
    headers is seen the way a client sees it: the chunks before the failure,
    then one frame carrying an ``error`` object (its ``type`` and ``code``
    name the failure), then ``[DONE]``.
    """

    chunks: int = 0
    """Frames with choices, in order of arrival."""
    finish_reason: str | None = None
    """The last finish reason any choice reported."""
    error: dict[str, Any] | None = None
    """The error frame's body, when the stream failed."""
    done: bool = False
    """Whether ``[DONE]`` closed the stream."""


def read_sse_chat(
    lines: Iterable[str], *, on_first_chunk: Callable[[], None] | None = None
) -> SseChatEnd:
    """Read a chat completion's SSE lines to their end; ``on_first_chunk`` runs
    once, at the first frame with choices. Lines that are not ``data:``
    (event names, comments, blanks) and frames that are not JSON are
    passed over."""
    end = SseChatEnd()
    for line in lines:
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if payload == "[DONE]":
            end.done = True
            break
        try:
            frame = json.loads(payload)
        except ValueError:
            continue
        if not isinstance(frame, dict):
            continue
        if isinstance(frame.get("error"), dict):
            end.error = frame["error"]
            continue
        choices = frame.get("choices")
        if not isinstance(choices, list) or not choices:
            continue  # a usage-only frame carries no choice
        end.chunks += 1
        if end.chunks == 1 and on_first_chunk is not None:
            on_first_chunk()
        for choice in choices:
            if isinstance(choice, dict) and choice.get("finish_reason"):
                end.finish_reason = choice["finish_reason"]
    return end


@dataclass
class StreamedResponse:
    """A Responses stream read to its end: every event, in order."""

    events: list[Any]

    @property
    def event_types(self) -> list[str]:
        """The events' types, in order (one entry per event)."""
        return [event.type for event in self.events]

    @property
    def completed_events(self) -> list[Any]:
        """The ``response.completed`` events (one, in a stream that ended well)."""
        return [event for event in self.events if event.type == "response.completed"]

    @property
    def final_response(self) -> Any:
        """The response the ``response.completed`` event carries; there must be one."""
        completed = self.completed_events
        assert completed, "no response.completed event in the stream"
        return completed[0].response


def collect_streamed_response(stream: Iterable[Any]) -> StreamedResponse:
    """Read a Responses stream to its end."""
    return StreamedResponse(events=list(stream))
