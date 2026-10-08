"""Server-Sent Events (SSE) parser.

Handles three SSE protocol variants:
1. Chat Completions: `data: {...}\\n\\n` lines, terminated by `data: [DONE]\\n\\n`
2. Anthropic: `event: type\\ndata: {...}\\n\\n` pairs
3. Responses API: `event: type\\ndata: {...}\\n\\n` pairs, ending with a
   `response.completed`, `response.incomplete`, or `response.failed` event and EOF.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    import httpx


@dataclass
class SseEvent:
    """A parsed SSE event."""

    data: str
    event: str | None = None

    def json(self) -> Any:
        """Parse the data field as JSON."""
        return json.loads(self.data)


def iter_sse_sync(response: httpx.Response) -> Iterator[SseEvent]:
    """Parse SSE events from a synchronous streaming response."""
    event_type: str | None = None
    data_lines: list[str] = []

    for line in response.iter_lines():
        if not line:
            # Empty line = dispatch current event
            if data_lines:
                data = "\n".join(data_lines)
                data_lines.clear()
                if data == "[DONE]":
                    return
                yield SseEvent(data=data, event=event_type)
            event_type = None
            continue

        if line.startswith(":"):
            continue

        if line.startswith("event:"):
            event_type = line[len("event:") :].strip()
            continue

        if line.startswith("data:"):
            value = line[len("data:") :]
            data_lines.append(value[1:] if value.startswith(" ") else value)
            continue

    # Flush any remaining buffered data at stream end.
    if data_lines:
        data = "\n".join(data_lines)
        if data != "[DONE]":
            yield SseEvent(data=data, event=event_type)


async def iter_sse_async(response: httpx.Response) -> AsyncIterator[SseEvent]:
    """Parse SSE events from an async streaming response."""
    event_type: str | None = None
    data_lines: list[str] = []

    async for line in response.aiter_lines():
        if not line:
            if data_lines:
                data = "\n".join(data_lines)
                data_lines.clear()
                if data == "[DONE]":
                    return
                yield SseEvent(data=data, event=event_type)
            event_type = None
            continue

        if line.startswith(":"):
            continue

        if line.startswith("event:"):
            event_type = line[len("event:") :].strip()
            continue

        if line.startswith("data:"):
            value = line[len("data:") :]
            data_lines.append(value[1:] if value.startswith(" ") else value)
            continue

    if data_lines:
        data = "\n".join(data_lines)
        if data != "[DONE]":
            yield SseEvent(data=data, event=event_type)
