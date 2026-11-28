"""Responses API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, overload

from smg_client._helpers import prepare_body
from smg_client._streaming import EventObject, ResponsesAsyncStream, ResponsesSyncStream
from smg_client.types import ResponsesResponse

if TYPE_CHECKING:
    from smg_client._transport import AsyncTransport, SyncTransport


def _input_items_params(
    limit: int | None,
    order: Literal["asc", "desc"] | None,
    after: str | None,
    include: list[str] | None,
) -> dict[str, Any] | None:
    """The list's query: the parameters the caller set, in the public API's order."""
    params: dict[str, Any] = {}
    if limit is not None:
        params["limit"] = limit
    if order is not None:
        params["order"] = order
    if after is not None:
        params["after"] = after
    if include:
        params["include"] = list(include)
    return params or None


# ---------------------------------------------------------------------------
# input_items sub-namespace (matches OpenAI SDK: client.responses.input_items.list(...))
# ---------------------------------------------------------------------------


class SyncInputItems:
    """Synchronous input_items namespace for responses."""

    def __init__(self, transport: SyncTransport) -> None:
        self._transport = transport

    def list(
        self,
        *,
        response_id: str,
        limit: int | None = None,
        order: Literal["asc", "desc"] | None = None,
        after: str | None = None,
        include: list[str] | None = None,
    ) -> EventObject:
        """List input items for a response, one page at a time.

        Returns an object with ``.data`` (the page's items), ``.first_id``,
        ``.last_id`` and ``.has_more``, matching the OpenAI SDK interface::

            items = client.responses.input_items.list(response_id=resp.id)
            while items.has_more:
                items = client.responses.input_items.list(
                    response_id=resp.id, after=items.last_id
                )

        Args:
            response_id: The response ID to list input items for.
            limit: The page size, 1 to 100 (the route's default is 20).
            order: ``asc`` (the order the request gave the items) or ``desc``
                (the route's default, the newest first).
            after: The id of the item the page follows (the previous page's
                ``last_id``).
            include: Extra fields to include, as the route names them.
        """
        resp = self._transport.request(
            "GET",
            f"/v1/responses/{response_id}/input_items",
            params=_input_items_params(limit, order, after, include),
        )
        return EventObject(resp.json())


class AsyncInputItems:
    """Async input_items namespace for responses."""

    def __init__(self, transport: AsyncTransport) -> None:
        self._transport = transport

    async def list(
        self,
        *,
        response_id: str,
        limit: int | None = None,
        order: Literal["asc", "desc"] | None = None,
        after: str | None = None,
        include: list[str] | None = None,
    ) -> EventObject:
        """List input items for a response, one page at a time (see the sync form)."""
        resp = await self._transport.request(
            "GET",
            f"/v1/responses/{response_id}/input_items",
            params=_input_items_params(limit, order, after, include),
        )
        return EventObject(resp.json())


# ---------------------------------------------------------------------------
# Main responses API
# ---------------------------------------------------------------------------


class SyncResponses:
    """Synchronous responses API.

    Matches the OpenAI SDK interface::

        # Non-streaming
        resp = client.responses.create(model="...", input="Hello")
        print(resp.output_text)

        # Streaming
        resp = client.responses.create(model="...", input="Hello", stream=True)
        for event in resp:
            print(event.type)

        retrieved = client.responses.retrieve(response_id=resp.id)
        items = client.responses.input_items.list(response_id=resp.id)
    """

    def __init__(self, transport: SyncTransport) -> None:
        self._transport = transport
        self.input_items = SyncInputItems(transport)

    @overload
    def create(self, *, stream: Literal[False] = ..., **kwargs: Any) -> ResponsesResponse: ...
    @overload
    def create(self, *, stream: Literal[True], **kwargs: Any) -> ResponsesSyncStream: ...

    def create(self, **kwargs: Any) -> ResponsesResponse | ResponsesSyncStream:
        """Create a response.

        Args:
            **kwargs: ResponsesRequest fields (model, input, etc.)
                stream: If True, returns a stream of SSE events. Default False.
                extra_body: Dict merged into the request body.
                extra_headers: Dict merged into request headers.

        Returns:
            ResponsesResponse for non-streaming, ResponsesSyncStream for streaming.
        """
        is_stream = kwargs.pop("stream", False)
        body, extra_headers = prepare_body(kwargs)

        if is_stream:
            body["stream"] = True
            resp = self._transport.request(
                "POST", "/v1/responses", json=body, stream=True, headers=extra_headers
            )
            return ResponsesSyncStream(resp)

        resp = self._transport.request("POST", "/v1/responses", json=body, headers=extra_headers)
        return ResponsesResponse.model_validate_json(resp.content)

    def retrieve(self, *, response_id: str) -> ResponsesResponse:
        """Retrieve a response by ID.

        Args:
            response_id: The response ID to retrieve.
        """
        resp = self._transport.request("GET", f"/v1/responses/{response_id}")
        return ResponsesResponse.model_validate_json(resp.content)

    def get(self, response_id: str) -> ResponsesResponse:
        """Retrieve a response by ID (backward-compat alias).

        Prefer ``retrieve(response_id=...)`` for OpenAI SDK compatibility.
        """
        return self.retrieve(response_id=response_id)

    def delete(self, response_id: str) -> None:
        """Delete a response. Note: not yet implemented on the server (returns 501)."""
        self._transport.request("DELETE", f"/v1/responses/{response_id}")

    def cancel(self, response_id: str) -> ResponsesResponse:
        resp = self._transport.request("POST", f"/v1/responses/{response_id}/cancel", json={})
        return ResponsesResponse.model_validate_json(resp.content)

    def list_input_items(self, response_id: str) -> EventObject:
        """List input items (backward-compat alias).

        Prefer ``input_items.list(response_id=...)`` for OpenAI SDK compatibility.
        """
        return self.input_items.list(response_id=response_id)


class AsyncResponses:
    """Async responses API."""

    def __init__(self, transport: AsyncTransport) -> None:
        self._transport = transport
        self.input_items = AsyncInputItems(transport)

    @overload
    async def create(self, *, stream: Literal[False] = ..., **kwargs: Any) -> ResponsesResponse: ...
    @overload
    async def create(self, *, stream: Literal[True], **kwargs: Any) -> ResponsesAsyncStream: ...

    async def create(self, **kwargs: Any) -> ResponsesResponse | ResponsesAsyncStream:
        """Create a response."""
        is_stream = kwargs.pop("stream", False)
        body, extra_headers = prepare_body(kwargs)

        if is_stream:
            body["stream"] = True
            resp = await self._transport.request(
                "POST", "/v1/responses", json=body, stream=True, headers=extra_headers
            )
            return ResponsesAsyncStream(resp)

        resp = await self._transport.request(
            "POST", "/v1/responses", json=body, headers=extra_headers
        )
        return ResponsesResponse.model_validate_json(resp.content)

    async def retrieve(self, *, response_id: str) -> ResponsesResponse:
        """Retrieve a response by ID."""
        resp = await self._transport.request("GET", f"/v1/responses/{response_id}")
        return ResponsesResponse.model_validate_json(resp.content)

    async def get(self, response_id: str) -> ResponsesResponse:
        """Retrieve a response by ID (backward-compat alias)."""
        return await self.retrieve(response_id=response_id)

    async def delete(self, response_id: str) -> None:
        """Delete a response."""
        await self._transport.request("DELETE", f"/v1/responses/{response_id}")

    async def cancel(self, response_id: str) -> ResponsesResponse:
        resp = await self._transport.request("POST", f"/v1/responses/{response_id}/cancel", json={})
        return ResponsesResponse.model_validate_json(resp.content)

    async def list_input_items(self, response_id: str) -> EventObject:
        """List input items (backward-compat alias)."""
        return await self.input_items.list(response_id=response_id)
