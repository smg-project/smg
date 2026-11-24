"""Optional SMG extension to vLLM's HTTP model list; no engine imports.

Enable with --middleware smg_grpc_servicer.http_metadata.PairingProtocolMiddleware.
Only the engine process's nonblank SMG_PAIRING_PROTOCOL is exposed. This is an
SMG metadata extension, not a field stock vLLM promises to emit.
"""

from __future__ import annotations

import json
from typing import Any

from .pd_pairing import pairing_protocol_from_env

MAX_METADATA_BYTES = 1024 * 1024


class PairingProtocolMiddleware:
    """Plain ASGI middleware: inference/streaming never enters the buffer."""

    def __init__(self, app):
        self.app = app
        self.protocol = pairing_protocol_from_env()

    async def __call__(self, scope, receive, send):
        if (
            not self.protocol
            or scope.get("type") != "http"
            or scope.get("method") != "GET"
            or scope.get("path") != "/v1/models"
        ):
            await self.app(scope, receive, send)
            return

        # We are outside inner middleware: request an uncompressed metadata
        # response before it is produced. An outer gzip middleware may safely
        # compress our final answer. Never decode an unexpected wire encoding.
        inner_scope = dict(scope)
        inner_scope["headers"] = [
            (key, value)
            for key, value in scope.get("headers", [])
            if key.lower() != b"accept-encoding"
        ] + [(b"accept-encoding", b"identity")]
        start: dict[str, Any] | None = None
        chunks = bytearray()
        failure = False
        passthrough = False
        complete = False

        async def emit_body(body: bytes, *, failed: bool):
            assert start is not None
            if failed:
                outgoing = {
                    "type": "http.response.start",
                    "status": 502,
                    "headers": [(b"content-type", b"application/json")],
                }
            else:
                outgoing = dict(start)
                outgoing.pop("trailers", None)
                invalidated = {
                    b"content-length",
                    b"content-type",
                    b"last-modified",
                    b"content-encoding",
                    b"transfer-encoding",
                    b"trailer",
                    b"etag",
                    b"content-md5",
                    b"digest",
                    b"content-digest",
                    b"repr-digest",
                }
                outgoing["headers"] = [
                    (key, value)
                    for key, value in start.get("headers", [])
                    if key.lower() not in invalidated
                ]
                outgoing["headers"].append((b"content-type", b"application/json"))
            outgoing["headers"].append((b"content-length", str(len(body)).encode("ascii")))
            await send(outgoing)
            await send({"type": "http.response.body", "body": body, "more_body": False})

        async def intercept(message):
            nonlocal start, failure, passthrough, complete
            if passthrough:
                await send(message)
                return
            if complete:
                # A declared trailer stream is refused as a whole. Do not
                # append its original trailers to our replacement 502 body.
                return
            if message["type"] == "http.response.start":
                start = dict(message)
                if start["status"] != 200:
                    passthrough = True
                    await send(message)
                    return
                headers = [(key.lower(), value) for key, value in start.get("headers", [])]
                encodings = [
                    value.strip().lower() for key, value in headers if key == b"content-encoding"
                ]
                content_types = [
                    value.split(b";", 1)[0].strip().lower()
                    for key, value in headers
                    if key == b"content-type"
                ]
                failure = (
                    bool(start.get("trailers"))
                    or any(value not in (b"", b"identity") for value in encodings)
                    or content_types != [b"application/json"]
                )
            elif message["type"] == "http.response.body":
                if start is None:
                    raise RuntimeError("metadata body arrived before response start")
                chunk = message.get("body", b"")
                if not failure and len(chunks) + len(chunk) <= MAX_METADATA_BYTES:
                    chunks.extend(chunk)
                else:
                    failure = True
                    chunks.clear()
                if message.get("more_body", False):
                    return
                try:
                    if failure:
                        raise ValueError("unsupported metadata wire")
                    payload = json.loads(chunks)
                    if (
                        not isinstance(payload, dict)
                        or payload.get("object") != "list"
                        or not isinstance(payload.get("data"), list)
                        or any(
                            not isinstance(card, dict) or not isinstance(card.get("id"), str)
                            for card in payload.get("data", [])
                        )
                    ):
                        raise ValueError("not a model list")
                    previous = payload.get("pairing_protocol")
                    if previous is not None and (
                        not isinstance(previous, str) or previous.strip() not in ("", self.protocol)
                    ):
                        raise ValueError("conflicting metadata extension")
                    payload["pairing_protocol"] = self.protocol
                    body = json.dumps(
                        payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")
                    ).encode("utf-8")
                except (ValueError, UnicodeError):
                    failure = True
                    body = (
                        b'{"error":{"code":"smg_http_metadata_invalid",'
                        b'"message":"Engine metadata extension could not be produced"}}'
                    )
                complete = True
                await emit_body(body, failed=failure)
            else:
                await send(message)

        # No task/stream reader is spawned. Disconnect/cancellation propagates
        # through the original app and releases its active request normally.
        await self.app(inner_scope, receive, intercept)
        if not complete and not passthrough:
            raise RuntimeError("metadata response did not finish")
