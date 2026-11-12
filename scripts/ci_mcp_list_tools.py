#!/usr/bin/env python3
"""List the tools an MCP server advertises over its streamable HTTP endpoint.

CI's shared-service check used to open the service's port and call it ready.
A server whose process is up but whose tool list is empty passes that check
and then fails every MCP case of the e2e rows twenty minutes later, with no
line that names the service. This probe does the MCP handshake the gateway
does (initialize, notifications/initialized, tools/list) and prints the tool
names, so the check step fails fast and says why.

Usage: ci_mcp_list_tools.py URL [--expect name[,name...]] [--timeout S] [--deadline S]

Exit 0 when the server lists at least one tool and every expected name is
among them, 1 when the list is empty or an expected name is missing, 2 when
the handshake fails (connection, HTTP status, malformed answer, the deadline).
Standard library only: the check step runs before the test dependencies land.

One deadline bounds the whole handshake: a process timer that interrupts
whatever call is blocking when it falls, so a peer that trickles its
headers, a line or a body a byte at a time inside the socket timeout still
meets it and the probe prints its own line naming the step; the deadline
is also checked
before every request and read and honoured by the session's close. The
per-request timeout bounds one idle socket operation. An event stream is
read event by event and left the moment the answer arrives, never to its end.
"""

from __future__ import annotations

import argparse
import http.client
import json
import signal
import sys
import time
import urllib.error
import urllib.request

PROTOCOL_VERSION = "2025-03-26"
CLIENT_INFO = {"name": "smg-ci-service-check", "version": "1"}
# Bytes of one answer the probe reads before it gives the server up: a tool
# list is a few kilobytes, a stream that passes this without the answer is
# not going to deliver it.
MAX_ANSWER_BYTES = 1024 * 1024
READ_CHUNK = 64 * 1024

NETWORK_ERRORS = (urllib.error.URLError, http.client.HTTPException, OSError)


class HandshakeError(Exception):
    """The server did not complete the MCP handshake."""


class Deadline:
    """One total bound over the handshake, checked before every request, read and the session's close."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self.end = time.monotonic() + seconds
        self.step = "the connection"  # the last step that checked in, for the timer's line

    def left(self) -> float:
        return self.end - time.monotonic()

    def remaining(self, step: str) -> float:
        self.step = step
        left = self.left()
        if left <= 0:
            raise HandshakeError(f"the deadline of {self.seconds:g} s passed during {step}")
        return left


def _is_answer(message: object, wanted_id: int) -> bool:
    return isinstance(message, dict) and message.get("id") == wanted_id


def _checked(message: dict) -> dict:
    if "error" in message:
        raise HandshakeError(f"JSON-RPC error: {message['error']}")
    return message


def _read_json(
    response: http.client.HTTPResponse, wanted_id: int, method: str, deadline: Deadline
) -> dict:
    """Return the JSON-RPC message with the wanted id from a JSON body (an object or a batch).

    The body is read in the pieces that have arrived, the deadline checked
    before each, so a peer that trickles bytes inside the socket timeout
    cannot run past it any more than a stream of keepalives can.
    """
    chunks: list[bytes] = []
    size = 0
    while True:
        deadline.remaining(f"the answer to {method}")
        try:
            chunk = response.read1(READ_CHUNK)
        except NETWORK_ERRORS as exc:
            raise HandshakeError(f"reading the answer to {method} failed: {exc}") from exc
        if not chunk:
            break
        chunks.append(chunk)
        size += len(chunk)
        if size > MAX_ANSWER_BYTES:
            raise HandshakeError(f"the answer to {method} passed {MAX_ANSWER_BYTES} bytes")
    body = b"".join(chunks).decode("utf-8", errors="replace")
    if not body.strip():
        raise HandshakeError(f"{method} answered HTTP {response.status} with no body")
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise HandshakeError(f"unparsable answer to {method}: {body[:200]!r}") from exc
    for message in parsed if isinstance(parsed, list) else [parsed]:
        if _is_answer(message, wanted_id):
            return _checked(message)
    raise HandshakeError(f"no answer with id {wanted_id} to {method} in {body[:200]!r}")


def _read_event(
    response: http.client.HTTPResponse, wanted_id: int, method: str, deadline: Deadline
) -> dict:
    """Return the JSON-RPC message with the wanted id from an SSE stream.

    Events are parsed as they arrive and the stream is left at the answer:
    a server that keeps it open afterwards (keepalive comments, later
    notifications) costs nothing, and one that never answers meets the
    deadline between two lines.
    """
    data: list[str] = []
    size = 0
    while True:
        deadline.remaining(f"the event stream of {method}")
        try:
            raw = response.readline(MAX_ANSWER_BYTES)
        except NETWORK_ERRORS as exc:
            raise HandshakeError(f"the event stream of {method} failed: {exc}") from exc
        size += len(raw)
        if size > MAX_ANSWER_BYTES:
            raise HandshakeError(
                f"the event stream of {method} passed {MAX_ANSWER_BYTES} bytes without an answer"
            )
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        if line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
        if raw and line != "":
            continue  # a comment (keepalive), another field, or more data to come
        if data:
            try:
                message = json.loads("\n".join(data))
            except json.JSONDecodeError:
                message = None
            data = []
            if _is_answer(message, wanted_id):
                return _checked(message)
        if not raw:
            raise HandshakeError(
                f"the event stream of {method} ended without an answer with id {wanted_id}"
            )


def _post(
    url: str, payload: dict, session: str | None, timeout: float, deadline: Deadline
) -> tuple[dict | None, str | None]:
    """POST one JSON-RPC message; return (its answer or None for a notification, session id)."""
    method = payload["method"]
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "MCP-Protocol-Version": PROTOCOL_VERSION,
    }
    if session:
        headers["Mcp-Session-Id"] = session
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers=headers, method="POST"
    )
    try:
        response = urllib.request.urlopen(request, timeout=min(timeout, deadline.remaining(method)))
    except urllib.error.HTTPError as exc:
        # A sized read returns what arrived instead of raising on a truncated
        # error body, and the guard keeps a stalled one from escaping this
        # handler as a traceback (an exception raised inside one except
        # clause is not caught by its siblings).
        try:
            detail = exc.read(200)
        except NETWORK_ERRORS:
            detail = b"(the error body could not be read)"
        raise HandshakeError(f"HTTP {exc.code} on {method}: {detail!r}") from exc
    except NETWORK_ERRORS as exc:
        raise HandshakeError(f"{method} failed: {exc}") from exc
    with response:
        new_session = response.headers.get("Mcp-Session-Id") or session
        if "id" not in payload:
            # A notification: any 2xx is the acknowledgement.
            return None, new_session
        if response.status == 202:
            raise HandshakeError(f"{method} answered HTTP 202 with no body")
        if response.headers.get("Content-Type", "").startswith("text/event-stream"):
            return _read_event(response, payload["id"], method, deadline), new_session
        return _read_json(response, payload["id"], method, deadline), new_session


def _close_session(url: str, session: str, timeout: float, deadline: Deadline) -> None:
    """Tell a stateful server the session is over (the protocol's DELETE), best effort.

    A stateless server answers 405 and a dead one nothing; neither changes the
    verdict, the tool list is already in hand. Without this every probe would
    leave a session in the shared server's memory until its pod restarts. The
    close gets what is left of the deadline, at most a few seconds, and is
    skipped once the deadline has passed.
    """
    left = deadline.left()
    if left < 0.5:
        return
    request = urllib.request.Request(
        url,
        method="DELETE",
        headers={"Mcp-Session-Id": session, "MCP-Protocol-Version": PROTOCOL_VERSION},
    )
    try:
        urllib.request.urlopen(request, timeout=min(timeout, 5.0, left)).close()
    except NETWORK_ERRORS + (HandshakeError,):
        pass  # the deadline falling during the close changes nothing either


def list_tools(url: str, timeout: float, deadline: Deadline) -> list[str]:
    """Run the handshake under the deadline and return the advertised tool names."""
    init = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": CLIENT_INFO,
        },
    }
    session = None
    try:
        _, session = _post(url, init, None, timeout, deadline)
        _post(
            url,
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            session,
            timeout,
            deadline,
        )
        answer, _ = _post(
            url,
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            session,
            timeout,
            deadline,
        )
    finally:
        if session:
            _close_session(url, session, timeout, deadline)
    result = (answer or {}).get("result")
    tools = result.get("tools") if isinstance(result, dict) else None
    if not isinstance(tools, list):
        raise HandshakeError(
            f"tools/list answered without a tools list: {json.dumps(answer)[:200]}"
        )
    return [tool.get("name", "?") for tool in tools if isinstance(tool, dict)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "url", help="the server's streamable HTTP endpoint, e.g. http://host:8080/mcp"
    )
    parser.add_argument(
        "--expect", default="", help="comma-separated tool names that must be listed"
    )
    parser.add_argument("--timeout", type=float, default=10.0, help="seconds per socket operation")
    parser.add_argument(
        "--deadline", type=float, default=30.0, help="seconds for the whole handshake"
    )
    args = parser.parse_args()

    def _expired(_signum, _frame):
        raise HandshakeError(f"the deadline of {args.deadline:g} s passed during {deadline.step}")

    # A process timer interrupts whatever call is blocking when the deadline
    # falls (an exception raised by a signal handler ends a blocked recv), so
    # a peer trickling its headers or a line a byte at a time inside the
    # socket timeout still meets it; its line names the step the handshake
    # was in, the last one that checked the deadline.
    deadline = Deadline(args.deadline)
    previous = signal.signal(signal.SIGALRM, _expired)
    signal.setitimer(signal.ITIMER_REAL, args.deadline)
    try:
        names = list_tools(args.url, args.timeout, deadline)
    except HandshakeError as exc:
        print(f"MCP handshake with {args.url} failed: {exc}")
        return 2
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    expected = [name for name in args.expect.split(",") if name]
    missing = [name for name in expected if name not in names]
    print(f"{args.url} lists {len(names)} tool(s): {', '.join(names) or '(none)'}")
    if not names:
        print(
            "the server is up but advertises no tools: the MCP cases of the e2e rows cannot bind to it"
        )
        return 1
    if missing:
        print(f"expected tool(s) not listed: {', '.join(missing)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
