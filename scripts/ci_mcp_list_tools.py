#!/usr/bin/env python3
"""List the tools an MCP server advertises over its streamable HTTP endpoint.

CI's shared-service check used to open the service's port and call it ready.
A server whose process is up but whose tool list is empty passes that check
and then fails every MCP case of the e2e rows twenty minutes later, with no
line that names the service. This probe does the MCP handshake the gateway
does (initialize, notifications/initialized, tools/list) and prints the tool
names, so the check step fails fast and says why.

Usage: ci_mcp_list_tools.py URL [--expect name[,name...]] [--timeout S]

Exit 0 when the server lists at least one tool and every expected name is
among them, 1 when the list is empty or an expected name is missing, 2 when
the handshake fails (connection, HTTP status, malformed answer).
Standard library only: the check step runs before the test dependencies land.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

PROTOCOL_VERSION = "2025-03-26"
CLIENT_INFO = {"name": "smg-ci-service-check", "version": "1"}


class HandshakeError(Exception):
    """The server did not complete the MCP handshake."""


def _post(
    url: str, payload: dict, session: str | None, timeout: float
) -> tuple[dict | None, str | None]:
    """POST one JSON-RPC message; return (parsed result message or None, session id)."""
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
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            new_session = response.headers.get("Mcp-Session-Id") or session
            content_type = response.headers.get("Content-Type", "")
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        raise HandshakeError(
            f"HTTP {exc.code} on {payload['method']}: {exc.read()[:200]!r}"
        ) from exc
    except (urllib.error.URLError, OSError) as exc:
        raise HandshakeError(f"{payload['method']} failed: {exc}") from exc
    if "id" not in payload:
        # A notification: any 2xx is the acknowledgement.
        return None, new_session
    if status == 202 or not body.strip():
        raise HandshakeError(f"{payload['method']} answered HTTP {status} with no body")
    return _parse_message(body, content_type, payload["id"]), new_session


def _parse_message(body: str, content_type: str, wanted_id: int) -> dict:
    """Return the JSON-RPC message with the wanted id from a JSON or SSE body."""
    candidates: list[dict] = []
    if content_type.startswith("text/event-stream"):
        for line in body.splitlines():
            if line.startswith("data:"):
                try:
                    candidates.append(json.loads(line[5:].strip()))
                except json.JSONDecodeError:
                    continue
    else:
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise HandshakeError(f"unparsable answer: {body[:200]!r}") from exc
        candidates = parsed if isinstance(parsed, list) else [parsed]
    for message in candidates:
        if isinstance(message, dict) and message.get("id") == wanted_id:
            if "error" in message:
                raise HandshakeError(f"JSON-RPC error: {message['error']}")
            return message
    raise HandshakeError(f"no answer with id {wanted_id} in {body[:200]!r}")


def list_tools(url: str, timeout: float) -> list[str]:
    """Run the handshake and return the advertised tool names."""
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
    _, session = _post(url, init, None, timeout)
    _post(url, {"jsonrpc": "2.0", "method": "notifications/initialized"}, session, timeout)
    answer, _ = _post(
        url, {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}, session, timeout
    )
    tools = (answer or {}).get("result", {}).get("tools")
    if not isinstance(tools, list):
        raise HandshakeError(f"tools/list answered without a tools list: {answer}")
    return [tool.get("name", "?") for tool in tools if isinstance(tool, dict)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "url", help="the server's streamable HTTP endpoint, e.g. http://host:8080/mcp"
    )
    parser.add_argument(
        "--expect", default="", help="comma-separated tool names that must be listed"
    )
    parser.add_argument("--timeout", type=float, default=10.0, help="seconds per request")
    args = parser.parse_args()
    try:
        names = list_tools(args.url, args.timeout)
    except HandshakeError as exc:
        print(f"MCP handshake with {args.url} failed: {exc}")
        return 2
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
