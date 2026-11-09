#!/usr/bin/env python3
"""A job-local proxy that holds the Anthropic API key so the Claude process never does.

    printf '%s\\n' "$ANTHROPIC_API_KEY" | sudo -n env -i /usr/bin/python3 scripts/ci_anthropic_proxy.py \\
        --port-file "$RUNNER_TEMP/anthropic-proxy.port"

The key arrives on stdin and lives only in this process; run it as root and the job's own user,
Claude Code and its tools included, can read neither its environment nor its memory. Claude Code
gets `ANTHROPIC_BASE_URL=http://127.0.0.1:<port>` and a placeholder key; every request is
forwarded to the upstream with the real key in `x-api-key`, any `authorization` header dropped,
and the response streamed back as it arrives. The log names method, path and status only.
"""

from __future__ import annotations

import argparse
import http.client
import http.server
import logging
import sys
import urllib.parse
from pathlib import Path

LOGGER = "ci_anthropic_proxy"
UPSTREAM = "https://api.anthropic.com"
HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}
# Never forwarded from the client: the hop-by-hop set, the framing we recompute, and every
# credential header, which the proxy replaces with the real key.
NOT_FORWARDED = HOP_BY_HOP | {"host", "content-length", "authorization", "x-api-key"}
CHUNK = 16384

log = logging.getLogger(LOGGER)


def serve(key: str, upstream: str, host: str, port: int) -> http.server.ThreadingHTTPServer:
    """A server bound to `host:port` (0 picks a free port) forwarding to `upstream` with `key`."""
    target = urllib.parse.urlsplit(upstream)
    if target.scheme not in ("http", "https") or not target.hostname:
        raise ValueError(f"upstream must be an http(s) URL, got {upstream!r}")
    connection_class = (
        http.client.HTTPSConnection if target.scheme == "https" else http.client.HTTPConnection
    )
    prefix = target.path.rstrip("/")

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args: object) -> None:  # the default logs client addresses
            return

        def forward(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            headers = {k: v for k, v in self.headers.items() if k.lower() not in NOT_FORWARDED}
            headers["x-api-key"] = key
            headers["Host"] = target.netloc
            if body:
                headers["Content-Length"] = str(len(body))
            connection = connection_class(target.hostname, target.port, timeout=600)
            try:
                connection.request(self.command, prefix + self.path, body=body, headers=headers)
                response = connection.getresponse()
            except OSError as error:
                log.warning(
                    "%s %s: upstream unreachable (%s)",
                    self.command,
                    self.path,
                    type(error).__name__,
                )
                self.send_error(502, "upstream unreachable")
                connection.close()
                return
            self.send_response(response.status)
            for name, value in response.getheaders():
                if name.lower() in HOP_BY_HOP or name.lower() == "content-length":
                    continue
                self.send_header(name, value)
            self.send_header("Connection", "close")
            self.end_headers()
            while chunk := response.read(CHUNK):
                self.wfile.write(chunk)
                self.wfile.flush()
            self.close_connection = True
            connection.close()
            log.info("%s %s -> %s", self.command, self.path, response.status)

        do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = forward

    return http.server.ThreadingHTTPServer((host, port), Handler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--upstream", default=UPSTREAM, help=f"the API to forward to (default {UPSTREAM})"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0, help="0 picks a free port")
    parser.add_argument("--port-file", type=Path, help="where to write the bound port")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stderr)
    key = sys.stdin.readline().strip()
    if not key:
        print("no API key on stdin", file=sys.stderr)
        return 2
    server = serve(key, args.upstream, args.host, args.port)
    port = server.server_address[1]
    if args.port_file:
        args.port_file.write_text(f"{port}\n")
    log.info("listening on %s:%s, forwarding to %s", args.host, port, args.upstream)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
