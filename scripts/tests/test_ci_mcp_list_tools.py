"""Tests for scripts/ci_mcp_list_tools.py.

Every verdict runs against a throwaway MCP server on a loopback port here, one
shape per test: JSON or SSE answers, an empty list, a missing tool, a failing
initialize, a null result, a stream kept open with keepalives after the answer,
a stream that never answers, a chunked HTTP/1.1 stream split mid-line, a JSON answer, a header
block or an event line that trickle in, a truncated error body, a closed port. Nothing here reaches a real server.
"""

from __future__ import annotations

import importlib.util
import json
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "ci_mcp_list_tools.py"
CHECK = SCRIPT.with_name("ci_agentic_svc_deps.sh")
TWO_TOOLS = [{"name": "brave_web_search"}, {"name": "brave_local_search"}]


def _load():
    spec = importlib.util.spec_from_file_location("ci_mcp_list_tools", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PROBE = _load()


class _Handler(BaseHTTPRequestHandler):
    """One MCP server shape per test, chosen by the server's mode."""

    def log_message(self, *_args) -> None:
        pass

    def _send(self, status, body=b"", content_type="application/json", session=None):
        self.send_response(status)
        if session:
            self.send_header("Mcp-Session-Id", session)
        if body:
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_DELETE(self):
        self.server.seen.append(("DELETE", self.headers.get("Mcp-Session-Id")))
        self._send(204)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length))
        self.server.seen.append((payload.get("method"), self.headers.get("Mcp-Session-Id")))
        mode = self.server.mode
        if "id" not in payload:
            self._send(202)
            return
        if payload["method"] == "initialize":
            if mode == "init-500":
                self._send(500, b"boom", "text/plain")
                return
            if mode == "init-502-truncated":
                # A proxy in front of a restarting pod: the status, a
                # Content-Length of 100, eleven bytes, then the close.
                self.send_response(502)
                self.send_header("Content-Length", "100")
                self.end_headers()
                self.wfile.write(b"bad gateway")
                self.wfile.flush()
                return
            result = {
                "protocolVersion": PROBE.PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "stub", "version": "0"},
            }
            answer = {"jsonrpc": "2.0", "id": payload["id"], "result": result}
            self._send(200, json.dumps(answer).encode(), session="session-1")
            return
        tools = {"empty": [], "missing": [{"name": "other_tool"}]}.get(mode, TWO_TOOLS)
        result = None if mode == "null-result" else {"tools": tools}
        answer = {"jsonrpc": "2.0", "id": payload["id"], "result": result}
        if mode == "headers-slow":
            raw = b"HTTP/1.0 200 OK\r\nContent-Type: application/json\r\n\r\n"
            self._slow_raw(raw + json.dumps(answer).encode())
            return
        if mode == "sse-slow-line":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self._slow_raw(f"data: {json.dumps(answer)}\n\n".encode())
            return
        if mode.startswith("sse"):
            self._event_stream(answer, mode)
        elif mode == "json-slow":
            self._trickle(json.dumps(answer).encode())
        else:
            self._send(200, json.dumps(answer).encode())

    def _trickle(self, body):
        """A JSON answer at a few bytes per second, each inside a socket timeout."""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            for i in range(len(body)):
                if self.server.stop.wait(0.2):
                    return
                self.wfile.write(body[i : i + 1])
                self.wfile.flush()
        except OSError:
            return

    def _slow_raw(self, raw):
        """Raw bytes a byte per 200 ms: a peer trickling its headers or a line inside a socket timeout."""
        try:
            for i in range(len(raw)):
                if self.server.stop.wait(0.2):
                    return
                self.wfile.write(raw[i : i + 1])
                self.wfile.flush()
        except OSError:
            return

    def _chunk(self, data):
        self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()

    def _chunked_event_stream(self, answer):
        """The real server's shape: HTTP/1.1, Transfer-Encoding chunked, the answer split mid-line."""
        self.protocol_version = "HTTP/1.1"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        event = f"event: message\ndata: {json.dumps(answer)}\n\n".encode()
        half = len(event) // 2
        try:
            self._chunk(b": a greeting comment\n\n")
            self._chunk(event[:half])
            self._chunk(event[half:])
            while not self.server.stop.wait(0.05):
                self._chunk(b": ping\n\n")
        except OSError:
            return

    def _event_stream(self, answer, mode):
        if mode == "sse-chunked":
            self._chunked_event_stream(answer)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        try:
            self.wfile.write(b": a greeting comment\n\n")
            if mode != "sse-silent":
                self.wfile.write(f"event: message\ndata: {json.dumps(answer)}\n\n".encode())
            self.wfile.flush()
            if mode in ("sse-open", "sse-silent"):
                # The stream stays open with a keepalive comment every 50 ms,
                # the shape a socket timeout never catches, until the test ends.
                while not self.server.stop.wait(0.05):
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
        except OSError:
            return


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, mode: str) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.mode = mode
        self.seen: list[tuple[str | None, str | None]] = []
        self.stop = threading.Event()
        self.url = f"http://127.0.0.1:{self.server_address[1]}/mcp"


@pytest.fixture
def server():
    started: list[_Server] = []

    def start(mode: str) -> _Server:
        srv = _Server(mode)
        threading.Thread(
            target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        ).start()
        started.append(srv)
        return srv

    yield start
    for srv in started:
        srv.stop.set()
        srv.shutdown()
        srv.server_close()


def _verdict(monkeypatch, capsys, url, *args):
    monkeypatch.setattr(sys, "argv", ["ci_mcp_list_tools.py", url, *args])
    code = PROBE.main()
    return code, capsys.readouterr().out


def test_a_json_answer_lists_the_tools_and_the_expected_name_passes(server, monkeypatch, capsys):
    code, out = _verdict(monkeypatch, capsys, server("json").url, "--expect", "brave_web_search")
    assert code == 0
    assert "lists 2 tool(s): brave_web_search, brave_local_search" in out


def test_the_session_rides_every_later_request_and_is_closed_at_the_end(
    server, monkeypatch, capsys
):
    srv = server("json")
    assert _verdict(monkeypatch, capsys, srv.url)[0] == 0
    assert srv.seen == [
        ("initialize", None),
        ("notifications/initialized", "session-1"),
        ("tools/list", "session-1"),
        ("DELETE", "session-1"),
    ]


def test_an_event_stream_answer_is_read_the_same(server, monkeypatch, capsys):
    code, out = _verdict(monkeypatch, capsys, server("sse").url, "--expect", "brave_web_search")
    assert code == 0
    assert "lists 2 tool(s)" in out


def test_an_empty_list_is_exit_1_with_the_named_line(server, monkeypatch, capsys):
    code, out = _verdict(monkeypatch, capsys, server("empty").url, "--expect", "brave_web_search")
    assert code == 1
    assert "lists 0 tool(s): (none)" in out
    assert "the server is up but advertises no tools" in out


def test_a_missing_expected_tool_is_exit_1(server, monkeypatch, capsys):
    code, out = _verdict(monkeypatch, capsys, server("missing").url, "--expect", "brave_web_search")
    assert code == 1
    assert "expected tool(s) not listed: brave_web_search" in out


def test_a_failing_initialize_is_exit_2_with_the_status(server, monkeypatch, capsys):
    code, out = _verdict(monkeypatch, capsys, server("init-500").url)
    assert code == 2
    assert "failed: HTTP 500 on initialize" in out


def test_a_closed_port_is_exit_2(monkeypatch, capsys):
    with socket.socket() as probe_socket:
        probe_socket.bind(("127.0.0.1", 0))
        port = probe_socket.getsockname()[1]
    code, out = _verdict(monkeypatch, capsys, f"http://127.0.0.1:{port}/mcp")
    assert code == 2
    assert "failed: initialize failed" in out


def test_a_truncated_error_body_is_the_named_line_not_a_traceback(server, monkeypatch, capsys):
    code, out = _verdict(monkeypatch, capsys, server("init-502-truncated").url)
    assert code == 2
    assert "failed: HTTP 502 on initialize: b'bad gateway'" in out


def test_a_json_answer_that_trickles_in_meets_the_deadline(server, monkeypatch, capsys):
    started = time.monotonic()
    code, out = _verdict(
        monkeypatch, capsys, server("json-slow").url, "--timeout", "10", "--deadline", "1"
    )
    assert code == 2
    assert "failed: the deadline of 1 s passed during the answer to tools/list" in out
    assert time.monotonic() - started < 5


def test_headers_that_trickle_in_meet_the_deadline(server, monkeypatch, capsys):
    started = time.monotonic()
    code, out = _verdict(
        monkeypatch, capsys, server("headers-slow").url, "--timeout", "10", "--deadline", "1"
    )
    assert code == 2
    assert "failed: the deadline of 1 s passed during tools/list" in out
    assert time.monotonic() - started < 5


def test_an_event_line_that_trickles_in_meets_the_deadline(server, monkeypatch, capsys):
    started = time.monotonic()
    code, out = _verdict(
        monkeypatch, capsys, server("sse-slow-line").url, "--timeout", "10", "--deadline", "1"
    )
    assert code == 2
    assert "failed: the deadline of 1 s passed during the event stream of tools/list" in out
    assert time.monotonic() - started < 5


def test_a_chunked_stream_split_mid_line_is_read_to_the_answer(server, monkeypatch, capsys):
    started = time.monotonic()
    code, out = _verdict(
        monkeypatch, capsys, server("sse-chunked").url, "--expect", "brave_web_search"
    )
    assert code == 0
    assert "lists 2 tool(s)" in out
    assert time.monotonic() - started < 5


def test_a_null_result_is_exit_2_with_the_named_line_not_a_traceback(server, monkeypatch, capsys):
    code, out = _verdict(monkeypatch, capsys, server("null-result").url)
    assert code == 2
    assert "tools/list answered without a tools list" in out


def test_a_stream_kept_open_after_the_answer_does_not_hold_the_probe(server, monkeypatch, capsys):
    started = time.monotonic()
    code, out = _verdict(monkeypatch, capsys, server("sse-open").url, "--deadline", "30")
    assert code == 0
    assert "lists 2 tool(s)" in out
    assert time.monotonic() - started < 5


def test_a_stream_that_never_answers_meets_the_deadline(server, monkeypatch, capsys):
    started = time.monotonic()
    code, out = _verdict(
        monkeypatch, capsys, server("sse-silent").url, "--timeout", "10", "--deadline", "1"
    )
    assert code == 2
    assert "failed: the deadline of 1 s passed during the event stream of tools/list" in out
    assert time.monotonic() - started < 5


def test_the_check_step_bounds_each_attempt_and_passes_the_deadline():
    text = CHECK.read_text()
    assert (
        'timeout 60 python3 "$probe" "$url" --expect "$expected" --timeout 10 --deadline 30' in text
    )
    assert 'check_mcp_tools "Brave Search MCP" "http://$host:8080/mcp" "brave_web_search"' in text
