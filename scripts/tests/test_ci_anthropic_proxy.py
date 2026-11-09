"""The job-local Anthropic proxy: the real key lives in it, Claude's process holds a placeholder."""

import http.server
import json
import logging
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ci_anthropic_proxy as proxy  # noqa: E402


class Upstream(http.server.BaseHTTPRequestHandler):
    """Records what it was sent; streams two chunks back, or an error when asked."""

    seen = []
    error = None

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        headers = {k.lower(): v for k, v in self.headers.items()}  # urllib capitalizes names
        Upstream.seen.append({"path": self.path, "headers": headers, "body": body})
        if Upstream.error:
            status, payload = Upstream.error
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        for chunk in (b"event: one\n\n", b"event: two\n\n"):
            self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
            self.wfile.flush()
        self.wfile.write(b"0\r\n\r\n")

    def log_message(self, *args):
        pass


@pytest.fixture
def servers():
    Upstream.seen = []
    Upstream.error = None
    upstream = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    upstream_url = f"http://127.0.0.1:{upstream.server_address[1]}"
    server = proxy.serve("real-key", upstream_url, "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    upstream.shutdown()


def call(base, path="/v1/messages", key="sk-ant-placeholder", body=b'{"model":"m"}'):
    request = urllib.request.Request(
        f"{base}{path}",
        data=body,
        method="POST",
        headers={
            "x-api-key": key,
            "authorization": "Bearer junk",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )
    return urllib.request.urlopen(request, timeout=10)


def test_the_proxy_swaps_in_the_real_key_and_streams_the_answer(servers):
    with call(servers) as response:
        assert response.status == 200
        assert response.read() == b"event: one\n\nevent: two\n\n"
    sent = Upstream.seen[0]
    assert sent["path"] == "/v1/messages"
    assert sent["headers"]["x-api-key"] == "real-key"
    assert "authorization" not in sent["headers"]
    assert sent["headers"]["anthropic-version"] == "2023-06-01"
    assert sent["body"] == b'{"model":"m"}'


def test_an_upstream_error_reaches_the_client_with_its_status_and_body(servers):
    Upstream.error = (401, json.dumps({"error": "bad key"}).encode())
    with pytest.raises(urllib.error.HTTPError) as failure:
        call(servers)
    assert failure.value.code == 401
    assert json.loads(failure.value.read()) == {"error": "bad key"}


def test_the_key_never_reaches_the_proxys_own_log(servers, caplog):
    with caplog.at_level(logging.INFO, logger=proxy.LOGGER):
        call(servers).read()
    assert caplog.records, "the proxy logs each request"
    assert "real-key" not in caplog.text and "placeholder" not in caplog.text
