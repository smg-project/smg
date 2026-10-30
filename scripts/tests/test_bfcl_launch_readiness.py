"""Exercise the real Bash/curl gate against a private HTTP gateway, without engines."""

import contextlib
import http.server
import os
import re
import shlex
import socket
import subprocess
import threading
import time
import unittest
from pathlib import Path

SCRIPT = Path(
    os.environ.get("BFCL_LAUNCH_SCRIPT", Path(__file__).parents[1] / "bfcl/launch_arm.sh")
)
SOURCE = SCRIPT.read_text()
WAIT_HTTP = re.search(r"(?ms)^wait_http\(\).*?^}", SOURCE).group()
GATE = re.search(r'wait_http "(http://127\.0\.0\.1:\$ARM_B_GW_PORT/[^\"]+)"', SOURCE).group(1)


@contextlib.contextmanager
def gateway(statuses, *, body_mode=None):
    class Handler(http.server.BaseHTTPRequestHandler):
        seen = []
        ready_reads = 0

        def do_GET(self):
            self.seen.append((self.path, self.headers.get("Authorization")))
            if self.path == "/health":
                code = 200
            elif self.path == "/readiness":
                code = statuses[min(type(self).ready_reads, len(statuses) - 1)]
                type(self).ready_reads += 1
            else:
                code = 404
            self.send_response(code)
            failing_body = self.path == "/readiness" and body_mode is not None
            self.send_header("Content-Length", "8" if failing_body else "0")
            self.end_headers()
            if failing_body and body_mode == "partial":
                self.wfile.write(b"x")
                self.wfile.flush()
                self.close_connection = True
            elif failing_body and body_mode == "stall":
                time.sleep(4)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port, Handler
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)
        assert not thread.is_alive()


def run_gate(port, limit=5):
    url = GATE.replace("$ARM_B_GW_PORT", str(port))
    return subprocess.run(
        ["/usr/bin/bash", "-c", WAIT_HTTP + "\nwait_http " + shlex.quote(url) + " " + str(limit)],
        capture_output=True,
        text=True,
        timeout=12,
        env={
            **os.environ,
            "PATH": "/usr/bin:/bin",
            "NO_PROXY": "127.0.0.1",
            "no_proxy": "127.0.0.1",
        },
    )


class LaunchReadiness(unittest.TestCase):
    def test_live_gateway_waits_for_worker_readiness(self):
        with gateway([503, 200]) as (port, handler):
            result = run_gate(port, 10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(handler.seen, [("/readiness", None), ("/readiness", None)])

    def test_permanently_unready_worker_times_out(self):
        with gateway([503]) as (port, handler):
            result = run_gate(port)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("TIMEOUT", result.stderr)
            self.assertEqual(handler.seen, [("/readiness", None)])

    def test_redirect_is_not_readiness(self):
        with gateway([302]) as (port, _):
            self.assertNotEqual(run_gate(port).returncode, 0)

    def test_unauthorized_is_not_readiness(self):
        with gateway([401]) as (port, _):
            self.assertNotEqual(run_gate(port).returncode, 0)

    def test_public_readiness_accepts_2xx_without_api_key(self):
        with gateway([204]) as (port, handler):
            self.assertEqual(run_gate(port).returncode, 0)
            self.assertEqual(handler.seen, [("/readiness", None)])

    def test_200_partial_transfer_is_not_readiness(self):
        with gateway([200], body_mode="partial") as (port, handler):
            result = run_gate(port)
            self.assertNotEqual(result.returncode, 0, result.stderr)
            self.assertIn("TIMEOUT", result.stderr)
            self.assertEqual(handler.seen, [("/readiness", None)])

    def test_200_stalled_transfer_is_not_readiness(self):
        with gateway([200], body_mode="stall") as (port, handler):
            result = run_gate(port)
            self.assertNotEqual(result.returncode, 0, result.stderr)
            self.assertIn("TIMEOUT", result.stderr)
            self.assertEqual(handler.seen, [("/readiness", None)])

    def test_connection_refusal_times_out(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        result = run_gate(port)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("TIMEOUT", result.stderr)


if __name__ == "__main__":
    unittest.main()
