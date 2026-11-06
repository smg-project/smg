#!/usr/bin/env python3
"""Bounded BFCL arm observations, without generation probes or result changes.

Only selected state is retained: no response dumps, worker labels, credentials,
prompts, environment or command lines. A failed observation stays unknown. PID
birth ticks distinguish an exited owned process from a reused PID; they do not
establish an exit code or an engine child's health. Full arm logs remain needed.
"""

import argparse
import datetime
import http.client
import io
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

MAX_BYTES = 1024 * 1024
MAX_OBSERVATION_SECONDS = 21600
ROLES = ("arm_a", "arm_b_worker", "arm_b_gateway")


def bounded_observer_duration(run_timeout):
    """Keep scoring's integer syntax and zero-as-uncapped meaning separate.

    Python's decimal integers handle leading zeroes and values beyond shell
    arithmetic without octal interpretation or overflow. The scoring setting
    is never modified: only observations are capped at the existing job bound.
    """
    try:
        seconds = int(run_timeout, 10)
    except ValueError as error:
        raise argparse.ArgumentTypeError("RUN_TIMEOUT must be a nonnegative integer") from error
    if seconds < 0:
        raise argparse.ArgumentTypeError("RUN_TIMEOUT must be a nonnegative integer")
    return min(seconds, MAX_OBSERVATION_SECONDS) if seconds else MAX_OBSERVATION_SECONDS


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def remaining_seconds(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("observation deadline expired")
    return remaining


class DeadlineReader(io.RawIOBase):
    def __init__(self, stream, sock, deadline):
        self.stream, self.sock, self.deadline = stream, sock, deadline

    def readable(self):
        return True

    def readinto(self, buffer):
        # SocketIO performs one recv, so headers and bodies cannot reset the
        # same relative timeout indefinitely by delivering another byte.
        self.sock.settimeout(remaining_seconds(self.deadline))
        return self.stream.readinto(buffer)

    def close(self):
        try:
            self.stream.close()
        finally:
            super().close()


class DeadlineConnection(http.client.HTTPConnection):
    def __init__(self, *args, deadline, **kwargs):
        super().__init__(*args, **kwargs)
        self.deadline = deadline

    def connect(self):
        super().connect()
        self.sock.settimeout(remaining_seconds(self.deadline))

    def send(self, data):
        if self.sock is not None:
            self.sock.settimeout(remaining_seconds(self.deadline))
        super().send(data)

    def response_class(self, sock, *args, **kwargs):
        response = http.client.HTTPResponse(sock, *args, **kwargs)
        response.fp.close()
        # Keep SocketIO's descriptor reference: urllib closes its connection
        # socket before the caller consumes the response body.
        response.fp = io.BufferedReader(
            DeadlineReader(sock.makefile("rb", buffering=0), sock, self.deadline)
        )
        return response


class DeadlineHTTPHandler(urllib.request.HTTPHandler):
    def __init__(self, deadline):
        super().__init__()
        self.deadline = deadline

    def http_open(self, request):
        def connection(host, **kwargs):
            return DeadlineConnection(host, deadline=self.deadline, **kwargs)

        return self.do_open(connection, request)


def loopback_url(value):
    parsed = urllib.parse.urlsplit(value)
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or not parsed.port
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("expected an unauthenticated HTTP loopback base URL")
    return value.rstrip("/")


def fetch(base, path, deadline=None):
    """Read local selected endpoints only; do not follow redirects or proxies."""
    try:
        request_deadline = time.monotonic() + 2
        if deadline is not None:
            request_deadline = min(request_deadline, deadline)
        timeout = remaining_seconds(request_deadline)
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), NoRedirect(), DeadlineHTTPHandler(request_deadline)
        )
        try:
            response = opener.open(loopback_url(base) + path, timeout=timeout)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            body = response.read(MAX_BYTES + 1)
            if len(body) > MAX_BYTES:
                return {"error": "oversize_response"}, None
            return {"http_status": response.code}, body
    except (OSError, ValueError, urllib.error.URLError, http.client.HTTPException) as error:
        return {"error": type(error).__name__}, None


def selected_json(base, path, model, deadline=None):
    status, body = fetch(base, path, deadline)
    if body is None:
        return status
    try:
        data = json.loads(body)
        if not isinstance(data, dict):
            raise ValueError("not an object")
        if path == "/readiness":
            status["ready"] = data.get("status") == "ready"
            status["models"] = [
                {
                    key: row[key]
                    for key in ("ready", "healthy_workers", "total_workers")
                    if key in row
                    and (type(row[key]) is bool if key == "ready" else type(row[key]) is int)
                }
                for row in data.get("models", [])
                if isinstance(row, dict) and row.get("model") == model
            ][:8]
        else:
            status["workers"] = [
                {
                    key: row[key]
                    for key in ("is_healthy", "status", "load", "stalled")
                    if key in row
                    and (
                        isinstance(row[key], (bool, int, type(None)))
                        or row[key]
                        in (
                            "pending",
                            "ready",
                            "not_ready",
                            "failed",
                            "draining",
                            "unreachable",
                            "wedged",
                        )
                    )
                }
                for row in data.get("workers", [])
                if isinstance(row, dict) and row.get("model_id") == model
            ][:8]
    except (ValueError, TypeError):
        status["error"] = "invalid_json_schema"
    return status


def pid_identity(pid):
    try:
        # comm may contain spaces or ')'; the final ')' ends field 2.
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return {
            "pid": pid,
            "start_ticks": int(fields[19]),
            "pgid": int(fields[2]),
            "state": fields[0],
        }
    except (OSError, ValueError, IndexError):
        return None


def owned_processes(run_dir):
    owned = {}
    for role in ROLES:
        try:
            pid = int((run_dir / f"{role}.pid").read_text())
            if pid > 0:
                owned[role] = pid_identity(pid)
        except (OSError, ValueError):
            pass
    return owned


def current_processes(owned):
    result = {}
    for role, original in owned.items():
        current = pid_identity(original["pid"]) if original else None
        same = bool(current and original["start_ticks"] == current["start_ticks"])
        result[role] = {
            "identity": original,
            "same_process_present": same,
            "state": current["state"] if same else None,
        }
    return result


def routing_states(metrics_base, worker_port, deadline=None):
    status, body = fetch(metrics_base, "/metrics", deadline)
    if body is None:
        return status
    status["states"] = []
    for line in body.decode("utf-8", errors="replace").splitlines():
        if not line.startswith("smg_worker_state{"):
            continue
        match = re.fullmatch(r"smg_worker_state\{(.*)\}\s+(\S+)", line)
        if not match:
            continue
        labels = dict(re.findall(r'(\w+)="([^"\\]*)"', match[1]))
        if labels.get("worker") != f"grpc://127.0.0.1:{worker_port}":
            continue
        if labels.get("state") not in ("available", "excluded"):
            continue
        if labels.get("reason") not in (
            "none",
            "pending",
            "not_ready",
            "failed",
            "draining",
            "overloaded",
            "unreachable",
            "wedged",
            "circuit_open",
        ):
            continue
        if match[2] in ("1", "1.0"):
            status["states"].append({key: labels[key] for key in ("state", "reason")})
    return status


def ready(observation):
    return (
        observation.get("http_status") == 200
        and observation.get("ready") is True
        and any(
            row.get("ready") is True
            and type(row.get("healthy_workers")) is int
            and row["healthy_workers"] >= 1
            for row in observation.get("models", [])
        )
    )


def main():
    if len(sys.argv) > 1 and sys.argv[1].split("=", 1)[0] == "--bounded-duration-for-run-timeout":
        # Pure preflight: no directories, process launches or HTTP requests.
        parser = argparse.ArgumentParser(
            description="Bound BFCL observations before engine startup"
        )
        parser.add_argument(
            "--bounded-duration-for-run-timeout", required=True, type=bounded_observer_duration
        )
        args = parser.parse_args()
        print(args.bounded_duration_for_run_timeout)
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, type=loopback_url)
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--wait-ready", type=float, default=0)
    parser.add_argument("--duration", type=float, default=14400)
    parser.add_argument("--metrics-url", type=loopback_url)
    parser.add_argument("--worker-port", type=int)
    args = parser.parse_args()
    if not 0 <= args.wait_ready <= 7200 or not 0 < args.duration <= MAX_OBSERVATION_SECONDS:
        parser.error("readiness bound must be 0..7200 s and duration must be > 0 and <= 21600 s")
    owned = owned_processes(args.run_dir)
    started = time.monotonic()
    deadline = started + (args.wait_ready or args.duration)
    # The existing always-upload BFCL arm-log artifact retains *.log files.
    with (args.run_dir / "arm_b_state.log").open("a", buffering=1) as output:
        while True:
            observation = selected_json(args.base_url, "/readiness", args.model, deadline)
            row = {
                # The launcher also runs on hosts whose python3 is Python 3.10.
                "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),  # noqa: UP017
                "elapsed_s": round(time.monotonic() - started, 3),
                "processes": current_processes(owned),
                "readiness": observation,
            }
            if args.wait_ready:
                output.write(json.dumps(row) + "\n")
                if ready(observation) and time.monotonic() <= deadline:
                    return 0
            else:
                row["workers"] = selected_json(args.base_url, "/workers", args.model, deadline)
                if args.metrics_url and args.worker_port:
                    row["routing"] = routing_states(args.metrics_url, args.worker_port, deadline)
                output.write(json.dumps(row) + "\n")
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(min(2 if args.wait_ready else 30, remaining))
            if time.monotonic() >= deadline:
                if args.wait_ready:
                    print(
                        "[bfcl arm probe] TIMEOUT waiting for target model readiness; "
                        f"latest observation: {json.dumps(observation)}; "
                        f"see {args.run_dir / 'arm_b_state.log'}",
                        file=sys.stderr,
                    )
                    return 1
                return 0


if __name__ == "__main__":
    raise SystemExit(main())
