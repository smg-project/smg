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
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

MAX_BYTES = 1024 * 1024
ROLES = ("arm_a", "arm_b_worker", "arm_b_gateway")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


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


def fetch(base, path):
    """Read local selected endpoints only; do not follow redirects or proxies."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    try:
        try:
            response = opener.open(loopback_url(base) + path, timeout=2)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            body = response.read(MAX_BYTES + 1)
            if len(body) > MAX_BYTES:
                return {"error": "oversize_response"}, None
            return {"http_status": response.code}, body
    except (OSError, ValueError, urllib.error.URLError, http.client.HTTPException) as error:
        return {"error": type(error).__name__}, None


def selected_json(base, path, model):
    status, body = fetch(base, path)
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
                    if key in row and isinstance(row[key], (bool, int))
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


def routing_states(metrics_base, worker_port):
    status, body = fetch(metrics_base, "/metrics")
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
            row.get("ready") is True and row.get("healthy_workers", 0) >= 1
            for row in observation.get("models", [])
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, type=loopback_url)
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--wait-ready", type=float, default=0)
    parser.add_argument("--duration", type=float, default=14400)
    parser.add_argument("--metrics-url", type=loopback_url)
    parser.add_argument("--worker-port", type=int)
    args = parser.parse_args()
    if not 0 <= args.wait_ready <= 7200 or not 0 < args.duration <= 21600:
        parser.error("readiness bound must be 0..7200 s and duration must be > 0 and <= 21600 s")
    owned = owned_processes(args.run_dir)
    started = time.monotonic()
    # The existing always-upload BFCL arm-log artifact retains *.log files.
    with (args.run_dir / "arm_b_state.log").open("a", buffering=1) as output:
        while True:
            observation = selected_json(args.base_url, "/readiness", args.model)
            row = {
                # The launcher also runs on hosts whose python3 is Python 3.10.
                "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),  # noqa: UP017
                "elapsed_s": round(time.monotonic() - started, 3),
                "processes": current_processes(owned),
                "readiness": observation,
            }
            if args.wait_ready:
                output.write(json.dumps(row) + "\n")
                if ready(observation):
                    return 0
                if time.monotonic() - started >= args.wait_ready:
                    print(
                        "[bfcl arm probe] TIMEOUT waiting for target model readiness; "
                        f"latest observation: {json.dumps(observation)}; "
                        f"see {args.run_dir / 'arm_b_state.log'}",
                        file=sys.stderr,
                    )
                    return 1
                time.sleep(2)
            else:
                row["workers"] = selected_json(args.base_url, "/workers", args.model)
                if args.metrics_url and args.worker_port:
                    row["routing"] = routing_states(args.metrics_url, args.worker_port)
                output.write(json.dumps(row) + "\n")
                remaining = args.duration - (time.monotonic() - started)
                if remaining <= 0:
                    return 0
                time.sleep(min(30, remaining))


if __name__ == "__main__":
    raise SystemExit(main())
