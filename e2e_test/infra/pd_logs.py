"""Log-scraping helpers shared by the PD KV-transfer e2e tests."""

from __future__ import annotations

import os
import re
import time
import uuid
from pathlib import Path

import pytest

LOG_FLUSH_TIMEOUT_S = 15.0


def assert_worker_logs_captured(worker_logs: str, what: str) -> None:
    """Empty worker logs are a broken-capture signal in CI, a skip locally.

    CI always writes worker log files (SHOW_WORKER_LOGS=0 + E2E_LOG_DIR), so a
    silent skip there would disable the transfer assertion exactly when log
    routing is broken.
    """
    if worker_logs:
        return
    msg = f"No worker log files captured (SHOW_WORKER_LOGS=1?); cannot assert on {what}"
    if os.environ.get("CI") == "true":
        pytest.fail(msg)
    pytest.skip(msg)


def worker_log_dir(default: Path) -> Path:
    """Worker logs go to E2E_LOG_DIR when set (CI); otherwise the gateway dir."""
    return Path(os.environ.get("E2E_LOG_DIR") or default)


def read_logs(log_dir: Path, pattern: str) -> str:
    return "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in sorted(log_dir.glob(pattern))
        if path.is_file()
    )


def wait_for_marker(log_dir: Path, pattern: str, marker: str | tuple[str, ...]) -> str:
    # Both the router file appender and worker pipes flush asynchronously
    markers = (marker,) if isinstance(marker, str) else marker
    deadline = time.monotonic() + LOG_FLUSH_TIMEOUT_S
    logs = ""
    while time.monotonic() < deadline:
        logs = read_logs(log_dir, pattern)
        if any(m in logs for m in markers):
            return logs
        time.sleep(0.5)
    return logs


def wait_for_pattern(log_dir: Path, pattern: str, regex: re.Pattern[str]) -> str:
    # Like wait_for_marker, but for assertions that need more than a substring
    # (e.g. a positive counter value)
    deadline = time.monotonic() + LOG_FLUSH_TIMEOUT_S
    logs = ""
    while time.monotonic() < deadline:
        logs = read_logs(log_dir, pattern)
        if regex.search(logs):
            return logs
        time.sleep(0.5)
    return logs


def wait_for_pattern_in_files(paths: list[str | Path | None], regex: re.Pattern[str]) -> str:
    """Poll only the supplied live-worker logs, without a directory fallback."""
    files = [Path(path) for path in paths if path]
    deadline = time.monotonic() + LOG_FLUSH_TIMEOUT_S
    logs = ""
    while time.monotonic() < deadline:
        logs = "\n".join(
            path.read_text(encoding="utf-8", errors="replace") for path in files if path.is_file()
        )
        if regex.search(logs):
            return logs
        time.sleep(0.5)
    return logs


def unique_prompt() -> str:
    # Unique filler defeats prefix caching so every request exercises a fresh
    # prefill -> transfer -> decode cycle
    filler = " ".join(uuid.uuid4().hex for _ in range(24))
    return (
        f"Session token: {filler}\n"
        "Ignoring the session token above, explain in two sentences why the "
        "sky appears blue during the day."
    )
