"""Process management utilities for E2E tests."""

from __future__ import annotations

import logging
import os
import signal
import socket
import subprocess
import time

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Port reservation utilities
# ---------------------------------------------------------------------------

# Port reservation to prevent the OS from returning the same port
# for sequential get_open_port() calls before the port is actually bound.
_reserved_ports: set[int] = set()


def get_open_port(max_attempts: int = 10) -> int:
    """Get an available port with reservation tracking.

    Finds an available port from the kernel and reserves it in our tracking set
    to prevent the OS from returning the same port on subsequent calls.

    Args:
        max_attempts: Maximum attempts to find an unreserved port.

    Returns:
        An available port number that is reserved until release_port() is called.

    Raises:
        RuntimeError: If unable to find an available port after max_attempts.
    """
    for attempt in range(max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("", 0))
            s.listen(1)
            port = s.getsockname()[1]

        if port not in _reserved_ports:
            _reserved_ports.add(port)
            logger.debug("Reserved port %d (attempt %d)", port, attempt + 1)
            return port

        logger.debug(
            "Port %d already reserved, retrying (attempt %d/%d)",
            port,
            attempt + 1,
            max_attempts,
        )

    raise RuntimeError(f"Failed to find available port after {max_attempts} attempts")


def release_port(port: int) -> None:
    """Release a reserved port back to the available pool.

    Should be called when the process using the port has terminated.

    Args:
        port: The port number to release.
    """
    _reserved_ports.discard(port)
    logger.debug("Released port %d", port)


def kill_process_tree(pid: int, sig: int = signal.SIGTERM) -> None:
    """Kill a process and all its children.

    Args:
        pid: Process ID to kill
        sig: Signal to send (default: SIGTERM)
    """
    try:
        import psutil

        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.send_signal(sig)
            except psutil.NoSuchProcess:
                pass
        parent.send_signal(sig)
    except ImportError:
        # Fallback if psutil not available
        os.kill(pid, sig)
    except Exception as e:
        logger.warning("Failed to kill process tree for PID %d: %s", pid, e)


def live_process_group_members(pgid: int, proc_root: str = "/proc") -> list[int]:
    """PIDs still running in ``pgid``, ignoring zombies that await reaping.

    Reads ``proc_root`` (``/proc`` on Linux). Elsewhere it falls back to
    probing the group with signal 0, which cannot tell a zombie from a live
    process.
    """
    if not os.path.isdir(proc_root):
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return []
        except PermissionError:
            pass
        return [pgid]
    members: list[int] = []
    for name in os.listdir(proc_root):
        if not name.isdigit():
            continue
        try:
            with open(
                os.path.join(proc_root, name, "stat"), encoding="utf-8", errors="replace"
            ) as f:
                stat = f.read()
        except OSError:
            continue
        # ``pid (comm) state ppid pgrp ...``: comm may contain spaces or
        # parentheses, so split after the last closing parenthesis.
        fields = stat.rpartition(")")[2].split()
        if len(fields) < 3:
            continue
        state, pgrp = fields[0], fields[2]
        if state != "Z" and pgrp == str(pgid):
            members.append(int(name))
    return members


def wait_for_process_group_exit(pgid: int, timeout: float, kill_after: float) -> list[int]:
    """Block until every live member of ``pgid`` has exited.

    Sends SIGKILL to the whole group once ``kill_after`` seconds pass without
    it emptying, then keeps waiting until ``timeout``. Returns the PIDs still
    alive at the deadline (empty on success).
    """
    start = time.monotonic()
    killed = False
    while True:
        members = live_process_group_members(pgid)
        elapsed = time.monotonic() - start
        if not members or elapsed >= timeout:
            return members
        if not killed and elapsed >= kill_after:
            logger.warning(
                "Process group %d still has %s alive after %.0fs; sending SIGKILL",
                pgid,
                members,
                elapsed,
            )
            try:
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
            killed = True
        time.sleep(0.2)


def gpu_memory_used_mib(gpu_ids: list[int]) -> dict[int, int] | None:
    """Used memory per GPU index via ``nvidia-smi``; ``None`` when unavailable."""
    if not gpu_ids:
        return {}
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
                "-i",
                ",".join(str(g) for g in gpu_ids),
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    used: dict[int, int] = {}
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            used[int(parts[0])] = int(parts[1])
        except ValueError:
            continue
    return used


def wait_for_gpu_memory_release(
    gpu_ids: list[int],
    baseline: dict[int, int],
    timeout: float,
    slack_mib: int = 2048,
) -> dict[int, int] | None:
    """Wait until each GPU's used memory is back within ``slack_mib`` of ``baseline``.

    An engine's child processes release their CUDA contexts after the launcher
    exits; a replacement started on the same GPU before that reads the old
    allocation as its own shortfall and dies on its startup memory check.
    Returns the GPUs still above the threshold (empty on success), or ``None``
    when ``nvidia-smi`` is unavailable. Gives up before ``timeout`` once that
    set has held the same values for 5 s: a region another live process still
    maps never drains, so a static figure is the answer, not a wait.
    """
    deadline = time.monotonic() + timeout
    last_over: dict[int, int] | None = None
    last_change = time.monotonic()
    while True:
        used = gpu_memory_used_mib(gpu_ids)
        if used is None:
            return None
        over = {gpu: mib for gpu, mib in used.items() if mib > baseline.get(gpu, 0) + slack_mib}
        now = time.monotonic()
        if not over or now >= deadline:
            return over
        # A region that another live process still maps (a decode holding a
        # dead prefill's KV) never drains; a teardown in progress shrinks the
        # figure every second. Give up once it has sat still for 5 s.
        if over != last_over:
            last_over, last_change = over, now
        elif now - last_change >= 5.0:
            return over
        time.sleep(0.5)


def terminate_process(proc: subprocess.Popen, timeout: float = 30) -> None:
    """Gracefully terminate a process, kill if needed.

    Args:
        proc: Process to terminate
        timeout: Seconds to wait before force-killing
    """
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    start = time.perf_counter()
    while proc.poll() is None:
        if time.perf_counter() - start > timeout:
            proc.kill()
            break
        time.sleep(1)


def wait_for_health(
    url: str,
    timeout: float = 60,
    api_key: str | None = None,
    check_interval: float = 1.0,
) -> None:
    """Wait for a server's /health endpoint to return 200.

    Args:
        url: Base URL of the server
        timeout: Seconds to wait before timing out
        api_key: Optional API key for auth header
        check_interval: Seconds between health checks
    """
    start = time.perf_counter()
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    with requests.Session() as session:
        while time.perf_counter() - start < timeout:
            try:
                resp = session.get(f"{url}/health", headers=headers, timeout=5)
                if resp.status_code == 200:
                    logger.info("Service healthy at %s", url)
                    return
            except requests.RequestException:
                pass
            time.sleep(check_interval)

    raise TimeoutError(f"Server at {url} did not become healthy within {timeout}s")


def wait_for_workers_ready(
    router_url: str,
    expected_workers: int,
    timeout: float = 300,
    api_key: str | None = None,
) -> None:
    """Wait for all workers to connect and for the router to become ready.

    Args:
        router_url: Base URL of the router
        expected_workers: Number of workers to wait for
        timeout: Seconds to wait before timing out
        api_key: Optional API key for auth header
    """
    start = time.perf_counter()
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    connected_workers = 0
    readiness_reason = "not checked"

    with requests.Session() as session:
        while time.perf_counter() - start < timeout:
            try:
                resp = session.get(f"{router_url}/workers", headers=headers, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    connected_workers = data.get("total", len(data.get("workers", [])))
                    if connected_workers >= expected_workers:
                        readiness_resp = session.get(
                            f"{router_url}/readiness", headers=headers, timeout=5
                        )
                        readiness_reason = f"status {readiness_resp.status_code}"
                        try:
                            readiness_data = readiness_resp.json()
                        except ValueError:
                            readiness_data = {}
                        if isinstance(readiness_data, dict):
                            reason = readiness_data.get("reason") or readiness_data.get("status")
                            if isinstance(reason, str):
                                readiness_reason = reason
                        else:
                            readiness_data = {}
                        healthy_workers = readiness_data.get("healthy_workers", 0)
                        if not isinstance(healthy_workers, int):
                            healthy_workers = 0
                        if (
                            readiness_data.get("status") == "ready"
                            and healthy_workers < expected_workers
                        ):
                            readiness_reason = (
                                f"healthy workers {healthy_workers}/{expected_workers}"
                            )
                        if (
                            readiness_resp.status_code == 200
                            and readiness_data.get("status") == "ready"
                            and healthy_workers >= expected_workers
                        ):
                            logger.info(
                                "All %d workers connected and gateway ready after %.1fs",
                                expected_workers,
                                time.perf_counter() - start,
                            )
                            return
                    else:
                        readiness_reason = "waiting for workers"
            except (requests.RequestException, ValueError):
                pass
            time.sleep(2)

    raise TimeoutError(
        f"Router at {router_url} did not become ready within {timeout}s "
        f"(workers: {connected_workers}/{expected_workers}, readiness: {readiness_reason})"
    )


def detect_ib_device() -> str | None:
    """Detect first active InfiniBand device (e.g., mlx5_0).

    Returns:
        Device name if found (e.g., "mlx5_0"), None otherwise.
    """
    try:
        subprocess.run(
            ["ibv_devinfo", "-l"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=1,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    for i in range(12):
        dev = f"mlx5_{i}"
        try:
            res = subprocess.run(
                ["ibv_devinfo", dev],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if res.returncode == 0 and "state:" in res.stdout:
                for line in res.stdout.splitlines():
                    if "state:" in line and "PORT_ACTIVE" in line:
                        logger.info("Detected IB device: %s", dev)
                        return dev
        except Exception:
            pass
    return None
