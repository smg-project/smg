import json
import os
import re
import shlex
import signal
import socket
import stat
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path


def proc(pid):
    try:
        raw = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return {"pid": pid, "pgid": int(raw[2]), "sid": int(raw[3]), "start": int(raw[19])}
    except (OSError, ValueError, IndexError):
        return None


# Keep a session leader alive until its engine descendants are gone. Its live
# start identity prevents a recycled engine PID/process group granting cleanup
# authority. No listening socket, polling daemon or global process matching.
SUPERVISOR = r"""
import json, os, pathlib, signal, subprocess, sys, time
record, command = pathlib.Path(sys.argv[1]), json.loads(sys.argv[2])
me = os.getpid()
stopping = False

def stop(*_):
    global stopping
    stopping = True

signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
record.with_suffix('.ready').touch(exist_ok=False)

def members():
    found = []
    for p in pathlib.Path('/proc').iterdir():
        if p.name.isdecimal() and int(p.name) != me:
            try:
                s = (p / 'stat').read_text().rsplit(')', 1)[1].split()
                if int(s[3]) == me and s[0] != 'Z':
                    found.append((int(p.name), s[19]))
            except (OSError, ValueError, IndexError):
                pass
    return found

def send(sig):
    for pid, start in members():
        fd = None
        try:
            fd = os.pidfd_open(pid)
            s = pathlib.Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
            if s[19] == start and int(s[3]) == me:
                signal.pidfd_send_signal(fd, sig)
        except (OSError, ValueError, IndexError):
            pass
        finally:
            if fd is not None:
                os.close(fd)

try:
    limit = time.monotonic() + 10
    while not record.with_suffix('.go').exists():
        if stopping or time.monotonic() >= limit:
            raise RuntimeError('owner registration did not complete')
        time.sleep(.02)
    child = subprocess.Popen(command)
    while child.poll() is None and not stopping:
        time.sleep(.05)
    # An engine exiting by itself does not abandon its descendants.
    send(signal.SIGTERM)
    limit = time.monotonic() + 10
    while members() and time.monotonic() < limit:
        child.poll()
        time.sleep(.05)
    send(signal.SIGKILL)
    limit = time.monotonic() + 5
    while members() and time.monotonic() < limit:
        child.poll()
        time.sleep(.05)
    complete = not members()
    status = {'cleanup_complete': complete, 'returncode': child.poll()}
except BaseException as exc:
    status = {'cleanup_complete': not members(), 'error': str(exc)}
status.update(cookie=os.environ['TAU2_OWNER_COOKIE'], scope=os.environ['TAU2_OWNER_SCOPE'])
temporary = record.with_suffix('.receipt-tmp')
temporary.write_text(json.dumps(status))
temporary.replace(record.with_suffix('.result'))
sys.exit(0 if status['cleanup_complete'] else 1)
"""


def directory(root, scope):
    # Never adopt an old generic /tmp/tau2_ab or a foreign/symlink directory.
    if not scope or not re.fullmatch(r"[A-Za-z0-9_.-]{1,160}", scope):
        raise ValueError("TAU2_SCOPE must identify this comparison job")
    if not root.is_absolute() or root.is_symlink():
        raise ValueError("TAU2_RUN_DIR must be an absolute non-symlink path")
    if root.resolve() != root:
        raise ValueError("TAU2_RUN_DIR cannot traverse symlinks")
    try:
        root.mkdir(mode=0o700)
        (root / "owner.json").write_text(json.dumps({"scope": scope, "uid": os.getuid()}))
    except FileExistsError:
        pass
    info = root.stat()
    owner = root / "owner.json"
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise ValueError("run directory is not private and owned")
    if owner.is_symlink() or json.loads(owner.read_text()) != {"scope": scope, "uid": os.getuid()}:
        raise ValueError("run directory belongs to another invocation")
    for entry in root.iterdir():
        if entry.is_symlink() or not entry.is_file() or entry.stat().st_uid != os.getuid():
            raise ValueError("unsafe run directory entry")


def atomic_record(path, value):
    # Exclusive creation, never overwrite another launch's authority.
    with path.open("x") as out:
        json.dump(value, out)


def owned(record, scope):
    value = json.loads(record.read_text())
    if value.get("scope") != scope or value.get("uid") != os.getuid():
        raise ValueError("foreign process record")
    identity = value["identity"]
    if (
        identity["pid"] <= 1
        or identity["sid"] != identity["pid"]
        or identity["pgid"] != identity["pid"]
    ):
        raise ValueError("invalid owned session")
    current = proc(identity["pid"])
    # A completed supervisor can remain an adopted zombie until init reaps it;
    # its environment is then unreadable. Its exact start identity plus the
    # private completion receipt is sufficient; never signal that zombie.
    if current == identity:
        state = Path(f"/proc/{identity['pid']}/stat").read_text().rsplit(")", 1)[1].split()[0]
        if state == "Z":
            current = None
    if current is not None:
        status = Path(f"/proc/{identity['pid']}/status").read_text()
        cookie = Path(f"/proc/{identity['pid']}/environ").read_bytes().split(b"\0")
        if not re.search(rf"^Uid:\s+{os.getuid()}\s", status, re.M) or (
            f"TAU2_OWNER_COOKIE={value['cookie']}".encode() not in cookie
            or f"TAU2_OWNER_SCOPE={scope}".encode() not in cookie
        ):
            raise ValueError("foreign session owner cookie/uid")
    if current is not None and current != identity:
        raise ValueError("process identity changed; refusing signal")
    return identity, current


def cleanup(root, scope):
    failed = False
    for record in sorted(root.glob("*.pid.json")):
        try:
            # Pin before validating /proc, then signal the pinned task only.
            # A recycled numeric PID can never receive our signal.
            fd = None
            try:
                value = json.loads(record.read_text())
                try:
                    fd = os.pidfd_open(value["identity"]["pid"])
                except ProcessLookupError:
                    pass
                identity, current = owned(record, scope)
                if current:
                    if fd is None:
                        raise ValueError("owner vanished before identity check")
                    try:
                        signal.pidfd_send_signal(fd, signal.SIGTERM)
                    except ProcessLookupError:
                        pass  # Natural exit: require its receipt below.
            finally:
                if fd is not None:
                    os.close(fd)
            result = record.with_suffix(".result")
            limit = time.monotonic() + 17
            while time.monotonic() < limit:
                now = proc(identity["pid"])
                if now is not None and now != identity:
                    raise ValueError("session identity changed while waiting")
                if result.exists():
                    break
                if now is None:
                    raise ValueError("owner disappeared without cleanup receipt")
                time.sleep(0.05)
            receipt = json.loads(result.read_text()) if result.exists() else {}
            value = json.loads(record.read_text())
            if (
                receipt.get("cleanup_complete") is not True
                or receipt.get("cookie") != value["cookie"]
                or receipt.get("scope") != scope
            ):
                raise ValueError("owned session cleanup incomplete")
            record.unlink()
            record.with_suffix(".go").unlink(missing_ok=True)
            result.unlink()
            record.with_suffix(".ready").unlink(missing_ok=True)
            record.with_suffix(".claim").unlink(missing_ok=True)
        except (OSError, ValueError, KeyError) as exc:
            print(f"[launch_arms] cleanup refused/failed: {exc}", file=sys.stderr)
            failed = True
    return not failed


def spawn(root, scope, name, command, env):
    record = root / f"{name}.pid.json"
    if record.exists():
        raise ValueError(f"{name} already registered")
    cookie = uuid.uuid4().hex
    env = dict(env, TAU2_OWNER_COOKIE=cookie, TAU2_OWNER_SCOPE=scope)
    claim = record.with_suffix(".claim")
    atomic_record(claim, {"cookie": cookie, "scope": scope})
    child = None
    try:
        with (root / f"{name}.log").open("ab") as log:
            child = subprocess.Popen(
                [sys.executable, "-c", SUPERVISOR, str(record), json.dumps(command)],
                env=env,
                stdout=log,
                stderr=log,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        deadline = time.monotonic() + 3
        while not record.with_suffix(".ready").exists():
            if child.poll() is not None or time.monotonic() >= deadline:
                raise ValueError("session owner failed to initialize")
            time.sleep(0.01)
        identity = proc(child.pid)
        if not identity or identity["sid"] != child.pid or identity["pgid"] != child.pid:
            raise ValueError("session owner unavailable")
        atomic_record(
            record, {"scope": scope, "uid": os.getuid(), "cookie": cookie, "identity": identity}
        )
        record.with_suffix(".go").touch(exist_ok=False)
    except BaseException:
        if child is not None:
            child.terminate()
            child.wait(timeout=17)
        if not record.exists():
            # No go file was issued, so no engine could have been launched.
            claim.unlink(missing_ok=True)
            record.with_suffix(".ready").unlink(missing_ok=True)
            record.with_suffix(".result").unlink(missing_ok=True)
        raise
    return record


def port(variable):
    value = os.environ.get(variable)
    if value:
        number = int(value)
        if not 0 < number < 65536:
            raise ValueError("invalid port")
        return number
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_ready(root, scope, record, number, http):
    timeout = float(os.environ.get("TAU2_STARTUP_TIMEOUT", "420"))
    if not 0 < timeout <= 3600:
        raise ValueError("startup timeout must be finite and <=3600s")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if record.with_suffix(".result").exists():
            raise RuntimeError("server exited before health check")
        try:
            remaining = max(0.01, min(1, deadline - time.monotonic()))
            if http:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{number}/health", timeout=remaining
                ) as response:
                    if response.status == 200:
                        return
            else:
                with socket.create_connection(("127.0.0.1", number), timeout=remaining):
                    return
        except OSError:
            pass
        time.sleep(min(0.1, max(0, deadline - time.monotonic())))
    raise TimeoutError("server startup timed out")


def main():
    arm = sys.argv[1] if len(sys.argv) == 2 else ""
    if arm not in {"a", "b", "stop"}:
        raise ValueError("usage: launch_arms.sh <a|b|stop>")
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        raise ValueError("Linux pidfd support is required; no numeric-PID fallback")
    root = Path(os.environ.get("TAU2_RUN_DIR", ""))
    scope = os.environ.get("TAU2_SCOPE", "")
    directory(root, scope)
    if arm == "stop":
        return 0 if cleanup(root, scope) else 2
    env = os.environ.copy()
    vendor = env.get("TAU2_GPU_VENDOR", "nvidia")
    if vendor not in {"nvidia", "amd"}:
        raise ValueError("unsupported GPU vendor")
    tp = int(env.get("TAU2_TP", "1"))
    gpu = env.get("TAU2_GPU", "0")
    devices = gpu.split(",")
    if (
        tp < 1
        or len(devices) != tp
        or len(set(devices)) != tp
        or any(not re.fullmatch(r"0|[1-9][0-9]*", d) for d in devices)
    ):
        raise ValueError("GPU list must contain TP unique numeric device IDs")
    if vendor == "amd":
        a, b = env.get("TAU2_GPU_A", ""), env.get("TAU2_GPU_B", "")
        for selection in (a, b):
            ids = selection.split(",")
            if (
                len(ids) != tp
                or len(set(ids)) != tp
                or any(not re.fullmatch(r"0|[1-9][0-9]*", x) for x in ids)
            ):
                raise ValueError("AMD requires both validated TP device sets")
        if set(a.split(",")) & set(b.split(",")) or gpu != (a if arm == "a" else b):
            raise ValueError("AMD concurrent arms must have disjoint matching device sets")
        if any(
            env.get(key)
            for key in ("CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES")
        ):
            raise ValueError("conflicting inherited device visibility")
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env.pop("ROCR_VISIBLE_DEVICES", None)
        env["HIP_VISIBLE_DEVICES"] = gpu
    else:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    model = env.get("TAU2_MODEL", "Qwen/Qwen3.8-27B")
    source = model
    if env.get("ROUTER_LOCAL_MODEL_PATH"):
        local = Path(env["ROUTER_LOCAL_MODEL_PATH"]) / model
        if local.is_dir():
            source = str(local)
    common = [
        "--served-model-name",
        model,
        "--host",
        "0.0.0.0",
        "--tensor-parallel-size",
        str(tp),
        "--max-model-len",
        env.get("TAU2_MAX_MODEL_LEN", "16384"),
        "--gpu-memory-utilization",
        env.get("TAU2_GPU_MEM_UTIL", "0.85"),
    ]
    extra = shlex.split(env.get("TAU2_VLLM_EXTRA", ""))
    # Validate all commands before the first spawn, including malformed quoting.
    gateway = shlex.split(env.get("SMG_LAUNCH", "smg launch"))
    if not gateway:
        raise ValueError("empty SMG launcher")
    startup = float(env.get("TAU2_STARTUP_TIMEOUT", "420"))
    if not 0 < startup <= 3600:
        raise ValueError("invalid startup timeout")
    try:
        if arm == "a":
            number = port("TAU2_ARM_A_PORT")
            cmd = [
                env.get("VLLM_BIN", "vllm"),
                "serve",
                source,
                *common,
                "--port",
                str(number),
                "--enable-auto-tool-choice",
                "--tool-call-parser",
                env.get("TAU2_VLLM_TOOL_PARSER") or "qwen3_xml",
            ]
            if env.get("TAU2_VLLM_REASONING_PARSER", "qwen3"):
                cmd += ["--reasoning-parser", env.get("TAU2_VLLM_REASONING_PARSER", "qwen3")]
            record = spawn(root, scope, "arm_a", cmd + extra, env)
            wait_ready(root, scope, record, number, True)
        else:
            worker_port, number, metrics = (
                port("TAU2_ARM_B_GRPC_PORT"),
                port("TAU2_ARM_B_GW_PORT"),
                port("TAU2_ARM_B_METRICS_PORT"),
            )
            cmd = [
                env.get("VLLM_PYTHON", "python"),
                "-m",
                "vllm.entrypoints.grpc_server",
                "--model",
                source,
                *common,
                "--port",
                str(worker_port),
                *extra,
            ]
            record = spawn(root, scope, "arm_b_worker", cmd, env)
            wait_ready(root, scope, record, worker_port, False)
            gateway += [
                "--model-path",
                source,
                "--worker-urls",
                f"grpc://127.0.0.1:{worker_port}",
                "--host",
                "0.0.0.0",
                "--port",
                str(number),
                "--prometheus-port",
                str(metrics),
            ]
            for key, flag, default in (
                ("TAU2_SMG_TOOL_PARSER", "--tool-call-parser", "qwen_xml"),
                ("TAU2_SMG_REASONING_PARSER", "--reasoning-parser", "qwen3"),
            ):
                if env.get(key, default):
                    gateway += [flag, env.get(key, default)]
            record = spawn(root, scope, "arm_b_gateway", gateway, env)
            wait_ready(root, scope, record, number, True)
        print(f"http://127.0.0.1:{number}")
    except BaseException:
        if not cleanup(root, scope):
            print("[launch_arms] OWNED CLEANUP INCOMPLETE", file=sys.stderr)
        raise
    return 0


if __name__ == "__main__":
    try:

        def cancel(_signum, _frame):
            raise KeyboardInterrupt("launch cancelled")

        signal.signal(signal.SIGTERM, cancel)
        signal.signal(signal.SIGINT, cancel)
        sys.exit(main())
    except (Exception, KeyboardInterrupt) as exc:
        print(f"[launch_arms] {exc}", file=sys.stderr)
        sys.exit(2)
