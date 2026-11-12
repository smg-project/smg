import contextlib
import ctypes
import errno
import importlib.util
import json
import os
import select
import signal
import socketserver
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "bfcl" / "arm_probe.py"
SPEC = importlib.util.spec_from_file_location("bfcl_arm_probe", SOURCE)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)
MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"


@contextlib.contextmanager
def server(responses):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            calls.append(self.path)
            status, body, headers = responses.get(self.path, (404, b"missing", {}))
            self.send_response(status)
            for key, value in headers.items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{http.server_port}", calls
    finally:
        http.shutdown()
        thread.join()
        http.server_close()


def response(data, status=200):
    return status, json.dumps(data).encode(), {}


@contextlib.contextmanager
def broken_http_server(payload):
    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            self.request.recv(8192)
            self.request.sendall(payload)

    http = socketserver.TCPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{http.server_address[1]}"
    finally:
        http.shutdown()
        thread.join()
        http.server_close()


def readiness(model=MODEL, healthy=1):
    return {
        "status": "ready",
        "models": [
            {
                "model": model,
                "ready": True,
                "healthy_workers": healthy,
                "total_workers": 1,
                "unknown": "PRIVATE",
            }
        ],
    }


def test_a_live_gateway_needs_the_target_model_and_healthy_worker():
    for data, status, expected in (
        (readiness(), 200, True),
        (readiness("another/model"), 200, False),
        (readiness(healthy=0), 200, False),
        (readiness(), 503, False),
        ({"status": "ready", "healthy_workers": 4}, 200, False),
    ):
        with server({"/readiness": response(data, status)}) as (url, _):
            actual = probe.selected_json(url, "/readiness", MODEL)
            assert probe.ready(actual) is expected
            assert "PRIVATE" not in json.dumps(actual)


def test_unready_json_remains_observable_without_becoming_ready():
    with server({"/readiness": response({"status": "not ready", "models": []}, 503)}) as (url, _):
        actual = probe.selected_json(url, "/readiness", MODEL)
    assert actual == {"http_status": 503, "ready": False, "models": []}


def test_worker_observation_excludes_urls_keys_labels_prompts_and_other_models():
    data = {
        "workers": [
            {
                "model_id": MODEL,
                "url": "grpc://user:PRIVATE@localhost:1234",
                "api_key": "PRIVATE",
                "labels": {"prompt": "PRIVATE"},
                "status": "not_ready",
                "is_healthy": False,
                "load": 2,
                "stalled": "wedged",
                "engine_load": {"secret": "PRIVATE"},
            },
            {"model_id": "other/model", "load": 999},
        ]
    }
    with server({"/workers": response(data)}) as (url, _):
        actual = probe.selected_json(url, "/workers", MODEL)
    assert actual == {
        "http_status": 200,
        "workers": [
            {
                "is_healthy": False,
                "status": "not_ready",
                "load": 2,
                "stalled": "wedged",
            }
        ],
    }
    assert "PRIVATE" not in json.dumps(actual)


def test_redirect_is_not_followed_and_error_body_is_not_retained():
    with server(
        {
            "/readiness": (302, b"PRIVATE", {"Location": "/private"}),
            "/private": response({"secret": "PRIVATE"}),
        }
    ) as (url, calls):
        actual = probe.selected_json(url, "/readiness", MODEL)
    assert calls == ["/readiness"]
    assert actual == {"http_status": 302, "error": "invalid_json_schema"}


@pytest.mark.parametrize("body", [b"not-json-PRIVATE", b"[]", b"x" * (probe.MAX_BYTES + 1)])
def test_bad_or_oversize_answer_is_unknown_and_bounded(body):
    with server({"/readiness": (200, body, {})}) as (url, _):
        actual = probe.selected_json(url, "/readiness", MODEL)
    assert actual.get("error") in ("invalid_json_schema", "oversize_response")
    assert not probe.ready(actual)
    assert "PRIVATE" not in json.dumps(actual)


@pytest.mark.parametrize("endpoint", ["readiness", "metrics"])
@pytest.mark.parametrize(
    "payload,error",
    [
        (b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n10\r\nPRIVATE", "IncompleteRead"),
        (b"PRIVATE\r\n\r\n", "BadStatusLine"),
    ],
)
def test_broken_http_is_unknown_without_ending_observation(endpoint, payload, error):
    with broken_http_server(payload) as url:
        if endpoint == "metrics":
            actual = probe.routing_states(url, 1234)
        else:
            actual = probe.selected_json(url, "/readiness", MODEL)
    assert actual == {"error": error}
    assert "PRIVATE" not in json.dumps(actual)


def test_routing_state_retains_only_this_worker_active_reason():
    body = (
        b'smg_worker_state{worker="grpc://127.0.0.1:1234",state="available",reason="none"} 0\n'
        b'smg_worker_state{worker="grpc://127.0.0.1:1234",state="excluded",reason="circuit_open"} 1\n'
        b'smg_worker_state{worker="grpc://PRIVATE@127.0.0.1:99",state="excluded",reason="failed"} 1\n'
        b'some_secret{key="PRIVATE"} 1\n'
    )
    with server({"/metrics": (200, body, {})}) as (url, _):
        actual = probe.routing_states(url, 1234)
    assert actual == {
        "http_status": 200,
        "states": [{"state": "excluded", "reason": "circuit_open"}],
    }
    assert "PRIVATE" not in json.dumps(actual)


def test_pid_birth_is_compared_and_zombie_does_not_imply_an_exit_code(tmp_path, monkeypatch):
    (tmp_path / "arm_b_worker.pid").write_text(str(os.getpid()))
    owned = probe.owned_processes(tmp_path)
    assert probe.current_processes(owned)["arm_b_worker"]["same_process_present"]
    original = owned["arm_b_worker"]
    monkeypatch.setattr(probe, "pid_identity", lambda pid: {**original, "state": "Z"})
    row = probe.current_processes(owned)["arm_b_worker"]
    assert row["state"] == "Z" and "exit_code" not in row
    monkeypatch.setattr(
        probe,
        "pid_identity",
        lambda pid: {
            **original,
            "start_ticks": original["start_ticks"] + 1,
        },
    )
    assert not probe.current_processes(owned)["arm_b_worker"]["same_process_present"]
    monkeypatch.setattr(probe, "pid_identity", lambda pid: None)
    assert not probe.current_processes(owned)["arm_b_worker"]["same_process_present"]


@pytest.fixture
def controlled_main(monkeypatch):
    def run(arguments, responses):
        now = [100.0]
        calls = []

        def sleep(seconds):
            now[0] += seconds

        def fetch(base, path, deadline=None):
            calls.append(path)
            status, body, _ = responses[path]
            return {"http_status": status}, body

        # Positive logging/schema assertions should not require a CI scheduler
        # to complete two real HTTP requests within a 10ms caller deadline.
        # Separate real CLI fixtures enforce that deadline under actual I/O.
        monkeypatch.setattr(probe.time, "monotonic", lambda: now[0])
        monkeypatch.setattr(probe.time, "sleep", sleep)
        monkeypatch.setattr(probe, "fetch", fetch)
        monkeypatch.setattr(sys, "argv", [str(SOURCE), *arguments])
        return probe.main(), calls

    return run


def test_monitor_main_writes_bounded_snapshots_without_generation(tmp_path, controlled_main):
    result, calls = controlled_main(
        [
            "--base-url",
            "http://127.0.0.1:1234",
            "--model",
            MODEL,
            "--run-dir",
            str(tmp_path),
            "--duration",
            "0.01",
        ],
        {
            "/readiness": response(readiness()),
            "/workers": response({"workers": []}),
        },
    )
    rows = [json.loads(line) for line in (tmp_path / "arm_b_state.log").read_text().splitlines()]
    assert result == 0
    assert 1 <= len(rows) <= 2
    assert set(calls) == {"/readiness", "/workers"}
    assert all(probe.ready(row["readiness"]) for row in rows)


def test_wait_main_timeout_does_not_accept_process_liveness(tmp_path, controlled_main, capsys):
    result, calls = controlled_main(
        [
            "--base-url",
            "http://127.0.0.1:1234",
            "--model",
            MODEL,
            "--run-dir",
            str(tmp_path),
            "--wait-ready",
            "0.01",
        ],
        {"/readiness": response({"status": "ready", "healthy_workers": 2})},
    )
    assert result == 1
    assert set(calls) == {"/readiness"}
    stderr = capsys.readouterr().err
    assert "TIMEOUT" in stderr
    assert '"http_status": 200' in stderr
    assert '"models": []' in stderr


def test_real_cli_setup_can_exhaust_10ms_without_an_http_observation(tmp_path):
    wrapper = """
import importlib.util, sys, time
source = sys.argv.pop(1)
spec = importlib.util.spec_from_file_location('private_bfcl_probe', source)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
original = probe.urllib.request.build_opener
def delayed(*args, **kwargs):
    time.sleep(0.03)
    return original(*args, **kwargs)
probe.urllib.request.build_opener = delayed
raise SystemExit(probe.main())
"""
    with server({"/readiness": response(readiness())}) as (url, calls):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                wrapper,
                str(SOURCE),
                "--base-url",
                url,
                "--model",
                MODEL,
                "--run-dir",
                str(tmp_path),
                "--wait-ready",
                "0.01",
            ],
            capture_output=True,
            timeout=1.2,
        )
    assert result.returncode == 1
    assert calls == []
    assert b"TIMEOUT" in result.stderr
    rows = [json.loads(line) for line in (tmp_path / "arm_b_state.log").read_text().splitlines()]
    assert len(rows) == 1
    # urllib can wrap the connection's absolute-deadline TimeoutError.
    assert rows[0]["readiness"]["error"] in ("TimeoutError", "URLError")
    assert not probe.ready(rows[0]["readiness"])


@contextlib.contextmanager
def trickle_http_server(phase):
    stopped = threading.Event()
    calls = []

    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            self.request.recv(8192)
            calls.append("request")
            body = json.dumps(readiness()).encode()
            header = b"HTTP/1.1 200 OK\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n"
            chunks = (
                [bytes([byte]) for byte in header] + [body]
                if phase == "header"
                else [header] + [bytes([byte]) for byte in body]
            )
            for chunk in chunks:
                if stopped.is_set():
                    break
                try:
                    self.request.sendall(chunk)
                except OSError:
                    break
                if stopped.wait(0.04):
                    break

    class Server(socketserver.ThreadingTCPServer):
        daemon_threads = True

    http = Server(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{http.server_address[1]}", calls
    finally:
        stopped.set()
        http.shutdown()
        thread.join()
        http.server_close()


@pytest.mark.parametrize("phase", ["header", "body"])
@pytest.mark.parametrize("mode", ["--wait-ready", "--duration"])
@pytest.mark.parametrize("budget", ["0.01", "0.2"])
def test_real_slow_http_cannot_overrun_the_whole_probe_budget(tmp_path, phase, mode, budget):
    with trickle_http_server(phase) as (url, calls):
        started = time.monotonic()
        result = subprocess.run(
            [
                sys.executable,
                str(SOURCE),
                "--base-url",
                url,
                "--model",
                MODEL,
                "--run-dir",
                str(tmp_path),
                mode,
                budget,
            ],
            capture_output=True,
            timeout=1.2,
        )
        elapsed = time.monotonic() - started
    assert elapsed < 1
    assert result.returncode == (1 if mode == "--wait-ready" else 0), result.stderr
    rows = [json.loads(line) for line in (tmp_path / "arm_b_state.log").read_text().splitlines()]
    assert len(rows) == 1
    # The deadline error is the proof the probe gave up inside its budget, whether
    # or not the server was ever reached (urllib can wrap the TimeoutError).
    assert rows[0]["readiness"]["error"] in ("TimeoutError", "URLError")
    assert not probe.ready(rows[0]["readiness"])
    # No further I/O once the shared caller budget is spent: the server saw at most
    # one connection (the handler counts connections, not requests). None when the
    # budget was spent before the probe's first connect (a slow runner spends the
    # 10 ms budget before the socket opens); one when the probe connected and gave
    # up before sending or while the response trickled; never two. The count is
    # not a clock.
    assert len(calls) <= 1
    assert "PRIVATE" not in json.dumps(rows)


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com:80",
        "http://user:PRIVATE@127.0.0.1:80",
        "http://127.0.0.1:80/private",
        "http://127.0.0.1:80?key=PRIVATE",
        "https://127.0.0.1:80",
        "http://127.0.0.1",
    ],
)
def test_probe_rejects_remote_or_authenticated_targets(url):
    with pytest.raises(ValueError):
        probe.loopback_url(url)


@contextlib.contextmanager
def real_startup_fixture(tmp_path, is_ready):
    """Private CPU processes only; adopt/reap our original RED's orphan children."""
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    assert libc.prctl(37, ctypes.byref(previous), 0, 0, 0) == 0
    assert libc.prctl(36, 1, 0, 0, 0) == 0
    identities = tmp_path / "identities"
    identities.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    bins = tmp_path / "bin"
    bins.mkdir()
    record = (
        "import json, os, sys, time\nfrom pathlib import Path\n"
        "fields = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()\n"
        "identity = {'pid': os.getpid(), 'start_ticks': int(fields[19])}\n"
    )
    service = bins / "private-service"
    service.write_text(
        f"#!{sys.executable}\n"
        + record
        + "role = 'gateway' if '--model-path' in sys.argv else 'worker'\n"
        "(Path(os.environ['BFCL_FIXTURE_PID_DIR']) / (role + '.json')).write_text(json.dumps(identity))\n"
        "time.sleep(10)\n"
    )
    service.chmod(0o755)
    tail = bins / "tail"
    tail.write_text(
        f"#!{sys.executable}\n"
        + record
        + "target = Path(os.environ['BFCL_FIXTURE_PID_DIR']) / ('tail-' + str(os.getpid()) + '.json')\n"
        "target.write_text(json.dumps(identity))\n"
        "os.execv('/usr/bin/tail', ['/usr/bin/tail', *sys.argv[1:]])\n"
    )
    tail.chmod(0o755)
    sentinel = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])
    (state / "foreign.pid").write_text(str(sentinel.pid))
    try:
        data = readiness() if is_ready else {"status": "not ready", "models": []}
        with server({"/readiness": response(data)}) as (url, _):
            port = url.rsplit(":", 1)[1]
            env = {
                **os.environ,
                "PATH": f"{bins}{os.pathsep}{os.environ['PATH']}",
                "BFCL_FIXTURE_PID_DIR": str(identities),
                "BFCL_RUN_DIR": str(state),
                "BFCL_MODEL": MODEL,
                "BFCL_STARTUP_TIMEOUT": "0.2",
                "BFCL_ARM_B_GRPC_PORT": port,
                "BFCL_ARM_B_GW_PORT": port,
                "BFCL_ARM_B_METRICS_PORT": port,
                "VLLM_PYTHON": str(service),
                "SMG_LAUNCH": str(service),
                "RUN_TIMEOUT": "2",
                "CUDA_VISIBLE_DEVICES": "",
                "HIP_VISIBLE_DEVICES": "",
                "ROCR_VISIBLE_DEVICES": "",
            }
            with (tmp_path / "stdout").open("wb") as out, (tmp_path / "stderr").open("wb") as err:
                launcher = subprocess.Popen(
                    ["bash", str(SOURCE.with_name("launch_arm.sh")), "b"],
                    env=env,
                    stdout=out,
                    stderr=err,
                )
                try:
                    launcher.wait(timeout=3)
                finally:
                    if launcher.poll() is None:
                        launcher.kill()
                        launcher.wait()
            # Only this invocation can create this private role pidfile.
            probe_pid = state / "arm_b_probe.pid"
            if probe_pid.exists():
                identity = probe.pid_identity(int(probe_pid.read_text()))
                if identity:
                    (identities / "probe.json").write_text(json.dumps(identity))
            yield launcher, identities, sentinel
    finally:
        # Exact identity + pidfd cleanup of our private RED fixtures; no group,
        # process-name, foreign pidfile or shared resource is used here.
        for path in identities.glob("*.json"):
            identity = json.loads(path.read_text())
            pid = identity["pid"]
            try:
                fd = os.pidfd_open(pid)
            except ProcessLookupError:
                continue
            try:
                current = probe.pid_identity(pid)
                if current and current["start_ticks"] == identity["start_ticks"]:
                    signal.pidfd_send_signal(fd, signal.SIGKILL)
            finally:
                os.close(fd)
            with contextlib.suppress(ChildProcessError):
                os.waitpid(pid, 0)
        sentinel.kill()
        sentinel.wait()
        assert libc.prctl(36, previous.value, 0, 0, 0) == 0


@pytest.mark.skipif(
    sys.platform != "linux" or not hasattr(os, "pidfd_open"), reason="needs Linux process handles"
)
@pytest.mark.parametrize("is_ready", [False, True], ids=["readiness-fails", "commit-success"])
def test_real_startup_cleanup_is_invocation_owned(tmp_path, is_ready):
    with real_startup_fixture(tmp_path, is_ready) as (launcher, identities, sentinel):
        assert launcher.returncode == (0 if is_ready else 1)
        for role in ("arm_b_worker", "arm_b_gateway"):
            assert (tmp_path / "state" / f"{role}.pid").exists() == is_ready
        assert (tmp_path / "state" / "foreign.pid").read_text() == str(sentinel.pid)
        assert (identities / "worker.json").exists()
        assert (identities / "gateway.json").exists()
        for path in identities.glob("*.json"):
            identity = json.loads(path.read_text())
            current = probe.pid_identity(identity["pid"])
            same = bool(current and current["start_ticks"] == identity["start_ticks"])
            if is_ready and path.stem in ("worker", "gateway"):
                assert same, f"successful startup unexpectedly stopped {path.stem}"
            elif not is_ready:
                assert not same, f"startup failure leaked {path.stem}"
        assert sentinel.poll() is None


@pytest.mark.skipif(sys.platform != "linux", reason="needs Linux process identity")
def test_startup_owner_does_not_require_pidfd(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "bfcl_startup_owner", SOURCE.with_name("startup_owner.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    owner = module.StartupOwner()
    sentinel = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])
    calls = []

    def unavailable(*args):
        calls.append(args)
        raise OSError(errno.ENOSYS, "private fixture pidfd unavailable")

    monkeypatch.setattr(module.os, "pidfd_open", unavailable, raising=False)
    try:
        handle, pid = owner.start(
            str(tmp_path / "child.log"), [sys.executable, "-c", "import time; time.sleep(10)"]
        )
        assert probe.pid_identity(pid) is not None
        owner.stop([handle])
        assert probe.pid_identity(pid) is None
        assert not calls  # The unreaped original leader, not a pidfd, anchors the PGID.
        assert sentinel.poll() is None
    finally:
        owner.stop(list(owner.children))
        owner.release()
        sentinel.kill()
        sentinel.wait()


@pytest.mark.skipif(
    sys.platform != "linux" or not hasattr(os, "pidfd_open"), reason="needs Linux process handles"
)
@pytest.mark.parametrize(
    "ending", ["eof", "malformed", "failed-start", "foreign-handle", "exited-leader"]
)
def test_startup_owner_aborts_only_its_retained_child(tmp_path, ending):
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    assert libc.prctl(37, ctypes.byref(previous), 0, 0, 0) == 0
    assert libc.prctl(36, 1, 0, 0, 0) == 0
    owner = subprocess.Popen(
        [sys.executable, str(SOURCE.with_name("startup_owner.py"))],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    sentinel = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])
    child_identity = None

    def send(*fields):
        owner.stdin.write(b"\0".join(os.fsencode(str(v)) for v in fields) + b"\0")
        owner.stdin.flush()

    try:
        marker = tmp_path / "child.json"
        code = (
            "import json, os, time; from pathlib import Path; "
            "s=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split(); "
            f"Path({str(marker)!r}).write_text(json.dumps({{'pid':os.getpid(),'start_ticks':int(s[19])}})); "
            "time.sleep(10)"
        )
        if ending == "exited-leader":
            code = "import os\nif os.fork():\n    os._exit(0)\n" + code
        send("start", tmp_path / "child.log", 3, sys.executable, "-c", code)
        assert select.select([owner.stdout], [], [], 1)[0]
        handle, pid = owner.stdout.readline().split()
        child_identity = probe.pid_identity(int(pid))
        deadline = time.monotonic() + 1
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        child_identity = json.loads(marker.read_text())
        if ending != "exited-leader":
            assert child_identity["pid"] == int(pid)
        if ending == "exited-leader":
            assert child_identity["pid"] != int(pid)
            assert probe.pid_identity(int(pid))["state"] == "Z"  # Still unreaped: PGID reserved.
            send("abort")
        elif ending == "eof":
            owner.stdin.close()
        elif ending == "malformed":
            send("invalid-command")
        elif ending == "failed-start":
            send("start", tmp_path / "failed.log", 1, "/must-not-exist-private-bfcl-executable")
        else:
            # The API accepts opaque returned handles, never arbitrary PIDs.
            assert int(handle) != sentinel.pid
            send("stop", sentinel.pid)
        assert owner.wait(timeout=2) == (0 if ending == "exited-leader" else 1)
        if ending == "exited-leader":
            deadline = time.monotonic() + 1
            while time.monotonic() < deadline:
                current = probe.pid_identity(child_identity["pid"])
                if current is None or current["state"] == "Z":
                    break
                time.sleep(0.01)
            assert current is None or current["state"] == "Z"
            with contextlib.suppress(ChildProcessError):
                os.waitpid(child_identity["pid"], 0)
        assert probe.pid_identity(child_identity["pid"]) is None
        assert sentinel.poll() is None
    finally:
        if owner.poll() is None:
            with contextlib.suppress(BrokenPipeError, ValueError):
                send("abort")
                owner.stdin.close()
            try:
                owner.wait(timeout=2)
            except subprocess.TimeoutExpired:
                owner.kill()
                owner.wait()
        if child_identity:
            try:
                fd = os.pidfd_open(child_identity["pid"])
            except ProcessLookupError:
                pass
            else:
                try:
                    current = probe.pid_identity(child_identity["pid"])
                    if current and current["start_ticks"] == child_identity["start_ticks"]:
                        signal.pidfd_send_signal(fd, signal.SIGKILL)
                finally:
                    os.close(fd)
            with contextlib.suppress(ChildProcessError):
                os.waitpid(child_identity["pid"], 0)
        sentinel.kill()
        sentinel.wait()
        for stream in (owner.stdin, owner.stdout, owner.stderr):
            stream.close()
        assert libc.prctl(36, previous.value, 0, 0, 0) == 0


def launcher_cli_without_engines(tmp_path, arm, run_timeout):
    """Run the actual launcher, replacing only process launch and arm-A curl.

    env records commands but never executes an engine, gateway or observer.
    The loopback fixture satisfies arm-B TCP/readiness checks. No model, GPU,
    production process or launcher stop/cleanup path is used.
    """
    stub_bin = tmp_path / "bin"
    stub_bin.mkdir()
    events_path = tmp_path / "launch-events.jsonl"
    launch_env = stub_bin / "env"
    launch_env.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "with open(os.environ['BFCL_FIXTURE_EVENTS'], 'a') as out:\n"
        "    out.write(json.dumps({'argv': sys.argv[1:], "
        "'RUN_TIMEOUT': os.environ.get('RUN_TIMEOUT')}) + '\\n')\n"
    )
    launch_env.chmod(0o755)
    curl = stub_bin / "curl"
    curl.write_text("#!/bin/sh\nexit 0\n")
    curl.chmod(0o755)
    run_dir = tmp_path / "state with spaces"
    env = {
        **os.environ,
        "PATH": f"{stub_bin}{os.pathsep}{os.environ['PATH']}",
        "BFCL_FIXTURE_EVENTS": str(events_path),
        "BFCL_RUN_DIR": str(run_dir),
        "BFCL_MODEL": MODEL,
        "BFCL_STARTUP_TIMEOUT": "1",
        "VLLM_BIN": "/must-not-run-a-real-engine",
        "VLLM_PYTHON": "/must-not-run-a-real-engine",
        "SMG_LAUNCH": "/must-not-run-a-real-gateway",
        "CUDA_VISIBLE_DEVICES": "",
        "HIP_VISIBLE_DEVICES": "",
        "ROCR_VISIBLE_DEVICES": "",
    }
    if run_timeout is None:
        env.pop("RUN_TIMEOUT", None)
    else:
        env["RUN_TIMEOUT"] = run_timeout
    with server({"/readiness": response(readiness())}) as (url, _):
        port = url.rsplit(":", 1)[1]
        for name in ("ARM_A_PORT", "ARM_B_GRPC_PORT", "ARM_B_GW_PORT", "ARM_B_METRICS_PORT"):
            env[f"BFCL_{name}"] = port
        result = subprocess.run(
            ["bash", str(SOURCE.with_name("launch_arm.sh")), arm],
            env=env,
            capture_output=True,
            timeout=10,
        )
    expected_events = 1 if arm == "a" else 3
    deadline = time.monotonic() + 1
    while True:
        events = (
            [json.loads(line) for line in events_path.read_text().splitlines()]
            if events_path.exists()
            else []
        )
        if len(events) >= expected_events or result.returncode or time.monotonic() >= deadline:
            return result, events, run_dir
        time.sleep(0.01)


@pytest.mark.parametrize("arm", ["a", "b"])
@pytest.mark.parametrize(
    "run_timeout,expected",
    [
        pytest.param(None, 14400, id="default"),
        pytest.param("", 14400, id="empty-default"),
        pytest.param("0", 21600, id="uncapped-scoring"),
        pytest.param("14400", 14400, id="normal"),
        pytest.param("21600", 21600, id="observer-cap"),
        pytest.param("21601", 21600, id="above-cap"),
        pytest.param("00008", 8, id="leading-zero-decimal"),
        pytest.param("0000", 21600, id="leading-zero-uncapped"),
        pytest.param("9" * 100, 21600, id="beyond-shell-integer-range"),
        pytest.param(" +0008 ", 8, id="integer-whitespace-and-plus"),
        pytest.param("1_000", 1000, id="python-integer-separator"),
    ],
)
def test_launcher_bounds_observer_before_start_without_changing_scoring(
    tmp_path, arm, run_timeout, expected
):
    result, events, _ = launcher_cli_without_engines(tmp_path, arm, run_timeout)
    assert result.returncode == 0, result.stderr
    assert len(events) == (1 if arm == "a" else 3)
    # Preserve the exact scoring setting, including zero and decimal spelling.
    assert all(event["RUN_TIMEOUT"] == run_timeout for event in events)
    bounded = f"RUN_TIMEOUT validated; arm B observer bound: {expected}s".encode()
    assert bounded in result.stderr
    assert result.stderr.index(bounded) < result.stderr.index(b"started arm_")
    if arm == "b":
        observer = next(event for event in events if "--duration" in event["argv"])
        assert observer["argv"][observer["argv"].index("--duration") + 1] == str(expected)


@pytest.mark.parametrize("arm", ["a", "b"])
@pytest.mark.parametrize("run_timeout", ["-1", "1.5", "1e4", "not-an-integer", "   "])
def test_launcher_rejects_invalid_timeout_before_any_process_start(tmp_path, arm, run_timeout):
    result, events, run_dir = launcher_cli_without_engines(tmp_path, arm, run_timeout)
    assert result.returncode == 2
    assert b"RUN_TIMEOUT" in result.stderr
    assert events == []
    assert not run_dir.exists()


def _readiness_with_fake_fetch(monkeypatch, data, status=200):
    calls = []

    def fake_fetch(base, path, deadline=None):
        calls.append((base, path, deadline))
        return {"http_status": status}, json.dumps(data).encode()

    monkeypatch.setattr(probe, "fetch", fake_fetch)
    selected = probe.selected_json("http://127.0.0.1:1234", "/readiness", MODEL)
    assert calls == [("http://127.0.0.1:1234", "/readiness", None)]
    return selected


@pytest.mark.parametrize(
    "shape,value,expected",
    [
        pytest.param("value", 1, True, id="int-one"),
        pytest.param("value", True, False, id="bool-true"),
        pytest.param("value", False, False, id="bool-false"),
        pytest.param("missing", None, False, id="missing"),
        pytest.param("value", None, False, id="none"),
        pytest.param("value", "1", False, id="string"),
        pytest.param("value", 1.5, False, id="float"),
        pytest.param("value", 0, False, id="int-zero"),
        pytest.param("value", -1, False, id="negative-int"),
    ],
)
def test_readiness_counts_require_real_integers(monkeypatch, shape, value, expected):
    data = readiness()
    if shape == "missing":
        del data["models"][0]["healthy_workers"]
    else:
        data["models"][0]["healthy_workers"] = value
    selected = _readiness_with_fake_fetch(monkeypatch, data)
    assert probe.ready(selected) is expected
    if shape == "value" and type(value) is int:
        assert selected["models"][0]["healthy_workers"] == value
    else:
        assert "healthy_workers" not in selected["models"][0]


def test_ready_defends_against_direct_noninteger_counts():
    for value in [True, False, None, "1", 1.5]:
        selected = {
            "http_status": 200,
            "ready": True,
            "models": [{"ready": True, "healthy_workers": value}],
        }
        assert not probe.ready(selected)
    assert probe.ready(
        {"http_status": 200, "ready": True, "models": [{"ready": True, "healthy_workers": 1}]}
    )


@pytest.mark.parametrize(
    "value,expected",
    [
        pytest.param(True, True, id="bool-true"),
        pytest.param(False, False, id="bool-false"),
        pytest.param(1, False, id="int-one"),
        pytest.param("true", False, id="string"),
        pytest.param(None, False, id="none"),
    ],
)
def test_model_ready_marker_requires_a_real_boolean(monkeypatch, value, expected):
    data = readiness()
    data["models"][0]["ready"] = value
    selected = _readiness_with_fake_fetch(monkeypatch, data)
    assert probe.ready(selected) is expected
    assert ("ready" in selected["models"][0]) is (type(value) is bool)


def test_total_worker_count_retains_only_real_integers(monkeypatch):
    for value in [True, False, None, "1", 1.5, 0, 1]:
        data = readiness()
        data["models"][0]["total_workers"] = value
        row = _readiness_with_fake_fetch(monkeypatch, data)["models"][0]
        assert ("total_workers" in row) is (type(value) is int)
        if type(value) is int:
            assert row["total_workers"] == value


def test_fake_readiness_preserves_http_target_and_gateway_contract(monkeypatch):
    for data, status, expected in [
        (readiness(), 200, True),
        (readiness("other/model"), 200, False),
        (readiness(), 503, False),
        ({**readiness(), "status": "not_ready"}, 200, False),
    ]:
        assert probe.ready(_readiness_with_fake_fetch(monkeypatch, data, status)) is expected
