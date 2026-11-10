"""Offline native-arm commands plus real CPU session ownership/cleanup controls."""

import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("native_arms", ROOT / "scripts/tau2/native_arms.py")
arms = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(arms)


@pytest.fixture
def invocation(tmp_path, monkeypatch):
    run = tmp_path / "run"
    capture = tmp_path / "capture.jsonl"
    fake = tmp_path / "engine"
    fake.write_text(
        f"#!{sys.executable}\n"
        "import json,os,sys,time,subprocess,signal\n"
        "if os.environ.get('ENGINE_IGNORE_TERM'): signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "with open(os.environ['CAPTURE'],'a') as f:\n"
        " f.write(json.dumps({'argv':sys.argv[1:], 'env':dict(os.environ), 'pid':os.getpid()})+'\\n')\n"
        "if os.environ.get('ENGINE_EXIT'): sys.exit(7)\n"
        "if os.environ.get('ENGINE_CHILD'):\n"
        " c=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)'])\n"
        " open(os.environ['CHILD_PID'],'w').write(str(c.pid))\n"
        "if os.environ.get('ENGINE_PARENT_EXIT'):\n"
        " time.sleep(.1);sys.exit(0)\n"
        "time.sleep(120)\n"
    )
    fake.chmod(0o755)
    for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        monkeypatch.delenv(key, raising=False)
    env = {
        "TAU2_RUN_DIR": str(run),
        "TAU2_SCOPE": "job-123-1-qwen",
        "VLLM_BIN": str(fake),
        "VLLM_PYTHON": str(fake),
        "SMG_LAUNCH": str(fake),
        "CAPTURE": str(capture),
        "TAU2_MODEL": "Qwen/Qwen3.8-27B",
        "TAU2_TP": "2",
        "TAU2_GPU": "0,1",
        "TAU2_GPU_A": "0,1",
        "TAU2_GPU_B": "2,3",
        "TAU2_MAX_MODEL_LEN": "auto",
        "TAU2_STARTUP_TIMEOUT": ".25",
        "CHILD_PID": str(tmp_path / "child.pid"),
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    yield run, capture, env
    if run.exists():
        # Test failures also release every real owned session.
        arms.cleanup(run, env["TAU2_SCOPE"])


def call(monkeypatch, arm):
    monkeypatch.setattr(sys, "argv", ["native_arms.py", arm])
    return arms.main()


def wait_capture(path, count):
    until = time.monotonic() + 3
    while time.monotonic() < until:
        if path.exists():
            lines = path.read_text().splitlines()
            if len(lines) >= count:
                return [json.loads(line) for line in lines]
        time.sleep(0.02)
    raise AssertionError("fake engine did not execute")


@pytest.mark.parametrize("vendor", ["nvidia", "amd"])
@pytest.mark.parametrize("arm", ["a", "b"])
def test_actual_executable_commands_and_visibility(invocation, monkeypatch, vendor, arm):
    run, capture, _ = invocation
    monkeypatch.setenv("TAU2_GPU_VENDOR", vendor)
    selected = "0,1" if arm == "a" else "2,3"
    monkeypatch.setenv("TAU2_GPU", selected)
    monkeypatch.setenv("TAU2_VLLM_EXTRA", '--enforce-eager --revision "fixed revision"')
    monkeypatch.setattr(arms, "wait_ready", lambda *_: None)
    assert call(monkeypatch, arm) == 0
    calls = wait_capture(capture, 1 if arm == "a" else 2)
    if arm == "b":
        calls.sort(key=lambda item: item["argv"][0] != "-m")
    engine = calls[0]
    assert engine["argv"][:2] == (
        ["serve", "Qwen/Qwen3.8-27B"] if arm == "a" else ["-m", "vllm.entrypoints.grpc_server"]
    )
    for flag, value in [
        ("--served-model-name", "Qwen/Qwen3.8-27B"),
        ("--tensor-parallel-size", "2"),
        ("--max-model-len", "auto"),
        ("--revision", "fixed revision"),
    ]:
        assert engine["argv"][engine["argv"].index(flag) + 1] == value
    key = "HIP_VISIBLE_DEVICES" if vendor == "amd" else "CUDA_VISIBLE_DEVICES"
    assert all(c["env"][key] == selected for c in calls)
    if vendor == "amd":
        assert all(
            "CUDA_VISIBLE_DEVICES" not in c["env"] and "ROCR_VISIBLE_DEVICES" not in c["env"]
            for c in calls
        )
    if arm == "a":
        assert engine["argv"][engine["argv"].index("--tool-call-parser") + 1] == "qwen3_xml"
    else:
        assert "--tool-call-parser" not in engine["argv"]
        assert calls[1]["argv"][calls[1]["argv"].index("--tool-call-parser") + 1] == "qwen_xml"
        assert calls[1]["argv"][calls[1]["argv"].index("--worker-urls") + 1].startswith(
            "grpc://127.0.0.1:"
        )
    assert call(monkeypatch, "stop") == 0
    assert call(monkeypatch, "stop") == 0
    assert not list(run.glob("*.pid.json"))


@pytest.mark.parametrize(
    "key,value",
    [
        ("TAU2_GPU", "0,0"),
        ("TAU2_GPU", "-1,2"),
        ("TAU2_TP", "1"),
        ("TAU2_GPU_B", "1,2"),
        ("TAU2_GPU_B", "2"),
        ("TAU2_GPU_VENDOR", "other"),
        ("HIP_VISIBLE_DEVICES", "0,1"),
        ("CUDA_VISIBLE_DEVICES", "0,1"),
        ("ROCR_VISIBLE_DEVICES", "0,1"),
        ("TAU2_STARTUP_TIMEOUT", "nan"),
        ("TAU2_VLLM_EXTRA", "'"),
    ],
)
def test_invalid_before_any_spawn(invocation, monkeypatch, key, value):
    run, capture, _ = invocation
    monkeypatch.setenv("TAU2_GPU_VENDOR", "amd")
    monkeypatch.setenv(key, value)
    with pytest.raises((ValueError, TimeoutError)):
        call(monkeypatch, "a")
    assert not capture.exists()
    assert not list(run.glob("*.pid.json"))


@pytest.mark.parametrize(
    "key", ["TAU2_VLLM_REASONING_PARSER", "TAU2_SMG_TOOL_PARSER", "TAU2_SMG_REASONING_PARSER"]
)
def test_explicit_empty_parser_remains_empty(invocation, monkeypatch, key):
    _, capture, _ = invocation
    monkeypatch.setenv(key, "")
    monkeypatch.setattr(arms, "wait_ready", lambda *_: None)
    arm = "a" if "VLLM" in key else "b"
    assert call(monkeypatch, arm) == 0
    calls = wait_capture(capture, 1 if arm == "a" else 2)
    flag = "--tool-call-parser" if "TOOL" in key else "--reasoning-parser"
    assert flag not in calls[-1]["argv"]


@pytest.mark.parametrize("failure", ["timeout", "worker_exit", "gateway_failure", "cancel"])
def test_partial_startup_releases_sessions(invocation, monkeypatch, failure):
    run, capture, env = invocation
    monkeypatch.setenv("ENGINE_CHILD", "1")
    if failure == "worker_exit":
        monkeypatch.setenv("ENGINE_EXIT", "1")
    elif failure in {"gateway_failure", "cancel"}:

        def ready(_root, _scope, record, *_):
            wait_capture(capture, 1)
            if failure == "cancel":
                raise KeyboardInterrupt("cancelled")
            if "gateway" in record.name:
                raise RuntimeError("gateway failed")

        monkeypatch.setattr(arms, "wait_ready", ready)
    with pytest.raises((TimeoutError, RuntimeError, KeyboardInterrupt)):
        call(monkeypatch, "b")
    assert not list(run.glob("*.pid.json"))
    childfile = Path(env["CHILD_PID"])
    if childfile.exists():
        pid = int(childfile.read_text())
        state = Path(f"/proc/{pid}/stat")
        assert not state.exists() or state.read_text().rsplit(")", 1)[1].split()[0] == "Z"


def test_engine_leader_exit_still_reaps_child(invocation, monkeypatch):
    run, capture, env = invocation
    monkeypatch.setenv("ENGINE_CHILD", "1")
    monkeypatch.setenv("ENGINE_PARENT_EXIT", "1")
    monkeypatch.setattr(arms, "wait_ready", lambda *_: None)
    assert call(monkeypatch, "a") == 0
    wait_capture(capture, 1)
    time.sleep(0.25)
    assert call(monkeypatch, "stop") == 0
    pid = int(Path(env["CHILD_PID"]).read_text())
    state = Path(f"/proc/{pid}/stat")
    assert not state.exists() or state.read_text().rsplit(")", 1)[1].split()[0] == "Z"
    assert not list(run.glob("*.pid.json"))


def test_foreign_identity_and_cookie_never_signalled(invocation, monkeypatch):
    run, _, env = invocation
    arms.directory(run, env["TAU2_SCOPE"])
    foreign = subprocess.Popen(
        [sys.executable, "-c", "import time;time.sleep(60)"], start_new_session=True
    )
    try:
        record = run / "foreign.pid.json"
        value = {
            "scope": env["TAU2_SCOPE"],
            "uid": os.getuid(),
            "cookie": "foreign",
            "identity": arms.proc(foreign.pid),
        }
        record.write_text(json.dumps(value))
        assert not arms.cleanup(run, env["TAU2_SCOPE"])
        assert foreign.poll() is None
        value["identity"]["start"] -= 1
        record.write_text(json.dumps(value))
        assert not arms.cleanup(run, env["TAU2_SCOPE"])
        assert foreign.poll() is None
        record.unlink()
    finally:
        foreign.terminate()
        foreign.wait(timeout=3)


def test_missing_owner_receipt_is_not_cleanup_success(invocation):
    run, _, env = invocation
    arms.directory(run, env["TAU2_SCOPE"])
    record = run / "gone.pid.json"
    record.write_text(
        json.dumps(
            {
                "scope": env["TAU2_SCOPE"],
                "uid": os.getuid(),
                "cookie": "gone",
                "identity": {"pid": 2147483646, "sid": 2147483646, "pgid": 2147483646, "start": 1},
            }
        )
    )
    assert not arms.cleanup(run, env["TAU2_SCOPE"])
    assert record.exists()
    record.unlink()


def test_scope_and_symlink_refused(invocation, monkeypatch, tmp_path):
    run, _, env = invocation
    arms.directory(run, env["TAU2_SCOPE"])
    with pytest.raises(ValueError, match="another invocation"):
        arms.directory(run, "other-job")
    (run / "foreign.pid.json").symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError, match="unsafe"):
        call(monkeypatch, "stop")
    (run / "foreign.pid.json").unlink()


def test_workflow_matrix_and_caller_contract():
    # Execute the actual matrix builder, without Actions/network/provider calls.
    text = (ROOT / ".github/workflows/nightly-tau2.yml").read_text()
    source = text.split("          import json, os\n", 1)[1].split("          PY\n", 1)[0]
    source = "import json, os\n" + "\n".join(line[10:] for line in source.splitlines())
    for enabled, expected in [("", 2), ("true", 3)]:
        proc = subprocess.run(
            [sys.executable, "-c", source],
            env=dict(
                os.environ,
                AMD_ENABLED=enabled,
                AMD_RUNNER="4-gpu-mi300x",
                EVENT_NAME="schedule",
                ONLY="",
            ),
            text=True,
            capture_output=True,
            check=True,
        )
        legs = json.loads(proc.stdout.removeprefix("matrix="))["include"]
        assert len(legs) == expected
        if enabled:
            nv, amd = legs[0], legs[-1]
            assert {
                k: v for k, v in amd.items() if k not in {"name", "runner", "vendor", "container"}
            } == {k: v for k, v in nv.items() if k not in {"name", "runner"}}
            assert "@sha256:749f6" in amd["container"]["image"]
            assert "/dev/kfd" in amd["container"]["options"]
    assert "shell: bash" in text
    assert "timeout-minutes: 360" in text
    assert (
        text.count(
            "TAU2_SCOPE: tau2-${{ github.run_id }}-${{ github.run_attempt }}-${{ matrix.name }}"
        )
        == 2
    )
    assert "run: bash scripts/tau2/launch_arms.sh stop || true" not in text
    assert "bash scripts/tau2/launch_arms.sh stop 9>&- || return 2" in text
    assert 'if [ "$TAU2_GPU_VENDOR" = amd ]; then' in text
    assert 'if [ "$TAU2_GPU_VENDOR" != amd ]; then' in text
    assert "run: bash scripts/ci_setup_vllm_rocm_image.sh" in text
    assert "if: matrix.vendor != 'amd' && matrix.vllm_commit != ''" in text
    assert '--expected-tasks "$RUNNER_TEMP/tau2_expected_tasks.json" --seed 300' in text
    assert "--user-llm gpt-5.2" in text
    assert "--ledger" in text and "--initialize" in text


def test_cli_sigterm_cleans_actual_sessions(invocation):
    run, capture, env = invocation
    child = subprocess.Popen(
        ["bash", str(ROOT / "scripts/tau2/launch_arms.sh"), "a"],
        env=dict(os.environ, TAU2_STARTUP_TIMEOUT="30", ENGINE_CHILD="1"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        wait_capture(capture, 1)
        child.send_signal(signal.SIGTERM)
        _, error = child.communicate(timeout=20)
        assert child.returncode == 2, error
        assert not list(run.glob("*.pid.json"))
        pidfile = Path(env["CHILD_PID"])
        if pidfile.exists():
            state = Path(f"/proc/{pidfile.read_text()}/stat")
            assert not state.exists() or state.read_text().rsplit(")", 1)[1].split()[0] == "Z"
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=3)


def test_pidfd_pinned_before_validation_never_numeric_signal(invocation, monkeypatch):
    run, capture, env = invocation
    monkeypatch.setattr(arms, "wait_ready", lambda *_: None)
    assert call(monkeypatch, "a") == 0
    wait_capture(capture, 1)
    events = []
    original_open, original_owned = os.pidfd_open, arms.owned
    original_send = signal.pidfd_send_signal

    def pin(pid, *args):
        events.append("pin")
        return original_open(pid, *args)

    def verify(*args):
        events.append("verify")
        return original_owned(*args)

    def send(fd, sig, *args):
        events.append("send")
        assert Path(f"/proc/self/fd/{fd}").readlink() == Path("anon_inode:[pidfd]")
        return original_send(fd, sig, *args)

    monkeypatch.setattr(os, "pidfd_open", pin)
    monkeypatch.setattr(arms, "owned", verify)
    monkeypatch.setattr(signal, "pidfd_send_signal", send)
    monkeypatch.setattr(os, "kill", lambda *_: pytest.fail("numeric-PID signal"))
    assert arms.cleanup(run, env["TAU2_SCOPE"])
    assert events == ["pin", "verify", "send"]


def test_recycled_identity_after_pin_cannot_receive_signal(invocation, monkeypatch):
    run, capture, env = invocation
    monkeypatch.setattr(arms, "wait_ready", lambda *_: None)
    assert call(monkeypatch, "a") == 0
    wait_capture(capture, 1)
    original_proc = arms.proc

    def recycled(pid):
        info = original_proc(pid)
        if info:
            info["start"] += 1
        return info

    with monkeypatch.context() as control:
        control.setattr(arms, "proc", recycled)
        control.setattr(
            signal, "pidfd_send_signal", lambda *_: pytest.fail("foreign identity signalled")
        )
        assert not arms.cleanup(run, env["TAU2_SCOPE"])
    assert arms.cleanup(run, env["TAU2_SCOPE"])


def test_nv_sequential_same_scope(invocation, monkeypatch):
    _, capture, _ = invocation
    monkeypatch.setattr(arms, "wait_ready", lambda *_: None)
    assert call(monkeypatch, "a") == 0
    wait_capture(capture, 1)
    assert call(monkeypatch, "stop") == 0
    assert call(monkeypatch, "b") == 0
    wait_capture(capture, 3)
    assert call(monkeypatch, "stop") == 0


def test_term_resistant_owned_descendants_are_killed(invocation, monkeypatch):
    run, capture, env = invocation
    monkeypatch.setenv("ENGINE_CHILD", "1")
    monkeypatch.setenv("ENGINE_IGNORE_TERM", "1")
    monkeypatch.setattr(arms, "wait_ready", lambda *_: None)
    assert call(monkeypatch, "a") == 0
    wait_capture(capture, 1)
    deadline = time.monotonic() + 2
    while not Path(env["CHILD_PID"]).exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    start = time.monotonic()
    assert call(monkeypatch, "stop") == 0
    assert 9 <= time.monotonic() - start < 18
    pid = Path(env["CHILD_PID"]).read_text()
    state = Path(f"/proc/{pid}/stat")
    assert not state.exists() or state.read_text().rsplit(")", 1)[1].split()[0] == "Z"
    assert not list(run.glob("*.pid.json"))
