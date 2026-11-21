"""READ host ownership/unknown-RPC/cancellation contracts, with no Docker or GPU."""

import ast
import datetime
import importlib.util
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("read_host", ROOT / "scripts/ci_moriio_read_host.py")
host = importlib.util.module_from_spec(spec)
spec.loader.exec_module(host)
CID = "a" * 64
IMAGE = "example.invalid/read@sha256:" + "b" * 64


def timestamp(ns):
    return (
        datetime.datetime.fromtimestamp(ns // 10**9, datetime.UTC).strftime("%Y-%m-%dT%H:%M:%S")
        + f".{ns % 10**9:09d}Z"
    )


def metadata(state, running=False):
    return {
        "id": CID,
        "image": "sha256:IMAGE",
        "name": "/" + state["name"],
        "labels": state["labels"],
        "created": timestamp(state["create_after_ns"]),
        "state": {"Pid": 999 if running else 0, "Running": running},
        "entrypoint": ["/usr/bin/python3"],
        "cmd": ["-I", "/workspace/scripts/ci_moriio_read_host.py", "inside", "--image", IMAGE],
        "user": "1001:1001",
        "healthcheck": {"Test": ["NONE"]},
        "host": {
            "NetworkMode": "host",
            "Runtime": "runc",
            "NanoCpus": 16000000000,
            "Memory": 128 * 1024**3,
            "MemorySwap": 128 * 1024**3,
            "PidsLimit": 4096,
            "CapDrop": ["ALL"],
            "IpcMode": "private",
            "CgroupnsMode": "host",
            "ReadonlyRootfs": True,
            "SecurityOpt": ["no-new-privileges"],
            "Ulimits": [{"Name": "memlock", "Hard": -1, "Soft": -1}],
            "ShmSize": 32 * 1024**3,
            "Devices": [
                {"PathOnHost": x, "PathInContainer": x, "CgroupPermissions": "rwm"}
                for x in state["devices"]
            ],
        },
        "mounts": [
            {"Source": k, "Destination": v[0], "RW": v[1]} for k, v in state["mounts"].items()
        ],
    }


@pytest.fixture
def job(tmp_path):
    value = host.Job(tmp_path, create=True)
    value.state = {
        "name": "smg-read-" + "c" * 32,
        "image": IMAGE,
        "image_id": "sha256:IMAGE",
        "labels": {"123abc": "", "smg.read.invocation": "c" * 32},
        "devices": [
            "/dev/kfd",
            "/dev/dri/renderD128",
            "/dev/infiniband/uverbs0",
            "/dev/infiniband/rdma_cm",
        ],
        "user": "1001:1001",
        "mounts": {"/OWN": ("/workspace", True)},
        "create_after_ns": time.time_ns() - 1000000,
        "create_intent": True,
    }
    value.persist()
    return value


@pytest.mark.parametrize("digits", [3, 6, 9])
def test_docker_fractional_created_portable(digits):
    assert host.created_ns("2026-11-14T09:02:25." + "123456789"[:digits] + "Z") % 10**9 == int(
        "123456789"[:digits].ljust(9, "0")
    )


@pytest.mark.parametrize(
    "text", ["2026-11-14T09:02:25.1234567890Z", "2026-11-14T09:02:25+01:00", "2026-13-14T09:02:25Z"]
)
def test_bad_created_rejected(text):
    with pytest.raises((RuntimeError, ValueError)):
        host.created_ns(text)


def test_owned_exact_positive(job):
    job.owned(metadata(job.state))


@pytest.mark.parametrize(
    "field,change",
    [
        ("labels", {"other": ""}),
        ("image", "foreign"),
        ("created", "2020-01-01T00:00:00Z"),
        ("name", "/foreign"),
        ("user", "0"),
    ],
)
def test_foreign_identity_never_adopted(job, field, change):
    value = metadata(job.state)
    value[field] = change
    with pytest.raises(RuntimeError):
        job.owned(value)


@pytest.mark.parametrize(
    "change",
    [
        lambda c: c["host"].update(NetworkMode="bridge"),
        lambda c: c["host"].update(CapAdd=["NET_ADMIN"]),
        lambda c: c["host"].update(PidMode="host"),
        lambda c: c["mounts"].append({"Source": "/", "Destination": "/host", "RW": True}),
        lambda c: c["host"]["Devices"].append({"PathOnHost": "/dev/dri/renderD999"}),
    ],
)
def test_runtime_boundary_rejects_extra_authority(job, change):
    value = metadata(job.state)
    change(value)
    with pytest.raises(RuntimeError):
        host.validate_runtime(value, job.state)


def test_unknown_create_reconciles_exact_name_then_removes_once(job, monkeypatch):
    values = iter([metadata(job.state), None])
    calls = []
    monkeypatch.setattr(job, "inspect", lambda: next(values))
    monkeypatch.setattr(job, "docker", lambda *a, **k: calls.append(a))
    job.cleanup()
    assert calls == [("remove", "rm", "-v", CID)] and job.state["closed"]
    assert job.state["remove_intent"]


def test_no_create_does_not_inspect_or_clean_foreign_name(job, monkeypatch):
    job.state.pop("create_intent")
    monkeypatch.setattr(
        job, "inspect", lambda: pytest.fail("No submitted creation, no resource adoption")
    )
    job.cleanup()
    assert job.state["closed"]


@pytest.mark.parametrize("running", [False, True])
def test_unknown_cleanup_ack_is_never_resent(job, monkeypatch, running):
    calls = []
    monkeypatch.setattr(job, "inspect", lambda: metadata(job.state, running))

    def unknown(*a, **k):
        calls.append(a)
        raise RuntimeError("lost ACK")

    monkeypatch.setattr(job, "docker", unknown)
    with pytest.raises(RuntimeError):
        job.cleanup()
    with pytest.raises(RuntimeError):
        job.cleanup()
    assert len(calls) == 1


def test_unknown_remove_then_absent_is_read_only_reconciliation(job, monkeypatch):
    job.state["remove_intent"] = True
    monkeypatch.setattr(job, "inspect", lambda: None)
    monkeypatch.setattr(job, "docker", lambda *a, **k: pytest.fail("No resend"))
    job.cleanup()
    assert job.state["closed"]


@pytest.mark.parametrize(
    "options",
    [
        "--privileged true",
        "--network host",
        "--volume /:/host",
        "--env GITHUB_TOKEN=secret",
        "--device /dev/sda",
        "--entrypoint /bin/bash",
    ],
)
def test_raw_docker_option_injection_rejected(options):
    with pytest.raises(ValueError):
        host.rdma_declarations(options)


def test_only_explicit_rdma_options_accepted():
    assert host.rdma_declarations(
        "--device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size=32g "
        "--volume /models:/models --env HF_HOME=/models --env HF_TOKEN --device=/dev/infiniband"
    ) == ["/dev/infiniband"]
    with pytest.raises(ValueError):
        host.rdma_declarations("--device=/dev/kfd")


def test_read_workflow_is_host_only_and_preserves_original_suite_contract():
    flow = yaml.safe_load((ROOT / ".github/workflows/e2e-moriio-read-host.yml").read_text())
    job = flow["jobs"]["run"]
    assert "container" not in job and job["timeout-minutes"] == 170
    source = (ROOT / "scripts/ci_moriio_read_host.py").read_text()
    tree = ast.parse(source)
    call = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "command" for t in n.targets)
    )
    args = [ast.literal_eval(n) for n in call.value.elts[1:]]
    assert args == [
        "-I",
        "-B",
        "-m",
        "pytest",
        "e2e_test/router/test_pd_topologies.py",
        "--ignore=e2e_test/benchmarks",
        "--ignore=e2e_test/bindings_go",
        "--reruns",
        "2",
        "--reruns-delay",
        "5",
        "-s",
        "-vv",
        "--junitxml=e2e-logs/moriio-read-junit.xml",
    ]
    assert "timeout=9000" in source
    assert "e2e_test/router/test_pd_topologies.py" in source and "--test-filter" not in source
    assert '--qualification-env "$GITHUB_ENV"' in source
    assert "flock" not in source and "systemctl" not in source
    cleanup = next(
        x for x in job["steps"] if x.get("name") == "Reconcile exact READ container cleanup"
    )
    assert "always()" in cleanup["if"]
    nightly = yaml.safe_load((ROOT / ".github/workflows/nightly-pd.yml").read_text())
    assert nightly["jobs"]["pd-topologies-read"]["with"]["image"] == "${{ matrix.container_image }}"
    assert (
        nightly["jobs"]["pd-topologies-read"]["with"]["options_json"]
        == "${{ needs.plan.outputs.amd_read_rdma_options_json }}"
    )


@pytest.mark.parametrize("failure", [KeyboardInterrupt, OSError])
def test_postspawn_setup_fault_reaps_only_own_cli_and_saves_raw(job, monkeypatch, failure):
    class Child:
        pid = 99999991
        returncode = None

        def __init__(self):
            streams = []
            for data in (b"original stdout", b"original stderr"):
                rd, wr = os.pipe()
                os.write(wr, data)
                os.close(wr)
                streams.append(os.fdopen(rd, "rb"))
            self.stdout, self.stderr = streams

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            self.returncode = -9
            return -9

    child = Child()
    signals = []
    monkeypatch.setattr(host.subprocess, "Popen", lambda *a, **k: child)

    def identity(pid):
        raise failure("synthetic post-spawn failure")

    monkeypatch.setattr(host, "identity", identity)

    def killpg(pid, signum):
        assert pid == child.pid
        signals.append(signum)
        if signum == 0 and child.returncode is not None:
            raise ProcessLookupError()

    monkeypatch.setattr(host.os, "killpg", killpg)
    with pytest.raises(RuntimeError, match="unknown"):
        job.command("fault", ["FAKE_NOT_EXECUTED"])
    ack = json.loads((job.root / "001-fault-ACK.json").read_text())
    assert ack["error"] == failure.__name__ and ack["reaped"] and ack["group_gone"]
    assert (job.root / "001-fault.stdout").read_bytes() == b"original stdout"
    assert (job.root / "001-fault.stderr").read_bytes() == b"original stderr"
    assert child.stdout.closed and child.stderr.closed and signals == [host.signal.SIGTERM, 0]


def test_cancellation_during_wait_reaches_exact_CID_cleanup(tmp_path, monkeypatch):
    workspace = tmp_path / "work"
    workspace.mkdir()
    (tmp_path / "output").touch()
    monkeypatch.chdir(workspace)
    for key, value in {
        "RUNNER_TEMP": str(tmp_path),
        "GITHUB_OUTPUT": str(tmp_path / "output"),
        "GITHUB_RUN_ID": "123",
        "GITHUB_RUN_ATTEMPT": "2",
        "GITHUB_TOKEN": "test-secret",
        "GITHUB_ACTOR": "test",
    }.items():
        monkeypatch.setenv(key, value)
    admission = {
        "label": "123abc",
        "cycle": "OLD_CURRENT_NORMAL_CYCLE",
        "worker_pid": 111,
        "worker_start": 222,
        "deadline": time.monotonic() + 20000,
    }
    monkeypatch.setattr(host, "binding", lambda p, command: (admission, {}))
    monkeypatch.setattr(
        host,
        "selected_devices",
        lambda p: (
            [
                {
                    "render": f"/dev/dri/renderD{128 + i}",  # codespell:ignore renderd
                    "uid": i + 1,
                    "pci": f"0000:{10 + i:02x}:00.0",
                }
                for i in range(4)
            ],
            ["/dev/infiniband/rdma_cm", "/dev/infiniband/uverbs0"],
        ),
    )
    original = host.Job
    instances = []

    class FakeDockerJob(original):
        present = False
        running = False

        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.calls = []
            instances.append(self)

        def inspect(self):
            if not self.present:
                return None
            value = metadata(self.state, self.running)
            value["image"] = self.state["image_id"]
            value["user"] = self.state["user"]
            return value

        def docker(self, name, *args, **kwargs):
            self.calls.append(name)
            if name == "pull":
                return b""
            if name == "image-id":
                return ("sha256:" + "d" * 64 + "\n").encode()
            if name == "image-volumes":
                return b"null\n"
            if name == "image-labels":
                return b'{"base-image-label":"kept"}\n'
            if name == "create":
                initial = (self.root / "environment").read_text().splitlines()
                assert (
                    "ROCR_VISIBLE_DEVICES=" + ",".join(f"GPU-{i:016x}" for i in range(1, 5))
                    in initial
                )
                self.present = True
                return (CID + "\n").encode()
            if name == "start":
                self.running = True
                return b""
            if name == "wait":
                raise KeyboardInterrupt("synthetic cancellation")
            if name == "stop":
                self.running = False
                return b""
            if name == "remove":
                self.present = False
                return b""
            pytest.fail(name)

    monkeypatch.setattr(host, "Job", FakeDockerJob)
    monkeypatch.setattr(host.os, "readlink", lambda p: "net:[HOST]")
    monkeypatch.setattr(host, "identity", lambda p: {"pid": p, "start": 33})
    previous = Path.read_text

    def read(path, *args, **kwargs):
        if str(path) == "/proc/999/cgroup":
            return f"0::/system.slice/docker-{CID}.scope\n"
        return previous(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    with pytest.raises(KeyboardInterrupt):
        host.run_host(
            SimpleNamespace(image=IMAGE, options_json=json.dumps("--device=/dev/infiniband"))
        )
    job = instances[0]
    assert job.calls == [
        "pull",
        "image-id",
        "image-volumes",
        "image-labels",
        "create",
        "start",
        "wait",
        "stop",
        "remove",
    ]
    assert job.state["labels"]["base-image-label"] == "kept"
    assert job.state["closed"] and job.state["private_files_removed"]
    assert not (job.root / "environment").exists()
    assert not (job.root / "home").exists() and not (job.root / "cache").exists()
    assert json.loads((job.root / "state.json").read_text())["primary_error"] == "KeyboardInterrupt"


def test_narrow_verifier_failure_has_no_metadata_or_docker_fallback(monkeypatch, tmp_path):
    seen = []

    def command(name, argv, *, seconds):
        seen.append((name, argv, seconds))
        return 1, b"", b"sudo policy denied"

    monkeypatch.setattr(host, "protected", lambda p: pytest.fail("no metadata fallback"))
    with pytest.raises(RuntimeError, match="narrow Worker verifier failed"):
        host.binding(tmp_path, command)
    assert seen == [("verify-started", host.VERIFY, 30)]


def test_pre_admission_failure_retains_ack_scope_and_never_calls_docker(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "outputs"))

    def denied(workspace, command):
        raise RuntimeError("existing sudo policy missing")

    monkeypatch.setattr(host, "binding", denied)
    monkeypatch.setattr(host.Job, "inspect", lambda *a: pytest.fail("no Docker before admission"))
    with pytest.raises(RuntimeError, match="existing sudo policy"):
        host.run_host(SimpleNamespace())
    root = next(tmp_path.glob("smg-read-*"))
    state = json.loads((root / "state.json").read_text())
    assert state["closed"] and state["private_files_removed"] and not state["create_intent"]
    host.Job(root).cleanup()  # always-step remains local-only after the same failure


def test_atomic_state_failure_preserves_prior_complete_json(job, monkeypatch):
    job.persist()
    previous = (job.root / "state.json").read_bytes()
    job.state["new_field"] = "must not truncate old state"

    def interrupted_replace(source, target):
        assert json.loads(Path(source).read_text())["new_field"] == job.state["new_field"]
        raise KeyboardInterrupt("cancel just before replace")

    monkeypatch.setattr(host.os, "replace", interrupted_replace)
    with pytest.raises(KeyboardInterrupt):
        job.persist()
    assert (job.root / "state.json").read_bytes() == previous
    assert not list(job.root.glob(".state.json-*.tmp"))


def test_atomic_state_success_replaces_complete_json(job):
    job.persist()
    previous = (job.root / "state.json").stat().st_ino
    job.state["new_field"] = "complete durable replacement"
    job.persist()
    assert json.loads((job.root / "state.json").read_text())["new_field"] == job.state["new_field"]
    assert (job.root / "state.json").stat().st_ino != previous
    assert not list(job.root.glob(".state.json-*.tmp"))


def test_always_cleanup_finishes_private_files_after_interrupted_finally(tmp_path, monkeypatch):
    root = tmp_path / ("smg-read-" + "1" * 32)
    root.mkdir()
    for name in ("home", "cache", "docker-config"):
        (root / name).mkdir()
        (root / name / "owned").write_text("private")
    (root / "environment").write_text("HF_TOKEN=not-for-artifacts")
    host.save(root / "state.json", {"closed": True, "create_intent": False})
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    monkeypatch.setattr(host.sys, "argv", ["read", "cleanup", "--state-dir", str(root)])
    monkeypatch.setattr(host.Job, "inspect", lambda *a: pytest.fail("never created; no Docker"))
    assert host.main() == 0
    assert json.loads((root / "state.json").read_text())["private_files_removed"]
    assert not (root / "environment").exists()
    assert not any((root / name).exists() for name in ("home", "cache", "docker-config"))
