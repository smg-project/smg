"""One READ-only Docker job inside the existing admitted AMD Runner.Worker.

No service control, extra lease, root-selected code, host installation, or broad
cleanup. The installed guardian owns the full Worker lease and crash recovery.
"""

from __future__ import annotations

import argparse
import base64
import calendar
import datetime
import hashlib
import importlib.util
import json
import os
import re
import selectors
import shlex
import shutil
import signal
import socket
import stat
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path


def load_sibling(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


image_gate = load_sibling("ci_moriio_read_image")

ACTIVE = Path("/run/smg-amd-ci/active-registration.json")
VERIFY = [
    "sudo",
    "-n",
    "/usr/bin/python3",
    "-I",
    "/usr/local/lib/smg-amd-ci-rollout-20261109/lifecycle.py",
    "verify-started",
]
INSPECT = "\n".join(
    "{{json " + field + "}}"
    for field in (
        ".Id",
        ".Image",
        ".Name",
        ".Config.Labels",
        ".State",
        ".HostConfig",
        ".Mounts",
        ".Created",
        ".Config.Entrypoint",
        ".Config.Cmd",
        ".Config.User",
        ".Config.Healthcheck",
    )
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data, *, exclusive=False, atomic=False):
    raw = data if isinstance(data, bytes) else (json.dumps(data, indent=2) + "\n").encode()
    target = path.with_name("." + path.name + "-" + uuid.uuid4().hex + ".tmp") if atomic else path
    try:
        fd = os.open(
            target,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_NOFOLLOW
            | (os.O_EXCL if exclusive or atomic else os.O_TRUNC),
            0o600,
        )
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if atomic:
            os.replace(target, path)
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if atomic:
            target.unlink(missing_ok=True)


def protected(path):
    path = Path(path)
    for part in (path, *path.parents):
        info = part.lstat()
        if info.st_uid != 0 or stat.S_ISLNK(info.st_mode) or info.st_mode & 0o022:
            raise RuntimeError("Untrusted installed admission metadata")
    return json.loads(path.read_text())


def identity(pid):
    try:
        row = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return {"pid": pid, "start": int(row[19]), "sid": int(row[3]), "pgid": int(row[2])}
    except (FileNotFoundError, ProcessLookupError):
        return None


def binding(workspace, command):
    # Fixed existing sudoers command, not arbitrary root Python or a new lease.
    code, stdout, _ = command("verify-started", VERIFY, seconds=30)
    if code:
        raise RuntimeError("Installed narrow Worker verifier failed; see private ACK")
    active = protected(ACTIVE)
    config_path = Path(active["config"])
    config = protected(config_path)
    state = protected(active["state"])
    if config["active_state"] != active["state"] or sha(config_path) != state["config_sha256"]:
        raise RuntimeError("Admission config/state drift")
    if state["cleanup_complete"] or state["cycle_id"] != active["cycle"]:
        raise RuntimeError("Not an active normal Worker cycle")
    if (
        state["profile"] != config["profile"]
        or state["registration_id"] != config["registration_id"]
    ):
        raise RuntimeError("Registration binding drift")
    if config["registration_id"] != {"amd-mi300x-1": 331464, "amd-mi300x-2": 331467}.get(
        socket.gethostname()
    ):
        raise RuntimeError("Unsupported AMD registration")
    if (state["lock_dev"], state["lock_ino"]) != (28, 6):
        raise RuntimeError("Unexpected original lease identity")
    if (
        not workspace.is_relative_to(Path(config["work_root"]).resolve())
        or Path(stdout.decode().strip()).resolve() != Path(config["work_root"]).resolve()
    ):
        raise RuntimeError("Workspace is outside this admitted registration")
    raw = (Path(config["runner_root"]) / ".runner").read_bytes()
    text = raw.decode("utf-8-sig")
    if not text.isascii() or hashlib.sha256(raw).hexdigest() != config["runner_config_sha256"]:
        raise RuntimeError("Installed .runner changed")
    label = hashlib.sha256(text.lower().encode()).hexdigest()[:6]
    if label != config["docker_owner_label"] or label != state["docker_owner_label"]:
        raise RuntimeError("DockerInstanceLabel mismatch")
    pid = os.getpid()
    while pid > 1:
        own = identity(pid)
        if own is None:
            raise RuntimeError("Worker ancestry disappeared")
        if pid == state["worker_pid"]:
            if (
                own["start"] != state["worker_start"]
                or sha(Path(f"/proc/{pid}/exe")) != config["worker_sha256"]
            ):
                raise RuntimeError("Worker identity mismatch")
            break
        pid = int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[1])
    else:
        raise RuntimeError("Helper is not a child of the admitted Worker")
    if Path("/proc/self/cgroup").read_text() != "0::" + state["child_cgroup"] + "\n":
        raise RuntimeError("Helper escaped the admitted job cgroup")
    return {
        "label": label,
        "cycle": state["cycle_id"],
        "worker_pid": state["worker_pid"],
        "worker_start": state["worker_start"],
        "deadline": min(state["job_deadline"], state["execution_deadline"]),
    }, config


def rdma_declarations(options):
    """Accept the existing READ options vocabulary, never arbitrary Docker args."""
    args = shlex.split(options)
    devices = []
    while args:
        arg = args.pop(0)
        key, sep, value = arg.partition("=")
        if not sep:
            if not args:
                raise ValueError("Missing READ option value")
            value = args.pop(0)
        if key == "--device":
            source = value.split(":", 1)[0]
            if source not in (
                "/dev/kfd",
                "/dev/dri",
                "/dev/infiniband",
                "/dev/infiniband/rdma_cm",
            ) and not re.fullmatch(
                r"/dev/(?:dri/renderD|infiniband/uverbs)[0-9]+",  # codespell:ignore renderd
                source,
            ):
                raise ValueError("Unreviewed READ device")
            if source.startswith("/dev/infiniband"):
                devices.append(source)
        elif (key, value) not in {
            ("--group-add", "video"),
            ("--group-add", "render"),
            ("--ipc", "host"),
            ("--ipc", "private"),
            ("--shm-size", "32g"),
            ("--volume", "/models:/models"),
            ("--env", "HF_HOME=/models"),
            ("--env", "HF_TOKEN"),
        }:
            raise ValueError("Unsupported READ option; no raw Docker passthrough")
    if "/dev/infiniband" not in devices and not (
        "/dev/infiniband/rdma_cm" in devices and any("uverbs" in x for x in devices)
    ):
        raise ValueError("Explicit RDMA devices required")
    return devices


def selected_devices(rdma):
    rows = []
    for node in Path("/sys/class/kfd/kfd/topology/nodes").iterdir():
        props = dict(line.split() for line in (node / "properties").read_text().splitlines())
        if int(props["simd_count"]) == 0:
            continue
        loc = int(props["location_id"])
        pci = f"{int(props['domain']):04x}:{loc >> 8:02x}:{(loc >> 3) & 31:02x}.{loc & 7}"
        render = Path("/dev/dri") / (
            "renderD" + props["drm_render_minor"]  # codespell:ignore renderd
        )
        if (
            Path("/sys/class/drm") / render.name / "device"
        ).resolve().name != pci or not stat.S_ISCHR(render.stat().st_mode):
            raise RuntimeError("Ambiguous KFD/PCI/render identity")
        rows.append({"pci": pci, "uid": int(props["unique_id"]), "render": str(render)})
    chosen = sorted(rows, key=lambda row: row["pci"])[:4]
    if (
        len(chosen) != 4
        or len({x["pci"] for x in chosen}) != 4
        or len({x["uid"] for x in chosen}) != 4
        or any(x["uid"] <= 0 for x in chosen)
    ):
        raise RuntimeError("Duplicate GPU identity")
    paths = {
        str(p)
        for declaration in rdma
        for p in (
            list(Path(declaration).iterdir())
            if declaration == "/dev/infiniband"
            else [Path(declaration)]
        )
        if re.fullmatch(r"(?:rdma_cm|uverbs[0-9]+)", p.name)
    }
    if (
        "/dev/infiniband/rdma_cm" not in paths
        or not any("uverbs" in x for x in paths)
        or not all(stat.S_ISCHR(Path(x).stat().st_mode) for x in paths)
    ):
        raise RuntimeError("Invalid RDMA character-device declarations")
    return chosen, sorted(paths)


def created_ns(value):
    match = re.fullmatch(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.(\d{1,9}))?(?:Z|\+00:00)", value)
    if not match:
        raise RuntimeError("Unknown Docker Created timestamp")
    seconds = calendar.timegm(datetime.datetime.strptime(match[1], "%Y-%m-%dT%H:%M:%S").timetuple())
    return seconds * 10**9 + int((match[2] or "").ljust(9, "0"))


class Job:
    def __init__(self, root, *, create=False):
        self.root = root
        self.serial = len(list(root.glob("*-ACK.json")))
        self.deadline = time.monotonic() + (10200 if create else 60)
        self.state = {} if create else json.loads((root / "state.json").read_text())

    def persist(self):
        save(self.root / "state.json", self.state, atomic=True)

    def command(self, name, args, *, seconds=30):
        if getattr(self, "unresolved_client", False):
            raise RuntimeError("Previous own CLI closure unknown; no further RPC")
        self.serial += 1
        stem = self.root / f"{self.serial:03d}-{name}"
        save(Path(str(stem) + "-intent.json"), {"args": args, "at": time.time()}, exclusive=True)
        process = None
        out = {"stdout": bytearray(), "stderr": bytearray()}
        error = None
        selector = None
        spawn_attempted = False
        try:
            until = min(self.deadline, time.monotonic() + seconds)
            if until <= time.monotonic():
                raise TimeoutError("No client budget remaining")
            spawn_attempted = True
            cli_env = {
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "LANG": "C.UTF-8",
                "DOCKER_CONFIG": str(self.root / "docker-config"),
            }
            process = subprocess.Popen(
                args,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
                env=cli_env,
            )
            save(
                Path(str(stem) + "-handle.json"),
                identity(process.pid) or {"pid": process.pid, "already_gone": True},
                exclusive=True,
            )
            selector = selectors.DefaultSelector()
            for kind, stream in (("stdout", process.stdout), ("stderr", process.stderr)):
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ, kind)
            while selector.get_map():
                if time.monotonic() >= until:
                    raise TimeoutError("Client deadline; no mutation replay")
                for key, _ in selector.select(min(0.2, until - time.monotonic())):
                    chunk = os.read(key.fd, 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    left = max(0, 1048576 - len(out[key.data]))
                    out[key.data].extend(chunk[:left])
                    if len(chunk) > left:
                        raise RuntimeError("Client raw-prefix limit")
            process.wait(timeout=1)
        except BaseException as exc:
            error = type(exc).__name__
        finally:
            if selector is not None:
                selector.close()
            if process is not None:
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                    except OSError as exc:
                        if not isinstance(exc, ProcessLookupError):
                            error = error or type(exc).__name__
                try:
                    process.wait(timeout=5)
                except BaseException as exc:
                    error = error or type(exc).__name__
                for stream in (process.stdout, process.stderr):
                    # Preserve a bounded prefix even if setup/handle creation failed.
                    try:
                        os.set_blocking(stream.fileno(), False)
                        kind = "stdout" if stream is process.stdout else "stderr"
                        while True:
                            chunk = os.read(stream.fileno(), 65536)
                            if not chunk:
                                break
                            left = max(0, 1048576 - len(out[kind]))
                            out[kind].extend(chunk[:left])
                            if len(chunk) > left:
                                error = error or "raw_prefix_cap"
                                break
                    except (OSError, ValueError):
                        pass
                    stream.close()
            gone = process is None
            if process is not None:
                try:
                    os.killpg(process.pid, 0)
                except ProcessLookupError:
                    gone = True
                except OSError as exc:
                    error = error or type(exc).__name__
            reaped = (not spawn_attempted) if process is None else process.poll() is not None
            self.unresolved_client = not (gone and reaped)
            if self.unresolved_client:
                self.state["local_client_closure_unknown"] = True
                self.persist()
            for kind, raw in out.items():
                save(Path(str(stem) + "." + kind), bytes(raw), exclusive=True)
            save(
                Path(str(stem) + "-ACK.json"),
                {
                    "exit": process.returncode if process else None,
                    "error": error,
                    "reaped": reaped,
                    "group_gone": gone,
                    "no_spawn": not spawn_attempted,
                },
                exclusive=True,
            )
            if self.unresolved_client:
                error = error or "own_cli_closure_unknown"
        if error:
            raise RuntimeError("Client outcome unknown; retained intent/ACK, no replay")
        return process.returncode, bytes(out["stdout"]), bytes(out["stderr"])

    def docker(self, name, *args, seconds=30):
        code, out, err = self.command(name, ["docker", *args], seconds=seconds)
        if code:
            raise RuntimeError("Docker command failed; see private ACK")
        return out

    def inspect(self):
        target = self.state.get("CID", self.state["name"])
        code, out, err = self.command("inspect", ["docker", "inspect", "--format", INSPECT, target])
        if code:
            if code == 1 and b"No such object:" in err:
                return None
            raise RuntimeError("Unknown inspect outcome")
        values = [json.loads(x) for x in out.splitlines()]
        if len(values) != 12:
            raise RuntimeError("Unexpected Docker metadata")
        return dict(
            zip(
                (
                    "id",
                    "image",
                    "name",
                    "labels",
                    "state",
                    "host",
                    "mounts",
                    "created",
                    "entrypoint",
                    "cmd",
                    "user",
                    "healthcheck",
                ),
                values,
            )
        )

    def owned(self, value):
        s = self.state
        if (
            value["name"] != "/" + s["name"]
            or value["labels"] != s["labels"]
            or value["image"] != s["image_id"]
            or (s.get("CID") and value["id"] != s["CID"])
        ):
            raise RuntimeError("CID ownership mismatch; do not touch")
        if not re.fullmatch("[a-f0-9]{64}", value["id"]):
            raise RuntimeError("Invalid CID")
        if not s["create_after_ns"] <= created_ns(value["created"]) <= time.time_ns():
            raise RuntimeError("CID Created is outside this exact create intent")
        validate_runtime(value, s)

    def cleanup(self):
        if self.state.get("local_client_closure_unknown"):
            raise RuntimeError("Own CLI closure unknown; installed guardian must reconcile")
        if not self.state.get("create_intent"):
            self.state["closed"] = True
            self.persist()
            return
        if self.state.get("closed"):
            # Always-step re-entry is read-only reconciliation, not another remove.
            if self.inspect() is not None:
                raise RuntimeError("Closed identity reappeared; do not adopt")
            return
        current = self.inspect()
        if current is None:
            self.state["closed"] = True
            self.persist()
            return
        self.owned(current)
        if current["state"]["Running"]:
            if self.state.get("stop_intent"):
                raise RuntimeError("Prior stop unknown; guardian reconciliation required")
            self.state["stop_intent"] = True
            self.persist()
            self.docker("stop", "stop", "--time", "30", current["id"], seconds=40)
            current = self.inspect()
            if current is None:
                self.state["closed"] = True
                self.persist()
                return
            self.owned(current)
        if current["state"]["Pid"] != 0 or current["state"]["Running"]:
            raise RuntimeError("Own container has not stopped")
        if self.state.get("remove_intent"):
            raise RuntimeError("Prior remove unknown; no resend")
        self.state["remove_intent"] = True
        self.persist()
        self.docker("remove", "rm", "-v", current["id"])
        if self.inspect() is not None:
            raise RuntimeError("Own CID still exists")
        self.state["closed"] = True
        self.persist()

    def remove_private_files(self):
        # After exact CID absence there can be no container cache/credential writer.
        if not self.state.get("closed"):
            raise RuntimeError("Private files still have a possible CID writer")

        def expired(signum, frame):
            raise TimeoutError("Bounded own cache removal")

        previous = signal.signal(signal.SIGALRM, expired)
        try:
            signal.alarm(15)
            for name in ("home", "cache", "docker-config"):
                path = self.root / name
                if path.exists():
                    if path.is_symlink() or path.stat().st_uid != os.geteuid():
                        raise RuntimeError("Private output ownership changed")
                    shutil.rmtree(path)
            self.state["private_files_removed"] = True
            self.persist()
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)


def validate_runtime(c, state):
    h = c["host"]
    if (
        h["NetworkMode"] != "host"
        or h["NanoCpus"] != 16000000000
        or h["Memory"] != h["MemorySwap"]
        or h["Memory"] != 137438953472
        or h["PidsLimit"] != 4096
    ):
        raise RuntimeError("READ resource contract mismatch")
    if (
        h["CapDrop"] != ["ALL"]
        or h.get("CapAdd")
        or h.get("Privileged")
        or h.get("DeviceRequests")
        or h.get("PortBindings")
        or h.get("PidMode")
        or h.get("UTSMode")
        or h["IpcMode"] != "private"
        or h["CgroupnsMode"] != "host"
        or not h["ReadonlyRootfs"]
    ):
        raise RuntimeError("READ isolation mismatch")
    if h["Runtime"] != "runc" or c["healthcheck"].get("Test") != ["NONE"]:
        raise RuntimeError("Unexpected runtime or automatic healthcheck")
    if h["SecurityOpt"] not in (["no-new-privileges"], ["no-new-privileges:true"]):
        raise RuntimeError("NoNewPrivileges required")
    if (
        h["Ulimits"] != [{"Name": "memlock", "Hard": -1, "Soft": -1}]
        or h["ShmSize"] != 32 * 1024**3
    ):
        raise RuntimeError("READ shared-memory/memlock contract mismatch")
    if {x["PathOnHost"] for x in h["Devices"]} != set(state["devices"]):
        raise RuntimeError("Unexpected GPU/RDMA ACL")
    if any(
        x["PathOnHost"] != x["PathInContainer"] or x["CgroupPermissions"] != "rwm"
        for x in h["Devices"]
    ):
        raise RuntimeError("Device mapping changed")
    if c["user"] != state["user"] or {
        x["Source"]: (x["Destination"], x["RW"]) for x in c["mounts"]
    } != {k: tuple(v) for k, v in state["mounts"].items()}:
        raise RuntimeError("Unexpected user or host mounts")
    if c["entrypoint"] != ["/usr/bin/python3"] or c["cmd"] != [
        "-I",
        "/workspace/scripts/ci_moriio_read_host.py",
        "inside",
        "--image",
        state["image"],
    ]:
        raise RuntimeError("Unreviewed entrypoint")


def run_host(args):
    token = uuid.uuid4().hex
    root = Path(os.environ["RUNNER_TEMP"]).resolve() / ("smg-read-" + token)
    root.mkdir(mode=0o700)
    job = Job(root, create=True)
    job.state = {"name": "smg-read-" + token, "create_intent": False, "closed": False}
    job.persist()
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as out:
        out.write(f"state_dir={root}\ninvocation={token}\n")
    try:
        return run_admitted(args, job, root, token)
    finally:
        try:
            if not job.state.get("closed"):
                job.cleanup()
            if not job.state.get("private_files_removed"):
                job.remove_private_files()
        finally:
            (root / "environment").unlink(missing_ok=True)


def run_admitted(args, job, root, token):
    workspace = Path.cwd().resolve()
    admitted, config = binding(workspace, job.command)
    job.deadline = min(job.deadline, admitted["deadline"] - 60)
    rdma = rdma_declarations(json.loads(args.options_json))
    gpus, rdma = selected_devices(rdma)
    for child in ("home", "cache"):
        (root / child).mkdir(mode=0o700)
    env = {
        "HOME": "/smg-home",
        "HF_HOME": "/models",
        "ROUTER_LOCAL_MODEL_PATH": "/models",
        "E2E_HOST": "127.0.0.1",
        "E2E_MORIIO_HOST": "127.0.0.1",
        "MORI_DISABLE_AUTO_XGMI": "1",
        "E2E_ENGINE": "vllm",
        "E2E_RUNTIME": "vllm",
        "E2E_GPU_VENDOR": "amd",
        "E2E_GPU_TIER": "4",
        "E2E_KV_BACKEND": "moriio",
        "E2E_VLLM_KV_BACKEND": "moriio",
        "E2E_VLLM_MORIIO_MODE": "read",
        "E2E_VLLM_MORIIO_BACKEND": "rdma",
        "E2E_JOB_MARKER": "read-" + token,
        "E2E_MIN_SELECTED": "0",
        "E2E_AMD_REQUIRE_PASSING_CALL": "1",
        "SHOW_WORKER_LOGS": "0",
        "SHOW_ROUTER_LOGS": "1",
        "E2E_LOG_DIR": "e2e-logs",
        "SMG_VLLM_SERVICER_WARMUP": "fast",
        "SMG_READ_GPUS": json.dumps(gpus, separators=(",", ":")),
        # Constrain even setup/imports/download metadata before the later ten-mask gate.
        "ROCR_VISIBLE_DEVICES": ",".join(f"GPU-{x['uid']:016x}" for x in gpus),
        "E2E_GPU_OFFSET": "0",
        "E2E_TEST_FILTER": "",
        "E2E_MODEL_TP_OVERRIDES": "",
        "E2E_VLLM_EXTRA_KV_BACKENDS": "",
        "E2E_VLLM_SERVICER_IMPL": "",
        "E2E_CONNECTION_MODE": "",
        "E2E_PARSER": "",
        "SMG_READ_DEADLINE": str(job.deadline),
    }
    for key in ("HF_TOKEN", "GITHUB_TOKEN"):
        if os.environ.get(key):
            env[key] = os.environ[key]
    if any("\n" in value for value in env.values()):
        raise ValueError("Multiline runtime environment rejected")
    save(
        root / "environment",
        "".join(k + "=" + v + "\n" for k, v in env.items()).encode(),
        exclusive=True,
    )
    (root / "docker-config").mkdir(mode=0o700)
    # Image inspection omits Env, which can contain secrets.
    image = image_gate.read_image(args.image)
    if image.startswith("ghcr.io/") and os.environ.get("GITHUB_TOKEN"):
        auth = base64.b64encode(
            (os.environ["GITHUB_ACTOR"] + ":" + os.environ["GITHUB_TOKEN"]).encode()
        ).decode()
        save(
            root / "docker-config/config.json",
            {"auths": {"ghcr.io": {"auth": auth}}},
            exclusive=True,
        )
    labels = {
        admitted["label"]: "",
        "smg.read.invocation": token,
        "smg.read.cycle": admitted["cycle"],
        "smg.read.run": os.environ["GITHUB_RUN_ID"],
        "smg.read.attempt": os.environ["GITHUB_RUN_ATTEMPT"],
    }
    devices = ["/dev/kfd", *[x["render"] for x in gpus], *rdma]
    mounts = {
        str(workspace): ("/workspace", True),
        str(root / "home"): ("/smg-home", True),
        str(root / "cache"): ("/smg-cache", True),
        "/models": ("/models", True),
    }
    user = f"{os.geteuid()}:{os.getegid()}"
    job.state = {
        "name": "smg-read-" + token,
        "image": image,
        "labels": labels,
        "devices": devices,
        "admission": admitted,
        "closed": False,
        "mounts": mounts,
        "user": user,
    }
    job.persist()
    create = [
        "create",
        "--pull=never",
        "--name",
        job.state["name"],
        "--network=host",
        "--cpus=16",
        "--memory=128g",
        "--memory-swap=128g",
        "--pids-limit=4096",
        "--read-only",
        "--runtime=runc",
        "--no-healthcheck",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--cgroupns=host",
        "--ipc=private",
        "--shm-size=32g",
        "--ulimit=memlock=-1:-1",
        "--tmpfs=/tmp:rw,nosuid,nodev,size=32g",
        "--user",
        f"{os.geteuid()}:{os.getegid()}",
        "--workdir=/workspace",
        "--env-file",
        str(root / "environment"),
    ]
    for group in os.getgroups():
        create += ["--group-add", str(group)]
    for key, value in labels.items():
        create += ["--label", key + "=" + value]
    for device in devices:
        create += ["--device", device + ":" + device + ":rwm"]
    for source, destination in (
        (workspace, "/workspace"),
        (root / "home", "/smg-home"),
        (root / "cache", "/smg-cache"),
        (Path("/models"), "/models"),
    ):
        create += ["--mount", f"type=bind,src={source},dst={destination}"]
    create += [
        "--entrypoint=/usr/bin/python3",
        image,
        "-I",
        "/workspace/scripts/ci_moriio_read_host.py",
        "inside",
        "--image",
        image,
    ]
    primary = None
    code = 1
    try:
        job.docker("pull", "pull", image, seconds=max(1, job.deadline - time.monotonic() - 60))
        job.state["image_id"] = (
            job.docker("image-id", "image", "inspect", "--format", "{{.Id}}", image)
            .decode()
            .strip()
        )
        if not re.fullmatch(r"sha256:[a-f0-9]{64}", job.state["image_id"]):
            raise RuntimeError("Invalid immutable image config ID")
        volumes = json.loads(
            job.docker(
                "image-volumes",
                "image",
                "inspect",
                "--format",
                '{{json (index .Config "Volumes")}}',
                image,
            )
        )
        if volumes:
            raise RuntimeError("Anonymous image volumes are not permitted")
        base_labels = (
            json.loads(
                job.docker(
                    "image-labels",
                    "image",
                    "inspect",
                    "--format",
                    '{{json (index .Config "Labels")}}',
                    image,
                )
            )
            or {}
        )
        if not isinstance(base_labels, dict) or set(base_labels) & set(labels):
            raise RuntimeError("Image labels collide with this exact job ownership")
        job.state["labels"] = {**base_labels, **labels}
        before, _ = binding(workspace, job.command)
        if before != admitted:
            raise RuntimeError("Current Worker admission changed")
        if job.inspect() is not None:
            raise RuntimeError("Fresh CID name exists; no adoption")
        job.state["create_intent"] = True
        job.state["create_after_ns"] = time.time_ns()
        job.persist()
        cid = job.docker("create", *create).decode().strip()
        if not re.fullmatch("[a-f0-9]{64}", cid):
            raise RuntimeError("Unknown create response; inspect exact fresh name only")
        job.state["CID"] = cid
        job.persist()
        current = job.inspect()
        job.owned(current)
        validate_runtime(current, job.state)
        if binding(workspace, job.command)[0] != admitted:
            raise RuntimeError("Admission changed before container start")
        job.state["start_intent"] = True
        job.persist()
        job.docker("start", "start", cid)
        current = job.inspect()
        job.owned(current)
        validate_runtime(current, job.state)
        pid = current["state"]["Pid"]
        if pid <= 1 or os.readlink(f"/proc/{pid}/ns/net") != os.readlink("/proc/self/ns/net"):
            raise RuntimeError("Own CID must use actual host network namespace")
        job.state["init"] = identity(pid)
        job.state["cgroup"] = Path(f"/proc/{pid}/cgroup").read_text()
        if job.state["cgroup"] != f"0::/system.slice/docker-{cid}.scope\n":
            raise RuntimeError("Own CID cgroup mismatch")
        job.persist()
        raw = job.docker(
            "wait", "wait", cid, seconds=max(0.01, job.deadline - time.monotonic() - 60)
        )
        code = int(raw.decode().strip())
        job.state["container_exit"] = code
    except BaseException as exc:
        primary = type(exc).__name__
        raise
    finally:
        job.state["primary_error"] = primary
        job.persist()
        try:
            job.cleanup()
            job.remove_private_files()
        finally:
            (root / "environment").unlink(missing_ok=True)
    return code


def inside(image):
    """All setup/install/model/native/test commands execute only inside own CID."""
    devices = load_sibling("ci_moriio_read_devices")

    out = Path("e2e-logs")
    out.mkdir(exist_ok=True)
    cache = Path("/smg-cache")
    # GITHUB_* file commands are private to this container, never host paths.
    env = dict(
        os.environ, GITHUB_ENV=str(cache / "github-env"), GITHUB_PATH=str(cache / "github-path")
    )
    setup = r"""set -euo pipefail
if [ -n "${GITHUB_TOKEN:-}" ]; then
  git config --global url."https://x-access-token:${GITHUB_TOKEN}@github.com/".insteadOf https://github.com/
fi
bash scripts/ci_setup_vllm_rocm_image.sh
source .venv/bin/activate
pip uninstall -y smg || true
pip install wheel/*.whl
bash scripts/ci_install_e2e_deps.sh
python3 scripts/ci_moriio_read_image.py verify --image "$1" --qualification-env "$GITHUB_ENV"
bash scripts/ci_download_model.sh --gpu-tier 4
"""
    if os.environ.get("SMG_READ_VENV_STAGE") != "1":
        with (cache / "setup.log").open("xb") as log:
            subprocess.run(
                ["bash", "-c", setup, "read-setup", image],
                env=env,
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=1100,
            )
        # Keep the venv spelling: resolving its symlink would select system Python.
        python = str(Path(".venv/bin/python").absolute())
        env["PATH"] = str(Path(".venv/bin").absolute()) + ":" + env.get("PATH", "/usr/bin:/bin")
        os.execve(
            python,
            [python, "-I", __file__, "inside", "--image", image],
            dict(env, SMG_READ_VENV_STAGE="1"),
        )
    exports = dict(line.split("=", 1) for line in (cache / "github-env").read_text().splitlines())
    if set(exports) != {
        image_gate.QUALIFICATION_ENV,
        "MORIIO_READ_IMAGE",
        "MORI_DISABLE_AUTO_XGMI",
    }:
        raise RuntimeError("Unexpected qualification file-command outputs")
    env.update(exports)
    if not image_gate.qualification_attempt(env):
        raise RuntimeError("Verified strict READ qualification intent required")
    selected = json.loads(env["SMG_READ_GPUS"])
    env.pop("HIP_VISIBLE_DEVICES", None)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["ROCR_VISIBLE_DEVICES"] = ",".join(f"GPU-{x['uid']:016x}" for x in selected)
    expected = devices.device_probe(env, cache, "initial-selected-four", ray=False)
    if len(expected) != 4:
        raise RuntimeError("Expected four selected GPUs")
    for observed, wanted in zip(expected, selected):
        uid = bytes.fromhex(observed["uuid"].replace("-", "")).decode("ascii")
        pci = (
            f"{observed['pci_domain']:04x}:{observed['pci_bus']:02x}:{observed['pci_device']:02x}.0"
        )
        if int(uid, 16) != wanted["uid"] or pci != wanted["pci"] or observed["gfx"] != "gfx942":
            raise RuntimeError("HIP/KFD physical identity mismatch")
    env, visibility = devices.pytest_device_env(env, expected, cache)
    env, copied = devices.writable_cache_env(
        env, cache, Path("/usr/local/lib/python3.12/dist-packages/aiter/jit")
    )
    save(
        out / "read-device-preflight.json",
        {
            "visibility": visibility,
            "cache_file_count": len(copied["aiter_copied_files"]),
            "scope": "Metadata/cache only, not READ/native/150-case PASS",
        },
    )
    prepared = image_gate.verify_installed()
    save(
        out / "read-source-prepared.json",
        {
            "image": image,
            "source_sha256": prepared,
            "scope": "Original installed-source/anti-shadow gate, not full native ABI qualification",
        },
    )
    subprocess.run(
        ["bash", "scripts/ci_killall_sglang.sh", "rocm"], env=env, check=True, timeout=60
    )
    if time.monotonic() + 9000 + 60 > float(env["SMG_READ_DEADLINE"]):
        raise RuntimeError(
            "Insufficient remaining budget for original full150-minute test; do not shrink it"
        )
    command = [
        sys.executable,
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
    with (
        (out / "original-full-topology.stdout").open("xb") as stdout,
        (out / "original-full-topology.stderr").open("xb") as stderr,
    ):
        try:
            result = subprocess.run(command, env=env, stdout=stdout, stderr=stderr, timeout=9000)
            code = result.returncode
        except subprocess.TimeoutExpired:
            code = 124  # PID namespace teardown/guardian still owns every child.
    final = image_gate.verify_installed()
    save(
        out / "read-source-final.json",
        {"image": image, "source_sha256": final, "same_as_prepared": final == prepared},
    )
    if final != prepared:
        raise RuntimeError("Installed source changed during original suite")
    save(
        out / "read-host-summary.json",
        {
            "pytest_exit": code,
            "image": image,
            "scope": (
                "Original full file/unaltered tests; selected counts, skips and real retries require raw/JUnit intake. "
                "No physical cleanup claim."
            ),
        },
    )
    return code


def failure_evidence():
    """Retain bounded useful setup/probe failures, never private env/config/cache."""
    out = Path("e2e-logs")
    out.mkdir(exist_ok=True)

    def redact(raw):
        text = raw.decode("utf-8", "replace")
        for key in ("GITHUB_TOKEN", "HF_TOKEN"):
            if os.environ.get(key):
                text = text.replace(os.environ[key], "[REDACTED]")
        return re.sub(r"https?://[^/\s@]+@", "https://[REDACTED]@", text)

    evidence = {
        "error_type": sys.exc_info()[0].__name__ if sys.exc_info()[0] else None,
        "traceback": redact(traceback.format_exc().encode()),
        "scope": "Failure only; no synthetic case or qualification PASS",
    }
    save(out / "read-host-failure.json", evidence)
    source = Path("/smg-cache/setup.log")
    if source.is_file() and not source.is_symlink():
        with source.open("rb") as stream:
            head = stream.read(131072)
            stream.seek(max(0, source.stat().st_size - 131072))
            tail = stream.read(131072)
        save(
            out / "read-setup.log", redact(head + b"\n[bounded setup head/tail]\n" + tail).encode()
        )
    snippets = []
    for source in sorted(Path("/smg-cache").glob("*.stderr")):
        if source.is_file() and not source.is_symlink():
            with source.open("rb") as stream:
                snippets.append(source.name + ":\n" + redact(stream.read(16384)))
    save(out / "read-preflight-diagnostic.log", "\n".join(snippets).encode())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "cleanup", "inside"))
    parser.add_argument("--image", default="")
    parser.add_argument("--options-json", default='""')
    parser.add_argument("--state-dir", type=Path)
    args = parser.parse_args()
    if args.command == "inside":
        try:
            return inside(args.image)
        except BaseException:
            failure_evidence()
            raise
    if args.command == "cleanup":
        root = args.state_dir.resolve()
        if (
            not root.is_relative_to(Path(os.environ["RUNNER_TEMP"]).resolve())
            or not re.fullmatch("smg-read-[a-f0-9]{32}", root.name)
            or root.is_symlink()
            or root.stat().st_uid != os.geteuid()
        ):
            raise RuntimeError("Not an owned READ control directory")
        job = Job(root)
        job.cleanup()
        job.remove_private_files()
        (root / "environment").unlink(missing_ok=True)
        return 0
    return run_host(args)


if __name__ == "__main__":

    def interrupted(signum, frame):
        raise KeyboardInterrupt("Owned READ job cancelled")

    for signum in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, interrupted)
    raise SystemExit(main())
