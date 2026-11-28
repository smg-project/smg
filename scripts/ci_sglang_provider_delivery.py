#!/usr/bin/env python3
"""Stage immutable provider data; guard actual provider mounts before installation."""

import argparse
import ast
import hashlib
import json
import os
import re
import shlex
import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKET = Path("scripts/ci_data/sglang-gptoss20-provider.json")
DEFAULT_OPTIONS = (
    "--device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size=32g "
    "--ulimit memlock=-1:-1 --volume /models:/models --env HF_HOME=/models --env HF_TOKEN"
)
RO_OPTIONS = (
    " --read-only --cap-drop=SYS_ADMIN --security-opt=no-new-privileges"
    " --tmpfs=/tmp:rw,exec,nosuid,nodev,size=8g"
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_packet(path, expected=None):
    path = Path(path).absolute()
    if path != path.resolve():
        raise ValueError("Aliased provider data path")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > 1024**2:
            raise ValueError("Bounded regular provider data required")
        body = stream.read(1024**2 + 1)
        actual = hashlib.sha256(body).hexdigest()
        if expected is not None and actual != expected:
            raise ValueError("Provider data digest differs")
    return json.loads(body), body, actual


def source_contract(packet, root):
    tree = ast.parse((root / "scripts/ci_sglang_cohort_bootstrap.py").read_text())
    names = ast.literal_eval(
        next(
            n.value
            for n in tree.body
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "SOURCES" for t in n.targets)
        )
    )
    expected = {name: digest(root / "scripts" / name) for name in names}
    if packet.get("source_sha256") != expected or packet.get("workflow_sha256") != digest(
        root / ".github/workflows/e2e-gpu-job.yml"
    ):
        raise ValueError("Provider data does not match this checkout Source")


def checkout(expected):
    if re.fullmatch(r"[0-9a-f]{40}", expected) is None:
        raise ValueError("Exact GitHub checkout SHA required")
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    )
    if result.stdout.strip() != expected:
        raise ValueError("Producer/consumer checkout differs from GitHub SHA")


def container_options(value):
    tokens = shlex.split(value or DEFAULT_OPTIONS)
    allowed = {
        "--device": {"/dev/kfd", "/dev/dri"},
        "--group-add": {"video"},
        "--ipc": {"host"},
        "--shm-size": {"32g"},
        "--ulimit": {"memlock=-1:-1"},
        "--volume": {"/models:/models", "/models:/models:rw"},
        "--env": {"HF_HOME=/models", "HF_TOKEN"},
    }
    pairs = []
    while tokens:
        item = tokens.pop(0)
        if "=" in item:
            key, value = item.split("=", 1)
        else:
            key = item
            if not tokens:
                raise ValueError("Incomplete container option")
            value = tokens.pop(0)
        if key not in allowed or value not in allowed[key]:
            raise ValueError("Unreviewed AMD container option")
        pairs.extend([key, value])
    return shlex.join(pairs) + RO_OPTIONS


def control_interfaces(environ, mountinfo, inspect=os.stat):
    """Refuse observed daemon entry points; deployment must also exclude remote APIs."""
    keys = (
        "DOCKER_HOST",
        "DOCKER_CONTEXT",
        "DOCKER_CONFIG",
        "DOCKER_TLS_VERIFY",
        "DOCKER_CERT_PATH",
        "CONTAINER_HOST",
        "CONTAINER_CONNECTION",
        "BUILDKIT_HOST",
    )
    if any(environ.get(key) for key in keys):
        raise ValueError("Configured container control interface is not admitted")
    paths = [
        "/var/run/docker.sock",
        "/run/docker.sock",
        "/run/podman/podman.sock",
        "/run/containerd/containerd.sock",
        "/run/buildkit/buildkitd.sock",
    ]
    if environ.get("XDG_RUNTIME_DIR"):
        paths.extend(
            str(Path(environ["XDG_RUNTIME_DIR"]) / name)
            for name in ("docker.sock", "podman/podman.sock", "buildkit/buildkitd.sock")
        )
    if not environ.get("HOME") or not Path(environ["HOME"]).is_absolute():
        raise ValueError("Explicit home required for container context inventory")
    paths.append(str(Path(environ["HOME"]) / ".docker"))
    for path in paths:
        try:
            inspect(path)
        except FileNotFoundError:
            continue
        raise ValueError("Container control path or context is present")
    if not mountinfo or len(mountinfo) > 1024**2:
        raise ValueError("Bounded complete mount inventory required")
    mounts = []
    for line in mountinfo.splitlines():
        parts = line.split(" - ")
        fields = parts[0].split()
        if len(parts) != 2 or len(fields) < 6 or len(parts[1].split()) < 3:
            raise ValueError("Malformed mount inventory")
        raw = fields[4]
        point = re.sub(r"\\(040|011|012|134)", lambda item: chr(int(item.group(1), 8)), raw)
        if not point.startswith("/") or "\\" in re.sub(r"\\(040|011|012|134)", "", raw):
            raise ValueError("Unknown mount path encoding")
        if stat.S_ISSOCK(inspect(point).st_mode):
            raise ValueError("Mounted socket control interface is not admitted")
        mounts.append(point)
    return {"configured_endpoint_keys_checked": list(keys), "mount_points_checked": mounts}


def provider_view(packet, status, statvfs=os.statvfs):
    fields = dict(line.split(":", 1) for line in status.splitlines() if ":" in line)
    masks = ["CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb"]
    if fields.get("NoNewPrivs", "").strip() != "1" or any(
        int(fields[name].strip(), 16) & (1 << 21) for name in masks
    ):
        raise ValueError("Read-only provider view requires no remount capability")
    paths = [
        *packet["base_sites"],
        packet["backend_hook"]["path"],
        str(Path("/proc/self/exe").resolve()),
    ]
    for item in packet["providers"].values():
        paths.extend([item["module_path"], item["metadata_path"]])
    for path in paths:
        if not (statvfs(path).f_flag & os.ST_RDONLY):
            raise ValueError("Provider path is not actually read-only")
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["publish", "guard"])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--sha256")
    parser.add_argument("--checkout", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    checkout(args.checkout)
    packet, body, sha = read_packet(args.input, args.sha256)
    source_contract(packet, ROOT)
    if args.phase == "publish":
        if args.input.resolve() != (ROOT / PACKET).resolve():
            raise ValueError("Tracked same-checkout provider data required")
        options = container_options(os.environ.get("SMG_RAW_CONTAINER_OPTIONS", ""))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("xb") as stream:
            stream.write(body)
        args.output.chmod(0o444)
        with Path(os.environ["GITHUB_OUTPUT"]).open("a") as stream:
            stream.write("sha256=" + sha + "\ncontainer_options=" + options + "\n")
    else:
        if not args.sha256:
            raise ValueError("Consumer packet digest required")
        with Path("/proc/self/mountinfo").open() as stream:
            mountinfo = stream.read(1024**2 + 1)
        controls = control_interfaces(os.environ, mountinfo)
        paths = provider_view(packet, Path("/proc/self/status").read_text())
        args.input.chmod(0o444)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(
                {
                    "provider_sha256": sha,
                    "read_only_provider_paths": paths,
                    "no_remount_capability_observed": True,
                    "control_interface_observation": controls,
                    "deployment_control_plane_isolation_qualified": False,
                    "SDK_or_native_qualified": False,
                },
                stream,
            )
            stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
