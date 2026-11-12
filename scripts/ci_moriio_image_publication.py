#!/usr/bin/env python3
"""Specific main-push CPU publisher for the reviewed MoRI READ recipe.

--check is entirely local. --publish is exclusively a normal smg main-push job;
its output is a new candidate image, never inherited A32 GPU qualification.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath

ENGINE = "f9e4ca9653da9bd731c8b5bf83483cb2083c8dc9"
BUILDKIT = "moby/buildkit@sha256:c5f8862f52499773af85bc07695b2379ef08e8002352b0df2c4864c91d78dabe"
MEMORY = 128 * 1024**3
DISK = 200 * 1024**3
QUOTA, PERIOD = 3200000, 100000
ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "docker/moriio-read"
CONTEXTS = {
    "stack": (
        "rocm/vllm-ci:stack-rocm-f0ff2bd31804e4af@sha256:"
        "482e36b7f45062dbd065095c59b78f4e2f3a8e0f0c7a971f52a492618b969e0d"
    ),
    "runtime": (
        "rocm/vllm-ci:runtime-rocm-c67ca115e7d64f7d@sha256:"
        "1e16e61342d903a2e509ed05c19a65d4e8176db2dcf4dc0e03cb2ee9d3ba335b"
    ),
    "test-deps": (
        "rocm/vllm-ci:test-deps-rocm-c67ca115e7d64f7d@sha256:"
        "4933c01636bc52d817c53a08d9b95b491c156ae45908f20a226b7c028fdc0ebd"
    ),
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(".new")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def check_recipe():
    manifest = json.loads((RECIPE / "manifest.json").read_text())
    for name, item in manifest["files"].items():
        relative = PurePosixPath(name)
        assert not relative.is_absolute() and ".." not in relative.parts
        path = RECIPE / name
        assert path.is_file() and not path.is_symlink()
        raw = path.read_bytes()
        assert len(raw) == item["bytes"] and digest(raw) == item["sha256"], name
    overlay = (RECIPE / "bake.hcl").read_text()
    for reference in CONTEXTS.values():
        assert overlay.count("docker-image://" + reference) == 2
    return digest((RECIPE / "manifest.json").read_bytes())


def cgroup_paths(pid):
    """Resolve actual host-visible controller paths, including hybrid cgroup v1."""
    membership = Path(f"/proc/{pid}/cgroup").read_text()
    controllers = {}
    for line in membership.splitlines():
        _, names, relative = line.split(":", 2)
        for name in names.split(",") if names else ["unified"]:
            controllers[name] = relative
    mounts = []
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        before, after = line.split(" - ", 1)
        fields, fs = before.split(), after.split()
        if fs[0] in ("cgroup", "cgroup2"):
            mounts.append((fields[3], fields[4], fs[0], fs[2].split(",")))
    paths = {}
    for name in (
        ("cpu", "memory") if "cpu" in controllers and "memory" in controllers else ("unified",)
    ):
        options = [
            m
            for m in mounts
            if (name == "unified" and m[2] == "cgroup2") or (m[2] == "cgroup" and name in m[3])
        ]
        assert len(options) == 1, (name, "unknown cgroup mount")
        root, mount, _, _ = options[0]
        path = Path(mount) / Path(controllers[name]).relative_to(root)
        assert path.resolve().is_relative_to(Path(mount).resolve())
        paths[name] = (path, Path(mount))
    return membership, paths


def limits(pid, capacity=False):
    membership, paths = cgroup_paths(pid)
    values = []
    for name, (path, mount) in paths.items():
        while True:
            if capacity and path == mount:
                values.append({"path": str(path), "root_controller_capacity": "unrestricted"})
                break
            if name == "cpu":
                quota = int((path / "cpu.cfs_quota_us").read_text())
                period = int((path / "cpu.cfs_period_us").read_text())
                values.append({"path": str(path), "cpu_quota": quota, "cpu_period": period})
                assert (
                    (quota < 0 or quota / period >= 32)
                    if capacity
                    else (quota, period) == (QUOTA, PERIOD)
                )
            elif name == "memory":
                memory = int((path / "memory.limit_in_bytes").read_text())
                values.append({"path": str(path), "memory": memory})
                if capacity:
                    assert memory >= MEMORY
                else:
                    swap = int((path / "memory.memsw.limit_in_bytes").read_text())
                    assert memory == swap == MEMORY
                    values[-1]["memory_memsw_total"] = swap
            else:
                cpu = (path / "cpu.max").read_text().split()
                memory = (path / "memory.max").read_text().strip()
                values.append({"path": str(path), "cpu": cpu, "memory": memory})
                if capacity:
                    assert cpu[0] == "max" or int(cpu[0]) / int(cpu[1]) >= 32
                    assert memory == "max" or int(memory) >= MEMORY
                else:
                    swap = (path / "memory.swap.max").read_text().strip()
                    assert (
                        cpu == [str(QUOTA), str(PERIOD)]
                        and int(memory) == MEMORY
                        and int(swap) == 0
                    )
                    values[-1]["swap"] = swap
            if not capacity or path == mount:
                break
            assert mount in path.parents
            path = path.parent
    assert Path(f"/proc/{pid}/cgroup").read_text() == membership
    return {"pid": pid, "membership": membership, "controllers": values}


def export_source(checkout, destination):
    assert (
        subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"]).decode().strip()
        == ENGINE
    )
    assert not subprocess.check_output(["git", "-C", str(checkout), "status", "--porcelain"])
    archive = destination.parent / "engine.tar"
    with archive.open("xb") as stream:
        subprocess.run(
            ["git", "-C", str(checkout), "archive", "--format=tar", ENGINE],
            stdout=stream,
            check=True,
            timeout=120,
        )
    destination.mkdir(mode=0o700)
    with tarfile.open(archive) as source:
        members = source.getmembers()
        files = [m for m in members if not m.isdir()]
        assert len(files) == 9173
        for member in members:
            p = PurePosixPath(member.name)
            assert not p.is_absolute() and ".." not in p.parts and str(p) not in ("", ".")
            assert member.isdir() or member.isfile() or member.issym()
            if member.issym():
                target = (destination / p.parent / member.linkname).resolve()
                assert target.is_relative_to(destination.resolve())
        # Prevent extraction through a symlink parent, even for immutable inputs.
        symlinks = {PurePosixPath(m.name) for m in members if m.issym()}
        assert all(
            not any(parent in symlinks for parent in PurePosixPath(m.name).parents) for m in members
        )
        for member in members:
            path = destination / member.name
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
            elif member.issym():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.symlink_to(member.linkname)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                with source.extractfile(member) as inp, path.open("xb") as out:
                    shutil.copyfileobj(inp, out)
                path.chmod(member.mode & 0o777)
    shutil.copyfile(RECIPE / "source.git_archival.txt", destination / ".git_archival.txt")
    return {
        "commit": ENGINE,
        "archive_sha256": digest(archive.read_bytes()),
        "source_blobs": len(files),
    }


class RegistryRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        assert urllib.parse.urlsplit(newurl).scheme == "https"
        redirected = super().redirect_request(request, fp, code, msg, headers, newurl)
        if urllib.parse.urlsplit(request.full_url).netloc != urllib.parse.urlsplit(newurl).netloc:
            redirected.remove_header("Authorization")
        return redirected


def registry_open(request, deadline):
    remaining = deadline - time.monotonic()
    assert remaining > 3, "registry engineering deadline expired"
    return urllib.request.build_opener(RegistryRedirect()).open(
        request, timeout=min(30, remaining - 2)
    )


def registry_get(url, deadline, token=None):
    headers = {
        "Accept": (
            "application/vnd.oci.image.manifest.v1+json, "
            "application/vnd.docker.distribution.manifest.v2+json, "
            "application/vnd.oci.image.index.v1+json, "
            "application/vnd.docker.distribution.manifest.list.v2+json"
        )
    }
    if token:
        headers["Authorization"] = "Bearer " + token
    with registry_open(urllib.request.Request(url, headers=headers), deadline) as response:
        assert int(response.headers.get("Content-Length", "0")) <= 2 * 1024**2
        raw = response.read(2 * 1024**2 + 1)
        assert len(raw) <= 2 * 1024**2
        return raw


def image_identity(repository, expected, deadline, actor=None, password=None):
    """Read exact small registry metadata, never layers or credentials in receipts."""
    ghcr = repository.startswith("ghcr.io/")
    repo = repository.removeprefix("ghcr.io/")
    host = "ghcr.io" if ghcr else "registry-1.docker.io"
    auth = "https://ghcr.io/token" if ghcr else "https://auth.docker.io/token"
    query = urllib.parse.urlencode(
        {"service": host if ghcr else "registry.docker.io", "scope": "repository:" + repo + ":pull"}
    )
    headers = {}
    if password:
        headers["Authorization"] = (
            "Basic " + base64.b64encode((actor + ":" + password).encode()).decode()
        )
    with registry_open(
        urllib.request.Request(auth + "?" + query, headers=headers), deadline
    ) as response:
        raw_token = response.read(65537)
        assert len(raw_token) <= 65536
        token = json.loads(raw_token)["token"]
    raw = registry_get(f"https://{host}/v2/{repo}/manifests/{expected}", deadline, token)
    assert "sha256:" + digest(raw) == expected
    manifest = json.loads(raw)
    result = {
        "requested_digest": expected,
        "manifest": manifest,
        "manifest_sha256": digest(raw),
        "manifest_raw_base64": base64.b64encode(raw).decode(),
    }
    if "manifests" in manifest:
        choices = [
            m
            for m in manifest["manifests"]
            if m.get("platform", {}).get("os") == "linux"
            and m.get("platform", {}).get("architecture") == "amd64"
        ]
        assert len(choices) == 1
        chosen = choices[0]["digest"]
        raw = registry_get(f"https://{host}/v2/{repo}/manifests/{chosen}", deadline, token)
        assert "sha256:" + digest(raw) == chosen
        manifest = json.loads(raw)
        result.update(
            linux_amd64_manifest=manifest,
            linux_amd64_manifest_sha256=digest(raw),
            linux_amd64_manifest_raw_base64=base64.b64encode(raw).decode(),
        )
    config_digest = manifest["config"]["digest"]
    config_raw = registry_get(f"https://{host}/v2/{repo}/blobs/{config_digest}", deadline, token)
    assert "sha256:" + digest(config_raw) == config_digest
    config = json.loads(config_raw)
    assert config["os"] == "linux" and config["architecture"] == "amd64"
    result.update(
        config=config,
        config_sha256=digest(config_raw),
        config_raw_base64=base64.b64encode(config_raw).decode(),
        layer_descriptors=manifest["layers"],
    )
    return result


def publish(engine_checkout, output):
    assert socket.gethostname().split(".")[0] not in ("amd-mi300x-1", "amd-mi300x-2"), (
        "Not a shared GPU runner operation"
    )
    assert os.environ.get("GITHUB_REPOSITORY") == "smg-project/smg"
    assert (
        os.environ.get("GITHUB_EVENT_NAME") == "push"
        and os.environ.get("GITHUB_REF") == "refs/heads/main"
    )
    assert re.fullmatch(r"[0-9a-f]{40}", os.environ["GITHUB_SHA"])
    assert (
        subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"]).decode().strip()
        == os.environ["GITHUB_SHA"]
    )
    assert output == Path(os.environ["RUNNER_TEMP"]).resolve() / (
        "moriio-publication-" + os.environ["GITHUB_RUN_ID"] + "-" + os.environ["GITHUB_RUN_ATTEMPT"]
    )
    assert not output.exists() and not output.is_symlink()
    output.mkdir(mode=0o700)
    state = {
        "status": "preflight",
        "new_candidate_only": True,
        "GPU_qualification": False,
        "public_pull_verified": False,
        "events": [],
    }
    record = output / "result.json"
    write_json(record, state)
    start = time.monotonic()
    deadline = start + 43200
    builder = "smg-moriio-" + os.environ["GITHUB_RUN_ID"] + "-" + os.environ["GITHUB_RUN_ATTEMPT"]
    assert re.fullmatch(r"smg-moriio-[0-9]+-[0-9]+", builder)
    name = "buildx_buildkit_" + builder + "0"
    owned_id = None
    creation_submitted = False
    docker = ["docker", "--host", "unix:///var/run/docker.sock"]
    env = {
        k: v
        for k, v in os.environ.items()
        if k
        not in ("DOCKER_CONTEXT", "DOCKER_HOST", "GH_TOKEN", "GITHUB_TOKEN", "MORIIO_PUBLISH_TOKEN")
    }
    env.update(
        CONTEXT=str(output / "source"),
        MORIIO_RECIPE_DIR=str(RECIPE),
        STACK="rocm",
        STACK_IMAGE="",
        RUNTIME="",
        TEST_DEPS="",
        CI_ARCH="gfx942",
        BUILDKITE_COMMIT="",
        STACK_WHEELS="",
        AITER_SPEC="",
        AITER_NIGHTLY="",
        VLLM_VERSION="",
        SCCACHE_BUCKET_NAME="",
        PUSH="false",
        CACHE="",
        CACHE_TO="",
        CACHE_CI="false",
        TAG="unused-moriio-local",
        BUILDX_BAKE_ENTITLEMENTS_FS="1",
    )

    def event(phase, **extra):
        state["events"].append({"phase": phase, **extra})
        write_json(record, state)

    def command(phase, argv, until=None, capture=False):
        until = deadline if until is None else until
        remaining = until - time.monotonic()
        assert remaining > 3, "engineering deadline expired"
        log = output / (phase + ".log")
        event(phase + "_submitted", argv=argv)
        with log.open("xb") as stream:
            proc = subprocess.Popen(
                argv,
                env=env,
                stdout=subprocess.PIPE if capture else stream,
                stderr=stream,
                start_new_session=True,
            )
            handle = {"pid": proc.pid, "process_group": proc.pid}
            try:
                handle["start_ticks"] = int(
                    Path(f"/proc/{proc.pid}/stat").read_text().rsplit(") ", 1)[1].split()[19]
                )
            except FileNotFoundError:
                handle["exited_before_identity_read"] = True
            event(phase + "_running", client_handle=handle)
            try:
                stdout, _ = proc.communicate(timeout=remaining - 3)
            except BaseException:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait(timeout=1)
                raise
        event(phase + "_returned", returncode=proc.returncode, log_sha256=digest(log.read_bytes()))
        assert proc.returncode == 0, phase + " failed (see owned log)"
        if capture:
            assert len(stdout) <= 4 * 1024**2, "metadata output too large"
        return stdout

    def inspect(phase, until=None):
        raw = command(phase, docker + ["inspect", name], until=until, capture=True)
        result = json.loads(raw)
        assert len(result) == 1 and result[0]["Name"] == "/" + name
        return result[0]

    def verify_owned(container):
        assert owned_id and container["Id"] == owned_id
        assert container["Config"]["Image"] == BUILDKIT and container["Name"] == "/" + name

    def interrupted(signum, _frame):
        raise RuntimeError("publisher interrupted " + str(signum))

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGALRM, interrupted)
    signal.setitimer(signal.ITIMER_REAL, max(0.01, deadline - time.monotonic()))
    failure = None
    try:
        state["recipe_manifest_sha256"] = check_recipe()
        assert len(os.sched_getaffinity(0)) >= 32, "CPU worker capacity below32"
        meminfo = dict(
            (k.rstrip(":"), int(v.split()[0]) * 1024)
            for k, v in (
                line.split(None, 1) for line in Path("/proc/meminfo").read_text().splitlines()
            )
        )
        assert meminfo["MemAvailable"] >= MEMORY + 2 * 1024**3, "insufficient actual RAM/headroom"
        state["worker_cgroup_capacity"] = limits(os.getpid(), capacity=True)
        info = json.loads(
            command("daemon", docker + ["info", "--format", "{{json .}}"], capture=True)
        )
        assert info["Name"].split(".")[0] == socket.gethostname().split(".")[0], "nonlocal daemon"
        assert info["NCPU"] >= 32 and info["MemTotal"] >= MEMORY
        assert info["OSType"] == "linux" and info["Architecture"] in ("x86_64", "amd64")
        assert (
            shutil.disk_usage(info["DockerRootDir"]).free >= DISK
            and shutil.disk_usage(output).free >= DISK
        )
        state["host_capacity"] = {
            "affinity_cpus": len(os.sched_getaffinity(0)),
            "MemAvailable": meminfo["MemAvailable"],
            "docker_root": info["DockerRootDir"],
            "docker_disk_free": shutil.disk_usage(info["DockerRootDir"]).free,
        }
        # Name existence is checked before create; no adoption of another builder/container.
        names = (
            command(
                "existing_containers",
                docker + ["ps", "-a", "--no-trunc", "--format", "{{.Names}}"],
                capture=True,
            )
            .decode()
            .splitlines()
        )
        assert name not in names
        builders = (
            command(
                "existing_builders",
                docker + ["buildx", "ls", "--format", "{{.Name}}"],
                capture=True,
            )
            .decode()
            .splitlines()
        )
        assert builder not in builders
        state["source"] = export_source(engine_checkout, output / "source")
        for kind, reference in CONTEXTS.items():
            repository, pin = reference.split("@")
            repository = repository.rsplit(":", 1)[0]
            event("material_read_submitted", material=kind, digest=pin)
            material = image_identity(repository, pin, deadline)
            write_json(output / (kind + "-material.json"), material)
        config = output / "buildkitd.toml"
        config.write_text("[worker.oci]\n  max-parallelism = 2\n")
        event("create_intent", builder=builder, container_name=name)
        creation_submitted = True
        command(
            "create",
            docker
            + [
                "buildx",
                "create",
                "--name",
                builder,
                "--driver",
                "docker-container",
                "--driver-opt",
                "image=" + BUILDKIT,
                "--driver-opt",
                "cpu-quota=3200000",
                "--driver-opt",
                "cpu-period=100000",
                "--driver-opt",
                "memory=128g",
                "--driver-opt",
                "memory-swap=128g",
                "--buildkitd-config",
                str(config),
            ],
        )
        command("bootstrap", docker + ["buildx", "inspect", builder, "--bootstrap"])
        container = inspect("builder_inspect")
        assert container["Config"]["Image"] == BUILDKIT
        owned_id = container["Id"]
        state["owned_builder_container_id"] = owned_id
        write_json(record, state)
        actual = container["HostConfig"]
        assert (
            actual["CpuQuota"],
            actual["CpuPeriod"],
            actual["Memory"],
            actual["MemorySwap"],
        ) == (QUOTA, PERIOD, MEMORY, MEMORY)
        assert (
            not actual.get("Devices")
            and not actual.get("DeviceRequests")
            and actual.get("NetworkMode") != "host"
            and actual.get("PidMode") != "host"
        )
        state["builder_kernel_limits"] = limits(container["State"]["Pid"])
        state["builder_ancestor_capacity"] = limits(container["State"]["Pid"], capacity=True)
        again = inspect("builder_membership_recheck")
        verify_owned(again)
        assert again["State"]["Pid"] == container["State"]["Pid"] and again["State"]["Running"]
        # Same builder/ordinary OCI RUN: never assume daemon DeviceRequests implies no GPU exposure.
        probe_context = output / "cpu-probe"
        probe_context.mkdir()
        probe = probe_context / "Dockerfile"
        probe.write_text(
            "FROM ubuntu:22.04@sha256:"
            "5ec03bb3441e8b0bf3b4f9cd4629a1ae763010dc3035bb8da3ae6cf026486401\n"
            "RUN test ! -e /dev/kfd && test ! -e /dev/dri && test ! -e /dev/accel "
            "&& test -z \"$(find /dev -maxdepth 1 -name 'nvidia*' -print -quit)\"\n"
        )
        command(
            "cpu_sandbox",
            docker
            + [
                "buildx",
                "build",
                "--builder",
                builder,
                "--no-cache",
                "--network=none",
                "--file",
                str(probe),
                str(probe_context),
            ],
        )
        bake = docker + [
            "buildx",
            "bake",
            "--builder",
            builder,
            "--allow=fs.read=" + str(RECIPE),
            "-f",
            str(RECIPE / "upstream-bake.hcl"),
            "-f",
            str(RECIPE / "bake.hcl"),
        ]
        plan = json.loads(
            command(
                "bake_plan",
                bake
                + [
                    "--set",
                    "test.output=type=cacheonly",
                    "--set",
                    "test-smoke.output=type=cacheonly",
                    "--print",
                    "test",
                    "test-smoke",
                ],
                capture=True,
            )
        )
        assert set(plan["target"]) == {"test", "test-smoke"}
        for target_name, target in plan["target"].items():
            assert target["contexts"] == {k: "docker-image://" + v for k, v in CONTEXTS.items()}
            assert target["dockerfile"] == str(RECIPE / "Dockerfile.rocm")
            assert target["context"] == str(output / "source") and target["target"] == target_name
            assert target["args"] == {
                "AITER_NIGHTLY": "",
                "AITER_ROCM_ARCH": "gfx942",
                "AITER_SPEC": "",
                "ARCH": "gfx942",
                "CI_ARCH": "gfx942",
                "PYTORCH_ROCM_ARCH": "gfx942",
                "SCCACHE_BUCKET_NAME": "",
                "SOURCE_DATE_EPOCH": "0",
                "STACK": "rocm",
                "VLLM_VERSION": "",
            }
            assert not target.get("cache-from") and not target.get("cache-to")
            assert not target.get("no-cache-filter")
            assert not target.get("entitlements") and target.get("network") != "host"
        write_json(output / "bake-plan.json", plan)
        # Both full targets must succeed BEFORE publication; this is one build, no retry.
        command(
            "full_build",
            bake
            + [
                "--set",
                "test.output=type=cacheonly",
                "--set",
                "test-smoke.output=type=cacheonly",
                "test",
                "test-smoke",
            ],
        )
        state["both_full_targets_passed"] = True
        identity = digest((ENGINE + state["recipe_manifest_sha256"]).encode())[:20]
        image = (
            "ghcr.io/smg-project/smg:ci-moriio-read-"
            + identity
            + "-"
            + os.environ["GITHUB_RUN_ID"]
            + "-"
            + os.environ["GITHUB_RUN_ATTEMPT"]
        )
        metadata = output / "publication-metadata.json"
        # Export the already-built full test target; same builder/cache, no second cold build.
        command(
            "publish_cached_target",
            bake
            + [
                "--metadata-file",
                str(metadata),
                "--set",
                "test.tags=" + image,
                "--set",
                "test.output=type=registry",
                "test",
            ],
        )
        publication = json.loads(metadata.read_text())["test"]
        pin = publication["containerimage.digest"]
        assert re.fullmatch(r"sha256:[0-9a-f]{64}", pin)
        state.update(published_candidate=image, candidate_digest=pin, actual_metadata=publication)
        write_json(record, state)
        event("candidate_metadata_read_submitted", image=image, digest=pin)
        candidate = image_identity(
            "ghcr.io/smg-project/smg",
            pin,
            deadline,
            os.environ["GITHUB_ACTOR"],
            os.environ["MORIIO_PUBLISH_TOKEN"],
        )
        write_json(output / "candidate-identity.json", candidate)
        # Private publication is not falsely reported as public qualification delivery.
        event("anonymous_pull_metadata_read_submitted", image=image, digest=pin)
        anonymous = image_identity("ghcr.io/smg-project/smg", pin, deadline)
        assert anonymous["config_sha256"] == candidate["config_sha256"]
        state.update(
            public_pull_verified=True,
            status="candidate_published_requires_fresh_actual_GPU_qualification",
        )
    except BaseException as error:
        failure = error
        state.update(
            status="failed_or_published_incomplete_requires_readback",
            error_type=type(error).__name__,
        )
        if isinstance(error, urllib.error.HTTPError):
            state["registry_http_status"] = error.code
    finally:
        cleanup_deadline = time.monotonic() + 60
        signal.setitimer(signal.ITIMER_REAL, 60)
        try:
            if creation_submitted:
                # A failed create/bootstrap may still have made our exact container.
                if owned_id is None:
                    container = inspect("cleanup_discover", cleanup_deadline - 10)
                    assert (
                        container["Config"]["Image"] == BUILDKIT and container["Name"] == "/" + name
                    )
                    owned_id = container["Id"]
                container = inspect("cleanup_identity", cleanup_deadline - 10)
                verify_owned(container)
                command(
                    "cleanup_stop",
                    docker + ["stop", "--time", "5", owned_id],
                    cleanup_deadline - 10,
                )
                container = inspect("cleanup_stopped", cleanup_deadline - 5)
                verify_owned(container)
                assert not container["State"]["Running"] and container["State"]["Pid"] == 0
                state["cleanup_verified_stopped_PID0"] = True
                # Only this newly created builder; no --all/prune/foreign removal.
                command(
                    "cleanup_remove_builder",
                    docker + ["buildx", "rm", builder],
                    cleanup_deadline - 3,
                )
                remaining = command(
                    "cleanup_final_absence",
                    docker
                    + [
                        "ps",
                        "-a",
                        "--no-trunc",
                        "--filter",
                        "id=" + owned_id,
                        "--format",
                        "{{.ID}}",
                    ],
                    cleanup_deadline,
                    capture=True,
                )
                assert not remaining.strip()
                state["cleanup_verified_absent"] = True
            else:
                state["cleanup_verified_stopped_PID0"] = True
                state["cleanup_verified_absent"] = True
        except BaseException as error:
            state["cleanup_failure"] = type(error).__name__
        signal.setitimer(signal.ITIMER_REAL, 0)
        state["elapsed_seconds"] = time.monotonic() - start
        if failure is not None or "cleanup_failure" in state:
            state["status"] = "failed_or_published_incomplete_requires_readback"
        write_json(record, state)
    assert (
        failure is None
        and state.get("public_pull_verified")
        and state.get("cleanup_verified_stopped_PID0")
        and state.get("cleanup_verified_absent")
        and "cleanup_failure" not in state
    ), "Candidate operation incomplete; inspect private result/logs"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--engine-checkout", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    assert args.check != args.publish
    if args.check:
        print(
            json.dumps(
                {
                    "recipe_manifest_sha256": check_recipe(),
                    "engine": ENGINE,
                    "targets": ["test", "test-smoke"],
                    "execution": False,
                }
            )
        )
    else:
        assert args.engine_checkout and args.output
        publish(args.engine_checkout.resolve(), args.output.absolute())


if __name__ == "__main__":
    main()
