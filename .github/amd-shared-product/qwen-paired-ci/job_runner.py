"""Single manual GHA-owned runner carrier for unchanged paired-v5, not a publisher."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tarfile
import time
import traceback

import ci_identity
import model_identity

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = "1473e2d4fae7084d0f744055ba29a1a4b9679ae3"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.sha256(stream.read()).hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def active_drm_devices(topology, proof):
    """Match active KFD GPUs to their real DRM character devices."""
    proof.update({"visible_render_entries": sum(1 for _ in Path("/dev/dri").glob("renderD*")),
                  "topology": topology, "devices": []})
    assert len(topology) == len({row["node"] for row in topology}) == len({row["gpu_id"] for row in topology}) == 8
    paths = [Path("/dev/kfd")]
    for row in topology:
        assert row["node"].isdigit()
        node = Path("/sys/class/kfd/kfd/topology/nodes") / row["node"]
        assert int((node / "gpu_id").read_text()) == row["gpu_id"]
        props = dict(line.split(None, 1) for line in (node / "properties").read_text().splitlines() if line.strip())
        assert int(props["gfx_target_version"]) == row["gfx_target_version"] == 90402
        minor = int(props["drm_render_minor"])
        assert minor >= 0
        path = Path("/dev/dri") / ("renderD" + str(minor))
        proof["checking_path"] = str(path)
        info = path.stat()
        device = (os.major(info.st_rdev), os.minor(info.st_rdev))
        sysfs = tuple(int(v) for v in (Path("/sys/class/drm") / path.name / "dev").read_text().strip().split(":"))
        proof["devices"].append({"node": row["node"], "gpu_id": row["gpu_id"],
                                 "path": str(path), "drm_render_minor": minor,
                                 "character": stat.S_ISCHR(info.st_mode),
                                 "device": list(device), "sysfs_device": list(sysfs), "gid": info.st_gid})
        assert stat.S_ISCHR(info.st_mode) and device == sysfs and device[1] == minor
        paths.append(path)
    assert len({path.name for path in paths}) == 9
    proof["checking_path"] = str(paths[0])
    info = paths[0].stat()
    proof["kfd"] = {"path": str(paths[0]), "character": stat.S_ISCHR(info.st_mode), "gid": info.st_gid}
    assert stat.S_ISCHR(info.st_mode)
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("run", "cleanup", "require-result"))
    args = parser.parse_args()
    assert __debug__
    run_id, attempt = os.environ["GITHUB_RUN_ID"], os.environ["GITHUB_RUN_ATTEMPT"]
    assert run_id.isdigit() and attempt == "1"
    stage = Path(os.environ["RUNNER_TEMP"]) / ("qwen-paired-v5-" + run_id + "-" + attempt)
    if args.mode == "require-result":
        result = json.loads((stage / "root-result.json").read_text())
        assert result["passed"] is True and result["cleanup"]["owned_CID_absent"] is True
        return 0
    if args.mode == "cleanup":
        if not (stage / "state.json").is_file():
            return 0  # Admission failed before any create; do not guess a container.
        state = json.loads((stage / "state.json").read_text())
        return cleanup(stage, state)
    admission = json.loads((HERE / "admission.json").read_text())
    assert all(admission[key] is True for key in ("ready", "root_manual_admission", "run_budget_admitted", "artifacts_pinned")), "source-only packet is not admitted"
    assert admission["NEW_runner_seconds"] == 2100 and admission["reservation_seconds"] == 2100
    assert admission["source_parent"] == SOURCE and os.environ["ADMISSION_NONCE"] == admission["admission_nonce"]
    assert re.fullmatch("[a-f0-9]{32}", admission["admission_nonce"])
    manifest_path = HERE / "source-manifest.json"
    assert sha(manifest_path) == admission["source_manifest_sha256"]
    manifest = json.loads(manifest_path.read_text())
    for relative, expected in manifest.items():
        path = HERE / relative
        assert not Path(relative).is_absolute() and ".." not in Path(relative).parts and not path.is_symlink() and sha(path) == expected
    assert (ROOT / ".github/workflows/pr-test-rust.yml").read_bytes() == (HERE / "workflow-template.yml").read_bytes(), "only the reviewed registered-workflow overlay may dispatch"
    peer_path = HERE / "review/ci-source-peer.json"
    assert admission["new_CI_peer_path"] == "review/ci-source-peer.json" and sha(peer_path) == admission["new_CI_peer_sha256"]
    peer = json.loads(peer_path.read_text())
    shared_sources = json.loads((HERE / "shared-product-sources.json").read_text())
    assert set(shared_sources) == {"artifact_io.py", "frozen_producer.py", "relay.py", "source.py", "wheel_record.py"}
    for name, expected in shared_sources.items():
        path = HERE.parent / name
        assert path.is_file() and not path.is_symlink() and sha(path) == expected
    paired = HERE / "paired"
    assert peer["result"].startswith("PASS") and peer["source_parent"] == SOURCE
    for key, filename in (("driver_sha256", "run_paired_ab.py"), ("envelope_sha256", "root_container_envelope.py"), ("inputs_sha256", "inputs.json")):
        assert peer[key] == sha(paired / filename)
    assert peer["CI_job_runner_sha256"] == sha(__file__) and peer["workflow_template_sha256"] == sha(HERE / "workflow-template.yml")
    assert peer["CI_source_manifest_sha256"] == sha(manifest_path)
    actual_files = {path.relative_to(HERE).as_posix() for path in HERE.rglob("*") if path.is_file()}
    assert actual_files == set(manifest) | {"admission.json", "source-manifest.json", "review/ci-source-peer.json", "review/root-budget.json"}
    budget = HERE / "review/root-budget.json"
    assert admission["root_budget_receipt"] == "review/root-budget.json" and sha(budget) == admission["root_budget_receipt_sha256"]
    budget_value = json.loads(budget.read_text())
    assert budget_value["root_admitted"] is True and budget_value["NEW_runner_seconds"] == 2100 and budget_value["GPU_jobs"] == 1
    assert budget_value["admission_nonce"] == admission["admission_nonce"] and budget_value["no_retired_allowance_reused"] is True
    assert budget_value["registered_single_job_observer_source_reviewed"] is True and budget_value["global_owned_GPU_runs_after_dispatch"] <= 2
    identity = ci_identity.actual_job(admission)
    started = datetime.fromisoformat(identity["job"]["started_at"].replace("Z", "+00:00")).timestamp()
    elapsed = time.time() - started
    assert 0 <= elapsed < 180 and not stage.exists(), "never reset allocated runner's clock or retry its output"
    origin = time.monotonic() - elapsed
    stage.mkdir(mode=0o700)
    evidence = stage / "evidence"
    evidence.mkdir(mode=0o700)
    nonce = "smg-qwen-ab-" + admission["admission_nonce"]
    window = "GHA-" + run_id + "-1-" + admission["admission_nonce"]
    os.environ["E2E_JOB_MARKER"] = "e2e-" + run_id + "-1-" + os.environ["RUNNER_NAME"]
    state = {"nonce": nonce, "window_id": window, "run_id": int(run_id), "job_id": identity["job"]["id"],
             "actual_GHA_head": os.environ["GITHUB_SHA"], "source_parent": SOURCE, "CID": None,
             "create_attempted": False, "image": json.loads((paired / "inputs.json").read_text())["image"],
             "job_started_epoch": started, "job_started_monotonic": origin,
             "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
             "root_admission_sha256": sha(HERE / "admission.json")}
    save(stage / "state.json", state)
    save(evidence / "actual-GHA-runner-job.json", identity)
    result = {"passed": False, "actual_GHA_run": int(run_id), "actual_GHA_job": identity["job"]["id"], "source_parent": SOURCE,
              "source_checkout_head": os.environ["GITHUB_SHA"], "commands": [], "coverage_increment": 0,
              "original_Qwen6_pytest_run": False, "paired_observations_planned": 12, "cleanup": {}}
    client = None

    def command(argv, cap=15, check=True):
        remaining = origin + 2040 - time.monotonic()  # Last60 of the2100 job belongs to evidence upload.
        assert remaining > 0
        row = {"argv": argv, "at_utc": stamp()}
        result["commands"].append(row)
        completed = subprocess.run(argv, capture_output=True, timeout=min(cap, remaining), check=False)
        row.update(exit_code=completed.returncode, stdout=completed.stdout[:262144].decode(errors="replace"), stderr=completed.stderr[:262144].decode(errors="replace"))
        save(stage / "root-result.json", result)
        assert not check or completed.returncode == 0, str(argv[:3]) + " failed"
        return completed

    def interrupted(signum, frame):
        raise TimeoutError("actual GHA allocation absolute work deadline or cancellation")

    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)}
    signal.setitimer(signal.ITIMER_REAL, max(0.001, origin + 1920 - time.monotonic()))
    execution_stage = "precreate_identity_and_model_metadata"
    try:
        assert not command(["docker", "ps", "-q"]).stdout.splitlines(), "preserve every foreign/ordinary container"
        kfd_before = ci_identity.check_kfd(admission, socket.gethostname())
        save(evidence / "KFD-before-create.json", kfd_before)
        node = admission["nodes"][socket.gethostname()]
        assert node["GPU_index"] == "0", "this minimal carrier admits only the one root-checked GPU0"
        model = model_identity.validate(node["model"])
        execution_stage = "image_identity"
        image = json.loads(command(["docker", "image", "inspect", state["image"]]).stdout)[0]
        assert "lmsysorg/sglang@" + state["image"].split("@", 1)[1] in image["RepoDigests"]
        state["actual_image_id"] = image["Id"]
        execution_stage = "source_archive_extract"
        workspace = stage / "source"
        workspace.mkdir(mode=0o755)
        tarpath = stage / "source.tar"
        with tarpath.open("wb") as stream:
            subprocess.run(["git", "archive", "--format=tar", "HEAD"], cwd=ROOT, stdout=stream, check=True, timeout=10)
        links = []
        with tarfile.open(tarpath) as archive:
            for member in archive.getmembers():
                relative = Path(member.name)
                assert not relative.is_absolute() and ".." not in relative.parts
                target = workspace / relative
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                if member.issym():
                    links.append((target, member.linkname))
                else:
                    assert member.isfile()
                    target.write_bytes(archive.extractfile(member).read())
                    target.chmod(member.mode & 0o777)
        execution_stage = "source_archive_symlink_containment"
        for path, target in links:
            assert not Path(target).is_absolute() and (path.parent / target).resolve().is_relative_to(workspace)
            path.symlink_to(target)
        hostname = "qwen-ab-" + admission["admission_nonce"][:16]
        env = {"GITHUB_SHA": os.environ["GITHUB_SHA"], "GITHUB_RUN_ID": run_id, "GITHUB_RUN_ATTEMPT": "1",
               "GITHUB_REPOSITORY": "smg-project/smg", "E2E_JOB_MARKER": os.environ["E2E_JOB_MARKER"],
               "HF_HOME": "/models", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HOME": "/evidence/home",
               "AITER_JIT_DIR": "/evidence/cache/aiter-jit",
               "MAX_JOBS": "2",
               "HF_TOKEN_PATH": "/evidence/home/hf-token",
               "XDG_CACHE_HOME": "/evidence/cache", "CUDA_VISIBLE_DEVICES": "0", "HIP_VISIBLE_DEVICES": "0", "ROCR_VISIBLE_DEVICES": "0",
               "PYTHONDONTWRITEBYTECODE": "1", "SPT_NOENV": "1", "PYTHONPATH": "/source/e2e_test"}
        for path in (evidence / "home", evidence / "cache", evidence / "facts"):
            path.mkdir(mode=0o755)
        argv = ["docker", "create", "--pull=never", "--name", nonce, "--hostname", hostname, "--runtime=runc",
            "--label", "smg.qwen.ab.owner=root", "--label", "smg.qwen.ab.nonce=" + nonce, "--label", "smg.qwen.ab.window=" + window,
            "--device=/dev/kfd", "--device=/dev/dri", "--group-add=video", "--ipc=private", "--shm-size=32g",
            "--network=bridge", "--cpus=8", "--memory=96g", "--pids-limit=1024", "--ulimit", "memlock=-1:-1",
            "--tmpfs", "/tmp/aiter_configs:rw,nosuid,nodev,mode=0700,uid=" + str(os.getuid()) + ",gid=" + str(os.getgid()),
            "--user", str(os.getuid()) + ":" + str(os.getgid()), "--workdir=/source"]
        execution_stage = "active_KFD_DRM_character_device_guard"
        result["device_guard"] = {}
        device_paths = active_drm_devices(kfd_before["topology"], result["device_guard"])
        save(evidence / "active-DRM-before-create.json", result["device_guard"])
        for group in sorted({path.stat().st_gid for path in device_paths}):
            argv += ["--group-add", str(group)]
        for key, value in env.items():
            argv += ["--env", key + "=" + value]
        for host, target, readonly in ((ROOT, "/checkout", True), (workspace, "/source", False), (evidence, "/evidence", False), (Path("/models"), "/models", True)):
            argv += ["--mount", "type=bind,src=" + str(host) + ",dst=" + target + (",readonly" if readonly else "")]
        argv += [state["image"], "/bin/sleep", "infinity"]
        execution_stage = "container_create_and_following_execution"
        state["create_attempted"] = True
        save(stage / "state.json", state)
        cid = command(argv).stdout.decode().strip()
        assert re.fullmatch("[0-9a-f]{64}", cid)
        state["CID"] = cid
        save(stage / "state.json", state)
        inspect = json.loads(command(["docker", "inspect", cid]).stdout)[0]
        assert inspect["Id"] == cid and inspect["Image"] == state["actual_image_id"] and inspect["Config"]["Hostname"] == hostname
        assert inspect["HostConfig"].get("PidMode", "") != "host" and not inspect["HostConfig"]["Privileged"] and not inspect["HostConfig"]["AutoRemove"]
        save(evidence / "container-inspect-created.json", inspect)
        command(["docker", "start", cid])
        # Only this short relay exec inherits the Actions token. The model container
        # and engine processes never receive it in their Config/environment.
        command(["docker", "exec", "--env", "GITHUB_TOKEN", cid, "/opt/venv/bin/python", "/checkout/.github/amd-shared-product/relay.py", "--out", "/evidence/relay"], cap=90)
        relay = json.loads((evidence / "relay/evidence/relay.json").read_text())
        inputs = json.loads((paired / "inputs.json").read_text())
        assert relay["source_trees"] == inputs["consumer19"] and relay["compiled_source_trees"] == inputs["compiled37"] and relay["producer_source_trees"] == inputs["producer14"]
        for source_dir, destination in ((evidence / "relay/smg-wheel", workspace / "wheel"),
                (evidence / "relay/python-client-types", workspace / "clients/python/smg_client/types"),
                (evidence / "relay/wasm-test-fixtures", workspace / "crates/wasm/tests/fixtures")):
            destination.mkdir(parents=True, exist_ok=True)
            for path in source_dir.iterdir():
                assert path.is_file() and not path.is_symlink()
                shutil.copyfile(path, destination / path.name)
        context = {**state, "runner_UID": os.getuid(), "runner_GID": os.getgid()}
        save(evidence / "facts/context.json", context)
        command(["docker", "exec", "--user", "0", cid, "/opt/venv/bin/python",
            "/source/.github/amd-shared-product/qwen-paired-ci/prepare_container.py", "--context", "/evidence/facts/context.json"], cap=max(1, origin + 180 - time.monotonic()))
        prepared = json.loads((evidence / "preparation-proof.json").read_text())
        assert prepared["actual_success"] is True and prepared["whole_pip"] == 0 and prepared["native_sha256"] == inputs["gateway_native"]["sha256"]
        now = stamp()
        common = {"window_id": window, "CID": cid, "checked_at_utc": now}
        container = {**common, "owner": "root", "root_approved": True, "image": state["image"], "hostname": hostname,
            "GPU_devices_selected": 1, "foreign_jobs_present": False, "actual_GHA_job": identity["job"]["id"], "actual_runner": identity["job"]["runner_name"]}
        bound = {**common, "source_parent": SOURCE, "consumer19": inputs["consumer19"], "compiled37": inputs["compiled37"], "producer14": inputs["producer14"],
            "producer_run": relay["external_producer"]["run"], "producer_head": relay["external_producer"]["head"], "four_artifacts": inputs["four_artifacts"],
            "native_sha256": prepared["native_sha256"], "actual_producer_status": relay["external_producer"]["run_status"], "actual_producer_conclusion": relay["external_producer"]["run_conclusion"], "actual_checkout_head": os.environ["GITHUB_SHA"],
            "producer_context": json.loads((evidence / "relay/evidence/frozen-producer-context.json").read_text())}
        for key, name in (("producer_run_api", "producer-run.json"), ("producer_build_job_api", "producer-build-job.json")):
            path = evidence / "relay/evidence" / name
            bound[key] = {"path": "/evidence/relay/evidence/" + name, "bytes": path.stat().st_size, "sha256": sha(path)}
        resources = {**common, "root_approved": True, "NEW_runner_seconds": 2100, "no_retired_allowance_reused": True,
            "one_selected_GPU": True, "ordinary_jobs_preserved": True, "original_raw_image_HTTPS_read_allowed": True, "external_model_API_calls": 0,
            "root_budget_receipt_sha256": sha(budget), "runner_ownership": "actual workflow_dispatch single selfhosted job + real Runner.Worker ancestry/agent join"}
        facts = {"container_proof": container, "source_binding_proof": bound, "resource_and_network_proof": resources,
                 "model_snapshot_proof": {**common, **model}, "preparation_proof": prepared}
        approval = {"root_manual_admission": True, "NEW_window": True, "window_seconds": 2100, "window_id": window,
            "issued_at_utc": now, "window_started_at_utc": identity["job"]["started_at"], "GPU_jobs": 1, "GPU_devices_simultaneous": 1,
            "source_parent": SOURCE, "source_worktree": "/source", "python_resolved": prepared["python_resolved"], "output": "/evidence/paired",
            "actual_GPU_masks": {key: "0" for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")},
            "driver_sha256": sha(paired / "run_paired_ab.py"), "envelope_sha256": sha(paired / "root_container_envelope.py"), "inputs_sha256": sha(paired / "inputs.json"),
            "independent_peer_path": "/source/.github/amd-shared-product/qwen-paired-ci/review/ci-source-peer.json",
            "independent_peer_sha256": sha(peer_path), "host_independent_peer_path": str(peer_path), "CID": cid, "container_nonce": nonce,
            "actual_image_id": state["actual_image_id"], "image": state["image"], "container_hostname": hostname,
            "host_output": str(evidence / "envelope"), "container_runner": "/source/.github/amd-shared-product/qwen-paired-ci/paired/run_paired_ab.py",
            "container_approval": "/evidence/facts/approval.json", "container_output": "/evidence/paired"}
        for name, value in facts.items():
            path = evidence / "facts" / (name + ".json")
            save(path, value)
            approval[name] = {"path": "/evidence/facts/" + path.name, "bytes": path.stat().st_size, "sha256": sha(path)}
        save(evidence / "facts/approval.json", approval)
        state["envelope_start_epoch"] = time.time()
        save(stage / "state.json", state)
        with (evidence / "envelope.stdout").open("xb") as stdout, (evidence / "envelope.stderr").open("xb") as stderr:
            client = subprocess.Popen([sys.executable, str(paired / "root_container_envelope.py"), "--approval", str(evidence / "facts/approval.json"), "--output", str(evidence / "envelope")],
                stdout=stdout, stderr=stderr, start_new_session=True)
            state["client"] = ci_identity.proc(client.pid)
            save(stage / "state.json", state)
            # Reserve original120 cleanup +60 artifact seconds inside the same
            # actual2100 allocation. Tightening never converts a timeout to pass.
            try:
                code = client.wait(timeout=max(0.001, origin + 1920 - time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(client.pid, signal.SIGTERM)
                code = client.wait(timeout=max(0.001, min(30, origin + 2040 - time.monotonic())))
                result["outer_work_deadline_reached"] = True
        envelope = json.loads((evidence / "envelope/result.json").read_text())
        result["envelope"] = envelope
        result["passed"] = code == 0 and envelope["passed"] is True and not result.get("outer_work_deadline_reached", False)
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)[:2000],
                           "stage": execution_stage, "traceback": traceback.format_exc(limit=12)[-8192:]}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            result["cleanup"]["owned_CID_absent"] = cleanup(stage, state) == 0
        except BaseException as error:
            result["cleanup_error"] = repr(error)
            result["cleanup"]["owned_CID_absent"] = False
        result["elapsed_allocated_job_seconds"] = time.monotonic() - origin
        result["passed"] = bool(result["passed"] and result["cleanup"]["owned_CID_absent"] and time.monotonic() < origin + 2040)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        save(stage / "root-result.json", result)
    return 0 if result["passed"] else 1


def cleanup(stage, state):
    """Idempotent finite removal of this workflow's one exact nonce container."""
    assert state["run_id"] == int(os.environ["GITHUB_RUN_ID"]) and os.environ["GITHUB_RUN_ATTEMPT"] == "1"
    assert Path("/proc/sys/kernel/random/boot_id").read_text().strip() == state["boot_id"]
    deadline = min(time.monotonic() + 30, state["job_started_monotonic"] + 2040)
    report = {"at_utc": stamp(), "CID": state["CID"], "removed": False}
    try:
        client = state.get("client")
        if client is not None:
            try:
                current = ci_identity.proc(client["pid"])
                assert current["start_ticks"] == client["start_ticks"], "refuse reused client PID"
                assert os.getpgid(client["pid"]) == client["pid"]
                os.killpg(client["pid"], signal.SIGKILL)
            except (FileNotFoundError, ProcessLookupError):
                pass
    except BaseException as error:
        report["client_cleanup_error"] = repr(error)
    # Keep this independent from every client-reap path, including failures.
    try:
        def docker(*args):
            remaining = deadline - time.monotonic()
            assert remaining > 0
            done = subprocess.run(["docker", *args], capture_output=True, timeout=min(8, remaining), check=False)
            assert done.returncode == 0, done.stderr[:1000].decode(errors="replace")
            return done.stdout
        cid = state["CID"]
        if cid is None and state["create_attempted"]:
            matches = docker("ps", "-a", "--no-trunc", "--filter", "label=smg.qwen.ab.nonce=" + state["nonce"], "--format", "{{.ID}}").decode().splitlines()
            assert len(matches) <= 1
            cid = matches[0] if matches else None
        if cid is not None:
            assert re.fullmatch("[0-9a-f]{64}", cid)
            matches = docker("ps", "-a", "--no-trunc", "--filter", "id=" + cid, "--format", "{{.ID}}").decode().splitlines()
            if matches:
                assert matches == [cid]
                row = json.loads(docker("inspect", cid))[0]
                labels = row["Config"]["Labels"]
                assert row["Id"] == cid and row["Image"] == state["actual_image_id"]
                assert labels["smg.qwen.ab.owner"] == "root" and labels["smg.qwen.ab.nonce"] == state["nonce"] and labels["smg.qwen.ab.window"] == state["window_id"]
                docker("rm", "--force", cid)
            assert docker("ps", "-a", "--no-trunc", "--filter", "id=" + cid, "--format", "{{.ID}}").decode().splitlines() == []
        report["removed"] = True
    except BaseException as error:
        report["error"] = repr(error)
    if report["removed"]:
        try:
            # Keep all raw diagnostics/relay metadata for upload and root review;
            # discard only this run's exported sources and artifact payload copies.
            removed = []
            for path in [stage / "source", stage / "source.tar", *[stage / "evidence/relay" / name for name in
                    ("smg-wheel", "go-ffi-library", "python-client-types", "wasm-test-fixtures")]]:
                assert time.monotonic() < deadline
                if not path.exists():
                    continue
                assert not path.is_symlink() and path.resolve().is_relative_to(stage.resolve())
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                removed.append(str(path))
            report["owned_transient_inputs_removed"] = removed
        except BaseException as error:
            report["transient_cleanup_error"] = repr(error)
    save(stage / "cleanup-result.json", report)
    return 0 if report["removed"] and not any(key in report for key in ("client_cleanup_error", "transient_cleanup_error")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
