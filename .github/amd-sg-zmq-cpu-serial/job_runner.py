"""Host step for one scheduled CPU replay; does not dispatch or modify CONTROL."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import worker_identity


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("run", "require-result"))
    args = parser.parse_args()
    assert os.getuid()==os.geteuid()==995 and os.getgid()==os.getegid()==985
    assert __debug__ and os.environ["GITHUB_RUN_ID"].isdigit() and os.environ["GITHUB_RUN_ATTEMPT"] == "1"
    admission = read(HERE / "admission.json")
    assert all(admission[k] is True for k in ("ready", "root_admitted", "run_budget_admitted", "artifacts_pinned")), "source only, not admitted"
    assert re.fullmatch("[0-9a-f]{32}", admission["window_nonce"])
    assert os.environ["ADMISSION_NONCE"] == admission["window_nonce"]
    assert admission["CPU_seconds_new"] == 600 and admission["GPU"] == 0 and admission["automatic_retry"] is False
    output = Path("/tmp/smg-sg-zmq-serial-output-" + admission["window_nonce"] + "-" + os.environ["GITHUB_RUN_ID"])
    if args.mode == "require-result":
        result = read(output / "driver-result.json")
        assert result["status"].startswith("PASS_") and result["within_NEW600"] is True
        assert result["elapsed_seconds"] <= 600 and result["coverage_increment"] == result["GPU"] == 0
        assert all(result["cleanup"].get(k) is True for k in ("owned_CID_absent_machine_readable", "owned_attach_PG_absent", "owned_tmpfs_work_removed_with_CID", "durable_evidence_retained_not_deleted"))
        assert sha(output / "CPU-evidence.tar") == result["evidence_archive"]["sha256"]
        return 0
    assert not output.exists(), "never reset/retry an admitted output"
    manifest = read(HERE / "source-manifest.json")
    assert sha(HERE / "source-manifest.json") == admission["source_manifest_sha256"]
    for name, digest in manifest.items():
        assert Path(name).name == name and not (HERE / name).is_symlink() and sha(HERE / name) == digest
    assert (ROOT / worker_identity.WORKFLOW).read_bytes() == (HERE / "workflow-template.yml").read_bytes()
    peer = HERE / "review/peer-independent.json"
    budget = HERE / "review/root-budget.json"
    assert sha(peer) == admission["source_peer_sha256"] and sha(budget) == admission["root_budget_sha256"]
    p, b = read(peer), read(budget)
    assert p["result"].startswith("PASS") and p["findings"] == []
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=10).strip() == os.environ["GITHUB_SHA"]
    assert subprocess.check_output(["git", "rev-parse", "HEAD^"], cwd=ROOT, text=True, timeout=10).strip() == p["carrier_head"]
    assert b["root_admitted"] is True and b["window_nonce"] == admission["window_nonce"]
    assert b["CPU_seconds_new"] == 600 and b["GPU"] == 0 and b["max_owned_runners_after"] <= 2
    assert b["sole_observer_source_reviewed"] is True and b["no_retired_allowance_reused"] is True
    packet = Path("/tmp/smg-sg-zmq-serial-packet-" + admission["window_nonce"])
    assert str(packet) == admission["packet_path"] and packet.is_dir() and not packet.is_symlink()
    data = Path('/tmp/smg-sg-zmq-serial-data-' + admission['window_nonce'])
    assert str(data)==admission['data_path'] and data.resolve(strict=True)==data
    assert data.is_dir() and not data.is_symlink() and data.stat().st_uid==0 and not data.stat().st_mode&0o022
    assert sha(data/'data-files.json')==admission['data_manifest_sha256']
    for entry in data.iterdir():
        assert entry.is_file() and not entry.is_symlink() and entry.stat().st_uid==0 and not entry.stat().st_mode&0o022
    assert packet.stat().st_uid == 0 and not packet.stat().st_mode & 0o022
    packet_manifest = read(packet / "packet-files.json")
    assert sha(packet / "packet-files.json") == admission["packet_manifest_sha256"]
    files = list(packet.rglob("*"))
    assert len(files) < 256 and all(not f.is_symlink() for f in files)
    assert all(stat.S_ISDIR(f.stat().st_mode) or stat.S_ISREG(f.stat().st_mode) for f in files)
    assert all(f.stat().st_uid == 0 and not f.stat().st_mode & 0o022 for f in files)
    regular = [f for f in files if f.is_file()]
    assert {f.relative_to(packet).as_posix() for f in regular} == set(packet_manifest) | {"packet-files.json"}
    assert sum(f.stat().st_size for f in regular) < 192 * 1024**2
    for name, digest in packet_manifest.items():
        rel = Path(name)
        assert name and not rel.is_absolute() and ".." not in rel.parts
        f = packet / rel
        assert stat.S_ISREG(f.stat().st_mode) and f.stat().st_nlink == 1 and f.stat().st_uid == 0
        assert not f.stat().st_mode & 0o022 and sha(f) == digest
    assert sha(packet / "run_owned_sg_zmq_cpu600.py") == sha(HERE / "run_owned_sg_zmq_cpu600.py") == p["driver_sha256"]
    assert sha(packet / "worker_identity.py") == sha(HERE / "worker_identity.py") == p["worker_identity_sha256"]
    for key, name in (("payload_sha256", "payload.py"), ("inputs_sha256", "inputs.json"), ("frozen_manifest_sha256", "frozen-files.json")):
        assert sha(packet / name) == p[key]
    actual = worker_identity.actual_job(admission)
    job_start = datetime.fromisoformat(actual["job"]["started_at"].replace("Z", "+00:00")).timestamp()
    created = datetime.fromisoformat(actual["run"]["created_at"].replace("Z", "+00:00")).timestamp()
    assert 0 <= job_start - created < 900 and 0 <= time.time() - created < 1800, "queued/overall admission expired"
    assert 0 <= time.time() - job_start < 30, "allocated job clock never reset"
    approval_path = output.with_suffix(".approval.json")
    actual_path = output.with_suffix(".actual-GHA-job.json")
    save(actual_path, actual)
    approval = {"root_admitted": True, "CPU_seconds_new": 600, "GPU": 0,
        "issued_epoch": job_start, "capacity_epoch": time.time(), "hostname": socket.gethostname(),
        "root_CPU_slot_registered_only_this_window": True, "controlled_network_for_normal_APT_and_PyPI": True,
        "owned_CID_default_root_capabilities_for_evidence_owner": admission["owned_CID_default_root_capabilities_for_evidence_owner"],
        "driver_sha256": p["driver_sha256"], "inputs_sha256": p["inputs_sha256"],
        "output_directory": str(output), "source_peer_path": str(peer),
        "source_peer": {"bytes": peer.stat().st_size, "sha256": sha(peer)},
        "packet_manifest_sha256": admission["packet_manifest_sha256"],
        "data_manifest_sha256": admission["data_manifest_sha256"],
        "image_id": read(packet / "inputs.json")["image"].split("@", 1)[1],
        "GHA_local_worker": actual["local"], "GHA_run_id": actual["run"]["id"],
        "GHA_head_sha": actual["run"]["head_sha"], "GHA_job_id": actual["job"]["id"],
        "GHA_workflow_path": worker_identity.WORKFLOW, "GHA_window_nonce": admission["window_nonce"],
        "GHA_identity_evidence_path": str(actual_path), "GHA_identity_evidence_sha256": sha(actual_path)}
    save(approval_path, approval)
    argv = ["/usr/bin/python3", "-I", "-B", str(packet / "run_owned_sg_zmq_cpu600.py"), "--packet", str(packet),
            "--data", str(data), "--approval", str(approval_path), "--approval-sha256", sha(approval_path), "--out", str(output)]
    save(output.with_suffix(".exact-argv.json"), {"argv": argv, "job_started_epoch": job_start, "window_nonce": admission["window_nonce"]})
    with Path(os.environ["GITHUB_ENV"]).open("a") as stream:
        stream.write("CPU_SERIAL_OUTPUT=" + str(output) + "\n")
    os.execv(argv[0], argv)


if __name__ == "__main__":
    raise SystemExit(main())
