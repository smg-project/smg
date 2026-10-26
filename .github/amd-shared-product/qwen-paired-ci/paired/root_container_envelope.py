"""Root's finite local-node envelope; SOURCE ONLY and approval defaults false.

The exact container must already have been created/prepared by root within the
same2100-second window, using real CI setup. Never create/pull/restart a container,
touch a service, or select a different node here. Finalize only this nonce CID.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

DOCKER = ["docker", "--host", "unix:///var/run/docker.sock"]
IMAGE = "lmsysorg/sglang:v0.5.21-rocm10-mi30x@sha256:4420a7e492fce623285c635b9a6a7e09ccf77a3c812006494e4414655d46662d"
SOURCE = "1473e2d4fae7084d0f744055ba29a1a4b9679ae3"

def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def main(approval_path, output):
    begin = time.monotonic()
    approval = json.loads(approval_path.read_text())
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    need(approval["root_manual_admission"] is True and approval["envelope_sha256"] == digest, "root envelope approval/hash required")
    need(approval["NEW_window"] is True and approval["window_seconds"] == 2100 and approval["GPU_jobs"] == 1, "new2100 only")
    need(approval["image"] == IMAGE and approval["source_parent"] == SOURCE, "exact4420/new frozen source cohort")
    peer_path = Path(approval["host_independent_peer_path"])
    need(hashlib.sha256(peer_path.read_bytes()).hexdigest() == approval["independent_peer_sha256"], "peer SHA")
    peer = json.loads(peer_path.read_text())
    need(peer["result"].startswith("PASS") and peer["source_parent"] == SOURCE and peer["driver_sha256"] == approval["driver_sha256"] and
         peer["envelope_sha256"] == digest and peer["inputs_sha256"] == approval["inputs_sha256"], "exact independent peer required")
    issued = datetime.fromisoformat(approval["issued_at_utc"].replace("Z", "+00:00"))
    now = datetime.now(timezone.utc)
    need(0 <= (now - issued).total_seconds() < 180, "stale root approval")
    window = datetime.fromisoformat(approval["window_started_at_utc"].replace("Z", "+00:00"))
    spent = (now - window).total_seconds()
    need(0 <= spent < 180, "CI preparation180s budget expired")
    deadline = begin + 2100 - spent
    cid, nonce = approval["CID"], approval["container_nonce"]
    need(len(cid) == 64 and all(c in "0123456789abcdef" for c in cid), "exact full CID required")
    need(nonce.startswith("smg-qwen-ab-") and len(nonce) < 100, "own container nonce required")
    need(str(output.resolve()) == approval["host_output"] and not output.exists(), "unused exact host output")
    output.mkdir(mode=0o700)
    report = {"passed": False, "CID": cid, "window": approval["window_id"], "operations": [], "GPU_jobs": 1}
    child = None

    def command(argv, cap=10):
        remaining = deadline - time.monotonic()
        need(remaining > 0, "absolute envelope deadline")
        result = subprocess.run(argv, capture_output=True, timeout=min(cap, remaining), check=False)
        report["operations"].append({"argv": argv, "exit": result.returncode})
        return result

    def inspect_own(running=True):
        result = command([*DOCKER, "inspect", cid], cap=3)
        need(result.returncode == 0, "bound container unavailable")
        rows = json.loads(result.stdout)
        need(len(rows) == 1 and rows[0]["Id"] == cid, "CID differs")
        row = rows[0]
        labels = row["Config"].get("Labels") or {}
        need(labels.get("smg.qwen.ab.owner") == "root" and labels.get("smg.qwen.ab.nonce") == nonce
             and labels.get("smg.qwen.ab.window") == approval["window_id"], "refuse foreign container")
        need(row["Image"] == approval["actual_image_id"] and row["Config"]["Image"] == approval["image"], "exact4420 image")
        need(row["HostConfig"].get("PidMode", "") != "host" and not row["HostConfig"].get("Privileged"), "own PID namespace/nonprivileged required")
        need((not running or row["State"]["Running"] is True) and row["Config"]["Hostname"] == approval["container_hostname"], "actual container/hostname")
        need(not row["HostConfig"].get("AutoRemove"), "root envelope owns final removal")
        return row

    def interrupted(signum, frame):
        raise TimeoutError("root signal/absolute envelope work deadline")

    previous = {sig: signal.signal(sig, interrupted) for sig in [signal.SIGTERM, signal.SIGINT, signal.SIGALRM]}
    signal.setitimer(signal.ITIMER_REAL, max(0.001, deadline - 20 - time.monotonic()))
    try:
        inspect_own()
        argv = [*DOCKER, "exec", cid, "/opt/venv/bin/python", approval["container_runner"],
                "--approval", approval["container_approval"], "--output", approval["container_output"]]
        with (output / "container.stdout").open("xb") as stdout, (output / "container.stderr").open("xb") as stderr:
            child = subprocess.Popen(argv, stdout=stdout, stderr=stderr, start_new_session=True)
            try:
                # Reserve20 seconds for exact-CID cleanup, even if the inner code hangs.
                code = child.wait(timeout=max(0.001, deadline - 20 - time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=2)
                raise TimeoutError("root envelope work deadline; exact owned CID will be removed")
        report["inner_exit"] = code
        report["passed"] = code == 0
    except BaseException as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)[:1024]}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=2)
        except BaseException as error:
            report["client_cleanup_error"] = {"type": type(error).__name__, "message": str(error)[:1024]}
            report["passed"] = False
        finally:
            try:
                # Re-inspect immutable ID and ownership; no container-name guessing.
                inspect_own(running=False)
                removed = command([*DOCKER, "rm", "--force", cid], cap=12)
                report["cleanup"] = removed.returncode == 0
            except BaseException as error:
                report["cleanup"] = False
                report["cleanup_error"] = str(error)[:1024]
            report["elapsed_seconds"] = time.monotonic() - begin
            report["total_window_seconds"] = spent + report["elapsed_seconds"]
            report["passed"] = bool(report["passed"] and report.get("cleanup") and time.monotonic() <= deadline)
            for sig, handler in previous.items():
                signal.signal(sig, handler)
            (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approval", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    raise SystemExit(main(args.approval, args.output))
