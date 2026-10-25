"""One scheduled, no-GPU real-metadata probe; all output is private to this job."""
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def git(*args):
    return subprocess.check_output(["/usr/bin/git", "-c", "safe.directory=", "-c", "safe.directory=" + str(ROOT), *args], cwd=ROOT, text=True, timeout=10).strip()

def process(pid):
    path = Path("/proc") / str(pid) / "stat"
    def read():
        fields = path.read_text().rsplit(") ", 1)[1].split()
        return {"PID":pid, "group":int(fields[2]), "start_ticks":fields[19], "state":fields[0]}
    first, second = read(), read()
    assert all(first[k] == second[k] for k in ("PID", "group", "start_ticks"))
    return second

def group_members(group):
    members = []
    deadline = time.monotonic() + 2
    for name in os.listdir("/proc"):
        assert time.monotonic() < deadline
        if not name.isdecimal():
            continue
        try:
            value = process(int(name))
        except FileNotFoundError:
            continue
        if value["group"] == group:
            members.append(value)
            assert len(members) <= 256
    return members

def run_probe(command, stdout, stderr):
    child = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True)
    anchor = process(child.pid)
    assert anchor["group"] == child.pid
    until = time.monotonic() + 125
    done = None
    while time.monotonic() < until:
        done = os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        if done is not None:
            break
        time.sleep(.1)
    timed_out = done is None
    def signal_owned(sig):
        current = process(child.pid)
        assert all(current[k] == anchor[k] for k in ("PID", "group", "start_ticks"))
        # WNOWAIT holds the original leader PID through the last group signal.
        os.killpg(child.pid, sig)
    members = group_members(child.pid)
    live = [v for v in members if v["state"] not in ("Z", "X")]
    if timed_out or any(v["PID"] != child.pid for v in live):
        signal_owned(signal.SIGTERM)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if not any(v["state"] not in ("Z", "X") for v in group_members(child.pid)):
                break
            time.sleep(.1)
        if any(v["state"] not in ("Z", "X") for v in group_members(child.pid)):
            signal_owned(signal.SIGKILL)
    code = child.wait(timeout=2)
    remaining = group_members(child.pid)
    cleanup = {"leader":anchor, "leader_held_unreaped_until_signals_finished":True,
               "remaining_members":remaining, "owned_group_absent":not remaining,
               "timed_out":timed_out}
    return (124 if timed_out else code if not remaining else 1), cleanup

def restore_output_owner(output):
    deadline = time.monotonic() + 3
    pending, count, total = [output], 0, 0
    while pending:
        directory = pending.pop()
        assert time.monotonic() < deadline
        info = directory.lstat()
        assert stat.S_ISDIR(info.st_mode) and not directory.is_symlink()
        count += 1
        assert count <= 512
        os.chown(directory, 995, 985, follow_symlinks=False)
        with os.scandir(directory) as entries:
            for entry in entries:
                assert time.monotonic() < deadline
                info = entry.stat(follow_symlinks=False)
                assert not entry.is_symlink()
                if stat.S_ISDIR(info.st_mode):
                    assert len(pending) + count < 512
                    pending.append(Path(entry.path))
                else:
                    assert stat.S_ISREG(info.st_mode) and info.st_nlink == 1
                    count += 1
                    total += info.st_size
                    assert count <= 512 and total <= 64 * 1024**2
                    os.chown(entry.path, 995, 985, follow_symlinks=False)
    return {"entries":count, "bytes":total, "actual_UID":995, "actual_GID":985,
            "lazy_nofollow_walk":True}

def main():
    assert __debug__ and os.geteuid() == 0
    a = json.loads((HERE / "admission.json").read_text())
    assert a["root_admitted"] is a["ready"] is True
    assert (a["CPU_new_seconds"], a["GPU"], a["automatic_retry"]) == (180, 0, False)
    nonce = os.environ["ADMISSION_NONCE"]
    assert re.fullmatch("[0-9a-f]{32}", nonce) and nonce == a["window_nonce"]
    assert os.environ["GITHUB_REPOSITORY"] == "smg-project/smg"
    assert os.environ["GITHUB_EVENT_NAME"] == "workflow_dispatch" and os.environ["GITHUB_RUN_ATTEMPT"] == "1"
    assert re.fullmatch("[0-9a-f]{40}", os.environ["GITHUB_SHA"])
    assert ROOT == ROOT.resolve(strict=True) and ROOT.name == ".smg-mc4-real-metadata-" + nonce
    assert git("rev-parse", "HEAD") == os.environ["GITHUB_SHA"]
    assert git("rev-parse", "HEAD^") == a["carrier_head"]
    assert a["product_head"] == "a03e3dba902a0be3caedd2b65f50af7a2804d3e6"
    changed = set(git("diff", "--name-only", "HEAD^", "HEAD").splitlines())
    assert changed == {".github/workflows/pr-test-rust.yml", ".github/amd-mc4-real-metadata-cpu/admission.json", ".github/amd-mc4-real-metadata-cpu/peer-independent.json"}
    assert (ROOT / ".github/workflows/pr-test-rust.yml").read_bytes() == (HERE / "workflow-template.yml").read_bytes()
    peer = HERE / "peer-independent.json"
    assert sha(peer) == a["peer_sha256"]
    p = json.loads(peer.read_text())
    assert p["result"].startswith("PASS") and p["findings"] == [] and p["carrier_head"] == a["carrier_head"]
    program = HERE / "real_metadata_check.py"
    assert sha(program) == a["program_sha256"] == p["program_sha256"]
    assert sha(Path(__file__).resolve()) == p["job_runner_sha256"]
    assert not Path("/dev/kfd").exists() and not list(Path("/dev/dri").glob("renderD*")), "no GPU devices"
    parent = Path(os.environ["RUNNER_TEMP"]).resolve(strict=True)
    output = parent / ("smg-mc4-real-metadata-" + nonce + "-" + os.environ["GITHUB_RUN_ID"])
    assert not output.exists() and not output.is_symlink()
    output.mkdir(mode=0o700)
    os.chown(output, 995, 985)
    with Path(os.environ["GITHUB_ENV"]).open("a") as stream:
        stream.write("MC4_METADATA_OUTPUT=" + str(output) + "\n")
    command = [sys.executable, "-I", "-B", str(program), "--source", str(ROOT), "--evidence", str(output / "real"), "--nonce", nonce, "--carrier-head", os.environ["GITHUB_SHA"]]
    identity = {k: os.environ[k] for k in ("GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT", "GITHUB_SHA", "GITHUB_REPOSITORY", "GITHUB_EVENT_NAME", "ADMISSION_NONCE")}
    identity.update(argv=command, GPU_devices=0, CPU_job_timeout_seconds=180, probe_timeout_seconds=125)
    (output / "scheduled-identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    start = time.monotonic()
    code = 1
    try:
        with (output / "probe.stdout").open("xb") as stdout, (output / "probe.stderr").open("xb") as stderr:
            code, cleanup = run_probe(command, stdout, stderr)
    finally:
        (output / "host-result.json").write_text(json.dumps({"exit_code":code, "probe_elapsed_seconds":time.monotonic()-start, "cleanup":locals().get("cleanup"), "GPU":0, "coverage_increment":0}, indent=2) + "\n")
        owner = restore_output_owner(output)
        (output / "evidence-owner.json").write_text(json.dumps(owner, indent=2) + "\n")
        os.chown(output / "evidence-owner.json",995,985)
    return code

if __name__ == "__main__":
    raise SystemExit(main())
