"""Observe the actual dispatch/assigned Runner.Worker, never reserve or stop it."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
from urllib.request import HTTPRedirectHandler, Request, build_opener


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        return None


def proc(pid):
    path = Path("/proc") / str(pid)
    raw = (path / "stat").read_text()
    fields = raw.rsplit(")", 1)[1].split()
    comm = (path / "comm").read_text().strip()
    again = (path / "stat").read_text().rsplit(")", 1)[1].split()
    assert (fields[1], fields[19]) == (again[1], again[19]), "process incarnation changed during observation"
    return {"pid": pid, "ppid": int(fields[1]), "start_ticks": fields[19], "comm": comm}


def get(path):
    assert path.startswith("actions/runs/") and ".." not in path
    request = Request("https://api.github.com/repos/smg-project/smg/" + path,
        headers={"Authorization": "Bearer " + os.environ["GITHUB_TOKEN"], "Accept": "application/vnd.github+json"})
    with build_opener(NoRedirect()).open(request, timeout=10) as response:
        raw = response.read(2 * 1024**2 + 1)
    assert len(raw) <= 2 * 1024**2
    return json.loads(raw)


def actual_job(admission):
    assert os.environ["GITHUB_EVENT_NAME"] == "workflow_dispatch" and os.environ["GITHUB_REPOSITORY"] == admission["repository"] == "smg-project/smg"
    assert os.environ["GITHUB_ACTOR"] == admission["approved_actor"]
    assert os.environ["GITHUB_REF"] == "refs/heads/" + admission["approved_branch"]
    run_id, attempt = os.environ["GITHUB_RUN_ID"], os.environ["GITHUB_RUN_ATTEMPT"]
    assert run_id.isdigit() and attempt == "1", "no workflow rerun/retry admission"
    run = get("actions/runs/" + run_id)
    assert run["id"] == int(run_id) and run["head_sha"] == os.environ["GITHUB_SHA"]
    assert run["head_branch"] == admission["approved_branch"] and run["event"] == "workflow_dispatch" and run["run_attempt"] == 1
    assert run["actor"]["login"] == admission["approved_actor"] and run["repository"]["full_name"] == admission["repository"]
    jobs = get("actions/runs/" + run_id + "/attempts/1/jobs?per_page=100")
    assert jobs["total_count"] == len(jobs["jobs"]) == 1, "exact single scheduled GPU job"
    job = jobs["jobs"][0]
    assert job["name"] == "qwen-paired-v5 / run" and job["status"] == "in_progress" and job["conclusion"] is None
    assert job["run_id"] == int(run_id) and job["head_sha"] == os.environ["GITHUB_SHA"] and job["runner_name"] == os.environ["RUNNER_NAME"]
    assert job["runner_name"] == socket.gethostname() and job["runner_name"] in admission["nodes"]
    processes = []
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            try:
                row = proc(int(path.name))
                if row["comm"] in {"Runner.Worker", "Runner.Listener"}:
                    processes.append(row)
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
    workers = [row for row in processes if row["comm"] == "Runner.Worker"]
    listeners = [row for row in processes if row["comm"] == "Runner.Listener"]
    assert len(workers) == len(listeners) == 1
    ancestors, pid = [], os.getpid()
    for _ in range(32):
        row = proc(pid)
        ancestors.append(row)
        if row["ppid"] <= 1:
            break
        pid = row["ppid"]
    assert any(row["pid"] == workers[0]["pid"] and row["start_ticks"] == workers[0]["start_ticks"] for row in ancestors), "current command is not owned by the assigned Runner.Worker"
    runner_root = Path("/proc", str(listeners[0]["pid"]), "exe").resolve(strict=True).parent.parent
    agent = json.loads((runner_root / ".runner").read_text(encoding="utf-8-sig"))
    assert agent["agentName"] == job["runner_name"] and agent["agentId"] == job["runner_id"]
    return {"run": run, "job": job, "processes": processes, "worker": workers[0], "ancestors": ancestors,
            "agent": {"name": agent["agentName"], "id": agent["agentId"]}, "checked_at_utc": datetime.now(timezone.utc).isoformat()}


def check_kfd(admission, hostname):
    allowed = {int(row["pid"]): row for row in admission["nodes"][hostname]["permitted_background_processes"]}
    observed = []
    root = Path("/sys/class/kfd/kfd/proc")
    assert root.is_dir()
    topology = []
    for node in Path("/sys/class/kfd/kfd/topology/nodes").iterdir():
        gpu_id = int((node / "gpu_id").read_text())
        if not gpu_id:
            continue
        props = dict(line.split(None, 1) for line in (node / "properties").read_text().splitlines() if line.strip())
        assert int(props["gfx_target_version"]) == 90402
        topology.append({"node": node.name, "gpu_id": gpu_id, "gfx_target_version": 90402})
    assert len(topology) == 8, "exact authorized MI300X topology"
    for path in root.iterdir():
        if not path.name.isdigit():
            continue
        row = proc(int(path.name))
        expected = allowed.get(row["pid"])
        assert expected is not None and row["start_ticks"] == expected["start_ticks"] and row["ppid"] == 1 and row["comm"] == "gpuagent"
        executable = Path("/proc", path.name, "exe").resolve(strict=True)
        assert str(executable) == expected["executable"]
        assert hashlib.sha256(executable.read_bytes()).hexdigest() == expected["executable_sha256"]
        vram = {item.name: int(item.read_text().strip()) for item in path.glob("vram_*")}
        assert vram and not any(vram.values()), "reported active GPU allocation is not idle"
        observed.append({**row, "vram_reported": vram, "executable": str(executable)})
    return {"observed_KFD_processes": observed, "topology": topology, "physical_HBM_reclamation_proven": False,
            "permission": "No signals/service/Listener changes; unrecognized process refuses admission."}
