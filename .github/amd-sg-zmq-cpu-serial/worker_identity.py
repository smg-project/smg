"""GET-only actual GHA job and sole ancestor Worker identity; no reservation."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
from urllib.request import HTTPRedirectHandler, Request, build_opener

REPO = "smg-project/smg"
WORKFLOW = ".github/workflows/pr-test-rust.yml"
JOB = "normal-sg-zmq-cpu"
RUNNERS = {"amd-mi300x-1": 331464, "amd-mi300x-2": 331467}


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        return None


def proc(pid):
    path = Path("/proc") / str(pid)
    fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
    comm = (path / "comm").read_text().strip()
    argv = (path / "cmdline").read_bytes()
    again = (path / "stat").read_text().rsplit(")", 1)[1].split()
    assert (fields[1], fields[19]) == (again[1], again[19]), "process incarnation changed"
    return {"pid": pid, "ppid": int(fields[1]), "start_ticks": fields[19], "comm": comm,
            "argv_sha256": hashlib.sha256(argv).hexdigest()}


def local_worker(hostname):
    assert socket.gethostname() == os.environ["RUNNER_NAME"] == hostname and hostname in RUNNERS
    rows = []
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            try:
                row = proc(int(path.name))
                if row["comm"] in {"Runner.Worker", "Runner.Listener"}:
                    rows.append(row)
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
    workers = [r for r in rows if r["comm"] == "Runner.Worker"]
    listeners = [r for r in rows if r["comm"] == "Runner.Listener"]
    assert len(workers) == len(listeners) == 1, "exact sole Worker and Listener required"
    pid, ancestors = os.getpid(), []
    for _ in range(32):
        row = proc(pid)
        ancestors.append(row)
        if row["ppid"] <= 1:
            break
        pid = row["ppid"]
    assert any(r == workers[0] for r in ancestors), "driver is not assigned Worker's descendant"
    runner_root = Path(os.readlink("/proc/" + str(listeners[0]["pid"]) + "/cwd"))
    path = runner_root / ".runner"
    assert path.is_file() and not path.is_symlink() and path.stat().st_size < 65536
    agent = json.loads(path.read_text(encoding="utf-8-sig"))
    assert agent["agentName"] == hostname and agent["agentId"] == RUNNERS[hostname]
    return {"worker": workers[0], "listener": listeners[0],
            "agent": {"name": hostname, "id": agent["agentId"]}}


def get(path):
    assert path.startswith("actions/runs/") and ".." not in path
    request = Request("https://api.github.com/repos/" + REPO + "/" + path,
        headers={"Authorization": "Bearer " + os.environ["GITHUB_TOKEN"], "Accept": "application/vnd.github+json"})
    with build_opener(NoRedirect()).open(request, timeout=10) as response:
        raw = response.read(2 * 1024**2 + 1)
    assert len(raw) <= 2 * 1024**2
    return json.loads(raw)


def actual_job(admission):
    assert __debug__ and os.environ["GITHUB_EVENT_NAME"] == "workflow_dispatch"
    assert os.environ["GITHUB_REPOSITORY"] == admission["repository"] == REPO
    assert os.environ["GITHUB_ACTOR"] == admission["approved_actor"]
    assert os.environ["GITHUB_REF"] == "refs/heads/" + admission["approved_branch"]
    run_id = os.environ["GITHUB_RUN_ID"]
    assert run_id.isdigit() and os.environ["GITHUB_RUN_ATTEMPT"] == "1"
    run = get("actions/runs/" + run_id)
    assert run["id"] == int(run_id) and run["head_sha"] == os.environ["GITHUB_SHA"]
    assert run["head_branch"] == admission["approved_branch"] and run["path"] == WORKFLOW
    assert run["event"] == "workflow_dispatch" and run["run_attempt"] == 1
    assert run["actor"]["login"] == admission["approved_actor"]
    assert run["status"] == "in_progress" and run["conclusion"] is None
    assert all(run[k]["full_name"] == REPO for k in ("repository", "head_repository"))
    jobs = get("actions/runs/" + run_id + "/attempts/1/jobs?per_page=100")
    assert jobs["total_count"] == len(jobs["jobs"]) == 1, "exact single scheduled CPU job"
    job = jobs["jobs"][0]
    assert job["name"] == JOB and job["status"] == "in_progress" and job["conclusion"] is None
    assert job["run_id"] == int(run_id) and job["head_sha"] == os.environ["GITHUB_SHA"]
    assert job.get("run_attempt", 1) == 1 and job["labels"] == ["1-gpu-mi300x"]
    local = local_worker(os.environ["RUNNER_NAME"])
    assert job["runner_name"] == local["agent"]["name"] and job["runner_id"] == local["agent"]["id"]
    assert job["started_at"] and job["completed_at"] is None
    return {"run": run, "job": job, "local": local,
            "checked_at_utc": datetime.now(timezone.utc).isoformat()}
