"""One device-free setup command under the actual job's absolute time budget."""

import datetime
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

from proof import require, write

ROOT = Path(__file__).resolve().parents[2]
IMAGE = "vllm/vllm-openai-rocm:v0.31.0@sha256:749f6f3f944f12af49966ac523c1f8573e4b229594b541954bd5f879b4496b1f"
COMMAND = ["bash", str(Path(__file__).with_name("run.sh"))]
INPUTS = Path(__file__).with_name("inputs.json")


def deadline(jobs, now):
    require(os.environ["GITHUB_RUN_ATTEMPT"] == "1", "reruns are not admitted")
    require(len(jobs) == 1, "expected exactly one consumer job")
    job = jobs[0]
    require(
        job["name"] == "cairo-product-setup"
        and job["runner_name"] == os.environ["RUNNER_NAME"]
        and job["head_sha"] == os.environ["GITHUB_SHA"]
        and job["status"] == "in_progress",
        "consumer identity differs",
    )
    started = datetime.datetime.fromisoformat(job["started_at"].replace("Z", "+00:00"))
    require(started.tzinfo is not None, "job start lacks timezone")
    elapsed = now - started.timestamp()
    require(0 <= elapsed < 235, "consumer core budget expired or start is in the future")
    return {
        "job_id": job["id"],
        "started_at": job["started_at"],
        "soft_deadline_epoch": started.timestamp() + 235,
        "hard_deadline_epoch": started.timestamp() + 240,
        "job_deadline_epoch": started.timestamp() + 300,
    }


def get_jobs():
    require(os.environ["GITHUB_REPOSITORY"] == "smg-project/smg", "unexpected repository")
    path = f"https://api.github.com/repos/smg-project/smg/actions/runs/{int(os.environ['GITHUB_RUN_ID'])}/attempts/1/jobs"
    request = urllib.request.Request(
        path,
        headers={
            "Authorization": "Bearer " + os.environ["GITHUB_TOKEN"],
            "Accept": "application/vnd.github+json",
        },
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.load(response)["jobs"]


def main():
    inputs = json.loads(INPUTS.read_text())
    require(inputs["ready"] is True, "ready=false; root admission is required")
    require(inputs["image"] == os.environ["SMG_QA_IMAGE"] == IMAGE, "image binding differs")
    require(
        [
            inputs[key]
            for key in [
                "job_seconds",
                "core_soft_seconds",
                "core_kill_grace_seconds",
                "core_hard_seconds",
                "cleanup_upload_seconds",
            ]
        ]
        == [300, 235, 5, 240, 60],
        "budget policy differs",
    )
    record = deadline(get_jobs(), time.time())
    write("job-budget.json", record)
    remaining = record["soft_deadline_epoch"] - time.time()
    require(remaining > 0, "consumer budget expired before setup")
    command = ["timeout", "--signal=INT", "--kill-after=5s", f"{remaining:.3f}s", *COMMAND]
    result = subprocess.run(command, cwd=ROOT, start_new_session=True)
    write(
        "execution.json",
        {"exit_code": result.returncode, "finished_at_epoch": time.time(), "command": command},
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
