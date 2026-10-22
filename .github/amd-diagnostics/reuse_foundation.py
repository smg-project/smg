"""Reuse only the audited foundation artifacts, preserving their producer identity."""

import datetime
import hashlib
import io
import json
import os
import stat
import subprocess
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
INPUTS = Path(__file__).with_name("qwen-vision-inputs.json")
API = "https://api.github.com/repos/smg-project/smg/"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_source(inputs):
    for path, oid in inputs["source_trees"].items():
        actual = subprocess.check_output(
            ["git", "rev-parse", f"HEAD:{path}"], cwd=ROOT, text=True
        ).strip()
        require(actual == oid, f"source identity differs: {path}")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT, text=True
    )
    require(not dirty, "diagnostic checkout must be clean before reuse")


def check_metadata(meta, artifact, inputs):
    producer = meta["workflow_run"]
    require(meta["id"] == artifact["id"] and meta["name"] == artifact["name"], "artifact identity")
    require(not meta["expired"], "artifact expired")
    require(meta["size_in_bytes"] == artifact["bytes"], "artifact archive size")
    require(meta["digest"] == "sha256:" + artifact["sha256"], "artifact archive digest")
    require(producer["id"] == inputs["run_id"], "producer run")
    require(producer["head_sha"] == inputs["producer_sha"], "producer head")


def unpack(data, artifact):
    require(
        len(data) == artifact["bytes"] and digest(data) == artifact["sha256"], "archive bytes/hash"
    )
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = archive.infolist()
        require(len(members) == len(artifact["files"]), "unexpected archive members")
        actual = {}
        for info in members:
            path = PurePosixPath(info.filename)
            mode = info.external_attr >> 16
            require(
                not path.is_absolute()
                and ".." not in path.parts
                and "\\" not in info.filename
                and not info.is_dir()
                and not stat.S_ISLNK(mode)
                and (stat.S_IFMT(mode) in (0, stat.S_IFREG))
                and not info.flag_bits & 1,
                "unsafe archive member",
            )
            require(
                info.filename not in actual and info.filename in artifact["files"],
                "unknown/duplicate file",
            )
            expected = artifact["files"][info.filename]
            require(info.file_size == expected["bytes"], "payload size")
            payload = archive.read(info)
            require(digest(payload) == expected["sha256"], "payload digest")
            actual[info.filename] = payload
        return actual


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def api_get(path):
    request = urllib.request.Request(
        API + path,
        headers={
            "Authorization": "Bearer " + os.environ["GITHUB_TOKEN"],
            "Accept": "application/vnd.github+json",
            "User-Agent": "smg-qwen-diagnostic",
        },
    )
    return urllib.request.build_opener(NoRedirect()).open(request, timeout=30)


def archive_bytes(artifact):
    try:
        with api_get(f"actions/artifacts/{artifact['id']}/zip") as response:
            return response.read(artifact["bytes"] + 1)
    except urllib.error.HTTPError as exc:
        if exc.code != 302:
            raise
        location = exc.headers["Location"]
    url = urllib.parse.urlsplit(location)
    require(
        url.scheme == "https" and url.hostname and not url.username and not url.password,
        "archive redirect",
    )
    # The GitHub bearer token is never forwarded to the signed blob URL.
    with urllib.request.urlopen(location, timeout=60) as response:
        return response.read(artifact["bytes"] + 1)


def main():
    inputs = json.loads(INPUTS.read_text())
    check_source(inputs)
    with api_get(f"actions/runs/{inputs['run_id']}") as response:
        run = json.load(response)
    require(
        run["head_sha"] == inputs["producer_sha"] and run["run_attempt"] == inputs["run_attempt"],
        "run identity/attempt",
    )
    require(run["status"] == "completed", "producer run not terminal")
    with api_get(f"actions/jobs/{inputs['build_job_id']}") as response:
        job = json.load(response)
    require(
        job["run_id"] == inputs["run_id"]
        and job["run_attempt"] == inputs["run_attempt"]
        and job["head_sha"] == inputs["producer_sha"]
        and job["name"] == "build-wheel"
        and job["status"] == "completed"
        and job["conclusion"] == "success",
        "producer build job not successful",
    )
    with api_get(
        f"actions/runs/{os.environ['GITHUB_RUN_ID']}/attempts/{os.environ['GITHUB_RUN_ATTEMPT']}/jobs"
    ) as response:
        consumer_jobs = json.load(response)["jobs"]
    own = [
        j
        for j in consumer_jobs
        if j["name"] == "qwen-vision-diagnostic"
        and j["runner_name"] == os.environ["RUNNER_NAME"]
        and j["status"] == "in_progress"
        and j["head_sha"] == os.environ["GITHUB_SHA"]
    ]
    require(len(own) == 1, "consumer job identity ambiguous")
    started = datetime.datetime.fromisoformat(own[0]["started_at"].replace("Z", "+00:00"))
    deadline = int(started.timestamp()) + 15 * 60 - 15
    require(
        deadline > int(datetime.datetime.now(datetime.UTC).timestamp()), "diagnostic budget elapsed"
    )
    with open(os.environ["GITHUB_ENV"], "a") as env:
        env.write(f"QWEN_DIAG_DEADLINE={deadline}\n")
    verified = []
    for artifact in inputs["artifacts"]:
        with api_get(f"actions/artifacts/{artifact['id']}") as response:
            check_metadata(json.load(response), artifact, inputs)
        files = unpack(archive_bytes(artifact), artifact)
        if artifact["name"] == "build-provenance":
            provenance = json.loads(files["build-provenance.json"])
            require(
                provenance["source_sha"] == inputs["producer_sha"]
                and provenance["run_id"] == str(inputs["run_id"])
                and provenance["run_attempt"] == str(inputs["run_attempt"]),
                "provenance producer",
            )
            require(
                all(
                    provenance["source_trees"].get(p) == oid
                    for p, oid in inputs["provenance_trees"].items()
                ),
                "provenance trees",
            )
        for name, payload in files.items():
            dest = ROOT / artifact["destination"] / name
            require(not dest.is_symlink(), "destination symlink")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(payload)
        verified.append(
            {"id": artifact["id"], "name": artifact["name"], "sha256": artifact["sha256"]}
        )
    out = ROOT / "diagnostic-results"
    out.mkdir(exist_ok=True)
    (out / "artifact-reuse.json").write_text(
        json.dumps(
            {
                "producer_run": inputs["run_id"],
                "producer_attempt": inputs["run_attempt"],
                "producer_sha": inputs["producer_sha"],
                "build_job": inputs["build_job_id"],
                "consumer_sha": os.environ["GITHUB_SHA"],
                "consumer_run": os.environ["GITHUB_RUN_ID"],
                "artifacts": verified,
                "producer_relabelled": False,
                "consumer_job": own[0]["id"],
                "consumer_started_at": own[0]["started_at"],
                "diagnostic_deadline_epoch": deadline,
                "cleanup_reserve_minutes": 5,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
