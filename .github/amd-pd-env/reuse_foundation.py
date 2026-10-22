"""Reuse only the audited foundation artifacts, preserving their producer identity."""

import datetime
import hashlib
import io
import json
import os
import re
import stat
import subprocess
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
INPUTS = Path(__file__).with_name("pd-close-inputs.json")
API = "https://api.github.com/repos/smg-project/smg/"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def git_output(*args):
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={ROOT}", *args], cwd=ROOT, text=True
    )


def check_source(inputs):
    for path, oid in inputs["source_trees"].items():
        actual = git_output("rev-parse", f"HEAD:{path}").strip()
        require(actual == oid, f"source identity differs: {path}")
    dirty = git_output("status", "--porcelain", "--untracked-files=all")
    require(not dirty, "diagnostic checkout must be clean before reuse")


def check_metadata(meta, artifact, inputs):
    producer = meta["workflow_run"]
    require(meta["id"] == artifact["id"] and meta["name"] == artifact["name"], "artifact identity")
    require(not meta["expired"], "artifact expired")
    expires = datetime.datetime.fromisoformat(meta["expires_at"].replace("Z", "+00:00"))
    require(
        expires.tzinfo is not None and expires > datetime.datetime.now(datetime.UTC),
        "artifact expiry time",
    )
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
            "User-Agent": "smg-pd-close-diagnostic",
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


def consumer_budget():
    with api_get(
        f"actions/runs/{os.environ['GITHUB_RUN_ID']}/attempts/{os.environ['GITHUB_RUN_ATTEMPT']}/jobs"
    ) as response:
        consumer_jobs = json.load(response)["jobs"]
    own = [
        j
        for j in consumer_jobs
        if j["name"] == "pd-nixl-env-close"
        and j["runner_name"] == os.environ["RUNNER_NAME"]
        and j["status"] == "in_progress"
        and j["head_sha"] == os.environ["GITHUB_SHA"]
    ]
    require(len(own) == 1, "consumer job identity ambiguous")
    started = datetime.datetime.fromisoformat(own[0]["started_at"].replace("Z", "+00:00"))
    now = datetime.datetime.now(datetime.UTC).timestamp()
    require(started.tzinfo is not None and started.timestamp() <= now, "consumer start time")
    deadline = started.timestamp() + 10 * 60 - 15
    require(deadline > now, "diagnostic budget elapsed")
    return {
        "consumer_job": own[0]["id"],
        "consumer_started_at": own[0]["started_at"],
        "consumer_sha": os.environ["GITHUB_SHA"],
        "consumer_run": os.environ["GITHUB_RUN_ID"],
        "consumer_attempt": os.environ["GITHUB_RUN_ATTEMPT"],
        "diagnostic_deadline_epoch": deadline,
        "cleanup_reserve_minutes": 5,
        "kill_grace_seconds": 15,
    }


def load_budget():
    budget = json.loads(Path(os.environ["SMG_PD_ENV_BUDGET_FILE"]).read_text())
    require(
        all(
            budget[key] == os.environ[env]
            for key, env in (
                ("consumer_sha", "GITHUB_SHA"),
                ("consumer_run", "GITHUB_RUN_ID"),
                ("consumer_attempt", "GITHUB_RUN_ATTEMPT"),
            )
        ),
        "budget consumer identity",
    )
    started = datetime.datetime.fromisoformat(budget["consumer_started_at"].replace("Z", "+00:00"))
    require(
        budget["diagnostic_deadline_epoch"] == started.timestamp() + 10 * 60 - 15
        and budget["cleanup_reserve_minutes"] == 5
        and budget["kill_grace_seconds"] == 15,
        "budget policy",
    )
    require(
        budget["diagnostic_deadline_epoch"] > datetime.datetime.now(datetime.UTC).timestamp(),
        "diagnostic budget elapsed",
    )
    return budget


def verify_cpu(files):
    evidence = {}
    for name, expected_tests, expected_skips in (
        ("servicer-cpu.xml", 1532, 27),
        ("closure-cpu.xml", 13, 0),
    ):
        root = ET.fromstring(files[name])
        suites = [root] if root.tag == "testsuite" else list(root)
        counts = {
            k: sum(int(s.get(k, 0)) for s in suites)
            for k in ("tests", "failures", "errors", "skipped")
        }
        require(
            counts == dict(tests=expected_tests, failures=0, errors=0, skipped=expected_skips),
            "producer CPU JUnit counts",
        )
        cases = list(root.iter("testcase"))
        require(len(cases) == expected_tests, "producer CPU case count")
        require(
            sum(c.find("skipped") is not None for c in cases) == expected_skips,
            "producer CPU skip count",
        )
        require(
            not any(c.find("failure") is not None or c.find("error") is not None for c in cases),
            "producer CPU failure",
        )
        evidence[name] = counts
    return evidence


def verify_environment_job(inputs):
    environment = inputs["environment_prerequisite"]
    require(
        environment["artifacts_pinned"] and isinstance(environment["artifact"], dict),
        "target-built environment artifact is not pinned",
    )
    require(
        isinstance(environment.get("cairo_runtime_package"), str)
        and re.fullmatch(r"libcairo2=[0-9A-Za-z.+:~_-]+", environment["cairo_runtime_package"]),
        "Cairo runtime package not bound",
    )
    with api_get(f"actions/runs/{environment['run_id']}") as response:
        run = json.load(response)
    require(
        run["id"] == environment["run_id"]
        and run["run_attempt"] == environment["run_attempt"]
        and run["head_sha"] == environment["producer_sha"]
        and run["head_branch"] == environment["branch"]
        and run["repository"]["full_name"] == "smg-project/smg"
        and run["event"] == "workflow_dispatch"
        and run["path"] == ".github/workflows/pr-test-rust.yml"
        and run["status"] == "completed"
        and run["conclusion"] == "success",
        "independent vLLM environment run not successful",
    )
    with api_get(f"actions/jobs/{environment['job_id']}") as response:
        job = json.load(response)
    require(
        job["id"] == environment["job_id"]
        and job["run_id"] == environment["run_id"]
        and job["run_attempt"] == environment["run_attempt"]
        and job["head_sha"] == environment["producer_sha"]
        and job["name"] == environment["job_name"] == "vllm-repair"
        and job["status"] == "completed"
        and job["conclusion"] == "success",
        "independent vLLM environment job not successful",
    )
    return {
        "run_id": run["id"],
        "run_attempt": run["run_attempt"],
        "head_sha": run["head_sha"],
        "job_id": job["id"],
        "status": run["status"],
        "conclusion": run["conclusion"],
    }


def main():
    inputs = json.loads(INPUTS.read_text())
    budget = load_budget()
    require(inputs.get("ready") is True, "producer artifacts/admission not frozen: ready=false")
    require(
        inputs["producer_policy"]
        == "successful_foundation_build_CPU_and_independent_vllm_environment",
        "unknown reuse policy",
    )
    environment = verify_environment_job(inputs)
    require(
        {a["name"] for a in inputs["artifacts"]}
        == {"build-provenance", "smg-wheel", "python-client-types", "servicer-cpu-results"}
        and len(inputs["artifacts"]) == 4,
        "exactly four required foundation artifacts",
    )
    require(
        {a["name"]: a["destination"] for a in inputs["artifacts"]}
        == {
            "build-provenance": "diagnostic-results/producer",
            "smg-wheel": "wheel",
            "python-client-types": "clients/python/smg_client/types",
            "servicer-cpu-results": "diagnostic-results/producer-cpu",
        },
        "artifact destinations",
    )
    require(
        git_output("rev-parse", "HEAD").strip() == os.environ["GITHUB_SHA"], "consumer checkout SHA"
    )
    check_source(inputs)
    with api_get(f"actions/runs/{inputs['run_id']}") as response:
        run = json.load(response)
    require(
        run["id"] == inputs["run_id"]
        and run["head_sha"] == inputs["producer_sha"]
        and run["run_attempt"] == inputs["run_attempt"]
        and run["head_branch"] == inputs["producer_branch"],
        "run identity/attempt",
    )
    require(run["status"] == "completed", "producer run not terminal")
    require(
        run["repository"]["full_name"] == "smg-project/smg"
        and run["path"] == ".github/workflows/pr-test-rust.yml"
        and run["event"] == "workflow_dispatch",
        "producer repository/workflow/event",
    )
    with api_get(f"actions/jobs/{inputs['build_job_id']}") as response:
        job = json.load(response)
    require(
        job["id"] == inputs["build_job_id"]
        and job["run_id"] == inputs["run_id"]
        and job["run_attempt"] == inputs["run_attempt"]
        and job["head_sha"] == inputs["producer_sha"]
        and job["name"] == "build-wheel"
        and job["status"] == "completed"
        and job["conclusion"] == "success",
        "producer build job not successful",
    )
    verified = []
    producer_cpu = None
    provenance = None
    for artifact in sorted(inputs["artifacts"], key=lambda a: a["name"] != "build-provenance"):
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
        require(provenance is not None, "provenance must be verified first")
        if artifact["name"] == "servicer-cpu-results":
            producer_cpu = verify_cpu(files)
        for name, payload in files.items():
            if artifact["name"] in {"smg-wheel", "python-client-types"}:
                original_path = (
                    "bindings/python/dist/" + name
                    if artifact["name"] == "smg-wheel"
                    else "clients/python/smg_client/types/" + name
                )
                rows = [f for f in provenance["files"] if f["path"] == original_path]
                require(
                    len(rows) == 1
                    and rows[0]["bytes"] == len(payload)
                    and rows[0]["sha256"] == digest(payload),
                    "actual provenance payload",
                )
            dest = ROOT / artifact["destination"] / name
            require(not dest.is_symlink(), "destination symlink")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(payload)
        verified.append(
            {"id": artifact["id"], "name": artifact["name"], "sha256": artifact["sha256"]}
        )
    require(producer_cpu is not None, "foundation CPU evidence missing")
    out = ROOT / "diagnostic-results"
    out.mkdir(exist_ok=True)
    (out / "artifact-reuse.json").write_text(
        json.dumps(
            {
                "producer_run": inputs["run_id"],
                "producer_attempt": inputs["run_attempt"],
                "producer_run_status": run["status"],
                "producer_run_conclusion": run["conclusion"],
                "producer_CPU": producer_cpu,
                "environment_prerequisite": environment,
                "prior_GPU_results_transferred": False,
                "producer_sha": inputs["producer_sha"],
                "build_job": inputs["build_job_id"],
                "consumer_sha": os.environ["GITHUB_SHA"],
                "consumer_run": os.environ["GITHUB_RUN_ID"],
                "artifacts": verified,
                "producer_relabelled": False,
                **budget,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
