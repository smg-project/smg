"""New dispatch-only producer context; archive bytes still need the relay's checks."""

import datetime
import re

from source import WORKFLOW, require

POLICY = "frozen_epd_registration_dispatch_22_steps_six_archives_v1"
EXPLICIT_STEPS = [
    "Run actions/checkout@v7",
    "Verify frozen build source",
    "Setup Rust",
    "Build Python wheel and Go FFI library",
    "Generate Python client types",
    "Build WASM test fixtures",
    "Upload wheel artifact",
    "Upload Go FFI library artifact",
    "Upload Python client types",
    "Upload WASM test fixtures",
    "Show sccache stats",
    "Set up Python",
    "Test wheel install",
    "Build EPD RPC source wheels",
    "Verify EPD RPC wheels on CPU",
    "Upload EPD proto wheel",
    "Upload EPD servicer wheel",
]
POST_STEPS = ["Post Set up Python", "Post Setup Rust", "Post Run actions/checkout@v7"]
EXPECTED_STEPS = {"Set up job", "Complete job", *EXPLICIT_STEPS, *POST_STEPS}
ARTIFACTS = {
    "smg-wheel",
    "go-ffi-library",
    "python-client-types",
    "wasm-test-fixtures",
    "epd-grpc-proto-wheel",
    "epd-grpc-servicer-wheel",
}


def verify_completed(run, jobs, inputs, artifacts, now):
    require(inputs["producer_policy"] == POLICY, "unknown producer policy")
    sha = inputs["producer_sha"]
    require(
        isinstance(sha, str)
        and re.fullmatch(r"[0-9a-f]{40}", sha)
        and sha != inputs["source_base"],
        "actual diagnostic head is not pinned",
    )
    require(
        run["id"] == inputs["run_id"]
        and run["head_sha"] == sha
        and run["run_attempt"] == inputs["run_attempt"]
        and run["event"] == "workflow_dispatch"
        and run["path"] == WORKFLOW
        and run["head_branch"] == inputs["producer_branch"]
        and run["status"] == "completed"
        and run["conclusion"] == "success",
        "producer run identity or state",
    )
    require(len(jobs) == 1, "producer must contain only build-wheel")
    job = jobs[0]
    require(
        job["id"] == inputs["build_job_id"]
        and job["run_id"] == run["id"]
        and job["run_attempt"] == run["run_attempt"]
        and job["head_sha"] == sha
        and job["name"] == "build-wheel"
        and job["status"] == "completed"
        and job["conclusion"] == "success",
        "producer job identity or state",
    )
    steps = job["steps"]
    names = [step["name"] for step in steps]
    require(len(names) == 22 and len(set(names)) == 22 and set(names) == EXPECTED_STEPS, "22 steps")
    for step in steps:
        require(
            step["status"] == "completed" and step["conclusion"] == "success",
            "producer step not successful: " + step["name"],
        )
    pins = inputs["artifacts"]
    require(
        inputs["artifacts_pinned"] is True
        and len(pins) == len(artifacts) == 6
        and {a["name"] for a in pins} == ARTIFACTS
        and len({a["id"] for a in pins}) == len({a["id"] for a in artifacts}) == 6,
        "six actual artifact pins required",
    )
    for pin in pins:
        matches = [a for a in artifacts if a["name"] == pin["name"]]
        require(len(matches) == 1, "artifact missing or duplicated")
        actual = matches[0]
        require(
            actual["id"] == pin["id"]
            and actual["size_in_bytes"] == pin["bytes"] > 0
            and re.fullmatch(r"[0-9a-f]{64}", pin["sha256"])
            and actual["digest"] == "sha256:" + pin["sha256"]
            and actual["expired"] is False
            and actual["expires_at"] == pin["expires_at"]
            and actual["workflow_run"]["id"] == run["id"]
            and actual["workflow_run"]["head_sha"] == sha
            and job["started_at"] <= actual["created_at"] <= job["completed_at"],
            "artifact metadata identity",
        )
        expiry = datetime.datetime.fromisoformat(actual["expires_at"].replace("Z", "+00:00"))
        require(expiry > now, "artifact expired")
    return {
        "producer_context_verified": True,
        "archive_bytes_verified": False,
        "source_base": inputs["source_base"],
        "actual_head": sha,
        "prior_cancelled_producer_accepted": False,
    }
