"""Explicit producer policies; archive bytes still need the relay's checks."""

import datetime
import re

from source import WORKFLOW, require

POLICY = "frozen_main19c_python173_build_only_all_18_steps_success_v1"
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
]
POST_STEPS = ["Post Set up Python", "Post Setup Rust", "Post Run actions/checkout@v7"]
EXPECTED_STEPS = {"Set up job", "Complete job", *EXPLICIT_STEPS, *POST_STEPS}
ARTIFACTS = {"smg-wheel", "go-ffi-library", "python-client-types", "wasm-test-fixtures"}


MAIN_PUSH_POLICY = "exact_9134_main_push_build_leg_all_17_steps_success_v1"
MAIN_PUSH_SHA = "9134d03216a53077668f373e2d9ed2546da57635"
MAIN_PUSH_RUN = 40795986051
MAIN_PUSH_BUILD_JOB = 123192466952
MAIN_PUSH_EXPECTED_STEPS = EXPECTED_STEPS - {"Verify frozen build source"}
MAIN_PUSH_ARTIFACTS = {'python-client-types': (12865295201, 34948, '425ad67ca0b09905cda43ce1087c9612ddb161033fe91b2155ae9a6f49be3bc6', '2026-10-30T01:02:00Z'), 'smg-wheel': (12864911107, 53281911, 'be379e40be5e978a0c94d07988b85e636d6a9d5f8394187d49c1b490f76bc5d7', '2026-10-30T01:01:29Z')}


F103_POLICY = "frozen_mainf103_complete_build_only_all_18_steps_success_v1"
F103_SOURCE_BASE = "f1038e99610647fa35a7e479e41f4dbc1821e481"
F103_PRODUCER_SHA = "69215ef26afdaf1bdf7b61b4dfd635db906e1eeb"
F103_RUN = 40312852744
F103_BUILD_JOB = 121538063467
F103_BRANCH = "ci/amd-main-f103-native-preparation-20261026"
F103_EXPECTED_STEPS = ["Set up job", *EXPLICIT_STEPS, *POST_STEPS, "Complete job"]


def select_build_job(jobs, inputs):
    if inputs.get("producer_policy") == MAIN_PUSH_POLICY:
        matches = [job for job in jobs if job["id"] == MAIN_PUSH_BUILD_JOB]
        require(len(matches) == 1, "exact main push build job missing or duplicated")
        return matches[0]
    require(inputs["producer_policy"] in (POLICY, F103_POLICY), "unknown producer policy")
    require(len(jobs) == 1, "producer must contain only build-wheel")
    return jobs[0]


def verify_completed(run, jobs, inputs, artifacts, now):
    is_main_push = inputs.get("producer_policy") == MAIN_PUSH_POLICY
    is_f103 = inputs.get("producer_policy") == F103_POLICY
    if is_main_push:
        sha = inputs["producer_sha"]
        require(
            sha == inputs["source_base"] == MAIN_PUSH_SHA
            and inputs["run_id"] == MAIN_PUSH_RUN
            and inputs["run_attempt"] == 1
            and inputs["build_job_id"] == MAIN_PUSH_BUILD_JOB
            and inputs["producer_branch"] == "main",
            "exact main push input identity",
        )
        require(
            run["id"] == MAIN_PUSH_RUN
            and run["head_sha"] == sha
            and run["run_attempt"] == 1
            and run["event"] == "push"
            and run["path"] == WORKFLOW
            and run["head_branch"] == "main",
            "main push producer run identity",
        )
        require(
            len(jobs) > 0
            and len({job["id"] for job in jobs}) == len(jobs)
            and all(
                job["run_id"] == run["id"] and job["run_attempt"] == 1 and job["head_sha"] == sha
                for job in jobs
            ),
            "main push attempt jobs identity",
        )
        require(
            {
                pin["name"]: (
                    pin["id"],
                    pin["bytes"],
                    pin["sha256"],
                    pin["expires_at"],
                )
                for pin in inputs["artifacts"]
            }
            == MAIN_PUSH_ARTIFACTS,
            "exact main push artifact pins",
        )
    else:
        require(inputs["producer_policy"] in (POLICY, F103_POLICY), "unknown producer policy")
        sha = inputs["producer_sha"]
        if is_f103:
            require(
                sha == F103_PRODUCER_SHA
                and inputs["source_base"] == F103_SOURCE_BASE
                and inputs["run_id"] == F103_RUN
                and inputs["run_attempt"] == 1
                and inputs["build_job_id"] == F103_BUILD_JOB
                and inputs["producer_branch"] == F103_BRANCH,
                "exact frozen f103 input identity",
            )
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
    job = select_build_job(jobs, inputs)
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
    expected_steps = MAIN_PUSH_EXPECTED_STEPS if is_main_push else EXPECTED_STEPS
    count = 17 if is_main_push else 18
    require(
        len(names) == count and len(set(names)) == count and set(names) == expected_steps,
        "17 steps" if is_main_push else "18 steps",
    )
    if is_f103:
        require(names == F103_EXPECTED_STEPS, "exact frozen f103 ordered 18 steps")
    for step in steps:
        require(
            step["status"] == "completed" and step["conclusion"] == "success",
            "producer step not successful: " + step["name"],
        )
    pins = inputs["artifacts"]
    required_artifacts = set(MAIN_PUSH_ARTIFACTS) if is_main_push else ARTIFACTS
    required_count = len(required_artifacts)
    require(
        inputs["artifacts_pinned"] is True
        and len(pins) == required_count
        and (len(artifacts) >= required_count if is_main_push else len(artifacts) == required_count)
        and {a["name"] for a in pins} == required_artifacts
        and len({a["id"] for a in pins}) == required_count
        and len({a["id"] for a in artifacts}) == len(artifacts),
        "complete policy-specific actual artifact pins required",
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
    context = {
        "producer_context_verified": True,
        "archive_bytes_verified": False,
        "source_base": inputs["source_base"],
        "actual_head": sha,
        "prior_cancelled_producer_accepted": False,
    }
    if is_main_push:
        context.update(
            producer_policy=MAIN_PUSH_POLICY,
            producer_build_leg_only=True,
            producer_parent_pipeline_accepted=False,
            producer_parent_status=run["status"],
            producer_parent_conclusion=run["conclusion"],
            producer_attempt_jobs_count=len(jobs),
            producer_parent_cancelled_observed=(
                run["status"] == "completed" and run["conclusion"] == "cancelled"
            ),
        )
    return context
