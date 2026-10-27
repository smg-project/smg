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


MAIN_PUSH_POLICY = "exact_d5f8_main_push_build_leg_all_17_steps_success_v1"
MAIN_PUSH_SHA = "d5f8f9e765dcdec1f56dd623b46e49e1e47862d4"
MAIN_PUSH_RUN = 40218847194
MAIN_PUSH_BUILD_JOB = 121246077837
MAIN_PUSH_EXPECTED_STEPS = EXPECTED_STEPS - {"Verify frozen build source"}
MAIN_PUSH_ARTIFACTS = {
    "smg-wheel": (
        12609313653,
        52600567,
        "ecce3937864f36c77ac5f08734bc73ef5795e5e8e23cd794fce8ea1cd30e2cb8",
        "2026-10-26T18:20:49Z",
    ),
    "wasm-test-fixtures": (
        12609142504,
        56596,
        "2169eab7ef97014996abb71a15eacfbd9217c77759c6fff16bafabfa5feb8ba3",
        "2026-10-26T18:20:59Z",
    ),
    "python-client-types": (
        12608937526,
        34700,
        "19cb852722d76789e6e69a2f7f53b11e4a618dbacb9646f34158c6498928f78e",
        "2026-10-26T18:20:57Z",
    ),
    "go-ffi-library": (
        12608627976,
        13845631,
        "30848e64a676f301ba428169ce96ede0106800d9114c5bca6db230fb31c6b876",
        "2026-10-26T18:20:53Z",
    ),
}


def select_build_job(jobs, inputs):
    if inputs.get("producer_policy") == MAIN_PUSH_POLICY:
        matches = [job for job in jobs if job["id"] == MAIN_PUSH_BUILD_JOB]
        require(len(matches) == 1, "exact main push build job missing or duplicated")
        return matches[0]
    require(inputs["producer_policy"] == POLICY, "unknown producer policy")
    require(len(jobs) == 1, "producer must contain only build-wheel")
    return jobs[0]


def verify_completed(run, jobs, inputs, artifacts, now):
    is_main_push = inputs.get("producer_policy") == MAIN_PUSH_POLICY
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
    for step in steps:
        require(
            step["status"] == "completed" and step["conclusion"] == "success",
            "producer step not successful: " + step["name"],
        )
    pins = inputs["artifacts"]
    require(
        inputs["artifacts_pinned"] is True
        and len(pins) == 4
        and (len(artifacts) >= 4 if is_main_push else len(artifacts) == 4)
        and {a["name"] for a in pins} == ARTIFACTS
        and len({a["id"] for a in pins}) == 4
        and len({a["id"] for a in artifacts}) == len(artifacts),
        "four actual artifact pins required",
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


# Delivery of existing d5 bytes is a separate, exact authenticated copy event.
# The unchanged MAIN_PUSH policy above still describes only the original build.
RENEWED_DELIVERY_POLICY = "exact_d5f8_origin17_copy40370788390_delivery_v1"
RENEWED_COPY_OWNER = {
    "run_id": 40370788390, "run_attempt": 1,
    "head_sha": "f4e9d7cc4fed43fb547fd9764d8cff2427c43f0d",
    "job_id": 121740660621, "branch": "ci/amd-qwen-artifact-renewal-20261026",
    "actor": "chunfangamd", "workflow": WORKFLOW, "runner_id": 1000073964,
}
RENEWED_COPY_ARCHIVES = {
    "smg-wheel": (12671394443, 52600567, "52eb291470e4ba37f40df62b603c8831407ffa9711e6b3b5c27f3d5dc2737b2c", "2026-11-02T17:40:50Z"),
    "go-ffi-library": (12671514143, 13845631, "1157f1b1d7b3d56830cdea0aa4cadddb7975cfc75ebe055690c278005e9d338e", "2026-11-02T17:40:54Z"),
    "python-client-types": (12671299803, 34700, "df79f4dfcf00c838434b0a5c940d29367987d62ecedc1030538f2f6268b294d0", "2026-11-02T17:40:57Z"),
    "wasm-test-fixtures": (12672711971, 56596, "2e4e738282f907d179e9fd9172955c17076d3accd7bc6cd8c66101e483427e8f", "2026-11-02T17:40:58Z"),
}
RENEWED_ORIGIN_EVIDENCE = (12671504010, 35942, "d98ee0714a465cf9e5555e5620c7388e8e076088e20efff194b6a882f2568f4d", "2026-11-02T17:41:00Z")
RENEWED_COPY_STEPS = [
    "Set up job", "Checkout exact isolated copy source",
    "Verify actual old producer and immutable payload bytes only",
    "Copy-upload original wheel payload", "Copy-upload original Go payloads",
    "Copy-upload original generated client types", "Copy-upload original WASM payloads",
    "Preserve origin verification evidence", "Post Checkout exact isolated copy source", "Complete job",
]
RENEWED_ORIGIN_VERIFIED_AT = "2026-10-26T17:40:50Z"
RENEWED_ORIGIN_VERIFICATION_CONSUMER19 = {
    ".cargo": "8a9d55bb8edfa19d9f90b313fbdc24326d3f667b",
    "Cargo.toml": "cb29df18a6eede1eed9f2d244367e804d20a2f14",
    "Cargo.lock": "7022fa76a1f1e4ef7aa812ee7f2d3ac159fd6aa4",
    "rust-toolchain.toml": "1b3e16ebff6f9fe991fc29854025255c597f785d",
    "Makefile": "0ae752629aa26b1b28cf23a0ad49fca5d4461845",
    "bindings": "a7570c34e19461de3579acc385541f7afc9de774",
    "clients": "599d7ce2727e5d6287825271f464a3b7b5e64ee9",
    "crates": "aa183216a4a450185252105434e4fe8f981ebd94",
    "model_gateway": "c9b0a8e790c4f71a760c41fd4e311f546db6bf48",
    "grpc_servicer": "7e23f8999c36292d7b0c8bfd11b29843352da2b0",
    "e2e_test": "492e701db279e840457d832af81f5e0377112109",
    "scripts": "2aaa0d62644a24b1580a7ef46ee6fd00499ff57b",
    ".github/actions": "a2c04716dbe54449fa894a9f7d4e519077efdbf4",
    ".github/workflows/e2e-gpu-job.yml": "453b36ebfd26262dc79c3f376dcb2469af6435a1",
    "README.md": "83b7df25632da03f0184bd5b4628b9c9a3ace803",
    "LICENSE": "9088c2872db9763b9e6533c6960e9adb0c9cb65f",
    "rustfmt.toml": "19e1ab31f8958c31292f0f568976a036af7e683b",
    "clippy.toml": "cfb9e85a5074e6b02e603df8da9653ee18dbcca3",
    "examples/wasm": "fc10fea21779371f9de247209af2028cf8173236"
}


def renewed_delivery_owner(inputs):
    delivery = inputs["artifact_delivery"]
    require(delivery["policy"] == RENEWED_DELIVERY_POLICY, "unknown artifact delivery policy")
    require(delivery["origin_verification_consumer19"] == RENEWED_ORIGIN_VERIFICATION_CONSUMER19,
            "exact historical copy verification source19")
    require(delivery["copy_owner"] == RENEWED_COPY_OWNER, "exact renewed-copy owner")
    require(inputs["producer_policy"] == MAIN_PUSH_POLICY, "copy origin must retain exact d5 policy")
    origin, current = delivery["origin_artifacts"], inputs["artifacts"]
    require(len(origin) == len(current) == 4, "copy needs exactly four original and delivered roles")
    def tuples(pins):
        require(len({p["name"] for p in pins}) == len(pins) and
                len({p["id"] for p in pins}) == len(pins), "duplicate copy role or identity")
        return {p["name"]: (p["id"], p["bytes"], p["sha256"], p["expires_at"]) for p in pins}
    require(tuples(origin) == MAIN_PUSH_ARTIFACTS, "copy original pins changed")
    require(tuples(current) == RENEWED_COPY_ARCHIVES, "exact renewed-copy archive pins")
    require({p["name"]: p["files"] for p in current} ==
            {p["name"]: p["files"] for p in origin}, "copy payload identities changed")
    evidence = delivery["origin_evidence"]
    require((evidence["id"], evidence["bytes"], evidence["sha256"], evidence["expires_at"]) ==
            RENEWED_ORIGIN_EVIDENCE and evidence["name"] == "d5-artifact-renewal-origin-evidence",
            "exact copy origin-evidence archive")
    require(len(evidence["files"]) == 14 and
            sum(p["bytes"] for p in evidence["files"].values()) <= 8 * 1024**2,
            "bounded copy origin evidence")
    require(delivery["copy_step_names"] == RENEWED_COPY_STEPS and
            delivery["origin_verified_step_completed_at"] == RENEWED_ORIGIN_VERIFIED_AT,
            "copy source-bound validation step identity")
    return {"run_id": RENEWED_COPY_OWNER["run_id"], "run_attempt": 1,
            "producer_sha": RENEWED_COPY_OWNER["head_sha"]}


def verify_renewed_delivery(run, jobs, inputs, copy_run, copy_jobs, copy_artifacts, origin_files, now):
    """Validate an exact copy and its historical origin, not a new build.

    The old policy is evaluated only at the real successful copy validation
    step's API timestamp, using the archived origin API records. Current time
    and current-copy expiration are checked separately below; no clock or
    original API expiry field is rewritten to make old ZIPs appear live.
    """
    owner = renewed_delivery_owner(inputs)
    delivery, expected = inputs["artifact_delivery"], RENEWED_COPY_OWNER
    require(copy_run["id"] == expected["run_id"] and
            copy_run["head_sha"] == expected["head_sha"] and
            copy_run["run_attempt"] == 1 and copy_run["head_branch"] == expected["branch"] and
            copy_run["event"] == "workflow_dispatch" and copy_run["path"] == WORKFLOW and
            copy_run["actor"]["login"] == expected["actor"] and
            copy_run["repository"]["full_name"] == copy_run["head_repository"]["full_name"] == "smg-project/smg" and
            copy_run["status"] == "completed" and copy_run["conclusion"] == "success",
            "actual successful exact copy run")
    require(len(copy_jobs) == 1, "copy must contain only its hosted upload job")
    job = copy_jobs[0]
    require(job["id"] == expected["job_id"] and job["run_id"] == copy_run["id"] and
            job["head_sha"] == copy_run["head_sha"] and job["run_attempt"] == 1 and
            job["head_branch"] == expected["branch"] and job["name"] == "renew-d5-artifacts-only" and
            job["status"] == "completed" and job["conclusion"] == "success" and
            job["runner_id"] == expected["runner_id"] and job["labels"] == ["ubuntu-latest"],
            "actual exact hosted copy job")
    steps = job["steps"]
    require([s["name"] for s in steps] == RENEWED_COPY_STEPS and
            all(s["status"] == "completed" and s["conclusion"] == "success" for s in steps),
            "all exact copy steps must succeed")
    verified_at = steps[2]["completed_at"]
    require(verified_at == RENEWED_ORIGIN_VERIFIED_AT, "actual copy validation timestamp changed")
    parse = lambda value: datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(parse(job["started_at"]) <= parse(verified_at) <= parse(job["completed_at"]) <= now,
            "copy validation/terminal time interval")
    pins = inputs["artifacts"] + [delivery["origin_evidence"]]
    require(len(copy_artifacts) == 5 and len({a["id"] for a in copy_artifacts}) == 5 and
            len({a["name"] for a in copy_artifacts}) == 5, "complete exact five copy artifacts")
    for pin in pins:
        matches = [a for a in copy_artifacts if a["id"] == pin["id"]]
        require(len(matches) == 1, "copy artifact absent or duplicated")
        meta = matches[0]
        require(meta["name"] == pin["name"] and meta["size_in_bytes"] == pin["bytes"] and
                meta["digest"] == "sha256:" + pin["sha256"] and meta["expired"] is False and
                meta["expires_at"] == pin["expires_at"] and parse(meta["expires_at"]) > now and
                meta["workflow_run"]["id"] == owner["run_id"] and
                meta["workflow_run"]["head_sha"] == owner["producer_sha"] and
                meta["workflow_run"]["head_branch"] == expected["branch"] and
                parse(job["started_at"]) <= parse(meta["created_at"]) <= parse(job["completed_at"]),
                "current copy metadata/owner/interval/expiry")
    require(set(origin_files) == set(delivery["origin_evidence"]["files"]), "copy proof file set")
    # JSON bytes were already checked by artifact_io.unpack against the pinned ZIP.
    import json
    data = {name: json.loads(value) for name, value in origin_files.items()}
    archived_run, archived_jobs = data["producer-run.json"], data["producer-attempt-jobs.json"]
    require(archived_run["id"] == run["id"] and archived_run["head_sha"] == run["head_sha"] and
            archived_run["status"] == run["status"] and archived_run["conclusion"] == run["conclusion"],
            "historical and current origin run identity/state")
    require(archived_jobs["total_count"] == len(archived_jobs["jobs"]) and
            archived_jobs["jobs"] == jobs and
            data["origin-attempt-pages/producer-attempt-jobs-page-1.json"] == archived_jobs,
            "complete historical original attempt jobs")
    origin_records = [data[f"origin-artifact-{p['id']}.json"] for p in delivery["origin_artifacts"]]
    original_page = data["origin-artifacts-page-1.json"]
    require(original_page["total_count"] == len(original_page["artifacts"]) and
            all(record in original_page["artifacts"] for record in origin_records),
            "complete historical origin artifact page")
    origin_inputs = {**inputs, "artifacts": delivery["origin_artifacts"]}
    context = verify_completed(run, jobs, origin_inputs, origin_records, parse(verified_at))
    require(context == data["origin-frozen-producer-context.json"], "original17 verification context differs")
    proof = data["origin-copy-proof.json"]
    original = proof["original_producer"]
    require(proof["passed"] is True and proof["stage"] == "all-original-payloads-verified-ready-for-copy-upload" and
            proof["admission_nonce"] == "0e0560c99481456189f8dd68cedeb26b" and
            proof["all_source_GPU_authorities_false"] is True and proof["build_executed"] is False and
            proof["tests_executed"] is False and proof["runtime_accepted"] is False and proof["coverage_increment"] == 0,
            "actual copy-only verification proof")
    require(original["run_id"] == MAIN_PUSH_RUN and original["job_id"] == MAIN_PUSH_BUILD_JOB and
            original["head_sha"] == MAIN_PUSH_SHA and original["attempt"] == 1 and
            original["all17_build_steps_success"] is True and
            original["parent_pipeline_status"] == "completed" and original["parent_pipeline_conclusion"] == "cancelled" and
            original["parent_pipeline_success_accepted"] is False and
            original["artifacts"] == delivery["origin_artifacts"] and original["native"] == inputs["gateway_native"],
            "original build/payload/native identity preserved")
    copied = proof["copy_job"]
    require(copied["run_id"] == owner["run_id"] and copied["run_attempt"] == 1 and
            copied["head_sha"] == owner["producer_sha"] and copied["job_id"] == job["id"] and
            copied["runner_id"] == expected["runner_id"] and copied["actor"] == expected["actor"] and
            copied["workflow"] == WORKFLOW and copied["only_upload_copy_job_not_producer_build"] is True,
            "origin proof exact copy owner")
    payloads = [{"role": p["name"], "name": name, **value}
                for p in delivery["origin_artifacts"] for name, value in p["files"].items()]
    require(proof["payloads"] == payloads and proof["source_maps"] == {
            "consumer19": RENEWED_ORIGIN_VERIFICATION_CONSUMER19, "compiled37": inputs["compiled_source_trees"],
            "producer14": inputs["producer_source_trees"]}, "copy six payloads/source19/37/14")
    archived_copy = data["copy-run-api.json"]
    archived_copy_jobs = data["copy-attempt-jobs-api.json"]
    require(archived_copy["id"] == owner["run_id"] and archived_copy["head_sha"] == owner["producer_sha"] and
            archived_copy["run_attempt"] == 1 and archived_copy["actor"]["login"] == expected["actor"] and
            archived_copy_jobs["total_count"] == len(archived_copy_jobs["jobs"]) == 1 and
            archived_copy_jobs["jobs"][0]["id"] == job["id"] and
            data["copy-attempt-pages/producer-attempt-jobs-page-1.json"] == archived_copy_jobs,
            "honest in-job copy API identity, separate from terminal API")
    return context, {
        "policy": RENEWED_DELIVERY_POLICY, "copy_owner": expected,
        "origin_verified_step_completed_at": verified_at,
        "origin_artifacts": delivery["origin_artifacts"], "four_artifacts": inputs["artifacts"],
        "origin_evidence": delivery["origin_evidence"], "origin_six_payloads_verified": True,
        "six_payloads_verified": False,
        "origin17_preserved": True, "new_build_executed": False, "GPU_credit": 0,
    }
