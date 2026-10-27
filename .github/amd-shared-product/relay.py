"""Verify external c9d artifacts on a hosted CPU before the real reusable GPU job."""

import argparse
import datetime
import json
import os
from pathlib import Path

import artifact_io as io
import frozen_producer
import source
import wheel_record

NAMES = {"smg-wheel", "python-client-types", "wasm-test-fixtures", "go-ffi-library"}


def admitted(inputs):
    io.require(
        all(
            inputs.get(key) is True for key in ("ready", "run_budget_admitted", "artifacts_pinned")
        ),
        "source, runner budget, and actual artifact pins must all be admitted",
    )
    io.require(
        inputs.get("runtime_cohort_accepted") is True,
        "actual normal-install/wholepip compatible engine cohort is unresolved",
    )
    artifacts = inputs["artifacts"]
    io.require(
        len(artifacts) == 4 and {a["name"] for a in artifacts} == NAMES,
        "four artifacts",
    )
    io.require(len({a["id"] for a in artifacts}) == 4, "duplicate artifact identity")
    io.require(isinstance(inputs["gateway_native"], dict), "actual native identity not pinned")


def metadata(meta, artifact, inputs):
    io.check_metadata(meta, artifact, inputs)
    io.require(meta["expires_at"] == artifact["expires_at"], "artifact expiry changed")
    expiry = datetime.datetime.fromisoformat(meta["expires_at"].replace("Z", "+00:00"))
    io.require(expiry > datetime.datetime.now(datetime.UTC), "artifact expiry elapsed")
    io.require(0 < artifact["bytes"] <= 256 * 1024**2, "archive size cap")


def get(path):
    with io.api_get(path) as response:
        return json.load(response)


def get_attempt_jobs(inputs, evidence):
    jobs, total = [], None
    for page in range(1, 11):
        batch = get(
            f"actions/runs/{inputs['run_id']}/attempts/{inputs['run_attempt']}"
            f"/jobs?per_page=100&page={page}"
        )
        (evidence / f"producer-attempt-jobs-page-{page}.json").write_text(
            json.dumps(batch, indent=2) + "\n"
        )
        count, items = batch["total_count"], batch["jobs"]
        io.require(type(count) is int and 0 <= count <= 1000, "attempt jobs total count")
        if total is None:
            total = count
        io.require(
            count == total and isinstance(items, list) and len(items) <= 100,
            "attempt jobs page metadata",
        )
        jobs.extend(items)
        io.require(
            len(jobs) <= total and len({job["id"] for job in jobs}) == len(jobs),
            "attempt jobs repeated or extra",
        )
        if len(jobs) == total:
            return {"total_count": total, "jobs": jobs}
        io.require(len(items) == 100, "attempt jobs page incomplete")
    raise ValueError("attempt jobs pagination cap")


def get_copy_artifacts(owner, evidence):
    records, total = [], None
    for page in range(1, 11):
        batch = get(f"actions/runs/{owner['run_id']}/artifacts?per_page=100&page={page}")
        (evidence / f"copy-artifacts-page-{page}.json").write_text(json.dumps(batch, indent=2) + "\n")
        count, items = batch["total_count"], batch["artifacts"]
        io.require(type(count) is int and 0 <= count <= 1000 and isinstance(items, list) and len(items) <= 100,
                   "copy artifacts page metadata")
        if total is None:
            total = count
        io.require(count == total, "copy artifacts total changed")
        records.extend(items)
        io.require(len(records) <= total and len({a["id"] for a in records}) == len(records),
                   "copy artifacts repeated or extra")
        if len(records) == total:
            return records
        io.require(len(items) == 100, "copy artifacts incomplete page")
    raise ValueError("copy artifact pagination cap")


def main(inputs, output):
    admitted(inputs)  # Fail before any network request, even for an incomplete candidate.
    io.require(
        io.git_output("rev-parse", "HEAD").strip() == os.environ["GITHUB_SHA"],
        "checkout SHA",
    )
    source.verify(inputs)
    output.mkdir(parents=True, exist_ok=False)
    evidence = output / "evidence"
    evidence.mkdir()
    run = get(f"actions/runs/{inputs['run_id']}")
    attempt_jobs = get_attempt_jobs(inputs, evidence)
    jobs = attempt_jobs["jobs"]
    io.require(attempt_jobs["total_count"] == len(jobs), "complete attempt jobs")
    job = frozen_producer.select_build_job(jobs, inputs)
    (evidence / "producer-attempt-jobs.json").write_text(json.dumps(attempt_jobs, indent=2) + "\n")
    for name, value in (("producer-run", run), ("producer-build-job", job)):
        (evidence / f"{name}.json").write_text(json.dumps(value, indent=2) + "\n")
    metadata_owner, artifact_job, delivery_context = inputs, job, None
    if "artifact_delivery" in inputs:
        metadata_owner = frozen_producer.renewed_delivery_owner(inputs)
        copy_evidence = evidence / "copy-api"
        copy_evidence.mkdir()
        copy_run = get(f"actions/runs/{metadata_owner['run_id']}")
        copy_attempt = get_attempt_jobs(metadata_owner, copy_evidence)
        records = get_copy_artifacts(metadata_owner, copy_evidence)
        io.require(copy_attempt["total_count"] == len(copy_attempt["jobs"]) == 1, "complete copy job")
        artifact_job = copy_attempt["jobs"][0]
        origin_pin = inputs["artifact_delivery"]["origin_evidence"]
        origin_meta = get(f"actions/artifacts/{origin_pin['id']}")
        metadata(origin_meta, origin_pin, metadata_owner)
        io.require(origin_meta in records, "origin evidence metadata differs from complete copy page")
        origin_files = io.unpack(io.archive_bytes(origin_pin), origin_pin)
        origin_dir = evidence / "copy-origin"
        origin_dir.mkdir()
        for name, payload in origin_files.items():
            path = origin_dir / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(payload)
        context, delivery_context = frozen_producer.verify_renewed_delivery(
            run, jobs, inputs, copy_run, copy_attempt["jobs"], records, origin_files,
            datetime.datetime.now(datetime.UTC)
        )
        for name, value in (("copy-run", copy_run), ("copy-job", artifact_job)):
            (evidence / f"{name}.json").write_text(json.dumps(value, indent=2) + "\n")
    else:
        records = []
        for page in range(1, 11):
            batch = get(f"actions/runs/{inputs['run_id']}/artifacts?per_page=100&page={page}")[
                "artifacts"
            ]
            records.extend(batch)
            if len(batch) < 100:
                break
        else:
            raise ValueError("artifact pagination cap")
        context = frozen_producer.verify_completed(
            run, jobs, inputs, records, datetime.datetime.now(datetime.UTC)
        )
    (evidence / "frozen-producer-context.json").write_text(json.dumps(context, indent=2) + "\n")
    origin_records = [json.loads(origin_files[f"origin-artifact-{p['id']}.json"])
                      for p in inputs["artifact_delivery"]["origin_artifacts"]] if delivery_context else records
    (evidence / "producer-artifacts.json").write_text(json.dumps(origin_records, indent=2) + "\n")
    if delivery_context:
        (evidence / "copy-artifacts.json").write_text(json.dumps(records, indent=2) + "\n")
    verified = []
    for artifact in inputs["artifacts"]:
        meta = get(f"actions/artifacts/{artifact['id']}")
        metadata(meta, artifact, metadata_owner)
        io.require(
            artifact_job["started_at"] <= meta["created_at"] <= artifact_job["completed_at"],
            "artifact was not created during the selected successful build or copy attempt",
        )
        (evidence / f"artifact-{artifact['id']}.json").write_text(json.dumps(meta, indent=2) + "\n")
        files = io.unpack(io.archive_bytes(artifact), artifact)
        destination = output / artifact["name"]
        destination.mkdir()
        for name, payload in files.items():
            path = destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(payload)
            if name.endswith(".whl"):
                record = wheel_record.verify(payload, inputs["gateway_native"])
                (evidence / "wheel-RECORD.json").write_text(json.dumps(record, indent=2) + "\n")
        verified.append(
            {"id": artifact["id"], "name": artifact["name"], "files": artifact["files"]}
        )
    io.require((evidence / "wheel-RECORD.json").is_file(), "wheel RECORD proof missing")
    if delivery_context:
        delivery_context["six_payloads_verified"] = True
        (evidence / "copy-delivery-context.json").write_text(json.dumps(delivery_context, indent=2) + "\n")
    (evidence / "relay.json").write_text(
        json.dumps(
            {
                "external_producer": {
                    "run": run["id"],
                    "attempt": run["run_attempt"],
                    "head": run["head_sha"],
                    "build_job": job["id"],
                    "run_status": run["status"],
                    "run_conclusion": run["conclusion"],
                    "policy": inputs["producer_policy"],
                    "source_base": inputs["source_base"],
                    "source_maps_verified": True,
                },
                "artifact_delivery": delivery_context,
                "relay_consumer": {
                    "run": os.environ["GITHUB_RUN_ID"],
                    "attempt": os.environ["GITHUB_RUN_ATTEMPT"],
                    "head": os.environ["GITHUB_SHA"],
                },
                "payloads_republished_without_modification": verified,
                "source_trees": inputs["source_trees"],
                "compiled_source_trees": inputs["compiled_source_trees"],
                "producer_source_trees": inputs["producer_source_trees"],
                "producer_relabelled": False,
                "prior_GPU_results_transferred": False,
                "product_installs_editable_current_Python_grpc_packages": True,
                "native_or_wheel_installed_by_relay": False,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(json.loads(Path(__file__).with_name("inputs.json").read_text()), args.out)
