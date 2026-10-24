"""Verify external c9d artifacts on a hosted CPU before the real reusable GPU job."""

import argparse
import datetime
import json
import os
from pathlib import Path

import artifact_io as io
import frozen_producer
from recipe import auxiliary_staging
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
    artifacts = inputs["artifacts"]
    io.require(len(artifacts) == 4 and {a["name"] for a in artifacts} == NAMES, "four artifacts")
    io.require(len({a["id"] for a in artifacts}) == 4, "duplicate artifact identity")
    io.require(isinstance(inputs["gateway_native"], dict), "actual native identity not pinned")
    io.require(inputs["private_mooncake_runtime"]["staging_accepted"] is True, "private Mooncake staging")
    auxiliary_staging(inputs)


def metadata(meta, artifact, inputs):
    io.check_metadata(meta, artifact, inputs)
    io.require(meta["expires_at"] == artifact["expires_at"], "artifact expiry changed")
    expiry = datetime.datetime.fromisoformat(meta["expires_at"].replace("Z", "+00:00"))
    io.require(expiry > datetime.datetime.now(datetime.UTC), "artifact expiry elapsed")
    io.require(0 < artifact["bytes"] <= 256 * 1024**2, "archive size cap")


def get(path):
    with io.api_get(path) as response:
        return json.load(response)


def main(inputs, output):
    admitted(inputs)  # Fail before any network request, even for an incomplete candidate.
    io.require(
        io.git_output("rev-parse", "HEAD").strip() == os.environ["GITHUB_SHA"], "checkout SHA"
    )
    source.verify(inputs)
    output.mkdir(parents=True, exist_ok=False)
    evidence = output / "evidence"
    evidence.mkdir()
    run = get(f"actions/runs/{inputs['run_id']}")
    attempt_jobs = get(
        f"actions/runs/{inputs['run_id']}/attempts/{inputs['run_attempt']}/jobs?per_page=100"
    )
    jobs = attempt_jobs["jobs"]
    io.require(attempt_jobs["total_count"] == len(jobs) == 1, "single build-only attempt")
    job = jobs[0]
    (evidence / "producer-attempt-jobs.json").write_text(json.dumps(attempt_jobs, indent=2) + "\n")
    for name, value in (("producer-run", run), ("producer-build-job", job)):
        (evidence / f"{name}.json").write_text(json.dumps(value, indent=2) + "\n")
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
    (evidence / "producer-artifacts.json").write_text(json.dumps(records, indent=2) + "\n")
    verified = []
    for artifact in inputs["artifacts"]:
        meta = get(f"actions/artifacts/{artifact['id']}")
        metadata(meta, artifact, inputs)
        io.require(
            job["started_at"] <= meta["created_at"] <= job["completed_at"],
            "artifact was not created during this successful build attempt",
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
                "product_python_grpc": {
                    "proto": "separate exact 8451 source39dea wheel; eight current schemas verified",
                    "servicer": "normal editable install from the exact consumer checkout",
                    "auxiliary_inputs": inputs["runtime_auxiliary_inputs"],
                    "auxiliary_is_a_fifth_445_archive": False,
                },
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
