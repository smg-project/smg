"""Bind the EPD consumer and its new producer independently to compiled dbc4."""

import subprocess

from artifact_io import ROOT, git_output, require

WORKFLOW = ".github/workflows/pr-test-rust.yml"


def check_map(revision, expected):
    for path, oid in expected.items():
        require(git_output("rev-parse", f"{revision}:{path}").strip() == oid, "source: " + path)


def verify(inputs):
    base, producer = inputs["source_base"], inputs["producer_sha"]
    require(len(inputs["compiled_source_trees"]) == 37, "compiled scope")
    require(len(inputs["producer_source_trees"]) == 14, "actual producer scope")
    require(len(inputs["consumer_source_trees"]) == 19, "consumer scope")
    check_map("HEAD", inputs["consumer_source_trees"])
    for revision in ("HEAD", base, producer):
        check_map(revision, inputs["compiled_source_trees"])
    check_map(producer, inputs["producer_source_trees"])
    check_map(base, inputs["producer_base_source_trees"])
    for revision in ("HEAD", producer):
        subprocess.check_call(
            ["git", "-c", f"safe.directory={ROOT}", "merge-base", "--is-ancestor", base, revision],
            cwd=ROOT,
        )
    allowed = {
        WORKFLOW,
        ".github/workflows/e2e-gpu-job.yml",
        "e2e_test/infra/worker.py",
        "e2e_test/fixtures/setup_backend.py",
        "scripts/ci_repair_sglang_rocm_native.py",
    }
    for path in git_output("diff", "--name-only", base, "HEAD").splitlines():
        require(
            path in allowed or path.startswith(".github/amd-epd-consumer/"), "consumer: " + path
        )
    for path in git_output("diff", "--name-only", base, producer).splitlines():
        require(
            path == WORKFLOW or path.startswith(".github/amd-epd-dbc4-producer/"),
            "producer: " + path,
        )
    require(not git_output("status", "--porcelain", "--untracked-files=all"), "dirty source")
