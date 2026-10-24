"""Bind the product caller, actual producer, and compiled inputs independently."""

import subprocess

from artifact_io import ROOT, git_output, require

WORKFLOW = ".github/workflows/pr-test-rust.yml"


def check_map(read_oid, revision, expected):
    for path, oid in expected.items():
        require(read_oid(revision, path) == oid, f"source identity differs: {revision}:{path}")


def verify(inputs):
    target, producer = inputs["target_product_commit"], inputs["producer_sha"]
    require(len(inputs["source_trees"]) == 19, "consumer source scope")
    require(len(inputs["compiled_source_trees"]) == 37, "compiled source scope")
    require(len(inputs["producer_source_trees"]) == 14, "producer source scope")

    def read_oid(revision, path):
        return git_output("rev-parse", f"{revision}:{path}").strip()

    check_map(read_oid, "HEAD", inputs["source_trees"])
    check_map(read_oid, target, inputs["source_trees"])
    check_map(read_oid, "HEAD", inputs["compiled_source_trees"])
    check_map(read_oid, producer, inputs["compiled_source_trees"])
    check_map(read_oid, producer, inputs["producer_source_trees"])
    # Only the caller, relay and explicit diagnostic observers may differ from the product tree.
    subprocess.check_call(
        ["git", "-c", f"safe.directory={ROOT}", "merge-base", "--is-ancestor", target, "HEAD"],
        cwd=ROOT,
    )
    for path in git_output("diff", "--name-only", target, "HEAD").splitlines():
        require(
            path.startswith(".github/amd-shared-product/")
            or path.startswith(".github/amd-p4-proof/")
            or path.startswith(".github/amd-nixl-environment/")
            or path == ".github/workflows/pr-test-rust.yml",
            f"unexpected product modification: {path}",
        )
    require(not git_output("status", "--porcelain", "--untracked-files=all"), "dirty relay source")
