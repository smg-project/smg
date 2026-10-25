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

    require(
        read_oid("HEAD", ".github/amd-gptoss-pilot") == inputs["pilot_namespace_oid"],
        "pilot source identity",
    )
    require(
        read_oid(target, ".github/amd-gptoss-pilot") == inputs["pilot_namespace_oid"],
        "pilot target identity",
    )
    evidence = ".github/amd-sg-zmq-evidence"
    require(
        read_oid("HEAD", evidence) == read_oid(target, evidence) == inputs["evidence_namespace_oid"],
        "passive original SG evidence identity",
    )
    check_map(read_oid, "HEAD", inputs["source_trees"])
    check_map(read_oid, target, inputs["source_trees"])
    check_map(read_oid, "HEAD", inputs["compiled_source_trees"])
    check_map(read_oid, producer, inputs["compiled_source_trees"])
    check_map(read_oid, producer, inputs["producer_source_trees"])
    # Only the dispatch caller and its relay may differ from the product tree.
    subprocess.check_call(
        ["git", "-c", f"safe.directory={ROOT}", "merge-base", "--is-ancestor", target, "HEAD"],
        cwd=ROOT,
    )
    for path in git_output("diff", "--name-only", target, "HEAD").splitlines():
        require(
            path.startswith(".github/amd-shared-product/")
            or path == ".github/workflows/pr-test-rust.yml",
            f"unexpected product modification: {path}",
        )
    require(not git_output("status", "--porcelain", "--untracked-files=all"), "dirty relay source")
