"""Bind the product caller, actual producer, and compiled inputs independently."""

import hashlib
import json
import subprocess

from artifact_io import ROOT, git_output, require

WORKFLOW = ".github/workflows/pr-test-rust.yml"
TEST_OVERLAY_FILES = {
    "e2e_test/infra/pd_logs.py",
    "e2e_test/infra/test_pd_logs.py",
    "e2e_test/router/test_pd_mooncake.py",
}


def product_test_source(inputs):
    if "test_overlay_sha256" not in inputs:
        return inputs["source_trees"], set()
    path = ROOT / ".github/amd-shared-product/test-overlay.json"
    require(
        hashlib.sha256(path.read_bytes()).hexdigest() == inputs["test_overlay_sha256"],
        "overlay manifest",
    )
    overlay = json.loads(path.read_text())
    require(overlay["policy"] == "mooncake_owned_logs_exact3_v1", "overlay policy")
    require(set(overlay["files"]) == TEST_OVERLAY_FILES, "exact three test overlays")
    target = inputs["target_product_commit"]
    require(overlay["base_product_commit"] == target, "overlay product base")
    changed = set(git_output("diff", "--name-only", target, "HEAD", "--", "e2e_test").splitlines())
    require(changed == TEST_OVERLAY_FILES, "unlisted e2e changes")
    for name, item in overlay["files"].items():
        before = git_output("ls-tree", target, "--", name).strip()
        require(before == item["base_ls_tree"], f"overlay base: {name}")
        require(
            hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == item["sha256"],
            f"overlay bytes: {name}",
        )
    return inputs["source_trees"] | {"e2e_test": overlay["base_e2e_tree"]}, TEST_OVERLAY_FILES


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
    product_source, test_overlay = product_test_source(inputs)
    check_map(read_oid, target, product_source)
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
            or path == ".github/workflows/pr-test-rust.yml"
            or path in test_overlay,
            # Only the three hash-bound test files may differ within e2e_test.
            f"unexpected product modification: {path}",
        )
    require(not git_output("status", "--porcelain", "--untracked-files=all"), "dirty relay source")
