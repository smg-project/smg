"""Validate the fixed main19c Python173 build source before installing Rust or starting compilation."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAMESPACE = ".github/amd-python173-19c-producer/"
WORKFLOW = ".github/workflows/pr-test-rust.yml"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def git(*args):
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={ROOT}", *args], cwd=ROOT, text=True
    ).strip()


def check_map(read_oid, revision, expected):
    for path, oid in expected.items():
        require(read_oid(revision, path) == oid, f"source differs: {revision}:{path}")


def verify(inputs, environment, read=git):
    require(inputs.get("ready") is True, "source has not been admitted")
    require(inputs.get("run_budget_admitted") is True, "CPU budget has not been admitted")
    require(
        environment["GITHUB_EVENT_NAME"] == "workflow_dispatch"
        and environment["GITHUB_REPOSITORY"] == "smg-project/smg"
        and environment["GITHUB_REF"] == "refs/heads/" + inputs["producer_branch"]
        and environment["GITHUB_SHA"] == read("rev-parse", "HEAD"),
        "build context differs",
    )
    base = inputs["source_base"]
    require(
        base != environment["GITHUB_SHA"],
        "diagnostic head must not be labelled as the main19c Python173 source base",
    )
    read("merge-base", "--is-ancestor", base, "HEAD")
    template = ROOT / NAMESPACE / "workflow-template.yml"
    require(hashlib.sha256(template.read_bytes()).hexdigest() == inputs["workflow_template_sha256"], "CPU template bytes differ")
    require((ROOT / WORKFLOW).read_bytes() == template.read_bytes(), "registered workflow has not received the manual-only CPU overlay")
    compiled, producer = inputs["compiled_source_trees"], inputs["producer_source_trees"]
    consumer = inputs["consumer_source_trees"]
    require(len(compiled) == 37 and len(producer) == 14 and len(consumer) == 19, "source scope differs")
    expected = inputs["expected_admitted_producer_source_trees"]
    require(set(expected) == set(producer), "admitted producer source scope differs")
    require(
        {p: oid for p, oid in expected.items() if p != ".github/workflows"}
        == {p: oid for p, oid in producer.items() if p != ".github/workflows"},
        "only the registered workflow may differ in admitted producer14",
    )

    def oid(revision, path):
        return read("rev-parse", f"{revision}:{path}")

    check_map(oid, base, consumer)
    check_map(oid, "HEAD", consumer)
    check_map(oid, "HEAD", inputs["expected_admitted_producer_source_trees"])
    check_map(oid, base, producer)
    check_map(oid, base, compiled)
    check_map(oid, "HEAD", compiled)
    check_map(oid, "HEAD", {p: value for p, value in producer.items() if p != ".github/workflows"})
    changed = read("diff", "--name-only", base, "HEAD").splitlines()
    require(WORKFLOW in changed, "independent workflow is missing")
    for path in changed:
        require(path == WORKFLOW or path.startswith(NAMESPACE), "unexpected change: " + path)
    require(not read("status", "--porcelain", "--untracked-files=all"), "dirty build source")
    return {"actual_head": environment["GITHUB_SHA"], "base": base, "compiled_inputs": 37}


if __name__ == "__main__":
    inputs = json.loads(Path(__file__).with_name("inputs.json").read_text())
    print(json.dumps(verify(inputs, os.environ), sort_keys=True))
