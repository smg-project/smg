"""Validate the fixed build source before installing Rust or starting compilation."""

import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAMESPACE = ".github/amd-epd-registration-producer/"
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
        "diagnostic head must not be labelled as the registration source base",
    )
    read("merge-base", "--is-ancestor", base, "HEAD")
    compiled, producer = inputs["compiled_source_trees"], inputs["producer_source_trees"]
    require(len(compiled) == 37 and len(producer) == 14, "source scope differs")

    def oid(revision, path):
        return read("rev-parse", f"{revision}:{path}")

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
