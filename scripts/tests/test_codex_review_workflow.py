"""The Codex review workflow keeps the key in a read-only job and the write token in another."""

import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/codex-code-review.yml"
MODES = ("review", "reply", "verify")


@pytest.fixture(scope="module")
def workflow():
    return yaml.safe_load(WORKFLOW.read_text())


def test_triggers_cover_pushes_the_command_threads_and_the_other_reviewers_activity(workflow):
    triggers = workflow[True]  # PyYAML reads the `on` key as a boolean
    assert set(triggers) == {
        "pull_request",
        "issue_comment",
        "pull_request_review_comment",
        "pull_request_review",
    }
    assert triggers["pull_request"]["types"] == [
        "opened",
        "synchronize",
        "reopened",
        "ready_for_review",
    ]


def test_the_codex_job_reads_and_the_post_job_writes(workflow):
    jobs = workflow["jobs"]
    assert jobs["codex"]["permissions"] == {"contents": "read", "pull-requests": "read"}
    assert jobs["post"]["permissions"] == {"contents": "read", "pull-requests": "write"}
    assert workflow["permissions"] == {}


def test_codex_runs_read_only_on_the_cluster_runner_with_a_pinned_cli(workflow):
    codex = workflow["jobs"]["codex"]
    step = next(
        s for s in codex["steps"] if str(s.get("uses", "")).startswith("openai/codex-action@")
    )
    assert step["with"]["permission-profile"] == ":read-only"
    assert step["with"]["allow-bot-users"] == "claude[bot]"
    assert "k8s-runner-cpu" in codex["runs-on"]
    assert workflow["env"]["CODEX_VERSION"] == "0.167.0"
    assert workflow["env"]["CODEX_MODEL"] == "gpt-6-astra"


def test_the_key_comes_from_the_runner_pod_not_a_repository_secret(workflow):
    codex = workflow["jobs"]["codex"]
    export = next(
        s for s in codex["steps"] if s.get("uses") == "./.github/actions/export-pod-api-key"
    )
    assert export["with"] == {"variable": "OPENAI_API_KEY"}
    assert "secrets." not in WORKFLOW.read_text()


def test_the_post_job_runs_only_after_a_successful_review_and_checks_out_the_default_branch(
    workflow,
):
    post = workflow["jobs"]["post"]
    assert post["needs"] == "codex"
    assert "needs.codex.result == 'success'" in post["if"]
    checkout = next(
        s for s in post["steps"] if str(s.get("uses", "")).startswith("actions/checkout@")
    )
    assert checkout["with"]["ref"] == "${{ github.event.repository.default_branch }}"


@pytest.mark.parametrize("mode", MODES)
def test_every_mode_has_a_prompt_and_a_closed_schema(mode):
    assert (ROOT / f".github/codex/{mode}-prompt.md").read_text().strip()
    schema = json.loads((ROOT / f".github/codex/{mode}-schema.json").read_text())
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])
