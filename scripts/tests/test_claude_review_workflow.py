"""The Claude review workflow runs from main, admits writers only, and never holds the key."""

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/claude-code-review.yml"
KEY_JOBS = ("review", "respond")


@pytest.fixture(scope="module")
def workflow():
    return yaml.safe_load(WORKFLOW.read_text())


def names(job):
    return [s.get("name") or s.get("uses") for s in job["steps"]]


def test_every_trigger_runs_the_workflow_from_the_default_branch(workflow):
    triggers = workflow[True]
    assert set(triggers) == {"pull_request_target", "issue_comment", "pull_request_review_comment"}
    assert "pull_request" not in triggers


def test_forks_and_bot_comments_are_refused_before_any_job_starts(workflow):
    review = workflow["jobs"]["review"]["if"]
    assert "github.event.pull_request.head.repo.full_name == github.repository" in review
    assert "github.event.comment.user.type != 'Bot'" in review
    assert "github.event.comment.user.type != 'Bot'" in workflow["jobs"]["respond"]["if"]


@pytest.mark.parametrize("job_name", KEY_JOBS)
def test_the_gate_runs_before_the_key_and_claude_runs_after_sudo_is_gone(workflow, job_name):
    job = workflow["jobs"][job_name]
    assert job["environment"] == "ai-review"
    order = names(job)
    gate = order.index("Gate the run")
    proxy = order.index("Start the Anthropic key proxy")
    drop = order.index("Drop sudo for the rest of the job")
    claude = next(
        i for i, s in enumerate(job["steps"]) if str(s.get("uses", "")).startswith("anthropics/")
    )
    assert gate < proxy < drop < claude
    for index in range(gate + 1, len(job["steps"])):
        assert "steps.gate.outputs.allowed == 'true'" in str(job["steps"][index].get("if", "")), (
            order[index]
        )
    proxy_step = job["steps"][proxy]
    assert proxy_step["env"] == {"ANTHROPIC_API_KEY": "${{ secrets.ANTHROPIC_API_KEY }}"}
    assert job["steps"][claude]["with"]["anthropic_api_key"] == "${{ env.PROXY_PLACEHOLDER_KEY }}"


def test_the_key_appears_nowhere_but_the_proxy_steps():
    text = WORKFLOW.read_text()
    assert text.count("secrets.ANTHROPIC_API_KEY") == len(KEY_JOBS)
    assert "export-pod-api-key" not in text
    assert "env.ANTHROPIC_API_KEY" not in text


def test_every_job_runs_on_the_review_runner_set(workflow):
    for name, job in workflow["jobs"].items():
        assert "k8s-runner-cpu" in job["runs-on"], name
