"""The Claude review workflow admits writers only, never holds the key, and fails loudly."""

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/claude-code-review.yml"
JOBS = ("review", "respond")
SIDECAR = "http://127.0.0.1:8787"


@pytest.fixture(scope="module")
def workflow():
    return yaml.safe_load(WORKFLOW.read_text())


def test_the_triggers_are_the_pull_request_event_and_the_comment_events(workflow):
    triggers = workflow[True]
    assert set(triggers) == {"pull_request", "issue_comment", "pull_request_review_comment"}
    # pull_request, not pull_request_target: the action's App-token exchange rejects the latter's OIDC token
    assert "pull_request_target" not in triggers


def test_a_failing_claude_action_fails_the_job(workflow):
    review = workflow["jobs"]["review"]
    claude = next(s for s in review["steps"] if str(s.get("uses", "")).startswith("anthropics/"))
    assert "continue-on-error" not in claude


def test_forks_and_bot_comments_are_refused_before_any_job_starts(workflow):
    review = workflow["jobs"]["review"]["if"]
    assert "github.event.pull_request.head.repo.full_name == github.repository" in review
    assert "github.event.comment.user.type != 'Bot'" in review
    assert "github.event.comment.user.type != 'Bot'" in workflow["jobs"]["respond"]["if"]


@pytest.mark.parametrize("job_name", JOBS)
def test_the_gate_runs_first_and_everything_after_it_is_conditional(workflow, job_name):
    job = workflow["jobs"][job_name]
    names = [s.get("name") or s.get("uses") for s in job["steps"]]
    gate = names.index("Gate the run")
    for index in range(gate + 1, len(job["steps"])):
        assert "steps.gate.outputs.allowed == 'true'" in str(job["steps"][index].get("if", "")), (
            names[index]
        )


@pytest.mark.parametrize("job_name", JOBS)
def test_claude_talks_to_the_sidecar_with_a_placeholder_key(workflow, job_name):
    job = workflow["jobs"][job_name]
    claude = next(s for s in job["steps"] if str(s.get("uses", "")).startswith("anthropics/"))
    assert claude["env"]["ANTHROPIC_BASE_URL"] == "${{ env.AI_KEY_PROXY }}/anthropic"
    assert claude["with"]["anthropic_api_key"] == "${{ env.PROXY_PLACEHOLDER_KEY }}"
    assert "environment" not in job
    assert "k8s-runner-cpu" in job["runs-on"]


def test_no_secret_and_no_pod_key_anywhere():
    text = WORKFLOW.read_text()
    assert "secrets." not in text
    assert "export-pod-api-key" not in text
    assert "env.ANTHROPIC_API_KEY" not in text
    assert "sudo" not in text
    workflow = yaml.safe_load(text)
    assert workflow["env"]["AI_KEY_PROXY"] == SIDECAR


def test_claude_tracks_progress_in_a_comment_and_requests_changes_on_important_findings(workflow):
    review = workflow["jobs"]["review"]
    action = next(
        s
        for s in review["steps"]
        if str(s.get("uses", "")).startswith("anthropics/claude-code-action@")
    )
    assert action["with"]["track_progress"] is True
    prompt = action["with"]["prompt"]
    assert "event=REQUEST_CHANGES" in prompt and "event=APPROVE" in prompt
    assert "dismissals" in prompt  # a later push that settles the findings lifts the request
