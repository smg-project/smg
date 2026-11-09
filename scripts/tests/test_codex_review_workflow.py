"""The Codex review workflow runs from main, admits writers only, and keeps the keys out of reach."""

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


def step(job, prefix):
    return next(s for s in job["steps"] if str(s.get("uses", "")).startswith(prefix))


def test_every_trigger_runs_the_workflow_from_the_default_branch(workflow):
    triggers = workflow[True]  # PyYAML reads the `on` key as a boolean
    assert set(triggers) == {
        "pull_request_target",
        "issue_comment",
        "pull_request_review_comment",
        "pull_request_review",
    }
    assert "pull_request" not in triggers
    assert triggers["pull_request_target"]["types"] == [
        "opened",
        "synchronize",
        "reopened",
        "ready_for_review",
    ]


def test_forks_and_bots_are_refused_before_any_job_starts(workflow):
    condition = workflow["jobs"]["codex"]["if"]
    assert (
        condition.count("github.event.pull_request.head.repo.full_name == github.repository") == 3
    )
    assert "github.event.comment.user.type != 'Bot'" in condition


def test_the_key_jobs_use_the_ai_review_environment_and_the_posting_job_has_no_key(workflow):
    jobs = workflow["jobs"]
    assert jobs["codex"]["environment"] == "ai-review"
    assert jobs["verify-by-claude"]["environment"] == "ai-review"
    assert "environment" not in jobs["post"]
    text = WORKFLOW.read_text()
    assert (
        text.count("secrets.OPENAI_API_KEY") == 1 and text.count("secrets.ANTHROPIC_API_KEY") == 1
    )
    assert "export-pod-api-key" not in text
    runs = "".join(str(s.get("run", "")) for s in jobs["codex"]["steps"])
    assert "API_KEY" not in runs  # no shell step touches a key, so none can reach the job env


def test_the_openai_key_is_only_the_actions_proxy_input(workflow):
    codex = workflow["jobs"]["codex"]
    run = step(codex, "openai/codex-action@")
    assert run["with"]["openai-api-key"] == "${{ secrets.OPENAI_API_KEY }}"
    assert run["with"]["permission-profile"] == ":read-only"
    assert run["with"]["working-directory"] == "pr"
    assert run["with"]["allow-bot-users"] == "claude[bot]"
    assert codex["permissions"] == {"contents": "read", "pull-requests": "read"}


def test_the_anthropic_key_feeds_a_root_proxy_and_sudo_is_dropped_before_claude(workflow):
    verify = workflow["jobs"]["verify-by-claude"]
    names = [s.get("name") or s.get("uses") for s in verify["steps"]]
    proxy = names.index("Start the Anthropic key proxy")
    drop = names.index("Drop sudo for the rest of the job")
    claude = next(
        i for i, s in enumerate(verify["steps"]) if str(s.get("uses", "")).startswith("anthropics/")
    )
    assert proxy < drop < claude
    proxy_step = verify["steps"][proxy]
    assert proxy_step["env"] == {"ANTHROPIC_API_KEY": "${{ secrets.ANTHROPIC_API_KEY }}"}
    assert "sudo -n env -i" in proxy_step["run"] and "printf" in proxy_step["run"]
    claude_step = verify["steps"][claude]
    assert claude_step["with"]["anthropic_api_key"].startswith("sk-ant-placeholder")


def test_the_post_job_runs_only_after_a_successful_review_from_the_default_branch(workflow):
    post = workflow["jobs"]["post"]
    assert post["needs"] == "codex"
    assert "needs.codex.result == 'success'" in post["if"]
    assert post["permissions"] == {"contents": "read", "pull-requests": "write"}
    checkout = step(post, "actions/checkout@")
    assert "ref" not in checkout.get("with", {})  # the default ref of a main-defined run is main


def test_only_the_configured_events_share_a_cancelling_group(workflow):
    group = workflow["concurrency"]["group"]
    assert "format('noop-{0}', github.run_id)" in group
    assert "contains(github.event.comment.body, '@codex review')" in group
    assert "github.event.review.user.login == 'claude[bot]'" in group


def test_every_mode_has_a_prompt_and_a_closed_schema():
    for mode in MODES:
        assert (ROOT / f".github/codex/{mode}-prompt.md").read_text().strip()
        schema = json.loads((ROOT / f".github/codex/{mode}-schema.json").read_text())
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])
    review = json.loads((ROOT / ".github/codex/review-schema.json").read_text())
    assert "prior_findings_still_open" in review["required"]


def test_every_job_runs_on_the_review_runner_set(workflow):
    for name, job in workflow["jobs"].items():
        assert "k8s-runner-cpu" in job["runs-on"], name


def test_the_acl_package_is_installed_before_codex_drops_sudo(workflow):
    codex = workflow["jobs"]["codex"]
    names = [s.get("name") or s.get("uses") for s in codex["steps"]]
    acl = names.index("Install the acl package drop-sudo needs")
    run = next(
        i
        for i, s in enumerate(codex["steps"])
        if str(s.get("uses", "")).startswith("openai/codex-action@")
    )
    assert acl < run
    assert "setfacl" in codex["steps"][acl]["run"]
