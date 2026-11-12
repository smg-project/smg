"""The Codex review workflow admits writers only, keeps the keys out of reach, and posts what it ran."""

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


def test_the_triggers_are_the_pull_request_events_and_the_comment_events(workflow):
    triggers = workflow[True]  # PyYAML reads the `on` key as a boolean
    assert set(triggers) == {
        "pull_request",
        "issue_comment",
        "pull_request_review_comment",
        "pull_request_review",
    }
    # pull_request, not pull_request_target: Claude's App-token exchange rejects the latter's OIDC token
    assert "pull_request_target" not in triggers
    assert triggers["pull_request"]["types"] == ["opened", "synchronize", "reopened"]


def test_the_artifact_upload_includes_the_dot_prefixed_review_directory(workflow):
    codex = workflow["jobs"]["codex"]
    upload = step(codex, "actions/upload-artifact@")
    assert upload["with"]["path"] == "pr/.codex-review/"
    assert upload["with"]["include-hidden-files"] is True


def test_forks_and_bots_are_refused_before_any_job_starts(workflow):
    condition = workflow["jobs"]["codex"]["if"]
    assert (
        condition.count("github.event.pull_request.head.repo.full_name == github.repository") == 3
    )
    assert "github.event.comment.user.type != 'Bot'" in condition


def test_no_job_uses_an_environment_or_a_secret_and_no_shell_step_handles_a_key(workflow):
    jobs = workflow["jobs"]
    assert all("environment" not in job for job in jobs.values())
    text = WORKFLOW.read_text()
    assert "secrets." not in text and "export-pod-api-key" not in text
    assert "Start the Anthropic key proxy" not in text and "Drop sudo" not in text
    runs = "".join(str(s.get("run", "")) for job in jobs.values() for s in job["steps"])
    assert "API_KEY" not in runs


def test_codex_reaches_the_sidecar_through_the_actions_proxy_with_a_placeholder_key(workflow):
    codex = workflow["jobs"]["codex"]
    run = step(codex, "openai/codex-action@")
    assert run["with"]["openai-api-key"] == "${{ env.OPENAI_PLACEHOLDER_KEY }}"
    assert run["with"]["responses-api-endpoint"] == "${{ env.AI_KEY_PROXY }}/openai/v1/responses"
    assert run["with"]["permission-profile"] == ":read-only"
    assert run["with"]["working-directory"] == "pr"
    assert run["with"]["allow-bot-users"] == "claude[bot]"
    assert codex["permissions"] == {"contents": "read", "pull-requests": "read"}
    assert workflow["env"]["AI_KEY_PROXY"] == "http://127.0.0.1:8787"


def test_claude_verifies_through_the_sidecar_with_a_placeholder_key(workflow):
    verify = workflow["jobs"]["verify-by-claude"]
    claude = step(verify, "anthropics/")
    assert claude["env"]["ANTHROPIC_BASE_URL"] == "${{ env.AI_KEY_PROXY }}/anthropic"
    assert claude["with"]["anthropic_api_key"] == "${{ env.ANTHROPIC_PLACEHOLDER_KEY }}"
    names = [s.get("name") or s.get("uses") for s in verify["steps"]]
    assert (
        "Start the Anthropic key proxy" not in names
        and "Drop sudo for the rest of the job" not in names
    )


def test_the_post_job_runs_only_after_a_successful_review_from_the_default_branch(workflow):
    post = workflow["jobs"]["post"]
    assert post["needs"] == "codex"
    assert "needs.codex.result == 'success'" in post["if"]
    # the App token posts, not the job token; id-token lets the job prove itself to the sidecar
    assert post["permissions"] == {"contents": "read", "id-token": "write"}
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


def test_the_post_job_posts_as_the_codex_app_with_a_token_the_sidecar_mints(workflow):
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import codex_review_post

    assert workflow["env"]["CODEX_BOT"] == codex_review_post.BOT == "smg-codex-review[bot]"
    post = workflow["jobs"]["post"]
    steps = post["steps"]
    mint = next(s for s in steps if s.get("id") == "app-token")
    assert mint["uses"] == "./.github/actions/codex-posting-token"
    # the codex job, where the model runs, has no id-token permission and so cannot prove
    # itself to the sidecar
    assert "id-token" not in workflow["jobs"]["codex"]["permissions"]
    posting = next(s for s in steps if "codex_review_post.py post" in str(s.get("run", "")))
    assert steps.index(mint) < steps.index(posting)
    assert posting["env"]["GITHUB_TOKEN"] == "${{ steps.app-token.outputs.token }}"
    text = WORKFLOW.read_text()
    assert "github-actions[bot]" not in text  # the verify prompt names the App's login via env
    assert "${{ env.CODEX_BOT }}" in text


def test_the_posting_token_action_proves_the_job_with_its_oidc_token_and_masks_the_result():
    action = yaml.safe_load((ROOT / ".github/actions/codex-posting-token/action.yml").read_text())
    assert action["runs"]["using"] == "composite"
    run = " ".join(str(s.get("run", "")) for s in action["runs"]["steps"])
    assert "ACTIONS_ID_TOKEN_REQUEST_URL" in run and "audience=ai-key-proxy" in run
    assert "Authorization: Bearer" in run and "/github/app-token" in run
    assert "::add-mask::" in run
    assert "2>/dev/null || true" in run  # a non-JSON reply still reaches the error line
    assert "token" in action["outputs"]


def test_codex_announces_itself_before_reviewing_and_reports_the_outcome(workflow):
    jobs = workflow["jobs"]
    announce = jobs["announce"]
    assert "needs" not in announce  # beside the review, not after it
    assert announce["permissions"] == {"contents": "read", "id-token": "write"}
    assert announce["runs-on"] == jobs["codex"]["runs-on"]
    assert "github.event_name == 'pull_request'" in announce["if"]
    assert "@codex review" in announce["if"]
    assert "pull_request_review_comment" not in announce["if"]  # a thread reply gets no note
    runs = " ".join(str(s.get("run", "")) for s in announce["steps"])
    assert "codex_review_post.py announce --state reviewing" in runs
    failed = jobs["finish-status"]
    assert failed["needs"] == "codex"
    assert "always()" in failed["if"] and "needs.codex.result == 'failure'" in failed["if"]
    assert "needs.codex.outputs.mode == 'review'" in failed["if"]
    runs = " ".join(str(s.get("run", "")) for s in failed["steps"])
    assert "codex_review_post.py announce --state failed" in runs
    for name in ("announce", "post", "finish-status"):
        mint = next(
            s
            for s in jobs[name]["steps"]
            if str(s.get("uses", "")).startswith("./.github/actions/codex-posting-token")
        )
        posting = next(
            s for s in jobs[name]["steps"] if "codex_review_post.py" in str(s.get("run", ""))
        )
        assert posting["env"]["GITHUB_TOKEN"] == "${{ steps." + mint["id"] + ".outputs.token }}"
