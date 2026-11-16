"""The one branch-name rule, as the pre-push hook and the pull-request workflow read it."""

import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "branch-name-check.sh"


def verdict(name):
    return subprocess.run(
        ["bash", str(SCRIPT), name], capture_output=True, text=True, check=False
    ).returncode


@pytest.mark.parametrize(
    "name",
    [
        "feat/add-auth",
        "fix/null-pointer",
        "someone/fix-routing",
        "dependabot/cargo/pyo3-0.28.1",
        "dependabot/docker/docker/ubuntu-fe81560",
        "fix/pd/rooms",
        "release/1.14",
        "main",
    ],
)
def test_a_lowercase_name_of_one_or_more_segments_is_accepted(name):
    assert verdict(name) == 0


@pytest.mark.parametrize(
    "name",
    [
        "Fix/rooms",
        "feat/Rooms",
        "feat",
        "feat/",
        "/rooms",
        "feat//rooms",
        "-feat/rooms",
        "feat/rooms with a space",
        "patch-1",
        "Dependabot/cargo/pyo3-0.28.1",
    ],
)
def test_a_capitalised_or_slashless_name_is_refused(name):
    assert verdict(name) == 1


@pytest.mark.parametrize(
    "name",
    [
        "dependabot/github_actions/Swatinem/rust-cache-3",
        "dependabot/github_actions/PyO3/maturin-action-2",
    ],
)
def test_a_dependabot_branch_keeps_the_tools_own_case(name):
    """The bot writes an action owner into the branch name as the workflow writes it;
    a refusal would close the bump, which the bot then takes for an ignored version."""
    assert verdict(name) == 0


def test_the_hook_and_the_workflow_read_the_script_and_hold_no_rule_of_their_own():
    hook = (ROOT / ".pre-commit-config.yaml").read_text()
    workflow = (ROOT / ".github" / "workflows" / "pr-naming-check.yml").read_text()
    assert "entry: scripts/branch-name-check.sh" in hook
    assert 'bash scripts/branch-name-check.sh "$BRANCH"' in workflow
    assert "[a-z0-9._-]*[a-z0-9])?/" not in hook
    assert "[a-zA-Z0-9._-]*[a-zA-Z0-9])?(/" not in workflow


WORKFLOW = ROOT / ".github" / "workflows" / "pr-naming-check.yml"


def naming_job_steps():
    return yaml.safe_load(WORKFLOW.read_text())["jobs"]["branch-naming"]["steps"]


def test_the_workflow_reads_the_script_from_the_pull_requests_own_tree():
    """A pull_request run takes the workflow from the pull request's tree, so the
    script comes from the same tree: a ref to another branch would judge a pull
    request that adds or changes the script by a copy without it."""
    checkout = next(
        step
        for step in naming_job_steps()
        if str(step.get("uses", "")).startswith("actions/checkout")
    )
    assert "ref" not in checkout["with"]
    assert checkout["with"]["persist-credentials"] is False


def run_naming_step(tmp_path, cwd, branch):
    """The job's shell step as the runner executes it, with a recording gh on PATH.

    Returns the step's exit status and the gh calls as `<command> <subcommand> <number>`."""
    step = next(step for step in naming_job_steps() if step.get("name") == "Validate branch name")
    script = re.sub(r"\$\{\{\s*github\.repository\s*\}\}", "owner/repo", step["run"])
    assert "${{" not in script
    (tmp_path / "step.sh").write_text(script)
    fake = tmp_path / "bin"
    fake.mkdir(exist_ok=True)
    log = tmp_path / "gh.log"
    (fake / "gh").write_text(
        f'#!/usr/bin/env bash\nprintf "%s %s %s\\n" "$1" "$2" "$3" >> "{log}"\n'
    )
    (fake / "gh").chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{fake}{os.pathsep}{os.environ['PATH']}",
        "BRANCH": branch,
        "PR_NUMBER": "7",
        "PR_AUTHOR": "someone",
        "GH_TOKEN": "none",
    }
    status = subprocess.run(
        ["bash", "--noprofile", "--norc", "-eo", "pipefail", str(tmp_path / "step.sh")],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    ).returncode
    calls = log.read_text().splitlines() if log.exists() else []
    return status, calls


def test_the_job_passes_an_accepted_name_without_touching_the_pull_request(tmp_path):
    assert run_naming_step(tmp_path, ROOT, "feat/rooms") == (0, [])


def test_the_job_comments_and_closes_on_a_refused_name(tmp_path):
    assert run_naming_step(tmp_path, ROOT, "Fix/rooms") == (1, ["pr comment 7", "pr close 7"])


def test_a_check_that_did_not_run_fails_the_job_and_leaves_the_pull_request_open(tmp_path):
    """The tree has no script (a run before the script's merge, a rename, a lost
    checkout): the step fails with the shell's status, and no call touches the pull request."""
    assert run_naming_step(tmp_path, tmp_path, "feat/rooms") == (127, [])


def test_the_tooling_workflow_starts_on_the_files_this_suite_reads():
    """A change to a reader alone must run the suite that guards it."""
    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "ci-tooling-tests.yml").read_text())
    paths = workflow.get("on", workflow.get(True))["pull_request"]["paths"]
    assert {
        ".github/workflows/pr-naming-check.yml",
        ".pre-commit-config.yaml",
        "scripts/**",
    } <= set(paths)


def test_a_plain_pre_commit_install_puts_the_pre_push_hook_in_place():
    """`pre-commit install` without a hook type installs only the types this key names."""
    config = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text())
    assert "pre-push" in config["default_install_hook_types"]


# Every hook whose manifest in pre-commit-hooks v5.0.0 puts it at push time; a
# rev bump re-reads the manifest's stages and updates this list.
MANIFEST_PUSH_TIME_HOOKS = {
    "trailing-whitespace",
    "end-of-file-fixer",
    "check-added-large-files",
    "check-executables-have-shebangs",
    "check-shebang-scripts-are-executable",
    "destroyed-symlinks",
}


def test_the_branch_name_hook_is_the_only_one_that_runs_at_push_time():
    """With the pre-push hook type installed by default, a hook that reaches push time
    runs on every push. pre-commit takes a hook's stages from this config, else from the
    hook's own manifest, else from default_stages; pre-commit-hooks' manifest puts several
    of its hooks at push time, so every hook taken from it names its stages here."""
    config = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text())
    at_push = {
        hook["id"]
        for repo in config["repos"]
        for hook in repo["hooks"]
        if "pre-push" in hook.get("stages", config["default_stages"])
    }
    assert at_push == {"branch-name-check"}
    repo = next(repo for repo in config["repos"] if repo["repo"].endswith("/pre-commit-hooks"))
    assert repo["rev"] == "v5.0.0", (
        "re-read the manifest's stages and update MANIFEST_PUSH_TIME_HOOKS"
    )
    used = {hook["id"] for hook in repo["hooks"]}
    named = {hook["id"] for hook in repo["hooks"] if "stages" in hook}
    assert MANIFEST_PUSH_TIME_HOOKS & used <= named
