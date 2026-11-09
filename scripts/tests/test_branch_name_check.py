"""The one branch-name rule, as the pre-push hook and the pull-request workflow read it."""

import subprocess
from pathlib import Path

import pytest

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
    ],
)
def test_a_capitalised_or_slashless_name_is_refused(name):
    assert verdict(name) == 1


def test_the_hook_and_the_workflow_read_the_script_and_hold_no_rule_of_their_own():
    hook = (ROOT / ".pre-commit-config.yaml").read_text()
    workflow = (ROOT / ".github" / "workflows" / "pr-naming-check.yml").read_text()
    assert "entry: scripts/branch-name-check.sh" in hook
    assert 'bash scripts/branch-name-check.sh "$BRANCH"' in workflow
    assert "[a-z0-9._-]*[a-z0-9])?/" not in hook
    assert "[a-zA-Z0-9._-]*[a-zA-Z0-9])?(/" not in workflow
