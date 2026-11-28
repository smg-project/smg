"""The engine watch hands over the engines no open issue reports, and fails when one stays so."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.skipif(
    shutil.which("jq") is None, reason="the watch reads its rows with jq"
)

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/engine_watch_pending.sh"
WORKFLOW = ROOT / ".github/workflows/engine-version-watch.yml"

SHA = "e87f7267c0b32836a5f51156309675dd1d7f60e8"
ROWS = [
    {"engine": "vllm", "current": "0.31.0", "latest": "0.33.0", "update": True},
    {
        "engine": "tokenspeed",
        "current": "7cd7ca0b3028dca2ce331c0599a0537379ab0e74",
        "latest": SHA,
        "update": True,
    },
    {"engine": "tensorrt-llm", "current": "1.3.0rc24", "latest": "1.4.0rc2.post1", "update": True},
]
ENGINES = [row["engine"] for row in ROWS]
REPORTED = [
    "[engine-watch] vllm 0.33.0 available (pinned: 0.31.0)",
    "[engine-watch] tokenspeed e87f726 available (pinned: 7cd7ca0)",
    "[engine-watch] tensorrt-llm 1.4.0rc2.post1 available (pinned: 1.3.0rc24)",
]
# Any call but the watch's own (its two labels, its one listing from the issues endpoint) fails,
# so these tests pin how the issues are listed; the listing's --jq runs over the one page.
FAKE_GH = """#!/bin/bash
[ "$1 $2" = "label create" ] && exit 0
[ "$1" = api ] || exit 2
shift
while [ $# -gt 0 ]; do
    case "$1" in
        --paginate) ;;
        --jq) filter=$2; shift ;;
        "repos/owner/repository/issues?state=open&labels=engine-watch&per_page=100") listed=1 ;;
        *) exit 2 ;;
    esac
    shift
done
[ -n "${listed:-}" ] || exit 2
jq -r "${filter:?}" "$ISSUES"
"""


@pytest.fixture
def env(tmp_path):
    """The environment of a step in a checkout at ``tmp_path``, where ``gh`` is the fake."""
    gh = tmp_path / "bin" / "gh"
    gh.parent.mkdir()
    gh.write_text(FAKE_GH)
    gh.chmod(0o755)
    (tmp_path / "scripts").symlink_to(ROOT / "scripts")
    (tmp_path / "issues.json").write_text("[]")
    return {
        **os.environ,
        "PATH": f"{gh.parent}{os.pathsep}{os.environ['PATH']}",
        "GITHUB_REPOSITORY": "owner/repository",
        "GITHUB_OUTPUT": str(tmp_path / "output"),
        "ISSUES": str(tmp_path / "issues.json"),
    }


def open_issues(env, titles, pull_requests=()):
    issues = [{"title": title} for title in titles]
    issues += [{"title": title, "pull_request": {}} for title in pull_requests]
    Path(env["ISSUES"]).write_text(json.dumps(issues))


def pending(env) -> list[str]:
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        input=json.dumps(ROWS),
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    return [row["engine"] for row in json.loads(result.stdout)]


def steps(job: str) -> list[dict]:
    return yaml.safe_load(WORKFLOW.read_text())["jobs"][job]["steps"]


def agent_of(filing: list[dict]) -> dict:
    return next(step for step in filing if str(step.get("uses", "")).startswith("anthropics/"))


def run_step(step: dict, env, checkout) -> subprocess.CompletedProcess:
    """The step's script as the runner runs a step that names no shell, from the checkout's root."""
    return subprocess.run(
        ["bash", "-e", "-c", step["run"]], cwd=checkout, capture_output=True, text=True, env=env
    )


def outputs(env) -> dict[str, str]:
    return dict(line.split("=", 1) for line in Path(env["GITHUB_OUTPUT"]).read_text().splitlines())


def test_every_engine_is_pending_while_no_issue_is_open(env):
    assert pending(env) == ENGINES


def test_an_engine_whose_latest_an_open_issue_names_is_not_pending(env):
    """A commit pin is named by a shortened SHA in the titles the watch has filed."""
    open_issues(env, REPORTED)
    assert pending(env) == []
    open_issues(env, [f"[engine-watch] tokenspeed {SHA} available (pinned: 7cd7ca0)"])
    assert pending(env) == ["vllm", "tensorrt-llm"]


def test_an_issue_for_another_version_or_engine_reports_nothing(env):
    """A candidate of the latest release, the release the latest one follows up, another commit,
    and the latest version of an engine whose name ends in this one's."""
    open_issues(
        env,
        [
            "[engine-watch] vllm 0.33.0rc1 available (pinned: 0.31.0)",
            "[engine-watch] tokenspeed 90eea27 available (pinned: e87f726)",
            "[engine-watch] tensorrt-llm 1.4.0rc2 available (pinned: 1.3.0rc24)",
            "[engine-watch] xvllm 0.33.0 available (pinned: 0.31.0)",
        ],
    )
    assert pending(env) == ENGINES


def test_a_pull_request_under_the_label_reports_nothing(env):
    open_issues(env, [], pull_requests=REPORTED)
    assert pending(env) == ENGINES


@pytest.mark.parametrize("stdin", ["", "not json", "{}"], ids=["nothing", "no JSON", "an object"])
def test_input_that_is_no_array_of_rows_fails(env, stdin):
    """A list that cannot be read must not read as nothing pending."""
    result = subprocess.run(
        ["bash", str(SCRIPT)], input=stdin, capture_output=True, text=True, env=env
    )
    assert result.returncode != 0
    assert result.stdout == ""


def test_the_detect_job_hands_over_the_engines_no_issue_reports(env, tmp_path):
    (dedup,) = [step for step in steps("detect") if step.get("id") == "dedup"]
    (tmp_path / "updates.json").write_text(json.dumps(ROWS))

    open_issues(env, REPORTED[:1])
    assert run_step(dedup, env, tmp_path).returncode == 0
    assert json.loads(outputs(env)["pending"]) == ROWS[1:]
    assert outputs(env)["has_pending"] == "true"

    open_issues(env, REPORTED)
    assert run_step(dedup, env, tmp_path).returncode == 0
    assert outputs(env)["has_pending"] == "false"


def test_a_rerun_of_the_filing_job_hands_the_agent_what_is_still_unreported(env, tmp_path):
    filing = steps("research-and-file")
    (unfiled,) = [step for step in filing if step.get("id") == "unfiled"]
    agent = agent_of(filing)
    assert filing.index(unfiled) < filing.index(agent)
    assert unfiled["env"]["PENDING"] == "${{ needs.detect.outputs.pending }}"
    assert "${{ steps.unfiled.outputs.rows }}" in agent["with"]["prompt"]
    assert "needs.detect.outputs" not in agent["with"]["prompt"]
    assert agent["if"] == "steps.unfiled.outputs.rows != '[]'"

    env["PENDING"] = json.dumps(ROWS)
    open_issues(env, REPORTED[:2])
    assert run_step(unfiled, env, tmp_path).returncode == 0
    assert json.loads(outputs(env)["rows"]) == ROWS[2:]

    open_issues(env, REPORTED)
    assert run_step(unfiled, env, tmp_path).returncode == 0
    assert outputs(env)["rows"] == "[]"


def test_the_filing_job_fails_when_an_engine_it_was_handed_stays_unreported(env, tmp_path):
    filing = steps("research-and-file")
    (check,) = [step for step in filing if "::error::" in str(step.get("run", ""))]
    assert filing.index(check) > filing.index(agent_of(filing))
    assert "if" not in check and "continue-on-error" not in check
    assert check["env"]["PENDING"] == "${{ needs.detect.outputs.pending }}"

    env["PENDING"] = json.dumps(ROWS)
    open_issues(env, REPORTED[:1])
    unfiled = run_step(check, env, tmp_path)
    assert unfiled.returncode == 1
    named = f"tokenspeed {SHA}, tensorrt-llm 1.4.0rc2.post1"
    assert f"::error::no engine-watch issue was filed for: {named}" in unfiled.stdout

    open_issues(env, REPORTED)
    assert run_step(check, env, tmp_path).returncode == 0

    env["PENDING"] = ""
    assert run_step(check, env, tmp_path).returncode != 0


def test_the_agents_report_goes_to_the_job_summary():
    """The agent step is green and logs counters when it files nothing: the report says why."""
    assert agent_of(steps("research-and-file"))["with"]["display_report"] is True
