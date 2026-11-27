"""Scheduling contracts for `.github/workflows/pr-test-rust.yml`.

Advisory 4-GPU benchmark jobs must opt in via `ci:bench` on pull requests so they
do not starve blocking 4-GPU E2E lanes or delay `finish`, while remaining enabled
on `push` and `workflow_dispatch`. Branch runs must not cancel in-progress runs so
back-to-back merges collapse in the single pending slot at zero compute cost.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "pr-test-rust.yml"
CANCEL_WORKFLOW = ROOT / ".github" / "workflows" / "cancel-pr-workflows.yml"


@pytest.fixture(scope="module")
def workflow() -> dict:
    """Load the parsed `pr-test-rust.yml` workflow document once per module."""
    return yaml.safe_load(WORKFLOW.read_text())


def _eval_gha_bool(
    expr: str,
    *,
    event_name: str,
    labels: tuple[str, ...] = (),
    docs_only: str = "false",
    card_tier_from: str = "",
    cancelled: bool = False,
    needs_results: dict[str, str] | None = None,
    vars_map: dict[str, str] | None = None,
) -> bool:
    """Evaluate a restricted GitHub Actions boolean `if:` expression for unit tests.

    Supports the exact operators and context references used by `benchmarks`,
    `benchmarks-amd`, `summarize-benchmarks`, and `concurrency.cancel-in-progress`.
    """
    needs_results = needs_results or {}
    vars_map = vars_map or {}

    stripped = expr.strip()
    if stripped.startswith("${{") and stripped.endswith("}}"):
        stripped = stripped[3:-2].strip()

    def _contains(haystack: object, needle: object) -> bool:
        return needle in haystack  # type: ignore[operator]

    def _from_json(raw: str) -> object:
        return json.loads(raw)

    # Translate GitHub Actions operators and context paths into a small Python expression.
    py = stripped
    py = py.replace("always()", "True")
    py = py.replace("!cancelled()", str(not cancelled))
    py = py.replace("cancelled()", str(cancelled))
    py = py.replace("&&", " and ")
    py = py.replace("||", " or ")
    py = re.sub(r"needs\.detect-changes\.outputs\.docs-only", repr(docs_only), py)
    py = re.sub(r"needs\.detect-changes\.outputs\.card-tier-from", repr(card_tier_from), py)
    py = re.sub(
        r"needs\.([a-z0-9_-]+)\.result",
        lambda m: repr(needs_results.get(m.group(1), "success")),
        py,
    )
    py = re.sub(
        r"vars\.([A-Z0-9_]+)",
        lambda m: repr(vars_map.get(m.group(1), "")),
        py,
    )
    py = py.replace("github.event_name", repr(event_name))
    py = py.replace("github.event.pull_request.labels.*.name", repr(list(labels)))

    result = eval(py, {"__builtins__": {}}, {"contains": _contains, "fromJSON": _from_json})
    return bool(result)


@pytest.mark.parametrize(
    ("event_name", "labels", "docs_only", "card_tier_from", "expected"),
    [
        ("pull_request", (), "false", "", False),
        ("pull_request", ("ci:full",), "false", "", False),
        ("pull_request", ("ci:bench",), "false", "", True),
        ("pull_request", ("ci:bench",), "true", "", False),
        ("pull_request", ("ci:bench",), "false", "a" * 40, False),
        ("push", (), "false", "", True),
        ("push", (), "false", "a" * 40, False),
        ("workflow_dispatch", (), "false", "", True),
    ],
)
def test_benchmarks_job_opts_in_on_prs_and_runs_on_push_or_dispatch(
    workflow: dict,
    event_name: str,
    labels: tuple[str, ...],
    docs_only: str,
    card_tier_from: str,
    expected: bool,
) -> None:
    """`benchmarks` skips PRs without `ci:bench`, runs on `ci:bench` PRs, and stays on for push/dispatch."""
    job = workflow["jobs"]["benchmarks"]
    assert job["continue-on-error"] is True
    expr = job["if"]
    # Without always(), GitHub Actions implicitly requires needs.* to succeed.
    assert "always()" not in expr
    assert (
        _eval_gha_bool(
            expr,
            event_name=event_name,
            labels=labels,
            docs_only=docs_only,
            card_tier_from=card_tier_from,
        )
        is expected
    )


@pytest.mark.parametrize(
    ("event_name", "labels", "amd_enabled", "docs_only", "card_tier_from", "expected"),
    [
        ("pull_request", (), True, "false", "", False),
        ("pull_request", ("ci:bench",), True, "false", "", True),
        ("pull_request", ("ci:bench",), False, "false", "", False),
        ("pull_request", ("ci:bench",), True, "true", "", False),
        ("pull_request", ("ci:bench",), True, "false", "a" * 40, False),
        ("push", (), True, "false", "", False),
    ],
)
def test_benchmarks_amd_job_requires_ci_bench_label_and_amd_vars(
    workflow: dict,
    event_name: str,
    labels: tuple[str, ...],
    amd_enabled: bool,
    docs_only: str,
    card_tier_from: str,
    expected: bool,
) -> None:
    """`benchmarks-amd` requires both the AMD repo variables and the `ci:bench` PR label."""
    job = workflow["jobs"]["benchmarks-amd"]
    assert job["continue-on-error"] is True
    flag = "true" if amd_enabled else "false"
    vars_map = {
        "SMG_RUN_AMD_LEGS": flag,
        "SMG_RUN_AMD_ON_PRS": flag,
        "SMG_RUN_AMD_PR_BENCHMARK": flag,
    }
    assert (
        _eval_gha_bool(
            job["if"],
            event_name=event_name,
            labels=labels,
            docs_only=docs_only,
            card_tier_from=card_tier_from,
            needs_results={"build-wheel": "success", "detect-changes": "success"},
            vars_map=vars_map,
        )
        is expected
    )


def test_pull_request_trigger_omits_labeled_to_avoid_cancellation_storms(workflow: dict) -> None:
    """Omitting `labeled` keeps labeler.yml and manual labels from cancelling active PR runs."""
    triggers = workflow.get("on", workflow.get(True))
    assert isinstance(triggers, dict)
    pr_types = triggers["pull_request"]["types"]
    assert "labeled" not in pr_types
    assert set(pr_types) == {"opened", "synchronize", "reopened"}


def test_summarize_benchmarks_and_finish_handle_skipped_benchmarks(workflow: dict) -> None:
    """Skipping `benchmarks` cleanly skips `summarize-benchmarks` and never blocks `finish`."""
    summarize = workflow["jobs"]["summarize-benchmarks"]
    assert summarize["needs"] == ["benchmarks"]
    assert (
        _eval_gha_bool(
            summarize["if"], event_name="pull_request", needs_results={"benchmarks": "skipped"}
        )
        is False
    )
    assert (
        _eval_gha_bool(
            summarize["if"], event_name="pull_request", needs_results={"benchmarks": "cancelled"}
        )
        is False
    )
    assert (
        _eval_gha_bool(
            summarize["if"], event_name="pull_request", needs_results={"benchmarks": "success"}
        )
        is True
    )
    assert (
        _eval_gha_bool(
            summarize["if"], event_name="pull_request", needs_results={"benchmarks": "failure"}
        )
        is True
    )

    finish_needs = set(workflow["jobs"]["finish"]["needs"])
    assert "benchmarks" not in finish_needs
    assert "benchmarks-amd" not in finish_needs


def test_concurrency_cancels_only_pull_requests_and_matches_cancel_workflow(workflow: dict) -> None:
    """PR runs cancel superseded pushes; branch runs complete while collapsing pending pushes."""
    concurrency = workflow["concurrency"]
    assert _eval_gha_bool(concurrency["cancel-in-progress"], event_name="pull_request") is True
    assert _eval_gha_bool(concurrency["cancel-in-progress"], event_name="push") is False
    assert (
        _eval_gha_bool(concurrency["cancel-in-progress"], event_name="workflow_dispatch") is False
    )

    cancel_doc = yaml.safe_load(CANCEL_WORKFLOW.read_text())
    entries = cancel_doc["jobs"]["cancel"]["strategy"]["matrix"]["include"]
    pr_test_entry = next(e for e in entries if e["workflow"] == workflow["name"])
    assert pr_test_entry["group"] == "gateway-tests-pr-{0}"
    assert "gateway-tests-" in concurrency["group"]
    assert "format('pr-{0}', github.event.pull_request.number)" in concurrency["group"]
