"""A stack layer cannot silently drop a baseline or installed component's tests."""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "tooling_test_runner", Path(__file__).resolve().parents[1] / "ci_run_tooling_tests.py"
)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def create(root, *paths):
    for path in paths:
        file = root / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.touch()


def test_early_stack_always_runs_baseline_and_reports_absent_suites(tmp_path, capsys):
    create(tmp_path, *RUNNER.BASELINE)
    assert RUNNER.test_paths(tmp_path) == list(RUNNER.BASELINE)
    output = capsys.readouterr().out
    for _, suite in RUNNER.COMPONENTS + RUNNER.REGRESSIONS:
        assert f"not tested): {suite}" in output


def test_late_stack_includes_every_available_component(tmp_path):
    create(tmp_path, *RUNNER.BASELINE)
    for source, suite in RUNNER.COMPONENTS + RUNNER.REGRESSIONS:
        create(tmp_path, source, suite)
    assert RUNNER.test_paths(tmp_path) == list(RUNNER.BASELINE) + [
        suite for _, suite in RUNNER.COMPONENTS + RUNNER.REGRESSIONS
    ]


@pytest.mark.parametrize("path", RUNNER.BASELINE)
def test_missing_baseline_is_an_error(tmp_path, path):
    create(tmp_path, *(p for p in RUNNER.BASELINE if p != path))
    with pytest.raises(ValueError, match="Required CPU suite is missing"):
        RUNNER.test_paths(tmp_path)


@pytest.mark.parametrize("source,suite", RUNNER.COMPONENTS)
def test_new_component_requires_its_regression_suite(tmp_path, source, suite):
    create(tmp_path, *RUNNER.BASELINE, source)
    with pytest.raises(ValueError, match="CPU test component is incomplete"):
        RUNNER.test_paths(tmp_path)


@pytest.mark.parametrize("source,suite", RUNNER.COMPONENTS + RUNNER.REGRESSIONS)
def test_suite_without_its_source_is_an_error(tmp_path, source, suite):
    create(tmp_path, *RUNNER.BASELINE, suite)
    with pytest.raises(ValueError, match="CPU test component is incomplete"):
        RUNNER.test_paths(tmp_path)
