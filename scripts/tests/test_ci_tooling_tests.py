"""The tooling runner runs the baseline and every component's suite, errors on a
source or suite the tree lacks instead of noting it, and names no path this tree
lacks; the main workflow's tooling step runs the whole test directory rather than a
list kept by hand."""

import importlib.util
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "pr-test-rust.yml"
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


def test_a_pair_the_tree_lacks_is_an_error_not_a_notice(tmp_path, capsys):
    create(tmp_path, *RUNNER.BASELINE)
    source, _ = (RUNNER.COMPONENTS + RUNNER.REGRESSIONS)[0]
    with pytest.raises(ValueError, match=f"CPU test component is incomplete: {source}"):
        RUNNER.test_paths(tmp_path)
    assert "not tested" not in capsys.readouterr().out


def test_every_component_runs_when_the_tree_holds_them_all(tmp_path):
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


def test_every_path_the_runner_names_exists_on_the_tree():
    """The tables name files of this tree only; the runner stops at the first
    missing path, this names them all."""
    named = set(RUNNER.BASELINE)
    for source, suite in RUNNER.COMPONENTS + RUNNER.REGRESSIONS:
        named.update((source, suite))
    missing = sorted(path for path in named if not (ROOT / path).is_file())
    assert not missing, f"the tooling runner names paths the tree lacks: {missing}"


def test_the_tooling_step_runs_the_whole_directory_and_every_shell_test():
    """The main workflow's tooling step runs `scripts/tests` as a directory and its
    shell tests by glob, so a new test file runs without a line added by hand."""
    jobs = yaml.safe_load(WORKFLOW.read_text())["jobs"]
    step = next(
        step
        for step in jobs["python-unit-tests"]["steps"]
        if step.get("name") == "Run CI tooling unit tests"
    )
    # `python -m pytest`: the console script leaves the repository root off sys.path,
    # and the tests that import `scripts.<module>` then fail to collect.
    assert "python -m pytest -q -rs scripts/tests\n" in step["run"], step["run"]
    assert 'for test in scripts/tests/test_*.sh; do bash "$test"; done' in step["run"], step["run"]
    assert sorted((ROOT / "scripts" / "tests").glob("test_*.sh")), "the glob has shell tests to run"
