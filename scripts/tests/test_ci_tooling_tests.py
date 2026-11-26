"""The tooling runner runs the baseline and every component's suite, errors on a
source or suite the tree lacks instead of noting it, and names no path this tree
lacks; the main workflow routes every script test to its supported interpreter."""

import fnmatch
import importlib.util
import shlex
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


def selected_tests(command):
    """Resolve the workflow's actual pytest selectors against this source tree."""
    tokens = shlex.split(command)
    selected = set()
    for token in tokens:
        if token == "scripts/tests":
            selected.update(ROOT.glob("scripts/tests/test_*.py"))
        elif token.startswith("scripts/tests/"):
            selected.update(ROOT.glob(token))
    ignores = [token.split("=", 1)[1] for token in tokens if token.startswith("--ignore-glob=")]
    return {
        path.relative_to(ROOT).as_posix()
        for path in selected
        if not any(fnmatch.fnmatch(path.relative_to(ROOT).as_posix(), ignore) for ignore in ignores)
    }


def test_the_blocking_tooling_job_runs_every_script_test_exactly_once():
    steps = yaml.safe_load(WORKFLOW.read_text())["jobs"]["python-unit-tests"]["steps"]
    general = next(step for step in steps if step.get("name") == "Run CI tooling unit tests")
    tau2 = next(step for step in steps if step.get("name") == "Run fixed tau2 tooling unit tests")
    general_tests = selected_tests(general["run"].splitlines()[0])
    tau2_tests = selected_tests(tau2["run"])
    expected = {path.relative_to(ROOT).as_posix() for path in ROOT.glob("scripts/tests/test_*.py")}
    assert general_tests and tau2_tests
    assert not general_tests & tau2_tests
    assert general_tests | tau2_tests == expected
    assert 'for test in scripts/tests/test_*.sh; do bash "$test"; done' in general["run"]
    assert sorted((ROOT / "scripts" / "tests").glob("test_*.sh"))
    for step in [general, tau2]:
        assert "if" not in step and not step.get("continue-on-error", False)
    assert "python -m pytest" in general["run"]  # repository imports remain available


def test_tau2_runs_after_the_other_python_tests_in_a_separate_fixed_environment():
    steps = yaml.safe_load(WORKFLOW.read_text())["jobs"]["python-unit-tests"]["steps"]
    switch = next(
        i for i, step in enumerate(steps) if step.get("name") == "Set up fixed tau2 Python"
    )
    harness = next(
        i for i, step in enumerate(steps) if step.get("name") == "Run e2e harness unit tests"
    )
    assert harness < switch
    assert steps[switch]["with"]["python-version"] == "3.12"
    assert any(step.get("with", {}).get("python-version") == "3.13" for step in steps[:switch])
    assert steps[-1]["name"] == "Run fixed tau2 tooling unit tests"
    assert '"$RUNNER_TEMP/tau2-budget-venv/bin/python" -m pytest' in steps[-1]["run"]


def test_both_blocking_tau2_entries_share_the_fixed_source_sdk_and_full_cohort():
    main_steps = yaml.safe_load(WORKFLOW.read_text())["jobs"]["python-unit-tests"]["steps"]
    budget = ROOT / ".github/workflows/tau2-budget-tests.yml"
    budget_steps = yaml.safe_load(budget.read_text())["jobs"]["budget"]["steps"]
    for steps in [main_steps, budget_steps]:
        source = next(
            step for step in steps if step.get("with", {}).get("path") == ".tau2-upstream"
        )
        assert source["with"] == {
            "repository": "sierra-research/tau2-bench",
            "ref": "5bfa7e37b36656b37dc6d022156be6563c1007f3",
            "path": ".tau2-upstream",
            "persist-credentials": False,
            "sparse-checkout": "src/tau2/utils/llm_utils.py",
            "sparse-checkout-cone-mode": False,
        }
    installs = [
        next(
            step
            for step in steps
            if step.get("name") == "Install fixed SDKs for fake-HTTP-only tests"
        )
        for steps in [main_steps, budget_steps]
    ]
    assert installs[0]["run"] == installs[1]["run"]
    assert 'python -m venv "$RUNNER_TEMP/tau2-budget-venv"' in installs[0]["run"]
    for requirement in [
        "litellm==1.81.11",
        "openai==2.20.0",
        "httpx==0.28.1",
        "websockets==15.0.1",
        "pydub==0.25.1",
        "5bfa7e37b36656b37dc6d022156be6563c1007f3.tar.gz",
    ]:
        assert requirement in installs[0]["run"]
    tau2_steps = [
        next(step for step in steps if step.get("name") == "Run fixed tau2 tooling unit tests")
        for steps in [main_steps, budget_steps]
    ]
    assert tau2_steps[0]["run"] == tau2_steps[1]["run"]
    assert selected_tests(tau2_steps[0]["run"]) == {
        path.relative_to(ROOT).as_posix() for path in ROOT.glob("scripts/tests/test_tau2*.py")
    }
