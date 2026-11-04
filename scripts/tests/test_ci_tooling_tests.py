"""CPU tooling selection and the workflow finish gate retain their required checks."""

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

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


_WORKFLOW = Path(__file__).resolve().parents[2] / ".github/workflows/pr-test-rust.yml"
_CLOSED_AMD_JOBS = (
    "e2e-1gpu-responses-memory-core-amd",
    "e2e-2gpu-pd-vllm-nixl-amd",
    "e2e-2gpu-pd-vllm-mooncake-amd",
    "e2e-4gpu-pd-vllm-nixl-amd",
)
_RESULT = re.compile(r"\$\{\{\s*needs\.([a-z0-9_-]+)\.result\s*\}\}")


def _run_finish_gate(tmp_path, *, source=None, results=None, step="Check CI result"):
    source = _WORKFLOW.read_text() if source is None else source
    finish = yaml.safe_load(source)["jobs"]["finish"]
    script = next(item["run"] for item in finish["steps"] if item.get("name") == step)
    values = dict.fromkeys(finish["needs"], "success")
    assert set(results or {}) <= values.keys()
    values.update(results or {})
    script = _RESULT.sub(lambda match: values[match[1]], script)
    assert "${{" not in script, "finish fixture must render every GitHub expression"
    work = tmp_path / "work"
    workflow = work / ".github/workflows/pr-test-rust.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text(source)
    binary = tmp_path / "bin"
    binary.mkdir()
    (binary / "python3").symlink_to(sys.executable)
    return subprocess.run(
        ["/bin/bash", "--noprofile", "--norc", "-e", "-o", "pipefail", "-c", script],
        cwd=work,
        env={"PATH": str(binary), "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=5,
    )


@pytest.mark.parametrize("job", _CLOSED_AMD_JOBS)
@pytest.mark.parametrize("result", ["success", "skipped", "failure", "cancelled", "", "unknown"])
def test_finish_optional_amd_result_is_closed(tmp_path, job, result):
    completed = _run_finish_gate(tmp_path, results={job: result})
    assert (completed.returncode == 0) is (result in ("success", "skipped")), completed.stdout


@pytest.mark.parametrize("job", ["python-lint", "e2e-1gpu-chat"])
@pytest.mark.parametrize("result", ["failure", "cancelled"])
def test_finish_still_rejects_other_cpu_and_nvidia_failures(tmp_path, job, result):
    completed = _run_finish_gate(tmp_path, results={job: result})
    assert completed.returncode != 0, completed.stdout


def test_finish_drift_check_accepts_the_actual_workflow(tmp_path):
    completed = _run_finish_gate(tmp_path, step="Gate-integrity lint")
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("job", _CLOSED_AMD_JOBS)
@pytest.mark.parametrize("omission", ["needs", "result"])
def test_finish_drift_check_rejects_an_omitted_job_reference(tmp_path, job, omission):
    source = _WORKFLOW.read_text()
    if omission == "needs":
        match = re.search(r"^  finish:\n    needs: \[([^\]]*)\]", source, re.MULTILINE)
        assert match is not None
        names = [name.strip() for name in match[1].split(",")]
        assert names.count(job) == 1
        source = (
            source[: match.start(1)]
            + ", ".join(name for name in names if name != job)
            + source[match.end(1) :]
        )
    else:
        reference = f"needs.{job}.result"
        assert reference in source
        source = source.replace(reference, "needs.omitted-job.result")
    completed = _run_finish_gate(tmp_path, source=source, step="Gate-integrity lint")
    assert completed.returncode != 0, completed.stdout + completed.stderr
    assert job in completed.stdout
