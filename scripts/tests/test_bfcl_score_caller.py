"""Exercise the actual BFCL Score caller with CPU-only process boundaries."""

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml


def score_script():
    workflow = (
        Path(__file__).resolve().parents[2] / ".github/workflows/nightly-bfcl.yml"
    )
    jobs = yaml.safe_load(workflow.read_text())["jobs"].values()
    marker = 'BFCL="$(command -v bfcl)"'
    scripts = [
        step["run"]
        for job in jobs
        for step in job.get("steps", [])
        if marker in step.get("run", "")
    ]
    assert len(scripts) == 1
    return scripts[0][scripts[0].index(marker) :]


def write_executable(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    path.chmod(0o700)


def run_score(tmp_path, mode="sequential", **faults):
    bin_dir = tmp_path / "bin"
    write_executable(bin_dir / "bfcl", "#!/bin/sh\nexit 99\n")
    write_executable(
        tmp_path / "scripts/bfcl/launch_arm.sh",
        """#!/bin/bash
printf 'launch:%s:%s\n' "$1" "${BFCL_GPU:-}" >> "$TRACE"
if [ "$1" = stop ]; then exit "${STOP_RC:-0}"; fi
printf 'http://%s.invalid\n' "$1"
if [ "$1" = "${FAIL_LAUNCH:-}" ]; then exit "$LAUNCH_RC"; fi
""",
    )
    write_executable(
        bin_dir / "python",
        """#!/bin/bash
phase=comparison
case "$*" in
  *"--score-arm vllm="*) phase=score-a ;;
  *"--score-arm smg="*) phase=score-b ;;
esac
printf 'python:%s\n' "$phase" >> "$TRACE"
if [ "$phase" = score-a ]; then exit "${SCORE_A_RC:-0}"; fi
if [ "$phase" = score-b ]; then exit 0; fi
printf '# complete comparison\n' > "$RUNNER_TEMP/bfcl_ab.md"
printf '{"complete":true}\n' > "$RUNNER_TEMP/bfcl_ab.json"
exit "${COMPARE_RC:-0}"
""",
    )
    env = {
        **os.environ,
        "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
        "TRACE": str(tmp_path / "trace"),
        "RUNNER_TEMP": str(tmp_path),
        "ARM_MODE": mode,
        "GPU_A": "0",
        "GPU_B": "1",
        "BFCL_MODEL_FC": "mock",
        "CATEGORIES": "mock",
        "RUN_TIMEOUT": "1",
        **{key: str(value) for key, value in faults.items()},
    }
    # Skip the real host/flock prelude and replace only its EXIT cleanup boundary.
    script = "set -euo pipefail\n"
    script += "trap 'bash scripts/bfcl/launch_arm.sh stop || true' EXIT\n"
    result = subprocess.run(
        ["bash", "-c", script + score_script()],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    trace = (tmp_path / "trace").read_text().splitlines()
    report = json.loads((tmp_path / "bfcl_ab.json").read_text())
    return result, trace, report


def assert_incomplete(tmp_path, result, trace, report, phase, status):
    assert result.returncode == status
    assert report == {
        "report_type": "caller-incomplete",
        "phase": phase,
        "exit_code": status,
        "complete": False,
    }
    summary = (tmp_path / "bfcl_ab.md").read_text()
    assert f"Caller phase: `{phase}`; exit: `{status}`." in summary
    assert any(event.startswith("launch:stop:") for event in trace)


@pytest.mark.parametrize("mode", ["sequential", "concurrent"])
@pytest.mark.parametrize("arm,status", [("a", 19), ("b", 23)])
def test_launch_failure_retains_status_and_partial_report(tmp_path, mode, arm, status):
    result, trace, report = run_score(tmp_path, mode, FAIL_LAUNCH=arm, LAUNCH_RC=status)
    assert_incomplete(tmp_path, result, trace, report, f"launch-{arm}", status)
    if arm == "a":
        assert not any(event.startswith("launch:b:") for event in trace)
        assert not any(event.startswith("python:") for event in trace)
    elif mode == "sequential":
        assert "python:score-a" in trace
        assert "python:score-b" not in trace
        assert "python:comparison" not in trace
    else:
        assert not any(event.startswith("python:") for event in trace)


@pytest.mark.parametrize(
    "score_rc,phase,status", [(11, "score-a", 11), (0, "stop-a", 17)]
)
def test_first_score_or_stop_failure_remains_hard(tmp_path, score_rc, phase, status):
    result, trace, report = run_score(tmp_path, SCORE_A_RC=score_rc, STOP_RC=17)
    assert_incomplete(tmp_path, result, trace, report, phase, status)
    assert not any(event.startswith("launch:b:") for event in trace)
    assert "python:comparison" not in trace


@pytest.mark.parametrize("comparison_rc", [0, 1])
def test_only_final_comparison_regression_is_advisory(tmp_path, comparison_rc):
    result, trace, report = run_score(tmp_path, "concurrent", COMPARE_RC=comparison_rc)
    assert result.returncode == 0
    assert report == {"complete": True}
    assert "python:comparison" in trace
    assert ("::warning::" in result.stdout) == bool(comparison_rc)
