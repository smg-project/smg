"""Exercise the actual BFCL Score caller with CPU-only process boundaries."""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml


def score_script():
    workflow = Path(__file__).resolve().parents[2] / ".github/workflows/nightly-bfcl.yml"
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


def run_score(tmp_path, mode="sequential", *, report_write_failure=False, **faults):
    bin_dir = tmp_path / "bin"
    write_executable(bin_dir / "bfcl", "#!/bin/sh\nexit 99\n")
    write_executable(
        tmp_path / "scripts/bfcl/launch_arm.sh",
        """#!/bin/bash
set -euo pipefail
[ "$#" -eq 1 ] || exit 98
case "$1" in a|b|stop) ;; *) exit 98 ;; esac
printf 'launch:%s:%s\n' "$1" "${BFCL_GPU:-}" >> "$TRACE"
if [ "$1" = stop ]; then
    rm -f -- "$RUNNER_TEMP/owned-a" "$RUNNER_TEMP/owned-b"
    exit "$STOP_RC"
fi
printf 'owned\n' > "$RUNNER_TEMP/owned-$1"
if [ "$1" = "$FAIL_LAUNCH" ]; then
    if [ "$EMPTY_URL" -ne 1 ]; then printf 'http://%s.invalid\n' "$1"; fi
    exit "$LAUNCH_RC"
fi
printf 'http://%s.invalid\n' "$1"
""",
    )
    write_executable(
        bin_dir / "python",
        f"#!{sys.executable}\n"
        + textwrap.dedent(
            """\
            import json
            import os
            import sys
            from pathlib import Path

            temp = Path(os.environ["RUNNER_TEMP"])
            root = temp / "bfcl_ab"
            args = sys.argv[1:]
            common = [
                "--bfcl-model", "mock", "--categories", "mock",
                "--bfcl", str(temp / "bin/bfcl"), "--project-root", str(root),
                "--run-timeout", "1",
            ]
            reports = [
                "--out", str(temp / "bfcl_ab.md"),
                "--json-out", str(temp / "bfcl_ab.json"),
            ]
            script = "scripts/bfcl/run_ab.py"
            if args[:2] == [script, "--score-arm"]:
                arm = "a" if args[2] == "vllm=http://a.invalid" else "b"
                name = "vllm" if arm == "a" else "smg"
                expected = [
                    script, "--score-arm", f"{name}=http://{arm}.invalid",
                    "--scores-out", str(root / f"scores_{name}.json"), *common,
                ]
                phase = f"score-{arm}"
            elif args[:2] == [script, "--diff-baseline"]:
                expected = [
                    script, "--diff-baseline", str(root / "scores_vllm.json"),
                    "--diff-candidate", str(root / "scores_smg.json"),
                    "--categories", "mock", *reports, "--tolerance", "0.02",
                ]
                phase = "comparison"
            else:
                expected = [
                    script, "--baseline", "vllm=http://a.invalid",
                    "--candidate", "smg=http://b.invalid", *common[:-2],
                    *reports, *common[-2:], "--tolerance", "0.02",
                ]
                phase = "comparison"
            if args != expected:
                print(f"Unexpected BFCL arguments: {args!r}", file=sys.stderr)
                sys.exit(98)
            with Path(os.environ["TRACE"]).open("a") as stream:
                stream.write(f"python:{phase}\\n")
            if phase == "score-a":
                sys.exit(int(os.environ["SCORE_A_RC"]))
            if phase == "score-b":
                sys.exit(0)
            (temp / "bfcl_ab.md").write_text("# complete comparison\\n")
            (temp / "bfcl_ab.json").write_text(json.dumps({"complete": True}) + "\\n")
            sys.exit(int(os.environ["COMPARE_RC"]))
            """
        ),
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
        "BFCL_GPU": "",
        "FAIL_LAUNCH": "",
        "LAUNCH_RC": "0",
        "EMPTY_URL": "0",
        "STOP_RC": "0",
        "SCORE_A_RC": "0",
        "COMPARE_RC": "0",
        **{key: str(value) for key, value in faults.items()},
    }
    if report_write_failure:
        (tmp_path / "bfcl_ab.json").symlink_to("/dev/full")
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
    report_path = tmp_path / "bfcl_ab.json"
    report = None if report_path.is_symlink() else json.loads(report_path.read_text())
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
    result, trace, report = run_score(
        tmp_path, mode, FAIL_LAUNCH=arm, LAUNCH_RC=status, EMPTY_URL=1
    )
    assert_incomplete(tmp_path, result, trace, report, f"launch-{arm}", status)
    assert "launch:a:0" in trace
    if arm == "b":
        assert f"launch:b:{'0' if mode == 'sequential' else '1'}" in trace
    assert not (tmp_path / "owned-a").exists()
    assert not (tmp_path / "owned-b").exists()
    if arm == "a":
        assert not any(event.startswith("launch:b:") for event in trace)
        assert not any(event.startswith("python:") for event in trace)
    elif mode == "sequential":
        assert "python:score-a" in trace
        assert "python:score-b" not in trace
        assert "python:comparison" not in trace
    else:
        assert not any(event.startswith("python:") for event in trace)


@pytest.mark.parametrize("score_rc,phase,status", [(11, "score-a", 11), (0, "stop-a", 17)])
def test_first_score_or_stop_failure_remains_hard(tmp_path, score_rc, phase, status):
    result, trace, report = run_score(tmp_path, SCORE_A_RC=score_rc, STOP_RC=17)
    assert_incomplete(tmp_path, result, trace, report, phase, status)
    assert not any(event.startswith("launch:b:") for event in trace)
    assert "python:comparison" not in trace


@pytest.mark.parametrize("comparison_rc", [0, 1])
@pytest.mark.parametrize("mode", ["sequential", "concurrent"])
def test_only_final_comparison_regression_is_advisory(tmp_path, mode, comparison_rc):
    result, trace, report = run_score(tmp_path, mode, COMPARE_RC=comparison_rc)
    assert result.returncode == 0
    assert report == {"complete": True}
    assert "python:comparison" in trace
    assert "launch:a:0" in trace
    assert f"launch:b:{'0' if mode == 'sequential' else '1'}" in trace
    if mode == "sequential":
        assert trace.index("python:score-a") < trace.index("launch:b:0")
        assert trace.index("python:score-b") < trace.index("python:comparison")
    assert ("::warning::" in result.stdout) == bool(comparison_rc)


@pytest.mark.parametrize("arm,status", [("a", 1), ("b", 2)])
def test_launch_error_survives_stop_and_diagnostic_write_failure(tmp_path, arm, status):
    result, trace, report = run_score(
        tmp_path,
        "concurrent",
        report_write_failure=True,
        FAIL_LAUNCH=arm,
        LAUNCH_RC=status,
        EMPTY_URL=1,
        STOP_RC=17,
    )
    assert result.returncode == status
    assert report is None
    summary = (tmp_path / "bfcl_ab.md").read_text()
    assert f"Caller phase: `launch-{arm}`; exit: `{status}`." in summary
    assert "partial report could not be written" in result.stdout
    assert any(event.startswith("launch:stop:") for event in trace)
    assert not any(event.startswith("python:") for event in trace)
    assert not (tmp_path / "owned-a").exists()
    assert not (tmp_path / "owned-b").exists()


def test_launch_failure_preserves_existing_reports_and_saved_scores(tmp_path):
    markdown = b"existing markdown\n"
    payload = b'{"existing":"comparison"}\n'
    (tmp_path / "bfcl_ab.md").write_bytes(markdown)
    (tmp_path / "bfcl_ab.json").write_bytes(payload)
    scores = tmp_path / "bfcl_ab/scores_vllm.json"
    scores.parent.mkdir()
    scores.write_bytes(b"existing arm A\n")
    result, trace, report = run_score(tmp_path, FAIL_LAUNCH="a", LAUNCH_RC=19, EMPTY_URL=1)
    assert result.returncode == 19
    assert report == {"existing": "comparison"}
    assert (tmp_path / "bfcl_ab.md").read_bytes() == markdown
    assert (tmp_path / "bfcl_ab.json").read_bytes() == payload
    assert scores.read_bytes() == b"existing arm A\n"
    assert not any(event.startswith("python:") for event in trace)


def run_actual_diff(
    tmp_path, candidate_score, report_error=None, *, diagnostic_write_failure=False
):
    for name, score in [("vllm", 1.0), ("smg", candidate_score)]:
        (tmp_path / f"{name}.json").write_text(
            json.dumps({"name": name, "scores": {"mock": score}, "counts": {"mock": 1}})
        )
    markdown = tmp_path / "comparison.md"
    report = tmp_path / "comparison.json"
    if report_error == "markdown":
        markdown.mkdir()
    elif report_error == "json":
        report.mkdir()
    driver = Path(__file__).resolve().parents[1] / "bfcl/run_ab.py"
    command = [
        sys.executable,
        str(driver),
        "--diff-baseline",
        str(tmp_path / "vllm.json"),
        "--diff-candidate",
        str(tmp_path / "smg.json"),
        "--categories",
        "mock",
        "--out",
        str(markdown),
        "--json-out",
        str(report),
    ]
    if diagnostic_write_failure:
        wrapper = tmp_path / "broken_stderr.py"
        wrapper.write_text(
            textwrap.dedent(
                """\
                import runpy
                import sys
                from pathlib import Path

                class BrokenStderr:
                    def write(self, _text):
                        Path(__file__).with_suffix(".attempted").write_text("attempted")
                        raise OSError("diagnostic writer unavailable")

                    def flush(self):
                        pass

                driver = sys.argv[1]
                sys.argv = sys.argv[1:]
                original_stderr = sys.stderr
                sys.stderr = BrokenStderr()
                try:
                    runpy.run_path(driver, run_name="__main__")
                finally:
                    sys.stderr = original_stderr
                """
            )
        )
        command.insert(1, str(wrapper))
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=10,
    )


@pytest.mark.parametrize("report_error", ["markdown", "json"])
def test_actual_driver_report_write_error_is_hard_incomplete(tmp_path, report_error):
    result = run_actual_diff(tmp_path, 1.0, report_error)
    assert result.returncode == 2
    assert "IsADirectoryError" in result.stderr


@pytest.mark.parametrize("candidate_score,status", [(1.0, 0), (0.5, 1)])
def test_actual_driver_preserves_clean_and_regression_exit_codes(tmp_path, candidate_score, status):
    result = run_actual_diff(tmp_path, candidate_score)
    assert result.returncode == status
    assert (tmp_path / "comparison.md").is_file()
    report = json.loads((tmp_path / "comparison.json").read_text())
    assert not report["incomplete"]


def test_actual_driver_diagnostic_write_failure_remains_hard_incomplete(tmp_path):
    result = run_actual_diff(tmp_path, 1.0, "markdown", diagnostic_write_failure=True)
    assert result.returncode == 2
    assert (tmp_path / "broken_stderr.attempted").read_text() == "attempted"
    assert not (tmp_path / "comparison.json").exists()
