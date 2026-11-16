"""The floor under the replay step of the Policy Recorded Rows workflow.

The step's verdict used to be the exit code of one ignored test. A run that
holds no such test (the test moved to another target, or behind a feature the
job does not name) prints `running 0 tests` and exits 0: the job stays green
and replays nothing. A test whose last assertion is lost prints its lines and
passes: the job stays green and checks nothing.

The test prints a line a sheet and a line a policy with a replay module, the
policies from the one table of modules in the test sources. The step fails
unless the output holds a policy line, every policy line counts rows and no
defect, and no line counts a defect. These cases run the step's own script,
read from the workflow file, with a `cargo` on PATH that prints a given output.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "policy-recorded-rows.yml"
STEP = "Replay every recorded row"

CLEAN = """\
running 1 test
/sheets/manual.json: 14 rows, 0 defects
/sheets/power_of_two.json: 44 rows, 0 defects
policy manual: 14 rows, 0 defects
policy power_of_two: 44 rows, 0 defects
test every_recorded_classic_row_replays ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 9 filtered out
"""
NO_TEST = """\
running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 9 filtered out
"""


def step_script() -> str:
    document = yaml.safe_load(WORKFLOW.read_text())
    steps = document["jobs"]["replay"]["steps"]
    return next(step["run"] for step in steps if step.get("name") == STEP)


def run_step(tmp_path: Path, printed: str, cargo_status: int = 0):
    """The step's script with a cargo that prints `printed` and exits `cargo_status`."""
    home = tmp_path / "home"
    (home / ".cargo").mkdir(parents=True)
    (home / ".cargo" / "env").write_text("")
    stubs = tmp_path / "bin"
    stubs.mkdir()
    output = tmp_path / "printed.txt"
    output.write_text(printed)
    cargo = stubs / "cargo"
    cargo.write_text(f'#!/bin/sh\ncat "{output}"\nexit {cargo_status}\n')
    cargo.chmod(0o755)
    work = tmp_path / "work"
    work.mkdir()
    # A workflow's `run` step is `bash -e` unless it names a shell.
    return subprocess.run(
        ["bash", "-e", "-c", step_script()],
        cwd=work,
        env={
            "PATH": f"{stubs}:/usr/bin:/bin",
            "HOME": str(home),
            "SMG_POLICY_CLASSIC_ROWS": "/sheets",
        },
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_a_clean_line_for_every_policy_passes(tmp_path):
    result = run_step(tmp_path, CLEAN)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "the floor: 2 policies" in result.stdout


@pytest.mark.parametrize(
    "printed,says",
    [
        # No test ran: the build line and the test line both exit 0.
        (NO_TEST, "named no policy"),
        # A policy's sheet is missing at the pin and the test passed all the same.
        (
            CLEAN.replace("/sheets/manual.json: 14 rows, 0 defects\n", "").replace(
                "policy manual: 14 rows, 0 defects", "policy manual: no sheet"
            ),
            "policy manual: no sheet",
        ),
        # A policy has two sheets.
        (
            CLEAN.replace("policy manual: 14 rows, 0 defects", "policy manual: 2 sheets"),
            "policy manual: 2 sheets",
        ),
        # A policy's sheet holds no row.
        (CLEAN.replace("14 rows, 0 defects", "0 rows, 0 defects"), "policy manual: 0 rows"),
        # A recorded row did not replay and the test passed all the same.
        (
            CLEAN.replace("44 rows, 0 defects", "44 rows, 1 defects"),
            "policy power_of_two: 44 rows, 1 defects",
        ),
        # A sheet no policy line stands for counts a defect.
        (
            CLEAN.replace("policy manual", "/sheets/stray.json: 0 rows, 1 defects\npolicy manual"),
            "a sheet counts a defect",
        ),
        # A policy line is read up to its colon: a name spelled with a dash is one too.
        (
            CLEAN.replace("policy manual", "policy least-load: no sheet\npolicy manual"),
            "policy least-load: no sheet",
        ),
    ],
)
def test_a_run_that_replays_or_checks_too_little_fails(tmp_path, printed, says):
    result = run_step(tmp_path, printed)
    assert result.returncode != 0, result.stdout + result.stderr
    assert says in result.stdout, result.stdout + result.stderr


def test_a_failed_test_fails_the_step_whatever_it_printed(tmp_path):
    result = run_step(tmp_path, CLEAN, cargo_status=101)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "the floor" not in result.stdout
