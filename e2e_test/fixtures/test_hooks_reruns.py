"""Unit tests for the rerun record (no GPU): the summary lines, the annotation,
the file under ``E2E_LOG_DIR``, and a real session whose case fails once under
``--reruns 1``.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fixtures import hooks

pytest_plugins = ["pytester"]


class _Terminal:
    """A terminal reporter that keeps the lines written to it."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def write_line(self, line: str, **_markup: Any) -> None:
        self.lines.append(line)


def _report(nodeid: str, outcome: str, attempt: int, message: str = "") -> Any:
    """A test report as pytest-rerunfailures logs it: ``rerun`` counts the
    attempts before this one; a failure carries its crash message."""
    crash = SimpleNamespace(message=message) if message else None
    return SimpleNamespace(
        nodeid=nodeid,
        outcome=outcome,
        when="call",
        rerun=attempt - 1,
        longrepr=SimpleNamespace(reprcrash=crash),
        longreprtext=message,
    )


def _summary(reporter: hooks._RerunReporter) -> list[str]:
    terminal = _Terminal()
    reporter.pytest_terminal_summary(terminal)  # type: ignore[arg-type]
    return terminal.lines


def test_a_session_without_reruns_counts_zero_and_writes_no_file(monkeypatch, tmp_path):
    monkeypatch.setenv("E2E_LOG_DIR", str(tmp_path))
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    reporter = hooks._RerunReporter()
    reporter.pytest_runtest_logreport(_report("t.py::test_a", "passed", attempt=1))
    reporter.pytest_runtest_logreport(_report("t.py::test_b", "failed", attempt=1, message="E"))
    assert _summary(reporter) == ["e2e reruns: 0"]
    assert not (tmp_path / "reruns.txt").exists()


def test_each_rerun_is_one_line_naming_case_attempt_and_first_failure_line(monkeypatch, tmp_path):
    """Two rerun reports of one attempt (its call and its teardown) are one
    rerun named by the first; the pass that ends the case adds nothing."""
    monkeypatch.setenv("E2E_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    reporter = hooks._RerunReporter()
    boom = "AssertionError: boom\n +  where 1 = one()"
    reporter.pytest_runtest_logreport(_report("t.py::test_a", "rerun", attempt=1, message=boom))
    reporter.pytest_runtest_logreport(
        _report("t.py::test_a", "rerun", attempt=1, message="OSError: teardown")
    )
    reporter.pytest_runtest_logreport(
        _report("t.py::test_a", "rerun", attempt=2, message="AssertionError: again")
    )
    reporter.pytest_runtest_logreport(_report("t.py::test_a", "passed", attempt=3))
    lines = _summary(reporter)
    assert lines == [
        "e2e reruns: 2",
        "e2e rerun: t.py::test_a attempt 1 failed: AssertionError: boom",
        "e2e rerun: t.py::test_a attempt 2 failed: AssertionError: again",
    ]
    record = (tmp_path / "logs" / "reruns.txt").read_text(encoding="utf-8")
    assert record.splitlines() == lines[1:]


def test_annotations_are_written_under_github_actions_only(monkeypatch):
    monkeypatch.delenv("E2E_LOG_DIR", raising=False)
    reporter = hooks._RerunReporter()
    reporter.pytest_runtest_logreport(
        _report("t.py::test_a[50%]", "rerun", attempt=1, message="AssertionError: boom")
    )
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    assert [line for line in _summary(reporter) if line.startswith("::")] == []
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    assert [line for line in _summary(reporter) if line.startswith("::")] == [
        "::warning title=e2e rerun::t.py::test_a[50%25] attempt 1 failed: AssertionError: boom"
    ]


def test_a_failure_without_a_crash_line_falls_back_to_its_text():
    report = SimpleNamespace(longrepr="worker died\nexit 137", longreprtext="worker died\nexit 137")
    assert hooks._first_failure_line(report) == "worker died"  # type: ignore[arg-type]
    empty = SimpleNamespace(longrepr=None, longreprtext="")
    assert hooks._first_failure_line(empty) == "(no failure text)"  # type: ignore[arg-type]


def test_a_long_first_line_is_cut_to_the_record_length():
    long = "AssertionError: " + "x" * 400
    report = _report("t.py::test_a", "rerun", attempt=1, message=long)
    assert hooks._first_failure_line(report) == long[: hooks._REASON_CHARS]
    assert len(hooks._first_failure_line(report)) == 200


def _isolated_session(pytester, monkeypatch, log_dir: Path):
    """A pytester session that loads only ``fixtures.hooks`` and the rerun
    plugin, with the record directed at ``log_dir``."""
    for key in list(os.environ):
        if key.startswith(("PYTEST_", "E2E_")):
            monkeypatch.delenv(key)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    hook = Path(__file__).with_name("hooks.py").resolve()
    monkeypatch.setenv("PYTHONPATH", str(hook.parent.parent))
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    monkeypatch.setenv("E2E_LOG_DIR", str(log_dir))
    pytester.makeconftest('pytest_plugins = ["fixtures.hooks"]\n')
    return pytester.makeini("[pytest]\n")


def test_a_real_session_records_the_case_that_passed_on_its_rerun(pytester, monkeypatch, tmp_path):
    """A case that fails on its first attempt and passes on the rerun ends
    the session green with the rerun named in the summary and the record."""
    log_dir = tmp_path / "logs"
    config = _isolated_session(pytester, monkeypatch, log_dir)
    case = pytester.makepyfile(
        test_case=(
            "import pathlib\n"
            "\n"
            "\n"
            "def test_flaky():\n"
            "    first = pathlib.Path(__file__).with_suffix('.first')\n"
            "    if not first.exists():\n"
            "        first.write_text('1')\n"
            "        raise AssertionError('first attempt')\n"
            "\n"
            "\n"
            "def test_sound():\n"
            "    pass\n"
        )
    )
    result = pytester.runpytest_subprocess(
        "-q",
        "-p",
        "no:cacheprovider",
        "-p",
        "rerunfailures",
        "-c",
        str(config),
        "--confcutdir",
        str(pytester.path),
        "--reruns",
        "1",
        str(case),
        timeout=30,
    )
    output = result.stdout.str() + result.stderr.str()
    assert result.ret == 0, output
    assert result.parseoutcomes() == {"passed": 2, "rerun": 1}
    rerun_line = (
        "e2e rerun: test_case.py::test_flaky attempt 1 failed: AssertionError: first attempt"
    )
    assert "e2e reruns: 1" in output.splitlines(), output
    assert rerun_line in output.splitlines(), output
    assert (log_dir / "reruns.txt").read_text(encoding="utf-8") == rerun_line + "\n"
