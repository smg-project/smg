"""Offline real-shell deadline controls; fake Python never installs or sends requests."""

import datetime
import io
import json
import sys
import time
from pathlib import Path

import diagnostic_budget as runner
import pytest
import reuse_foundation as reuse

SCRIPT = Path(__file__).with_name("run_qwen_diagnostic.sh")


@pytest.fixture
def job(monkeypatch, tmp_path):
    for key, value in {
        "GITHUB_RUN_ID": "17",
        "GITHUB_RUN_ATTEMPT": "2",
        "GITHUB_SHA": "a" * 40,
        "RUNNER_NAME": "offline-runner",
        "RUNNER_TEMP": str(tmp_path),
    }.items():
        monkeypatch.setenv(key, value)
    now = datetime.datetime.now(datetime.UTC)
    own = {
        "id": 19,
        "name": "qwen-vision-diagnostic",
        "runner_name": "offline-runner",
        "status": "in_progress",
        "head_sha": "a" * 40,
        "started_at": (now - datetime.timedelta(seconds=60)).isoformat(),
    }

    def fake_api(path):
        assert path == "actions/runs/17/attempts/2/jobs"
        return io.StringIO(json.dumps({"jobs": [own]}))

    monkeypatch.setattr(reuse, "api_get", fake_api)
    monkeypatch.setattr(reuse, "ROOT", tmp_path)
    return own


def test_budget_counts_time_before_script(job):
    budget = reuse.consumer_budget()
    started = datetime.datetime.fromisoformat(job["started_at"]).timestamp()
    assert budget["diagnostic_deadline_epoch"] == started + 885
    assert budget["diagnostic_deadline_epoch"] + 15 + 300 == started + 20 * 60
    assert budget["diagnostic_deadline_epoch"] - time.time() < 825


@pytest.mark.parametrize("change", ["head", "runner", "future", "expired"])
def test_bad_consumer_never_launches(job, monkeypatch, change):
    if change in ("head", "runner"):
        job[{"head": "head_sha", "runner": "runner_name"}[change]] = "different"
    else:
        delta = 60 if change == "future" else -901
        job["started_at"] = (
            datetime.datetime.now(datetime.UTC) + datetime.timedelta(seconds=delta)
        ).isoformat()
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: pytest.fail("core launched"))
    with pytest.raises(ValueError):
        runner.main()


@pytest.mark.parametrize("field", ["consumer_attempt", "diagnostic_deadline_epoch"])
def test_budget_receipt_cannot_extend_or_change_consumer(job, tmp_path, monkeypatch, field):
    budget = reuse.consumer_budget()
    budget[field] = "different" if field == "consumer_attempt" else budget[field] + 60
    path = tmp_path / "budget.json"
    path.write_text(json.dumps(budget))
    monkeypatch.setenv("QWEN_DIAG_BUDGET_FILE", str(path))
    with pytest.raises(ValueError):
        reuse.load_budget()


def test_budget_expiry_during_bootstrap_never_launches(job, monkeypatch, tmp_path):
    budget = reuse.consumer_budget()
    monkeypatch.setattr(runner.time, "time", lambda: budget["diagnostic_deadline_epoch"] + 1)
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: pytest.fail("core launched"))
    with pytest.raises(ValueError, match="before execution"):
        runner.main()
    assert (tmp_path / "diagnostic-results/job-budget.json").is_file()


@pytest.mark.parametrize("limited", [False, True])
def test_real_shell_shares_deadline_across_download_install_and_pytest(
    job, tmp_path, monkeypatch, limited
):
    fake = tmp_path / "python3"
    fake.write_text(
        f"#!{sys.executable}\n"
        "import json, os, signal, sys, time\n"
        "from pathlib import Path\n"
        "def interrupted(*_):\n"
        "    Path('interrupted').write_text('SIGINT')\n"
        "    raise SystemExit(130)\n"
        "signal.signal(signal.SIGINT, interrupted)\n"
        "with open('commands.jsonl', 'a') as out:\n"
        "    out.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "budget = json.loads(Path(os.environ['QWEN_DIAG_BUDGET_FILE']).read_text())\n"
        "assert budget['consumer_job'] == 19\n"
        "Path('diagnostic-results').mkdir(exist_ok=True)\n"
        "if os.environ['SLOW_PHASES'] == '1': time.sleep(0.7)\n"
    )
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + ":/usr/bin:/bin")
    monkeypatch.setenv("SLOW_PHASES", "1" if limited else "0")
    monkeypatch.setattr(runner, "COMMAND", ["bash", str(SCRIPT)])
    if limited:
        job["started_at"] = (
            datetime.datetime.now(datetime.UTC) - datetime.timedelta(seconds=883.8)
        ).isoformat()
    before = time.monotonic()
    result = runner.main()
    elapsed = time.monotonic() - before
    commands = [json.loads(line) for line in (tmp_path / "commands.jsonl").read_text().splitlines()]
    assert commands[0] == [".github/amd-diagnostics/reuse_foundation.py"]
    receipt = json.loads((tmp_path / "diagnostic-results/job-budget.json").read_text())
    assert receipt["consumer_started_at"] == job["started_at"]
    if limited:
        assert result == 124
        assert (tmp_path / "interrupted").read_text() == "SIGINT"
        assert commands[1][:3] == ["-m", "pip", "install"]
        assert not any(args[:2] == ["-m", "pytest"] for args in commands)
        assert elapsed < 3
    else:
        assert result == 0
        assert commands[-1][:2] == ["-m", "pytest"]
        assert "no:rerunfailures" in commands[-1]
        assert len(commands) == 8
