"""A prior fixture's successful transfer cannot satisfy the current workers."""

import re

import pytest
from infra import pd_logs

SUCCESS = re.compile(r"Num successful transfers=([1-9]\d*)")


@pytest.fixture(autouse=True)
def short_poll(monkeypatch):
    monkeypatch.setattr(pd_logs, "LOG_FLUSH_TIMEOUT_S", 0.001)


def test_foreign_positive_does_not_satisfy_owned_zero(tmp_path):
    owned = tmp_path / "worker-owned.log"
    owned.write_text("Num successful transfers=0\n")
    (tmp_path / "worker-foreign.log").write_text("Num successful transfers=99\n")
    assert SUCCESS.search(pd_logs.wait_for_pattern(tmp_path, "worker-*.log", SUCCESS))
    logs = pd_logs.wait_for_pattern_in_files([owned], SUCCESS)
    assert logs == owned.read_text()
    assert not SUCCESS.search(logs)


def test_positive_from_owned_worker_is_accepted(tmp_path):
    owned = tmp_path / "owned.log"
    owned.write_text("Num successful transfers=2\n")
    assert SUCCESS.search(pd_logs.wait_for_pattern_in_files([owned], SUCCESS)).group(1) == "2"


def test_missing_owned_capture_is_red_in_ci(monkeypatch, tmp_path):
    monkeypatch.setenv("CI", "true")
    (tmp_path / "worker-foreign.log").write_text("Num successful transfers=99\n")
    logs = pd_logs.wait_for_pattern_in_files([None, tmp_path / "missing.log"], SUCCESS)
    assert logs == ""
    with pytest.raises(pytest.fail.Exception, match="No worker log files captured"):
        pd_logs.assert_worker_logs_captured(logs, "Mooncake transfer")


def test_missing_owned_capture_retains_local_skip(monkeypatch):
    monkeypatch.delenv("CI", raising=False)
    logs = pd_logs.wait_for_pattern_in_files([None], SUCCESS)
    with pytest.raises(pytest.skip.Exception, match="No worker log files captured"):
        pd_logs.assert_worker_logs_captured(logs, "Mooncake transfer")
