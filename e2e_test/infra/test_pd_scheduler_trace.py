"""CPU checks of the prototype transport; no engine imports or worker startup."""

import json
from types import SimpleNamespace

import pytest
from infra.pd_scheduler_trace import SchedulerTrace, nonnegative_int


def environment(monkeypatch, tmp_path):
    for key, value in {
        "RUN": "cpu",
        "ATTEMPT": "2",
        "WORKER": "p-0",
        "ROLE": "prefill",
        "DIR": str(tmp_path),
    }.items():
        monkeypatch.setenv(f"SMG_PD_TRACE_{key}", value)


def test_environment_sink_records_explicit_identity_and_no_acceptance(monkeypatch, tmp_path):
    environment(monkeypatch, tmp_path)
    trace = SchedulerTrace.from_env()
    trace.emit("startup", async_scheduling=True)
    trace.emit("finish_requests_return", requests=[])
    files = list(tmp_path.glob("scheduler-*.jsonl"))
    assert len(files) == 1
    first, second = [json.loads(line) for line in files[0].read_text().splitlines()]
    assert first["seq"] == 1 and second["seq"] == 2
    assert first["worker"] == "p-0" and first["attempt"] == "2"
    assert first["role"] == "prefill" and first["run"] == "cpu"
    assert first["pid"] > 0 and first["instance"]
    assert first["observation"] == second["observation"] == "partial"
    assert trace.unknown is None


@pytest.mark.parametrize("missing", ["RUN", "ATTEMPT", "WORKER", "ROLE", "DIR"])
def test_missing_identity_or_directory_disables_trace_as_unknown(monkeypatch, tmp_path, missing):
    environment(monkeypatch, tmp_path)
    monkeypatch.delenv(f"SMG_PD_TRACE_{missing}")
    trace = SchedulerTrace.from_env()
    trace.emit("finish_requests_return", requests=[])
    assert trace.unknown is not None
    assert list(tmp_path.iterdir()) == []


def test_unknown_role_and_unavailable_directory_fail_without_creating_paths(monkeypatch, tmp_path):
    environment(monkeypatch, tmp_path)
    monkeypatch.setenv("SMG_PD_TRACE_ROLE", "other")
    assert SchedulerTrace.from_env().unknown == "setup:ValueError"
    monkeypatch.setenv("SMG_PD_TRACE_ROLE", "decode")
    monkeypatch.setenv("SMG_PD_TRACE_DIR", str(tmp_path / "missing"))
    assert SchedulerTrace.from_env().unknown == "setup:FileNotFoundError"
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("value", [float("nan"), SimpleNamespace(request_id="object")])
def test_non_json_or_non_finite_record_is_unknown_and_cannot_recover_to_success(value):
    lines = []
    trace = SchedulerTrace(lines.append, {})
    trace.emit("bad", value=value)
    assert trace.unknown is not None
    trace.emit("finish_requests_return", requests=[])
    assert lines == []


@pytest.mark.parametrize("value", [None, True, -1, 1.0, "1"])
def test_unknown_scheduler_numbers_are_not_fabricated_as_zero(value):
    with pytest.raises(ValueError):
        nonnegative_int(value)
