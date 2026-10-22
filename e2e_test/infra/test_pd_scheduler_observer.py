"""CPU contract-stub checks; these do not import or execute the vLLM backend."""

import json
from types import SimpleNamespace

import pytest
from infra.pd_scheduler_observer import SchedulerObserverMixin
from infra.pd_scheduler_test_support import (
    NativeContract,
    Status,
    actual_aborted_ids,
    events,
    observed,
    request,
)
from infra.pd_scheduler_trace import SchedulerTrace


def test_one_shot_ids_are_not_consumed_and_unknown_id_is_not_acknowledged():
    scheduler, trace, records = observed()
    req = request()
    scheduler.add_request(req)
    ids = iter(["wrong-request"])
    result = scheduler.finish_requests(ids, Status.FINISHED_ABORTED)
    assert result == []
    assert scheduler.calls[-1][1] is ids
    assert req.status == Status.RUNNING
    assert actual_aborted_ids(records) == []
    assert trace.unknown is None
    scheduler.finish_requests(iter([req.request_id]), Status.FINISHED_ABORTED)
    assert actual_aborted_ids(records) == [req.request_id]


def test_ignored_abort_then_natural_length_never_becomes_abort_ack():
    scheduler, trace, records = observed()
    req = request()
    scheduler.add_request(req)
    scheduler.ignore_abort = True
    assert scheduler.finish_requests(req.request_id, Status.FINISHED_ABORTED) == []
    req.num_output_tokens = req.max_tokens
    req.status = Status.FINISHED_LENGTH_CAPPED
    scheduler._free_request(req)
    assert actual_aborted_ids(records) == []
    assert (
        events(records, "free_request_return")[-1]["request"]["status"] == "FINISHED_LENGTH_CAPPED"
    )
    assert scheduler.requests == {}
    assert trace.unknown is None


def test_unfinished_native_result_does_not_copy_requested_abort_status(monkeypatch):
    scheduler, trace, records = observed()
    req = request()
    scheduler.add_request(req)
    result = [req]
    monkeypatch.setattr(NativeContract, "finish_requests", lambda *args: result)
    assert scheduler.finish_requests(req.request_id, Status.FINISHED_ABORTED) is result
    assert actual_aborted_ids(records) == []
    assert (
        events(records, "finish_requests_return")[-1]["requests"][0]["request"]["status"]
        == "RUNNING"
    )
    assert trace.unknown is None


def test_native_exception_identity_and_no_post_return_ack():
    scheduler, trace, records = observed()
    error = RuntimeError("native failure")
    scheduler.exception = error
    with pytest.raises(RuntimeError) as caught:
        scheduler.finish_requests("request", Status.FINISHED_ABORTED)
    assert caught.value is error
    assert [call[0] for call in scheduler.calls].count("finish") == 1
    assert events(records, "finish_requests_return") == []
    assert trace.unknown is None


def test_trace_factory_failure_does_not_skip_native_initialization():
    class BrokenTrace(SchedulerObserverMixin, NativeContract):
        def _make_observer_trace(self):
            raise OSError("cannot initialize trace")

    scheduler = BrokenTrace(SimpleNamespace(async_scheduling=False), marker="native")
    assert scheduler.calls == [("init", "native")]
    assert scheduler._smg_pd_trace.unknown == "setup:OSError"


@pytest.mark.parametrize("actual", [None, True, 0, "false"])
def test_wrong_or_unresolved_scheduler_class_is_unknown(actual):
    scheduler, trace, records = observed()
    scheduler.scheduler_config.async_scheduling = actual

    # Exercise the same constructor guard with its actual resolved config.
    class Mismatched(SchedulerObserverMixin, NativeContract):
        _observer_expected_async = False

        def _make_observer_trace(self):
            return SchedulerTrace(records.append, {})

    wrong = Mismatched(scheduler.scheduler_config)
    assert wrong.scheduler_config.async_scheduling == actual
    assert wrong._smg_pd_trace.unknown == "startup:ValueError"


@pytest.mark.parametrize(
    "missing", ["request_id", "status", "num_output_tokens", "abort_immediately"]
)
def test_missing_request_fields_make_observation_unknown_without_changing_native_add(missing):
    scheduler, trace, records = observed()
    req = request()
    if missing == "request_id":
        req.request_id = ""
    else:
        delattr(req, missing)
    scheduler.add_request(req)
    assert scheduler.requests[req.request_id] is req
    assert trace.unknown is not None
    assert events(records, "add_request_return") == []


def test_trace_failure_is_sticky_and_does_not_replace_native_exception():
    def failed_sink(line):
        raise OSError("disk full")

    scheduler, trace, records = observed(sink=failed_sink)
    assert trace.unknown == "startup:OSError"
    error = ValueError("native")
    scheduler.exception = error
    with pytest.raises(ValueError) as caught:
        scheduler.finish_requests("id", Status.FINISHED_ABORTED)
    assert caught.value is error
    assert records == []


def test_snapshots_keep_exact_ids_and_primitives_without_retaining_request():
    scheduler, trace, records = observed()
    req = request("chatcmpl-other-attempt-ffffffff")
    scheduler.add_request(req)
    scheduler.finish_requests(req.request_id, Status.FINISHED_ABORTED)
    req.request_id = "mutated-after-event"
    req.num_output_tokens = 99
    assert actual_aborted_ids(records) == ["chatcmpl-other-attempt-ffffffff"]
    assert "chatcmpl-nonce-0123abcd" not in actual_aborted_ids(records)
    assert (
        events(records, "finish_requests_return")[-1]["requests"][0]["request"]["num_output_tokens"]
        == 2
    )
    assert all(json.loads(line)["attempt"] == "1" for line in records)
    assert all("accepted" not in json.loads(line) for line in records)
    assert trace.unknown is None
