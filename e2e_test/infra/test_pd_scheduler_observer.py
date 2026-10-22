"""CPU contract-stub checks; these do not import or execute the vLLM backend."""

import importlib
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
from infra.pd_scheduler_observer import SchedulerObserverMixin
from infra.pd_scheduler_test_support import (
    AsyncNativeContract,
    NativeContract,
    Status,
    actual_aborted_ids,
    events,
    observed,
    request,
)
from infra.pd_scheduler_trace import SchedulerTrace


@pytest.mark.parametrize(
    "base,asynchronous", [(NativeContract, False), (AsyncNativeContract, True)]
)
def test_delegation_preserves_native_order_return_identity_and_async_path(base, asynchronous):
    instrumented, trace, records = observed(base, asynchronous)
    native = base(SimpleNamespace(async_scheduling=asynchronous), marker=instrumented.marker)
    req, plain_req = request(), request()
    instrumented.add_request(req)
    native.add_request(plain_req)
    instrumented.schedule_result.num_scheduled_tokens = {req.request_id: 1}
    native.schedule_result.num_scheduled_tokens = {plain_req.request_id: 1}
    assert instrumented.schedule(throttle_prefills=True) is instrumented.schedule_result
    native.schedule(throttle_prefills=True)
    assert (
        instrumented.finish_requests(req.request_id, Status.FINISHED_ABORTED)
        is instrumented.finished
    )
    native.finish_requests(plain_req.request_id, Status.FINISHED_ABORTED)
    assert instrumented.calls == native.calls
    assert instrumented.requests == native.requests == {}
    assert instrumented.freed == native.freed
    assert trace.unknown is None
    assert actual_aborted_ids(records) == [req.request_id]
    assert events(records, "first_scheduled")[0]["requests"][0]["scheduled_tokens"] == 1
    assert [json.loads(line)["seq"] for line in records] == list(range(1, len(records) + 1))


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


@pytest.mark.parametrize(
    "method,args",
    [
        ("add_request", (request(),)),
        ("schedule", ()),
        ("_free_request", (request(),)),
        ("_free_blocks", (request(),)),
        ("_free_request_blocks", (request(),)),
        ("_drain_deferred_frees", ()),
    ],
)
def test_all_other_hooks_preserve_native_exception_and_call_once(monkeypatch, method, args):
    error = RuntimeError("original native exception")
    calls = []

    def fail(self, *actual_args, **kwargs):
        calls.append((actual_args, kwargs))
        raise error

    monkeypatch.setattr(NativeContract, method, fail)
    scheduler, trace, records = observed()
    with pytest.raises(RuntimeError) as caught:
        getattr(scheduler, method)(*args)
    assert caught.value is error
    assert len(calls) == 1
    assert calls[0][0] == args
    assert len(records) == 1  # Startup is the only completed observation.
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


def test_connector_hold_and_deferred_fence_are_distinct_from_request_removal():
    scheduler, trace, records = observed()
    req = request()
    scheduler.add_request(req)
    scheduler.connector_holds = True
    scheduler.finish_requests(req.request_id, Status.FINISHED_ABORTED)
    assert events(records, "free_request_return")[-1]["registered"] is True
    assert events(records, "free_request_blocks_return") == []
    scheduler.defer = True
    scheduler._free_blocks(req)
    assert events(records, "free_blocks_return")[-1]["registered"] is False
    deferred = events(records, "free_request_blocks_return")[-1]
    assert deferred["deferred_after"] == [{"fence": 3, "blocks": [7]}]
    scheduler._drain_deferred_frees()
    assert (
        events(records, "drain_deferred_return")[-1]["deferred_after"] == deferred["deferred_after"]
    )
    scheduler.processed_step_seq = 3
    scheduler._drain_deferred_frees()
    drained = events(records, "drain_deferred_return")[-1]
    assert drained["deferred_after"] == []
    assert drained["transfer_ack"] is None and drained["peer_memory_released"] is None
    assert trace.unknown is None


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


def test_real_entry_module_keeps_both_native_parent_classes(monkeypatch):
    # Import the actual entry module against explicit contract stubs, not vLLM.
    monkeypatch.setattr(
        "infra.pd_scheduler_compat.require_installed_contract", lambda: {"stub": "cpu"}
    )
    monkeypatch.setattr("infra.pd_scheduler_compat.require_loaded_contract", lambda *args: None)
    for name, base in (("scheduler", NativeContract), ("async_scheduler", AsyncNativeContract)):
        module = ModuleType(f"vllm.v1.core.sched.{name}")
        setattr(module, "Scheduler" if name == "scheduler" else "AsyncScheduler", base)
        monkeypatch.setitem(sys.modules, module.__name__, module)
    sys.modules.pop("infra.pd_vllm_scheduler", None)
    entry = importlib.import_module("infra.pd_vllm_scheduler")
    try:
        assert entry.ObservedScheduler.__mro__[2] is NativeContract
        assert entry.ObservedAsyncScheduler.__mro__[2] is AsyncNativeContract

        class CustomAsync(AsyncNativeContract):
            pass

        for native, expected in (
            (NativeContract, entry.ObservedScheduler),
            (AsyncNativeContract, entry.ObservedAsyncScheduler),
        ):
            config = SimpleNamespace(
                scheduler_config=SimpleNamespace(
                    async_scheduling=True, get_scheduler_cls=lambda: native
                )
            )
            assert entry.observer_class_for_config(config) is expected
        for unsupported in (None, False, True, "true", entry.ObservedScheduler, CustomAsync):
            config = SimpleNamespace(
                scheduler_config=SimpleNamespace(
                    async_scheduling=True, get_scheduler_cls=lambda: unsupported
                )
            )
            with pytest.raises(ValueError):
                entry.observer_class_for_config(config)
        for unresolved in (None, 0, 1, "true"):
            config = SimpleNamespace(
                scheduler_config=SimpleNamespace(
                    async_scheduling=unresolved,
                    get_scheduler_cls=lambda: pytest.fail("unresolved config must not call getter"),
                )
            )
            with pytest.raises(ValueError, match="unresolved native scheduler configuration"):
                entry.observer_class_for_config(config)
    finally:
        sys.modules.pop("infra.pd_vllm_scheduler", None)
