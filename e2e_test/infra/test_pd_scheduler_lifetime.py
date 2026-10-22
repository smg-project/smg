"""CPU contract checks of the observer's per-registration scheduling marker."""

from collections import deque

from infra.pd_scheduler_test_support import Status, events, observed, request


def test_same_id_fresh_request_is_observed_after_native_registration_is_removed():
    scheduler, trace, records = observed()
    original = request()
    scheduler.add_request(original)
    scheduler.schedule_result.num_scheduled_tokens = {original.request_id: 1}
    scheduler.schedule()
    scheduler.finish_requests(original.request_id, Status.FINISHED_ABORTED)
    assert original.request_id not in scheduler.requests
    assert scheduler._smg_pd_scheduled == set()
    replacement = request(original.request_id)
    scheduler.add_request(replacement)
    scheduler.schedule()
    assert scheduler.requests[original.request_id] is replacement
    assert len(events(records, "first_scheduled")) == 2
    assert [call[0] for call in scheduler.calls].count("schedule") == 2
    assert trace.unknown is None


def test_connector_hold_does_not_clear_original_registration_marker():
    scheduler, trace, records = observed()
    original = request()
    scheduler.add_request(original)
    scheduler.schedule_result.num_scheduled_tokens = {original.request_id: 1}
    scheduler.schedule()
    scheduler.connector_holds = True
    scheduler.finish_requests(original.request_id, Status.FINISHED_ABORTED)
    assert scheduler.requests[original.request_id] is original
    assert scheduler._smg_pd_scheduled == {original.request_id}
    assert len(events(records, "first_scheduled")) == 1
    assert events(records, "free_blocks_return") == []
    assert trace.unknown is None


def test_streaming_update_keeps_original_registration_and_first_schedule_marker():
    scheduler, trace, records = observed()
    original = request()
    original.streaming_queue = deque()
    scheduler.add_request(original)
    scheduler.schedule_result.num_scheduled_tokens = {original.request_id: 1}
    scheduler.schedule()
    update = request(original.request_id)
    scheduler.add_request(update)
    scheduler.schedule()
    assert scheduler.requests[original.request_id] is original
    assert scheduler.streaming_updates == [original.request_id]
    assert list(original.streaming_queue) == [update]
    assert scheduler._smg_pd_scheduled == {original.request_id}
    assert len(events(records, "first_scheduled")) == 1
    assert events(records, "add_request_return")[-1]["registered"] is False
    assert trace.unknown is None
