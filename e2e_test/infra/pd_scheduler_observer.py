"""Delegation hooks; CPU tests use contract stubs, not a running vLLM backend."""

from .pd_local_kv_compat import enabled as local_kv_enabled
from .pd_scheduler_trace import (
    SchedulerTrace,
    deferred_snapshot,
    nonnegative_int,
    request_snapshot,
)


class SchedulerObserverMixin:
    _observer_expected_async = None

    def _make_observer_trace(self):
        return SchedulerTrace.from_env()

    def __init__(self, *args, **kwargs):
        try:
            self._smg_pd_trace = self._make_observer_trace()
        except Exception as exc:
            self._smg_pd_trace = SchedulerTrace(None, {})
            self._smg_pd_trace.mark_unknown("setup", exc)
        self._smg_pd_scheduled = set()
        self._smg_local_kv = None
        if local_kv_enabled():
            from .pd_local_kv_tracker import LocalKVTracker

            self._smg_local_kv = LocalKVTracker(self)
        super().__init__(*args, **kwargs)

        def startup():
            actual = self.scheduler_config.async_scheduling
            if type(actual) is not bool or actual is not self._observer_expected_async:
                raise ValueError("observer does not match resolved default scheduler")
            return {
                "async_scheduling": actual,
                "mro": [f"{cls.__module__}.{cls.__qualname__}" for cls in type(self).__mro__],
                "source_contract": getattr(self, "_observer_source_contract", None),
            }

        self._observe("startup", startup)

    def _capture(self, phase, read):
        try:
            return read()
        except Exception as exc:
            self._smg_pd_trace.mark_unknown(phase, exc)
            return None

    def _observe(self, event, read):
        fields = self._capture(event, read)
        if fields is not None:
            self._smg_pd_trace.emit(event, **fields)

    def _request_record(self, request):
        return {
            "request": request_snapshot(request),
            "registered": self.requests.get(request.request_id) is request,
        }

    def _local_kv(self, method, *args):
        tracker = self._smg_local_kv
        if tracker is None or tracker.unknown:
            return None
        try:
            return getattr(tracker, method)(*args)
        except Exception as exc:
            tracker.unknown = True
            tracker.pending.clear()
            self._smg_pd_trace.emit("local_kv_unknown", phase=method, error=type(exc).__name__)
            return None

    def _local_kv_records(self, method, *args):
        for fields in self._local_kv(method, *args) or ():
            self._smg_pd_trace.emit("local_kv_return", **fields)

    def add_request(self, request):
        result = super().add_request(request)
        self._observe("add_request_return", lambda: self._request_record(request))
        return result

    def schedule(self, *args, **kwargs):
        result = super().schedule(*args, **kwargs)

        def first_scheduled():
            records = []
            for request_id, count in result.num_scheduled_tokens.items():
                if request_id not in self._smg_pd_scheduled:
                    records.append(
                        {
                            "request": request_snapshot(self.requests[request_id]),
                            "scheduled_tokens": nonnegative_int(count),
                        }
                    )
                    self._smg_pd_scheduled.add(request_id)
            return {"requests": records}

        fields = self._capture("schedule_return", first_scheduled)
        if fields is not None and fields["requests"]:
            self._smg_pd_trace.emit("first_scheduled", **fields)
        return result

    def finish_requests(self, request_ids, finished_status):
        # Do not consume a caller's one-shot request-ID iterable before delegation.
        result = super().finish_requests(request_ids, finished_status)

        def finished():
            if type(result) is not list:
                raise ValueError("unexpected native finish_requests result")
            return {"requests": [self._request_record(request) for request in result]}

        self._observe("finish_requests_return", finished)
        return result

    def _free_request(self, request, delay_free_blocks=False):
        result = super()._free_request(request, delay_free_blocks=delay_free_blocks)
        self._observe("free_request_return", lambda: self._request_record(request))
        return result

    def _free_blocks(self, request):
        result = super()._free_blocks(request)

        def after():
            if request.request_id not in self.requests:
                self._smg_pd_scheduled.discard(request.request_id)
            return self._request_record(request)

        self._observe("free_blocks_return", after)
        return result

    def _free_request_blocks(self, request):
        local = self._local_kv("before_free", request)
        before = self._capture("free_request_blocks_before", lambda: deferred_snapshot(self))
        result = super()._free_request_blocks(request)
        self._local_kv_records("after_free", local)

        def after():
            if before is None:
                raise ValueError("missing pre-free observation")
            return {
                **self._request_record(request),
                "deferred_before": before,
                "deferred_after": deferred_snapshot(self),
                "processed_step_seq": nonnegative_int(self.processed_step_seq),
                "sched_step_seq": nonnegative_int(self.sched_step_seq),
                "transfer_ack": None,
                "peer_memory_released": None,
            }

        self._observe("free_request_blocks_return", after)
        return result

    def _drain_deferred_frees(self):
        local = self._local_kv("before_drain")
        before = self._capture("drain_before", lambda: deferred_snapshot(self))
        result = super()._drain_deferred_frees()
        self._local_kv_records("after_drain", local or ())
        if before:
            self._observe(
                "drain_deferred_return",
                lambda: {
                    "deferred_before": before,
                    "deferred_after": deferred_snapshot(self),
                    "processed_step_seq": nonnegative_int(self.processed_step_seq),
                    "transfer_ack": None,
                    "peer_memory_released": None,
                },
            )
        return result
