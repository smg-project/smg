"""Delegation hooks; CPU tests use contract stubs, not a running vLLM backend."""

from .pd_scheduler_trace import SchedulerTrace, request_snapshot


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
        super().__init__(*args, **kwargs)

        def startup():
            actual = self.scheduler_config.async_scheduling
            if type(actual) is not bool or actual is not self._observer_expected_async:
                raise ValueError("observer does not match resolved default scheduler")
            return {
                "async_scheduling": actual,
                "mro": [f"{cls.__module__}.{cls.__qualname__}" for cls in type(self).__mro__],
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

    def add_request(self, request):
        result = super().add_request(request)
        self._observe("add_request_return", lambda: self._request_record(request))
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
